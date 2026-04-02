#!/usr/bin/env python3
"""
Build a PyG HeteroData graph with node features and split-aware phage-host edges.

This version avoids full-table pandas loads for the large protein parquet and
streams the larger edge TSVs in chunks to reduce peak memory while improving
throughput.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

try:
    import orjson
except ImportError:  # pragma: no cover - optional acceleration
    orjson = None


DEFAULT_PARQUET_BATCH_SIZE = int(os.getenv("RAGAP_GRAPH_PARQUET_BATCH_SIZE", "8192"))
DEFAULT_EDGE_CHUNK_SIZE = int(os.getenv("RAGAP_GRAPH_EDGE_CHUNK_SIZE", "250000"))
PROGRESS_EVERY_BATCHES = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phage_catalog", required=True)
    parser.add_argument("--host_catalog", required=True)
    parser.add_argument("--protein_clusters", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--edge_dir", required=True)
    parser.add_argument("--pairs_train", required=True)
    parser.add_argument("--pairs_val", required=True)
    parser.add_argument("--pairs_test", required=True)
    parser.add_argument("--out", default="hetero_graph_with_features_splits.pt")
    parser.add_argument("--map_out", default="node_maps.json")
    parser.add_argument("--parquet_batch_size", type=int, default=DEFAULT_PARQUET_BATCH_SIZE)
    parser.add_argument("--edge_chunk_size", type=int, default=DEFAULT_EDGE_CHUNK_SIZE)
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def df_to_emb_matrix(
    df: pd.DataFrame,
    id_col: str,
    emb_col: str = "embedding",
    prefix_dim: str = "dim_",
) -> tuple[list[str], np.ndarray, list[str], pd.DataFrame]:
    dim_cols = [column for column in df.columns if column.startswith(prefix_dim)]

    if emb_col in df.columns:
        ids = df[id_col].astype(str).tolist()
        raw = df[emb_col].tolist()
        kept_ids: list[str] = []
        rows: list[np.ndarray] = []
        dropped: list[str] = []

        for idx, value in enumerate(raw):
            if value is None:
                dropped.append(ids[idx])
                continue
            try:
                array = np.asarray(value, dtype=np.float32)
            except Exception:
                dropped.append(ids[idx])
                continue
            if array.ndim == 0:
                array = array.reshape(1)
            kept_ids.append(ids[idx])
            rows.append(array)

        if not rows:
            raise RuntimeError(f"No valid embeddings found in column {emb_col}")

        lengths = [row.shape[0] for row in rows]
        dim = max(lengths)
        if not all(length == dim for length in lengths):
            rows = [
                np.pad(row, (0, dim - row.shape[0]), mode="constant")
                if row.shape[0] < dim
                else row[:dim]
                for row in rows
            ]
        matrix = np.vstack(rows).astype(np.float32, copy=False)
        filtered = df[df[id_col].astype(str).isin(kept_ids)].copy()
        return kept_ids, matrix, dropped, filtered

    if dim_cols:
        ids = df[id_col].astype(str).tolist()
        matrix = df[dim_cols].to_numpy(dtype=np.float32, copy=False)
        return ids, matrix, [], df.copy()

    raise RuntimeError(
        f"Embedding column '{emb_col}' not found and no '{prefix_dim}*' columns in dataframe."
    )


def safe_load_small_parquet_embeddings(
    path: str,
    id_col: str,
    emb_col: str,
) -> tuple[list[str], torch.Tensor, list[str], pd.DataFrame]:
    df = pd.read_parquet(path)
    if id_col not in df.columns:
        raise RuntimeError(f"{path} missing id column '{id_col}'")
    ids, matrix, dropped, filtered = df_to_emb_matrix(df, id_col=id_col, emb_col=emb_col)
    return ids, torch.from_numpy(matrix), dropped, filtered


def _list_array_to_matrix(
    array: Any,
    expected_dim: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    size = len(array)
    if size == 0:
        if expected_dim is None:
            raise RuntimeError("Cannot infer embedding dimension from an empty batch")
        return np.empty((0, expected_dim), dtype=np.float32), np.zeros(0, dtype=bool), expected_dim

    lengths = array.value_lengths()
    if array.null_count:
        lengths = lengths.fill_null(-1)
    lengths_np = lengths.to_numpy(zero_copy_only=False)

    if expected_dim is None:
        valid_lengths = lengths_np[lengths_np > 0]
        if valid_lengths.size == 0:
            raise RuntimeError("Unable to infer embedding dimension from parquet batches")
        expected_dim = int(valid_lengths[0])

    valid_mask = lengths_np == expected_dim

    if valid_mask.all():
        values = array.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        return values.reshape(size, expected_dim), valid_mask, expected_dim

    valid_rows = [
        np.asarray(array[idx].as_py(), dtype=np.float32)
        for idx in np.flatnonzero(valid_mask)
    ]
    if valid_rows:
        matrix = np.vstack(valid_rows).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, expected_dim), dtype=np.float32)
    return matrix, valid_mask, expected_dim


def stream_large_parquet_embeddings(
    path: str,
    id_col: str,
    emb_col: str,
    batch_size: int,
    label: str,
) -> tuple[torch.Tensor, dict[str, int], int]:
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    expected_dim: int | None = None
    tensor: torch.Tensor | None = None
    id_map: dict[str, int] = {}
    write_pos = 0
    dropped = 0

    for batch_index, batch in enumerate(
        parquet.iter_batches(
            batch_size=batch_size,
            columns=[id_col, emb_col],
            use_threads=True,
        ),
        start=1,
    ):
        batch_ids = [str(value) for value in batch.column(0).to_pylist()]
        batch_matrix, valid_mask, expected_dim = _list_array_to_matrix(
            batch.column(1), expected_dim
        )

        if tensor is None:
            tensor = torch.empty((total_rows, expected_dim), dtype=torch.float32)

        if valid_mask.all():
            batch_len = len(batch_ids)
            tensor[write_pos : write_pos + batch_len].copy_(torch.from_numpy(batch_matrix))
            id_map.update(zip(batch_ids, range(write_pos, write_pos + batch_len)))
            write_pos += batch_len
        else:
            valid_indices = np.flatnonzero(valid_mask)
            valid_ids = [batch_ids[idx] for idx in valid_indices]
            valid_count = len(valid_ids)
            dropped += len(batch_ids) - valid_count
            if valid_count:
                tensor[write_pos : write_pos + valid_count].copy_(torch.from_numpy(batch_matrix))
                id_map.update(zip(valid_ids, range(write_pos, write_pos + valid_count)))
                write_pos += valid_count

        if batch_index % PROGRESS_EVERY_BATCHES == 0:
            print(
                f"  {label}: processed {write_pos:,}/{total_rows:,} rows "
                f"(dropped {dropped:,})"
            )

    if tensor is None or expected_dim is None:
        raise RuntimeError(f"No embeddings loaded from {path}")
    if write_pos == 0:
        raise RuntimeError(f"All embeddings were dropped from {path}")
    if write_pos != total_rows:
        tensor = tensor[:write_pos].clone()

    print(
        f"  {label}: kept {write_pos:,}, dropped {dropped:,}, dim={tensor.shape[1]}"
    )
    return tensor, id_map, dropped


def load_host_nodes(
    host_catalog: str,
) -> tuple[
    list[str],
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
    dict[str, int],
    torch.Tensor,
    torch.Tensor,
]:
    print("Loading host embeddings from", host_catalog)
    host_df = pd.read_parquet(
        host_catalog,
        columns=["host_gcf", "sequence_id", "host_species_taxid", "host_dna_emb"],
    ).copy()
    host_df["host_gcf"] = host_df["host_gcf"].astype(str)
    host_df["sequence_id"] = host_df["sequence_id"].astype(str)

    host_unique = host_df[["host_gcf", "host_species_taxid"]].drop_duplicates("host_gcf")
    host_ids = host_unique["host_gcf"].tolist()
    host_map = {host_id: idx for idx, host_id in enumerate(host_ids)}

    sequence_source = host_df.drop_duplicates("sequence_id", keep="first").reset_index(drop=True)
    sequence_ids, sequence_mat, seq_dropped, _ = df_to_emb_matrix(
        sequence_source,
        id_col="sequence_id",
        emb_col="host_dna_emb",
    )
    sequence_map = {sequence_id: idx for idx, sequence_id in enumerate(sequence_ids)}
    sequence_x = torch.from_numpy(sequence_mat)

    host_index = pd.Index(host_ids)
    host_idx = host_index.get_indexer(host_df["host_gcf"].to_numpy())
    seq_idx_series = host_df["sequence_id"].map(sequence_map)
    valid_links = (host_idx >= 0) & seq_idx_series.notna().to_numpy()
    host_edge_src = host_idx[valid_links].astype(np.int64, copy=False)
    host_edge_dst = seq_idx_series[valid_links].astype(np.int64).to_numpy(copy=False)

    host_x_np = np.zeros((len(host_ids), sequence_mat.shape[1]), dtype=np.float32)
    counts = np.zeros(len(host_ids), dtype=np.int64)
    np.add.at(host_x_np, host_edge_src, sequence_mat[host_edge_dst])
    np.add.at(counts, host_edge_src, 1)
    nonzero_mask = counts > 0
    if nonzero_mask.any():
        host_x_np[nonzero_mask] /= counts[nonzero_mask, None]

    host_taxid = pd.to_numeric(
        host_unique["host_species_taxid"], errors="coerce"
    ).fillna(-1)
    host_taxid_tensor = torch.from_numpy(host_taxid.to_numpy(dtype=np.int64, copy=False))
    host_x = torch.from_numpy(host_x_np)
    host_sequence_edge_index = (
        torch.from_numpy(np.vstack([host_edge_src, host_edge_dst]))
        if host_edge_src.size
        else torch.empty((2, 0), dtype=torch.long)
    ).long()

    print(
        "  host: host_gcf count="
        f"{len(host_ids):,}, host_sequence count={len(sequence_ids):,}, "
        f"dropped_sequences={len(seq_dropped):,}"
    )
    print(
        f"  host: zero-feature nodes={(~nonzero_mask).sum():,}/{len(host_ids):,}, "
        f"dim={host_x.shape[1]}"
    )
    print(f"  host-sequence edges: {host_sequence_edge_index.shape[1]:,}")

    return (
        host_ids,
        host_x,
        host_taxid_tensor,
        host_map,
        sequence_map,
        sequence_x,
        host_sequence_edge_index,
    )


def load_edge_index(
    path: str,
    src_map: dict[str, int],
    dst_map: dict[str, int],
    chunk_size: int,
    *,
    reverse: bool = False,
    label: str,
) -> tuple[torch.Tensor, int]:
    if not os.path.exists(path):
        print(f"  WARNING: edge file {path} not found, skipping relation {label}")
        return torch.empty((2, 0), dtype=torch.long), 0

    src_chunks: list[torch.Tensor] = []
    dst_chunks: list[torch.Tensor] = []
    dropped = 0
    kept = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1],
        dtype=str,
        chunksize=chunk_size,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        src_series = chunk.iloc[:, 1] if reverse else chunk.iloc[:, 0]
        dst_series = chunk.iloc[:, 0] if reverse else chunk.iloc[:, 1]

        src_idx = src_series.map(src_map)
        dst_idx = dst_series.map(dst_map)
        valid_mask = src_idx.notna() & dst_idx.notna()

        dropped += int((~valid_mask).sum())
        if valid_mask.any():
            src_chunks.append(
                torch.from_numpy(
                    src_idx[valid_mask].astype(np.int64).to_numpy(copy=False)
                )
            )
            dst_chunks.append(
                torch.from_numpy(
                    dst_idx[valid_mask].astype(np.int64).to_numpy(copy=False)
                )
            )
            kept += int(valid_mask.sum())

        if chunk_index % PROGRESS_EVERY_BATCHES == 0:
            print(f"  {label}: kept {kept:,} edges so far (dropped {dropped:,})")

    if not src_chunks:
        print(f"  {label}: produced 0 edges (all dropped)")
        return torch.empty((2, 0), dtype=torch.long), dropped

    edge_index = torch.vstack([torch.cat(src_chunks), torch.cat(dst_chunks)]).long()
    print(f"  {label}: added {edge_index.shape[1]:,} edges (dropped {dropped:,})")
    return edge_index, dropped


def load_split_pairs(
    path: str,
    phage_map: dict[str, int],
    host_map: dict[str, int],
    chunk_size: int,
    label: str,
) -> tuple[torch.Tensor, int]:
    if not os.path.exists(path):
        return torch.empty((2, 0), dtype=torch.long), 0

    src_chunks: list[torch.Tensor] = []
    dst_chunks: list[torch.Tensor] = []
    dropped = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1],
        dtype=str,
        chunksize=chunk_size,
    )
    for chunk in reader:
        src_idx = chunk.iloc[:, 0].map(phage_map)
        dst_idx = chunk.iloc[:, 1].map(host_map)
        valid_mask = src_idx.notna() & dst_idx.notna()
        dropped += int((~valid_mask).sum())
        if valid_mask.any():
            src_chunks.append(
                torch.from_numpy(
                    src_idx[valid_mask].astype(np.int64).to_numpy(copy=False)
                )
            )
            dst_chunks.append(
                torch.from_numpy(
                    dst_idx[valid_mask].astype(np.int64).to_numpy(copy=False)
                )
            )

    if not src_chunks:
        print(f"  {label}: 0 edges")
        return torch.empty((2, 0), dtype=torch.long), dropped

    edge_index = torch.vstack([torch.cat(src_chunks), torch.cat(dst_chunks)]).long()
    print(f"  {label}: {edge_index.shape[1]:,} edges (dropped {dropped:,})")
    return edge_index, dropped


def unique_edge_index(*edge_indices: torch.Tensor) -> torch.Tensor:
    nonempty = [edge_index for edge_index in edge_indices if edge_index.numel()]
    if not nonempty:
        return torch.empty((2, 0), dtype=torch.long)
    merged = torch.cat(nonempty, dim=1)
    unique_pairs = torch.unique(merged.t(), dim=0)
    return unique_pairs.t().contiguous()


def assign_edge_index(data: HeteroData, edge_type: tuple[str, str, str], edge_index: torch.Tensor) -> bool:
    if edge_index.numel() == 0:
        return False
    data[edge_type].edge_index = edge_index
    return True


def assign_reverse_edge_index(
    data: HeteroData,
    forward_edge_type: tuple[str, str, str],
    reverse_edge_type: tuple[str, str, str],
) -> bool:
    if forward_edge_type not in data.edge_types:
        return False
    forward_edge_index = data[forward_edge_type].edge_index
    if forward_edge_index.numel() == 0:
        return False
    data[reverse_edge_type].edge_index = forward_edge_index[[1, 0], :].contiguous()
    return True


def write_json(path: str, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    if orjson is not None:
        with open(path, "wb") as handle:
            handle.write(orjson.dumps(payload))
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def build_hetero(args: argparse.Namespace) -> HeteroData:
    ensure_parent_dir(args.out)
    ensure_parent_dir(args.map_out)

    print("Loading phage embeddings from", args.phage_catalog)
    phage_ids, phage_x, phage_dropped, _ = safe_load_small_parquet_embeddings(
        args.phage_catalog,
        id_col="phage_id",
        emb_col="phage_dna_emb",
    )
    phage_map = {phage_id: idx for idx, phage_id in enumerate(phage_ids)}
    print(f"  phage: kept {len(phage_ids):,}, dropped {len(phage_dropped):,}")

    print("Loading protein cluster embeddings from", args.protein_clusters)
    protein_x, protein_map, protein_dropped = stream_large_parquet_embeddings(
        args.protein_clusters,
        id_col="protein_id",
        emb_col="embedding",
        batch_size=args.parquet_batch_size,
        label="protein",
    )

    print("Loading taxonomy embeddings from", args.taxonomy)
    tax_ids, taxonomy_x, tax_dropped, _ = safe_load_small_parquet_embeddings(
        args.taxonomy,
        id_col="taxid",
        emb_col="tangent_emb",
    )
    tax_map = {str(tax_id): idx for idx, tax_id in enumerate(tax_ids)}
    print(f"  taxonomy: kept {len(tax_ids):,}, dropped {len(tax_dropped):,}")

    (
        host_ids,
        host_x,
        host_taxid,
        host_map,
        sequence_map,
        sequence_x,
        host_sequence_edge_index,
    ) = load_host_nodes(args.host_catalog)

    data = HeteroData()
    data["phage"].x = phage_x
    data["host"].x = host_x
    data["host"].taxid = host_taxid
    data["host_sequence"].x = sequence_x
    data["protein"].x = protein_x
    data["taxonomy"].x = taxonomy_x
    assign_edge_index(data, ("host", "has_sequence", "host_sequence"), host_sequence_edge_index)
    assign_reverse_edge_index(
        data,
        ("host", "has_sequence", "host_sequence"),
        ("host_sequence", "sequence_of", "host"),
    )

    print(
        "Node counts: phage "
        f"{data['phage'].x.shape[0]:,}, host {data['host'].x.shape[0]:,}, "
        f"host_sequence {data['host_sequence'].x.shape[0]:,}, "
        f"protein {data['protein'].x.shape[0]:,}, taxonomy {data['taxonomy'].x.shape[0]:,}"
    )

    edge_specs = [
        (
            ("phage", "interacts", "phage"),
            os.path.join(args.edge_dir, "phage_phage_edges.tsv"),
            phage_map,
            phage_map,
            False,
        ),
        (
            ("host", "interacts", "host"),
            os.path.join(args.edge_dir, "host_host_edges.tsv"),
            host_map,
            host_map,
            False,
        ),
        (
            ("phage", "encodes", "protein"),
            os.path.join(args.edge_dir, "phage_protein_edges.tsv"),
            phage_map,
            protein_map,
            False,
        ),
        (
            ("host", "encodes", "protein"),
            os.path.join(args.edge_dir, "host_protein_edges.tsv"),
            host_map,
            protein_map,
            False,
        ),
        (
            ("protein", "similar", "protein"),
            os.path.join(args.edge_dir, "protein_protein_edges.tsv"),
            protein_map,
            protein_map,
            False,
        ),
        (
            ("host", "belongs_to", "taxonomy"),
            os.path.join(args.edge_dir, "host_taxonomy_edges.tsv"),
            host_map,
            tax_map,
            False,
        ),
        (
            ("taxonomy", "related", "taxonomy"),
            os.path.join(args.edge_dir, "taxonomy_taxonomy_edges.tsv"),
            tax_map,
            tax_map,
            False,
        ),
        (
            ("phage", "belongs_to", "taxonomy"),
            os.path.join(args.edge_dir, "phage_taxonomy_edges.tsv"),
            phage_map,
            tax_map,
            False,
        ),
    ]

    for edge_type, path, src_map, dst_map, reverse in edge_specs:
        edge_index, _ = load_edge_index(
            path,
            src_map=src_map,
            dst_map=dst_map,
            chunk_size=args.edge_chunk_size,
            reverse=reverse,
            label=str(edge_type),
        )
        assign_edge_index(data, edge_type, edge_index)

    assign_reverse_edge_index(
        data,
        ("phage", "encodes", "protein"),
        ("protein", "encoded_by_phage", "phage"),
    )
    assign_reverse_edge_index(
        data,
        ("host", "encodes", "protein"),
        ("protein", "encoded_by_host", "host"),
    )
    assign_reverse_edge_index(
        data,
        ("host", "belongs_to", "taxonomy"),
        ("taxonomy", "contains_host", "host"),
    )
    assign_reverse_edge_index(
        data,
        ("phage", "belongs_to", "taxonomy"),
        ("taxonomy", "contains_phage", "phage"),
    )

    print("Processing phage-host train/val/test splits...")
    train_edge_index, train_dropped = load_split_pairs(
        args.pairs_train, phage_map, host_map, args.edge_chunk_size, "train split"
    )
    val_edge_index, val_dropped = load_split_pairs(
        args.pairs_val, phage_map, host_map, args.edge_chunk_size, "val split"
    )
    test_edge_index, test_dropped = load_split_pairs(
        args.pairs_test, phage_map, host_map, args.edge_chunk_size, "test split"
    )
    if train_dropped or val_dropped or test_dropped:
        print(
            "  dropped split edges because nodes were missing: "
            f"train={train_dropped:,}, val={val_dropped:,}, test={test_dropped:,}"
        )

    infects_edge_index = unique_edge_index(train_edge_index, val_edge_index, test_edge_index)
    assign_edge_index(data, ("phage", "infects", "host"), infects_edge_index)
    data[("phage", "infects", "host")].edge_index_train = train_edge_index
    data[("phage", "infects", "host")].edge_index_val = val_edge_index
    data[("phage", "infects", "host")].edge_index_test = test_edge_index
    print(f"  infects union edges: {infects_edge_index.shape[1]:,}")

    node_maps = {
        "phage_map": phage_map,
        "host_map": host_map,
        "host_sequence_map": sequence_map,
        "protein_map": protein_map,
        "tax_map": tax_map,
    }
    write_json(args.map_out, node_maps)
    print("Saved node maps to", args.map_out)

    torch.save(data, args.out)
    print("Saved HeteroData to", args.out)
    print(f"Protein embeddings dropped: {protein_dropped:,}")
    return data


if __name__ == "__main__":
    build_hetero(parse_args())
