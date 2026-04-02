from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch


PARQUET_ID_BATCH_SIZE = int(os.getenv("RAGAP_GRAPH_VALIDATE_PARQUET_BATCH_SIZE", "131072"))
EDGE_VALIDATE_CHUNK_SIZE = int(os.getenv("RAGAP_GRAPH_VALIDATE_EDGE_CHUNK_SIZE", "250000"))


def inputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["graph"]
    return [
        cfg["phage_catalog"],
        cfg["host_catalog"],
        cfg["protein_clusters"],
        cfg["taxonomy"],
        cfg["edge_dir"],
        cfg["pairs_train"],
        cfg["pairs_val"],
        cfg["pairs_test"],
    ]


def outputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["graph"]
    return [cfg["out"], cfg["map_out"]]


def params(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = config["graph"]
    ignore = {
        "mode",
        "script",
        "python",
        "phage_catalog",
        "host_catalog",
        "protein_clusters",
        "taxonomy",
        "edge_dir",
        "pairs_train",
        "pairs_val",
        "pairs_test",
        "out",
        "map_out",
        "validate",
        "deps",
    }
    return {key: value for key, value in cfg.items() if key not in ignore}


def script_path(config: dict[str, Any], stage_name: str) -> str:
    return str(config["graph"]["script"])


def command(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["graph"]
    return [
        cfg.get("python") or config.get("python_bin", "python"),
        cfg["script"],
        "--phage_catalog",
        cfg["phage_catalog"],
        "--host_catalog",
        cfg["host_catalog"],
        "--protein_clusters",
        cfg["protein_clusters"],
        "--taxonomy",
        cfg["taxonomy"],
        "--edge_dir",
        cfg["edge_dir"],
        "--pairs_train",
        cfg["pairs_train"],
        "--pairs_val",
        cfg["pairs_val"],
        "--pairs_test",
        cfg["pairs_test"],
        "--out",
        cfg["out"],
        "--map_out",
        cfg["map_out"],
    ]


def _load_id_set_from_parquet(path: str, column: str) -> set[str]:
    ids: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=PARQUET_ID_BATCH_SIZE,
        columns=[column],
        use_threads=True,
    ):
        ids.update(str(value) for value in batch.column(0).to_pylist() if value is not None)
    return ids


def _validate_edge_file(path: Path, allowed_src: set[str], allowed_dst: set[str]) -> None:
    has_rows = False
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=[0, 1],
        chunksize=EDGE_VALIDATE_CHUNK_SIZE,
    ):
        has_rows = True
        src_values = chunk.iloc[:, 0].astype(str)
        dst_values = chunk.iloc[:, 1].astype(str)

        invalid_src = src_values[~src_values.isin(allowed_src)]
        if not invalid_src.empty:
            raise RuntimeError(
                f"{path.name} contains ids missing from source node table: "
                f"{invalid_src.iloc[0]}"
            )

        invalid_dst = dst_values[~dst_values.isin(allowed_dst)]
        if not invalid_dst.empty:
            raise RuntimeError(
                f"{path.name} contains ids missing from destination node table: "
                f"{invalid_dst.iloc[0]}"
            )

    if not has_rows:
        return


def _validate_split_file(path: str, phage_ids: set[str], host_ids: set[str]) -> None:
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=[0, 1])
    if df.empty:
        raise RuntimeError(f"graph split file is empty: {path}")
    phage_series = df.iloc[:, 0].astype(str)
    host_series = df.iloc[:, 1].astype(str)
    invalid_phage = phage_series[~phage_series.isin(phage_ids)]
    if not invalid_phage.empty:
        raise RuntimeError(f"split file contains phage ids missing from phage catalog: {path}")
    invalid_host = host_series[~host_series.isin(host_ids)]
    if not invalid_host.empty:
        raise RuntimeError(f"split file contains host ids missing from host catalog: {path}")


def pre_run(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = config["graph"]
    phage_ids = _load_id_set_from_parquet(cfg["phage_catalog"], "phage_id")
    host_ids = _load_id_set_from_parquet(cfg["host_catalog"], "host_gcf")
    protein_ids = _load_id_set_from_parquet(cfg["protein_clusters"], "protein_id")
    taxonomy_ids = _load_id_set_from_parquet(cfg["taxonomy"], "taxid")
    edge_specs = {
        "phage_phage_edges.tsv": (phage_ids, phage_ids),
        "host_host_edges.tsv": (host_ids, host_ids),
        "phage_protein_edges.tsv": (phage_ids, protein_ids),
        "host_protein_edges.tsv": (host_ids, protein_ids),
        "protein_protein_edges.tsv": (protein_ids, protein_ids),
        "host_taxonomy_edges.tsv": (host_ids, taxonomy_ids),
        "phage_taxonomy_edges.tsv": (phage_ids, taxonomy_ids),
        "taxonomy_taxonomy_edges.tsv": (taxonomy_ids, taxonomy_ids),
    }
    for filename, (allowed_src, allowed_dst) in edge_specs.items():
        path = Path(cfg["edge_dir"]) / filename
        if not path.exists():
            raise RuntimeError(f"graph input edge file missing: {path}")
        _validate_edge_file(path, allowed_src, allowed_dst)
    for split_path in (cfg["pairs_train"], cfg["pairs_val"], cfg["pairs_test"]):
        _validate_split_file(split_path, phage_ids, host_ids)
    return {
        "phage_nodes": len(phage_ids),
        "host_nodes": len(host_ids),
        "protein_nodes": len(protein_ids),
        "taxonomy_nodes": len(taxonomy_ids),
    }


def _load_graph(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _incoming_edge_counts(data: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge_type in data.edge_types:
        dst_type = edge_type[2]
        edge_index = data[edge_type].edge_index
        counts[dst_type] = counts.get(dst_type, 0) + int(edge_index.shape[1])
    return counts


def _validate_reverse_relation(
    data: Any,
    forward_edge_type: tuple[str, str, str],
    reverse_edge_type: tuple[str, str, str],
) -> int:
    has_forward = forward_edge_type in data.edge_types
    has_reverse = reverse_edge_type in data.edge_types
    if has_forward != has_reverse:
        raise RuntimeError(
            f"graph reverse-edge mismatch: {forward_edge_type} and {reverse_edge_type} must either both exist or both be absent"
        )
    if not has_forward:
        return 0

    forward_edge_index = data[forward_edge_type].edge_index
    reverse_edge_index = data[reverse_edge_type].edge_index
    if forward_edge_index.shape[1] != reverse_edge_index.shape[1]:
        raise RuntimeError(
            f"graph reverse-edge count mismatch: {forward_edge_type} has {forward_edge_index.shape[1]} edges but {reverse_edge_type} has {reverse_edge_index.shape[1]}"
        )
    if not torch.equal(forward_edge_index[[1, 0], :], reverse_edge_index):
        raise RuntimeError(
            f"graph reverse-edge content mismatch: {reverse_edge_type} is not the exact flipped view of {forward_edge_type}"
        )
    return int(forward_edge_index.shape[1])


def post_run(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = config["graph"]
    graph_path = Path(cfg["out"])
    data = _load_graph(str(graph_path))
    incoming_counts = _incoming_edge_counts(data)
    zero_edge_types = [
        edge_type
        for edge_type in data.edge_types
        if int(data[edge_type].edge_index.shape[1]) == 0
    ]
    infects_edges = (
        data[("phage", "infects", "host")].edge_index.shape[1]
        if ("phage", "infects", "host") in data.edge_types
        else 0
    )

    if zero_edge_types:
        raise RuntimeError(
            f"graph contains zero-edge relations that should have been omitted: {zero_edge_types}"
        )
    if ("phage", "interacts", "host") in data.edge_types or ("host", "interacts", "phage") in data.edge_types:
        raise RuntimeError(
            "graph contains phage-host 'interacts' relations; phage-host supervision must use only ('phage', 'infects', 'host')"
        )
    if infects_edges == 0:
        raise RuntimeError(
            "graph script did not load phage-host supervision edges into ('phage', 'infects', 'host')"
        )

    reverse_edge_counts = {
        "host_sequence_reverse_edges": _validate_reverse_relation(
            data,
            ("host", "has_sequence", "host_sequence"),
            ("host_sequence", "sequence_of", "host"),
        ),
        "phage_protein_reverse_edges": _validate_reverse_relation(
            data,
            ("phage", "encodes", "protein"),
            ("protein", "encoded_by_phage", "phage"),
        ),
        "host_protein_reverse_edges": _validate_reverse_relation(
            data,
            ("host", "encodes", "protein"),
            ("protein", "encoded_by_host", "host"),
        ),
        "host_taxonomy_reverse_edges": _validate_reverse_relation(
            data,
            ("host", "belongs_to", "taxonomy"),
            ("taxonomy", "contains_host", "host"),
        ),
        "phage_taxonomy_reverse_edges": _validate_reverse_relation(
            data,
            ("phage", "belongs_to", "taxonomy"),
            ("taxonomy", "contains_phage", "phage"),
        ),
    }

    if incoming_counts.get("phage", 0) == 0:
        raise RuntimeError(
            "graph leaves phage without incoming message-passing edges; "
            "check reverse auxiliary edges plus phage_phage/phage_taxonomy assets"
        )

    return {
        "infects_edges": int(infects_edges),
        "incoming_edge_counts": incoming_counts,
        **reverse_edge_counts,
    }
