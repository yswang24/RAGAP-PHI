from __future__ import annotations

import csv
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..execution import conda_bin
from ..utils import ensure_dir, iter_fasta_files, list_files, load_pickle, to_float_list


EDGE_FILENAMES = [
    "phage_phage_edges.tsv",
    "host_host_edges.tsv",
    "phage_host_edges.tsv",
    "phage_protein_edges.tsv",
    "host_protein_edges.tsv",
    "protein_protein_edges.tsv",
    "host_taxonomy_edges.tsv",
    "phage_taxonomy_edges.tsv",
    "taxonomy_taxonomy_edges.tsv",
]

PROTEIN_SCHEMA = pa.schema(
    [
        pa.field("protein_id", pa.string()),
        pa.field("source_type", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("embedding", pa.list_(pa.float64())),
    ]
)
PHAGE_EDGE_SCHEMA = pa.schema(
    [
        pa.field("phage_id", pa.string()),
        pa.field("protein_id", pa.string()),
    ]
)
HOST_EDGE_SCHEMA = pa.schema(
    [
        pa.field("host_id", pa.string()),
        pa.field("protein_id", pa.string()),
    ]
)


def inputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["cluster_assets"]
    values = [
        cfg["phage_embedding_dir"],
        cfg["host_embedding_dir"],
        config["inputs"]["phage_fasta_dir"],
        config["inputs"]["host_fasta_dir"],
        config["build_catalogs"]["phage_catalog"],
        config["build_catalogs"]["host_catalog"],
        config["inputs"]["taxonomy_graph_parquet"],
    ]
    values.extend(str(path) for path in cfg.get("cluster_member_tables", []))
    for value in cfg.get("edge_sources", {}).values():
        if value:
            values.append(str(value))
    return values


def outputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["cluster_assets"]
    return [
        cfg["protein_catalog_out"],
        cfg["phage_protein_edges_out"],
        cfg["host_protein_edges_out"],
        cfg["cluster_protein_catalog_out"],
        cfg["edge_dir"],
    ]


def params(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = config["cluster_assets"]
    ignore = {
        "mode",
        "phage_embedding_dir",
        "host_embedding_dir",
        "protein_catalog_out",
        "phage_protein_edges_out",
        "host_protein_edges_out",
        "cluster_protein_catalog_out",
        "edge_dir",
        "edge_sources",
        "validate",
        "deps",
    }
    return {key: value for key, value in cfg.items() if key not in ignore}


def script_path(config: dict[str, Any], stage_name: str) -> str:
    return str(Path(__file__).resolve())


def _int_setting(value: Any, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = default
    resolved = max(minimum, resolved)
    if maximum is not None:
        resolved = min(maximum, resolved)
    return resolved


def _available_cpu_count() -> int:
    for env_name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            continue
    return max(1, os.cpu_count() or 1)


class _BufferedParquetWriter:
    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        batch_rows: int,
        row_group_size: int,
        compression: str,
    ) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        if self.path.exists():
            self.path.unlink()
        self._schema = schema
        self._batch_rows = max(1, batch_rows)
        self._row_group_size = max(1, row_group_size)
        self._buffer: list[dict[str, Any]] = []
        self._rows_written = 0
        self._closed = False
        self._writer = pq.ParquetWriter(
            self.path,
            schema,
            compression=compression,
            use_dictionary=False,
            write_statistics=False,
        )

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def write_row(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table, row_group_size=self._row_group_size)
        self._rows_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        if self._closed:
            return
        if self._buffer or self._rows_written == 0:
            table = pa.Table.from_pylist(self._buffer, schema=self._schema)
            self._writer.write_table(table, row_group_size=self._row_group_size)
            self._rows_written += len(self._buffer)
            self._buffer.clear()
        self._writer.close()
        self._closed = True


def _open_edge_tsv_writer(path: Path) -> tuple[Any, csv.writer]:
    ensure_dir(path.parent)
    handle = path.open("w", encoding="utf-8", newline="", buffering=1024 * 1024)
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["src_id", "dst_id", "edge_type", "weight"])
    return handle, writer


def _parquet_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    batch_rows = _int_setting(cfg.get("parquet_batch_rows"), default=2048)
    row_group_size = _int_setting(cfg.get("parquet_row_group_size"), default=batch_rows)
    compression = str(cfg.get("parquet_compression", "snappy"))
    return {
        "batch_rows": batch_rows,
        "row_group_size": row_group_size,
        "compression": compression,
    }


def _load_subset_ids(paths: list[str]) -> set[str]:
    if not paths:
        return set()
    result: set[str] = set()
    candidate_columns = ("Sequence_ID", "protein_id", "seq_id")
    for raw_path in paths:
        path = Path(raw_path)
        frames: list[pd.DataFrame] = []
        for separator in ("\t", ","):
            try:
                frames.append(pd.read_csv(path, sep=separator, dtype=str))
            except Exception:
                continue
        try:
            frames.append(pd.read_csv(path, sep=None, engine="python", dtype=str))
        except Exception:
            pass
        for df in frames:
            for column in candidate_columns:
                if column in df.columns:
                    result.update(df[column].dropna().astype(str))
                    break
            else:
                continue
            break
    return result


def _write_edge_tsv(path: Path, rows: list[tuple[str, str, str, float]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["src_id", "dst_id", "edge_type", "weight"])
        for row in rows:
            writer.writerow(row)


def _normalize_edge_source(
    source_path: str | None,
    destination: Path,
    allowed_src: set[str],
    allowed_dst: set[str],
    default_edge_type: str,
) -> int:
    rows: list[tuple[str, str, str, float]] = []
    if source_path:
        df = pd.read_csv(source_path, sep="\t", dtype=str)
        if df.shape[1] >= 2:
            for _, row in df.iterrows():
                src_id = str(row.iloc[0])
                dst_id = str(row.iloc[1])
                if src_id not in allowed_src or dst_id not in allowed_dst:
                    continue
                edge_type = str(row.iloc[2]) if df.shape[1] >= 3 else default_edge_type
                weight = float(row.iloc[3]) if df.shape[1] >= 4 else 1.0
                rows.append((src_id, dst_id, edge_type, weight))
    _write_edge_tsv(destination, rows)
    return len(rows)


def _similarity_cfg(config: dict[str, Any], source_type: str) -> dict[str, Any]:
    cfg = config["cluster_assets"].get("similarity_edges", {})
    defaults = {
        "enabled": True,
        "sourmash_bin": "sourmash",
        "sourmash_env": "sourmash_env",
        "scaled": 1000,
        "threshold": 0.8,
        "workers": None,
    }
    merged = dict(defaults)
    merged.update(cfg.get(source_type, {}))
    merged.update({k: v for k, v in cfg.items() if k in defaults})
    return merged


def _strip_name(path_value: str) -> str:
    return Path(path_value).stem


def _load_similarity_matrix(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "files"):
        key = "data" if "data" in data.files else data.files[0]
        return data[key]
    return data


def _run_sourmash_command(config: dict[str, Any], env_name: str, command: list[str]) -> None:
    subprocess.run(
        [conda_bin(config), "run", "--no-capture-output", "-n", env_name, *command],
        check=True,
    )


def _sketch_signature(task: tuple[dict[str, Any], str, str, int, int, Path, Path]) -> str:
    config, env_name, sourmash_bin_name, kmer_size, scaled, fasta_path, sig_path = task
    _run_sourmash_command(
        config,
        env_name,
        [
            sourmash_bin_name,
            "sketch",
            "dna",
            "-p",
            f"k={kmer_size},scaled={scaled},abund",
            "-o",
            str(sig_path),
            str(fasta_path),
        ],
    )
    return str(sig_path)


def _build_similarity_edges(
    config: dict[str, Any],
    fasta_dir: str,
    allowed_ids: set[str],
    output_path: Path,
    work_dir: Path,
    edge_type: str,
    settings: dict[str, Any],
) -> int:
    fasta_files = [path for path in iter_fasta_files(fasta_dir) if path.stem in allowed_ids]
    if len(fasta_files) < 2:
        _write_edge_tsv(output_path, [])
        return 0

    if work_dir.exists():
        shutil.rmtree(work_dir)
    sig_dir = work_dir / "signatures"
    compare_dir = work_dir / "compare"
    ensure_dir(sig_dir)
    ensure_dir(compare_dir)

    sourmash_bin_name = str(settings.get("sourmash_bin", "sourmash"))
    sourmash_env_name = str(settings.get("sourmash_env", "sourmash_env"))
    kmer_size = int(settings["kmer_size"])
    scaled = int(settings.get("scaled", 1000))
    threshold = float(settings.get("threshold", 0.8))
    sketch_workers = _int_setting(
        settings.get("workers"),
        default=min(len(fasta_files), min(_available_cpu_count(), 16)),
        maximum=len(fasta_files),
    )

    tasks = [
        (
            config,
            sourmash_env_name,
            sourmash_bin_name,
            kmer_size,
            scaled,
            fasta_path,
            sig_dir / f"{fasta_path.stem}.sig",
        )
        for fasta_path in fasta_files
    ]
    if sketch_workers == 1:
        sig_paths = [_sketch_signature(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=sketch_workers) as executor:
            sig_paths = list(executor.map(_sketch_signature, tasks))

    matrix_path = compare_dir / "compare_matrix.npz"
    _run_sourmash_command(
        config,
        sourmash_env_name,
        [sourmash_bin_name, "compare", "-k", str(kmer_size), *sig_paths, "-o", str(matrix_path)],
    )

    labels_path = Path(str(matrix_path) + ".labels.txt")
    if not labels_path.exists():
        raise RuntimeError(f"sourmash compare labels not found: {labels_path}")

    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sim_matrix = _load_similarity_matrix(matrix_path)
    if sim_matrix.shape != (len(labels), len(labels)):
        raise RuntimeError(f"sourmash matrix shape mismatch: {sim_matrix.shape} vs {len(labels)} labels")

    rows: list[tuple[str, str, str, float]] = []
    for i in range(len(labels)):
        src_id = _strip_name(labels[i])
        if src_id not in allowed_ids:
            continue
        for j in range(i + 1, len(labels)):
            dst_id = _strip_name(labels[j])
            if dst_id not in allowed_ids:
                continue
            weight = float(sim_matrix[i, j])
            if weight >= threshold:
                rows.append((src_id, dst_id, edge_type, round(weight, 6)))
    _write_edge_tsv(output_path, rows)
    return len(rows)


def _stream_cluster_catalogs(
    cfg: dict[str, Any],
    subset_ids: set[str],
    phage_ids: set[str],
    host_ids: set[str],
) -> dict[str, Any]:
    parquet_settings = _parquet_settings(cfg)
    protein_writer = _BufferedParquetWriter(
        cfg["protein_catalog_out"],
        PROTEIN_SCHEMA,
        batch_rows=parquet_settings["batch_rows"],
        row_group_size=parquet_settings["row_group_size"],
        compression=parquet_settings["compression"],
    )
    cluster_writer = _BufferedParquetWriter(
        cfg["cluster_protein_catalog_out"],
        PROTEIN_SCHEMA,
        batch_rows=parquet_settings["batch_rows"],
        row_group_size=parquet_settings["row_group_size"],
        compression=parquet_settings["compression"],
    )
    phage_edge_writer = _BufferedParquetWriter(
        cfg["phage_protein_edges_out"],
        PHAGE_EDGE_SCHEMA,
        batch_rows=max(4096, parquet_settings["batch_rows"]),
        row_group_size=max(4096, parquet_settings["row_group_size"]),
        compression=parquet_settings["compression"],
    )
    host_edge_writer = _BufferedParquetWriter(
        cfg["host_protein_edges_out"],
        HOST_EDGE_SCHEMA,
        batch_rows=max(4096, parquet_settings["batch_rows"]),
        row_group_size=max(4096, parquet_settings["row_group_size"]),
        compression=parquet_settings["compression"],
    )
    phage_tsv_handle, phage_tsv_writer = _open_edge_tsv_writer(Path(cfg["edge_dir"]) / "phage_protein_edges.tsv")
    host_tsv_handle, host_tsv_writer = _open_edge_tsv_writer(Path(cfg["edge_dir"]) / "host_protein_edges.tsv")

    seen_protein_ids: set[str] = set()
    cluster_protein_ids: set[str] = set()
    duplicate_proteins = 0
    all_protein_count = 0
    cluster_protein_count = 0
    phage_edge_count = 0
    host_edge_count = 0

    try:
        for source_type, directory, valid_sources in (
            ("phage", cfg["phage_embedding_dir"], phage_ids),
            ("host", cfg["host_embedding_dir"], host_ids),
        ):
            for path in list_files(directory, ".pkl"):
                source_id = path.stem
                payload = load_pickle(path)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Embedding pickle must contain a dict: {path}")
                for protein_id, embedding in payload.items():
                    protein_id_str = str(protein_id)
                    include_in_cluster = not subset_ids or protein_id_str in subset_ids
                    if protein_id_str not in seen_protein_ids:
                        seen_protein_ids.add(protein_id_str)
                        record = {
                            "protein_id": protein_id_str,
                            "source_type": source_type,
                            "source_id": source_id,
                            "embedding": to_float_list(embedding),
                        }
                        protein_writer.write_row(record)
                        all_protein_count += 1
                        if include_in_cluster:
                            cluster_writer.write_row(record)
                            cluster_protein_count += 1
                            if subset_ids:
                                cluster_protein_ids.add(protein_id_str)
                    else:
                        duplicate_proteins += 1

                    if not include_in_cluster or source_id not in valid_sources:
                        continue
                    if source_type == "phage":
                        edge_record = {"phage_id": source_id, "protein_id": protein_id_str}
                        phage_edge_writer.write_row(edge_record)
                        phage_tsv_writer.writerow((source_id, protein_id_str, "phage-protein", 1.0))
                        phage_edge_count += 1
                    else:
                        edge_record = {"host_id": source_id, "protein_id": protein_id_str}
                        host_edge_writer.write_row(edge_record)
                        host_tsv_writer.writerow((source_id, protein_id_str, "host-protein", 1.0))
                        host_edge_count += 1

        return {
            "all_proteins": all_protein_count,
            "cluster_protein_ids": cluster_protein_ids if subset_ids else seen_protein_ids,
            "cluster_proteins": cluster_protein_count,
            "phage_edges": phage_edge_count,
            "host_edges": host_edge_count,
            "duplicate_proteins": duplicate_proteins,
            "parquet_batch_rows": parquet_settings["batch_rows"],
            "parquet_row_group_size": parquet_settings["row_group_size"],
            "parquet_compression": parquet_settings["compression"],
        }
    finally:
        protein_writer.close()
        cluster_writer.close()
        phage_edge_writer.close()
        host_edge_writer.close()
        phage_tsv_handle.close()
        host_tsv_handle.close()


def run_internal(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = config["cluster_assets"]
    ensure_dir(cfg["edge_dir"])
    subset_ids = _load_subset_ids(cfg.get("cluster_member_tables", []))
    phage_ids = set(pd.read_parquet(config["build_catalogs"]["phage_catalog"], columns=["phage_id"])["phage_id"].astype(str))
    host_ids = set(pd.read_parquet(config["build_catalogs"]["host_catalog"], columns=["host_gcf"])["host_gcf"].astype(str))
    taxonomy_ids = set(pd.read_parquet(config["inputs"]["taxonomy_graph_parquet"], columns=["taxid"])["taxid"].astype(str))
    catalog_stats = _stream_cluster_catalogs(cfg, subset_ids, phage_ids, host_ids)
    cluster_protein_ids = catalog_stats["cluster_protein_ids"]

    sources = cfg.get("edge_sources", {})
    sourmash_root = Path(cfg.get("sourmash_work_dir", Path(cfg["edge_dir"]).parent / "sourmash"))
    phage_similarity_cfg = _similarity_cfg(config, "phage")
    host_similarity_cfg = _similarity_cfg(config, "host")
    counts = {
        "phage_phage_edges.tsv": _build_similarity_edges(
            config,
            config["inputs"]["phage_fasta_dir"],
            phage_ids,
            Path(cfg["edge_dir"]) / "phage_phage_edges.tsv",
            sourmash_root / "phage_phage",
            "phage-phage",
            phage_similarity_cfg,
        )
        if phage_similarity_cfg.get("enabled", True)
        else _normalize_edge_source(
            sources.get("phage_phage_edges"),
            Path(cfg["edge_dir"]) / "phage_phage_edges.tsv",
            phage_ids,
            phage_ids,
            "phage-phage",
        ),
        "host_host_edges.tsv": _build_similarity_edges(
            config,
            config["inputs"]["host_fasta_dir"],
            host_ids,
            Path(cfg["edge_dir"]) / "host_host_edges.tsv",
            sourmash_root / "host_host",
            "host-host",
            host_similarity_cfg,
        )
        if host_similarity_cfg.get("enabled", True)
        else _normalize_edge_source(
            sources.get("host_host_edges"),
            Path(cfg["edge_dir"]) / "host_host_edges.tsv",
            host_ids,
            host_ids,
            "host-host",
        ),
        "phage_host_edges.tsv": _normalize_edge_source(
            sources.get("phage_host_edges"),
            Path(cfg["edge_dir"]) / "phage_host_edges.tsv",
            phage_ids,
            host_ids,
            "phage-host",
        ),
        "protein_protein_edges.tsv": _normalize_edge_source(
            sources.get("protein_protein_edges"),
            Path(cfg["edge_dir"]) / "protein_protein_edges.tsv",
            cluster_protein_ids,
            cluster_protein_ids,
            "protein-protein",
        ),
        "host_taxonomy_edges.tsv": _normalize_edge_source(
            sources.get("host_taxonomy_edges"),
            Path(cfg["edge_dir"]) / "host_taxonomy_edges.tsv",
            host_ids,
            taxonomy_ids,
            "host-taxonomy",
        ),
        "phage_taxonomy_edges.tsv": _normalize_edge_source(
            sources.get("phage_taxonomy_edges"),
            Path(cfg["edge_dir"]) / "phage_taxonomy_edges.tsv",
            phage_ids,
            taxonomy_ids,
            "phage-taxonomy",
        ),
        "taxonomy_taxonomy_edges.tsv": _normalize_edge_source(
            sources.get("taxonomy_taxonomy_edges"),
            Path(cfg["edge_dir"]) / "taxonomy_taxonomy_edges.tsv",
            taxonomy_ids,
            taxonomy_ids,
            "taxonomy-taxonomy",
        ),
    }

    return {
        "all_proteins": catalog_stats["all_proteins"],
        "cluster_proteins": catalog_stats["cluster_proteins"],
        "phage_edges": catalog_stats["phage_edges"],
        "host_edges": catalog_stats["host_edges"],
        "duplicate_proteins": catalog_stats["duplicate_proteins"],
        "parquet_batch_rows": catalog_stats["parquet_batch_rows"],
        "parquet_row_group_size": catalog_stats["parquet_row_group_size"],
        "parquet_compression": catalog_stats["parquet_compression"],
        "copied_edges": counts,
    }
