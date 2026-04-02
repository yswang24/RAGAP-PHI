from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch

from .utils import list_files


def _result(errors: list[str], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": not errors, "errors": errors, "summary": summary or {}}


def _section_config(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    mapping = {
        "dna_embed_phage": config["dna_embedding"]["phage"],
        "dna_embed_host": config["dna_embedding"]["host"],
        "build_catalogs": config["build_catalogs"],
        "build_pairs": config["pairs"],
        "prepare_phage_proteins": config["phage_protein_prep"],
        "prepare_host_proteins": config["host_protein_prep"],
        "embed_phage_proteins": config["phage_protein_embedding"],
        "embed_host_proteins": config["host_protein_embedding"],
        "build_cluster_assets": config["cluster_assets"],
        "build_graph": config["graph"],
        "train": config["train"],
    }
    return mapping[stage_name]


def _has_embedding(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return len(value) > 0
    return True


def _validate_custom_rules(validate_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for item in validate_cfg.get("dir_contains", []):
        path = Path(item["path"])
        if not path.is_dir():
            errors.append(f"validation dir missing: {path}")
            continue
        for name in item.get("names", []):
            if not (path / name).exists():
                errors.append(f"validation missing '{name}' in {path}")
    for item in validate_cfg.get("parquet_columns", []):
        path = Path(item["path"])
        if not path.exists():
            errors.append(f"validation parquet missing: {path}")
            continue
        columns = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [column for column in item.get("columns", []) if column not in columns]
        if missing:
            errors.append(f"validation parquet columns missing in {path}: {', '.join(missing)}")
    for item in validate_cfg.get("tsv_columns", []):
        path = Path(item["path"])
        if not path.exists():
            errors.append(f"validation tsv missing: {path}")
            continue
        df = pd.read_csv(path, sep="\t", nrows=2)
        missing = [column for column in item.get("columns", []) if column not in df.columns]
        if missing:
            errors.append(f"validation tsv columns missing in {path}: {', '.join(missing)}")
    for raw_path in validate_cfg.get("nonempty_files", []):
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"validation file missing: {path}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"validation file empty: {path}")
    return errors


def _validate_dna_dir(out_dir: str, id_column: str = "sequence_id") -> dict[str, Any]:
    errors: list[str] = []
    files = list_files(out_dir, ".parquet")
    if not files:
        return _result([f"dna output directory is empty: {out_dir}"])
    row_count = 0
    for file_path in files:
        table = pd.read_parquet(file_path, columns=[id_column, "embedding"])
        if table.empty:
            errors.append(f"dna parquet empty: {file_path}")
            continue
        if id_column not in table.columns or "embedding" not in table.columns:
            errors.append(f"dna parquet missing required columns: {file_path}")
            continue
        if table[id_column].isna().any():
            errors.append(f"dna parquet contains null ids: {file_path}")
        if not table["embedding"].map(_has_embedding).all():
            errors.append(f"dna parquet contains empty embeddings: {file_path}")
        row_count += len(table)
    return _result(errors, {"files": len(files), "rows": row_count})


def _validate_catalogs(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["build_catalogs"]
    phage_path = Path(cfg["phage_catalog"])
    host_path = Path(cfg["host_catalog"])
    errors: list[str] = []
    if not phage_path.exists():
        errors.append(f"phage catalog missing: {phage_path}")
    if not host_path.exists():
        errors.append(f"host catalog missing: {host_path}")
    if errors:
        return _result(errors)

    phage_df = pd.read_parquet(phage_path)
    host_df = pd.read_parquet(host_path)
    for column in ("phage_id", "phage_dna_emb"):
        if column not in phage_df.columns:
            errors.append(f"phage catalog missing column: {column}")
    for column in ("host_gcf", "sequence_id", "host_species_taxid", "host_dna_emb"):
        if column not in host_df.columns:
            errors.append(f"host catalog missing column: {column}")
    if "phage_id" in phage_df.columns and phage_df["phage_id"].astype(str).duplicated().any():
        errors.append("phage catalog contains duplicate phage_id values")
    if {"host_gcf", "sequence_id"} <= set(host_df.columns):
        if host_df[["host_gcf", "sequence_id"]].astype(str).duplicated().any():
            errors.append("host catalog contains duplicate (host_gcf, sequence_id) pairs")
    if "host_species_taxid" in host_df.columns and host_df["host_species_taxid"].isna().any():
        errors.append("host catalog contains null host_species_taxid")
    return _result(
        errors,
        {
            "phage_rows": len(phage_df),
            "host_rows": len(host_df),
            "host_unique_gcf": int(host_df["host_gcf"].astype(str).nunique()) if "host_gcf" in host_df.columns else 0,
        },
    )


def _validate_pairs(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["pairs"]
    split = cfg.get("split", "random")
    suffix = "" if split == "random" else "_taxa"
    paths = {
        "pairs_all": Path(cfg["out_dir"]) / "pairs_all.tsv",
        "pairs_train": Path(cfg["out_dir"]) / f"pairs_train{suffix}.tsv",
        "pairs_val": Path(cfg["out_dir"]) / f"pairs_val{suffix}.tsv",
        "pairs_test": Path(cfg["out_dir"]) / f"pairs_test{suffix}.tsv",
    }
    errors: list[str] = []
    counts: dict[str, int] = {}
    required = {"phage_id", "host_gcf", "label"}
    for label, path in paths.items():
        if not path.exists():
            errors.append(f"pairs output missing: {path}")
            continue
        df = pd.read_csv(path, sep="\t")
        counts[label] = len(df)
        missing = sorted(required - set(df.columns))
        if missing:
            errors.append(f"{path} missing columns: {', '.join(missing)}")
        if label != "pairs_all" and df.empty:
            errors.append(f"{path} is empty")
    return _result(errors, counts)


def _validate_faa_dir(path_value: str, label: str) -> tuple[list[str], int]:
    errors: list[str] = []
    files = list_files(path_value, ".faa")
    if not files:
        errors.append(f"{label} FAA directory is empty: {path_value}")
    return errors, len(files)


def _validate_prepare_phage(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["phage_protein_prep"]
    errors, count = _validate_faa_dir(cfg["faa_dir"], "phage")
    backend = str(cfg.get("backend", "phanotate_direct")).strip().lower()
    aliases = {"phanotate": "phanotate_direct", "direct": "phanotate_direct"}
    backend = aliases.get(backend, backend)
    if backend not in {"phanotate_direct", "pharokka"}:
        errors.append(f"prepare_phage_proteins backend is unsupported: {backend}")
    if cfg.get("keep_annotation", False):
        annotation_dir = Path(cfg["annotation_dir"])
        if not annotation_dir.is_dir():
            errors.append(f"phage annotation directory missing: {annotation_dir}")
        elif not any(annotation_dir.iterdir()):
            errors.append(f"phage annotation directory is empty: {annotation_dir}")
        elif backend == "phanotate_direct":
            if not any(annotation_dir.rglob("phanotate_out.txt")):
                errors.append("phage annotation directory missing phanotate_out.txt files")
        elif backend == "pharokka":
            if not any(annotation_dir.rglob("*.faa")):
                errors.append("phage annotation directory missing Pharokka FAA outputs")
    return _result(errors, {"faa_files": count, "backend": backend})


def _validate_protein_embedding(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = config["phage_protein_embedding"] if stage_name == "embed_phage_proteins" else config["host_protein_embedding"]
    faa_dir = cfg["faa_dir"]
    pkl_dir = cfg["out_dir"]
    failure_report = Path(cfg["failure_report"])
    expected = {path.stem for path in list_files(faa_dir, ".faa")}
    produced = {path.stem for path in list_files(pkl_dir, ".pkl")}
    errors: list[str] = []
    if not failure_report.exists():
        errors.append(f"failure report missing: {failure_report}")
        failure_payload = {"missing": sorted(expected - produced), "failed": []}
    else:
        failure_payload = json.loads(failure_report.read_text(encoding="utf-8"))
    if expected != produced:
        missing = sorted(expected - produced)
        errors.append(f"FAA/PKL count mismatch for {stage_name}: expected {len(expected)}, found {len(produced)}")
        if failure_payload.get("missing") != missing:
            errors.append(f"failure report missing set does not match actual missing files for {stage_name}")
    return _result(
        errors,
        {
            "faa_files": len(expected),
            "pkl_files": len(produced),
            "failed_files": len(failure_payload.get("missing", [])),
        },
    )


def _read_edge_ids(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] < 2:
        raise ValueError(f"edge file must have at least 2 columns: {path}")
    return df.iloc[:, 0].astype(str).tolist(), df.iloc[:, 1].astype(str).tolist()


def _validate_cluster_assets(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["cluster_assets"]
    edge_dir = Path(cfg["edge_dir"])
    catalog_path = Path(cfg["cluster_protein_catalog_out"])
    protein_catalog_path = Path(cfg["protein_catalog_out"])
    phage_edges_path = Path(cfg["phage_protein_edges_out"])
    host_edges_path = Path(cfg["host_protein_edges_out"])
    required_edge_files = {
        "phage_phage_edges.tsv",
        "host_host_edges.tsv",
        "phage_protein_edges.tsv",
        "host_protein_edges.tsv",
        "protein_protein_edges.tsv",
        "host_taxonomy_edges.tsv",
        "phage_taxonomy_edges.tsv",
        "taxonomy_taxonomy_edges.tsv",
    }
    errors: list[str] = []
    for path in (catalog_path, protein_catalog_path, phage_edges_path, host_edges_path):
        if not path.exists():
            errors.append(f"cluster asset missing: {path}")
    if not edge_dir.is_dir():
        errors.append(f"cluster edge_dir missing: {edge_dir}")
    else:
        for filename in sorted(required_edge_files):
            if not (edge_dir / filename).exists():
                errors.append(f"cluster edge file missing: {edge_dir / filename}")
    if errors:
        return _result(errors)

    cluster_df = pd.read_parquet(catalog_path)
    protein_df = pd.read_parquet(protein_catalog_path)
    phage_catalog = pd.read_parquet(config["build_catalogs"]["phage_catalog"])
    host_catalog = pd.read_parquet(config["build_catalogs"]["host_catalog"])
    taxonomy_df = pd.read_parquet(config["inputs"]["taxonomy_graph_parquet"])

    for column in ("protein_id", "source_type", "source_id", "embedding"):
        if column not in cluster_df.columns:
            errors.append(f"cluster catalog missing column: {column}")
    phage_ids = set(phage_catalog["phage_id"].astype(str))
    host_ids = set(host_catalog["host_gcf"].astype(str))
    protein_ids = set(cluster_df["protein_id"].astype(str))
    taxids = set(taxonomy_df["taxid"].astype(str))

    if cluster_df.empty:
        errors.append("cluster catalog is empty")
    if protein_df.empty:
        errors.append("protein catalog is empty")

    phage_edge_df = pd.read_parquet(phage_edges_path)
    host_edge_df = pd.read_parquet(host_edges_path)
    if phage_edge_df.empty:
        errors.append("phage protein edges parquet is empty")
    if host_edge_df.empty:
        errors.append("host protein edges parquet is empty")
    if "phage_id" in phage_edge_df.columns and "protein_id" in phage_edge_df.columns:
        if not set(phage_edge_df["phage_id"].astype(str)).issubset(phage_ids):
            errors.append("phage protein edges contain unknown phage_id")
        if not set(phage_edge_df["protein_id"].astype(str)).issubset(protein_ids):
            errors.append("phage protein edges contain unknown protein_id")
    if "host_id" in host_edge_df.columns and "protein_id" in host_edge_df.columns:
        if not set(host_edge_df["host_id"].astype(str)).issubset(host_ids):
            errors.append("host protein edges contain unknown host_id")
        if not set(host_edge_df["protein_id"].astype(str)).issubset(protein_ids):
            errors.append("host protein edges contain unknown protein_id")

    for filename in ("phage_protein_edges.tsv", "host_protein_edges.tsv"):
        src_ids, dst_ids = _read_edge_ids(edge_dir / filename)
        if filename.startswith("phage"):
            if not set(src_ids).issubset(phage_ids):
                errors.append("phage_protein_edges.tsv contains unknown phage_id")
        else:
            if not set(src_ids).issubset(host_ids):
                errors.append("host_protein_edges.tsv contains unknown host_id")
        if not set(dst_ids).issubset(protein_ids):
            errors.append(f"{filename} contains unknown protein_id")

    src_ids, dst_ids = _read_edge_ids(edge_dir / "host_taxonomy_edges.tsv")
    if not set(src_ids).issubset(host_ids):
        errors.append("host_taxonomy_edges.tsv contains unknown host_id")
    if not set(dst_ids).issubset(taxids):
        errors.append("host_taxonomy_edges.tsv contains unknown taxid")

    src_ids, dst_ids = _read_edge_ids(edge_dir / "phage_taxonomy_edges.tsv")
    if not set(src_ids).issubset(phage_ids):
        errors.append("phage_taxonomy_edges.tsv contains unknown phage_id")
    if not set(dst_ids).issubset(taxids):
        errors.append("phage_taxonomy_edges.tsv contains unknown taxid")

    src_ids, dst_ids = _read_edge_ids(edge_dir / "taxonomy_taxonomy_edges.tsv")
    if not set(src_ids).issubset(taxids) or not set(dst_ids).issubset(taxids):
        errors.append("taxonomy_taxonomy_edges.tsv contains unknown taxid")

    return _result(
        errors,
        {
            "cluster_proteins": len(cluster_df),
            "all_proteins": len(protein_df),
            "phage_protein_edges": len(phage_edge_df),
            "host_protein_edges": len(host_edge_df),
        },
    )


def _node_count(data: Any, node_type: str) -> int:
    node = data[node_type]
    if hasattr(node, "num_nodes") and node.num_nodes is not None:
        return int(node.num_nodes)
    if hasattr(node, "x"):
        return int(node.x.size(0))
    raise ValueError(f"cannot determine node count for {node_type}")


def _validate_graph(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["graph"]
    graph_path = Path(cfg["out"])
    map_path = Path(cfg["map_out"])
    errors: list[str] = []
    if not graph_path.exists():
        errors.append(f"graph output missing: {graph_path}")
    if not map_path.exists():
        errors.append(f"node map missing: {map_path}")
    if errors:
        return _result(errors)

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    with map_path.open("r", encoding="utf-8") as handle:
        node_maps = json.load(handle)

    if "phage_map" not in node_maps or "host_map" not in node_maps:
        errors.append("node_maps.json missing phage_map or host_map")
    node_counts = {
        "phage": _node_count(data, "phage"),
        "host": _node_count(data, "host"),
        "protein": _node_count(data, "protein"),
        "taxonomy": _node_count(data, "taxonomy"),
    }
    relation = data[("phage", "infects", "host")]
    for split_name in ("train", "val", "test"):
        attr = f"edge_index_{split_name}"
        edge_index = getattr(relation, attr, None)
        if edge_index is None or edge_index.numel() == 0:
            errors.append(f"graph split is empty: {split_name}")
            continue
        if int(edge_index[0].max()) >= node_counts["phage"] or int(edge_index[1].max()) >= node_counts["host"]:
            errors.append(f"graph split index out of bounds: {split_name}")
    return _result(errors, node_counts)


def _validate_train(config: dict[str, Any], statuses: dict[str, Any]) -> dict[str, Any]:
    if statuses["build_graph"].state != "valid":
        return _result([f"graph is not valid: {statuses['build_graph'].reason}"])
    cfg = config["train"]
    out_dir = Path(cfg["out_dir"])
    log_path = out_dir / "run.log"
    checkpoint_path = out_dir / Path(str(cfg["save_path"])).name
    errors: list[str] = []
    if not out_dir.is_dir():
        errors.append(f"train out_dir missing: {out_dir}")
    if not log_path.exists():
        errors.append(f"train log missing: {log_path}")
    if not checkpoint_path.exists():
        errors.append(f"train checkpoint missing: {checkpoint_path}")
    return _result(errors, {"out_dir": str(out_dir), "checkpoint": str(checkpoint_path)})


def validate_stage(stage_name: str, config: dict[str, Any], statuses: dict[str, Any]) -> dict[str, Any]:
    extra_errors = _validate_custom_rules(_section_config(config, stage_name).get("validate", {}))
    if stage_name == "dna_embed_phage":
        result = _validate_dna_dir(config["dna_embedding"]["phage"]["out_dir"])
    elif stage_name == "dna_embed_host":
        result = _validate_dna_dir(config["dna_embedding"]["host"]["out_dir"])
    elif stage_name == "build_catalogs":
        result = _validate_catalogs(config)
    elif stage_name == "build_pairs":
        result = _validate_pairs(config)
    elif stage_name == "prepare_phage_proteins":
        result = _validate_prepare_phage(config)
    elif stage_name == "prepare_host_proteins":
        errors, count = _validate_faa_dir(config["host_protein_prep"]["faa_dir"], "host")
        if config["host_protein_prep"].get("backend") != "prodigal":
            errors.append("prepare_host_proteins backend must be prodigal")
        result = _result(errors, {"faa_files": count})
    elif stage_name in {"embed_phage_proteins", "embed_host_proteins"}:
        result = _validate_protein_embedding(config, stage_name)
    elif stage_name == "build_cluster_assets":
        result = _validate_cluster_assets(config)
    elif stage_name == "build_graph":
        result = _validate_graph(config)
    elif stage_name == "train":
        result = _validate_train(config, statuses)
    else:
        result = _result([])

    result["errors"].extend(extra_errors)
    result["ok"] = not result["errors"]
    return result
