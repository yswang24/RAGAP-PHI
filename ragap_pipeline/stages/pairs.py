from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import ensure_dir, to_float_list


def _parse_gcfs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _load_host_taxid_map(metadata_path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    df = pd.read_csv(metadata_path, sep=None, engine="python", dtype=str)
    tax_column = None
    for candidate in ("host_species_taxid", "host_taxid", "host_tax_id"):
        if candidate in df.columns:
            tax_column = candidate
            break
    if tax_column is None:
        return mapping
    for _, row in df.iterrows():
        taxid = str(row[tax_column]).strip()
        if not taxid or taxid.lower() == "nan":
            continue
        if "host_gcf" in df.columns and row.get("host_gcf"):
            mapping[str(row["host_gcf"]).strip()] = taxid
        if "sequence_id" in df.columns and row.get("sequence_id"):
            mapping[str(row["sequence_id"]).strip()] = taxid
        if "Extracted_GCFs" in df.columns:
            for gcf in _parse_gcfs(row.get("Extracted_GCFs")):
                mapping[gcf] = taxid
    return mapping


def _load_taxonomy_map(taxonomy_path: str) -> dict[str, list[float]]:
    df = pd.read_parquet(taxonomy_path)
    if "tangent_emb" not in df.columns:
        raise RuntimeError(f"taxonomy parquet missing tangent_emb column: {taxonomy_path}")
    mapping: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        mapping[str(row["taxid"])] = to_float_list(row["tangent_emb"])
    return mapping


def _load_dna_frames(directory: str) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for path in sorted(Path(directory).glob("*.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        frames.append((path.stem, df))
    return frames


def _canonical_pair_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    out_dir = Path(cfg["out_dir"])
    return {
        "pairs_all": out_dir / "pairs_all.tsv",
        "pairs_train": out_dir / "pairs_train.tsv",
        "pairs_val": out_dir / "pairs_val.tsv",
        "pairs_test": out_dir / "pairs_test.tsv",
    }


def _actual_pair_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = _canonical_pair_paths(cfg)
    if cfg.get("split", "random") != "taxa":
        return paths
    out_dir = Path(cfg["out_dir"])
    return {
        "pairs_all": paths["pairs_all"],
        "pairs_train": out_dir / "pairs_train_taxa.tsv",
        "pairs_val": out_dir / "pairs_val_taxa.tsv",
        "pairs_test": out_dir / "pairs_test_taxa.tsv",
    }


def inputs(config: dict[str, Any], stage_name: str) -> list[str]:
    if stage_name == "build_catalogs":
        cfg = config["build_catalogs"]
        values = [cfg["phage_dir"], cfg["host_dir"], cfg["taxonomy_parquet"]]
        metadata = cfg.get("host_metadata_tsv")
        if metadata:
            values.append(metadata)
        return values
    cfg = config["pairs"]
    return [cfg["raw_pairs"], cfg["host_catalog"], cfg["taxonomy_parquet"]]


def outputs(config: dict[str, Any], stage_name: str) -> list[str]:
    if stage_name == "build_catalogs":
        cfg = config["build_catalogs"]
        return [cfg["phage_catalog"], cfg["host_catalog"]]
    return [str(path) for path in _canonical_pair_paths(config["pairs"]).values()]


def params(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name == "build_catalogs":
        cfg = config["build_catalogs"]
        ignore = {
            "mode",
            "phage_dir",
            "host_dir",
            "taxonomy_parquet",
            "host_metadata_tsv",
            "out_dir",
            "phage_catalog",
            "host_catalog",
            "validate",
            "deps",
        }
        return {key: value for key, value in cfg.items() if key not in ignore}
    cfg = config["pairs"]
    ignore = {
        "mode",
        "script",
        "python",
        "raw_pairs",
        "host_catalog",
        "taxonomy_parquet",
        "out_dir",
        "validate",
        "deps",
    }
    return {key: value for key, value in cfg.items() if key not in ignore}


def script_path(config: dict[str, Any], stage_name: str) -> str:
    if stage_name == "build_catalogs":
        return str(Path(__file__).resolve())
    return str(config["pairs"]["script"])


def command(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["pairs"]
    cmd = [
        cfg.get("python") or config.get("python_bin", "python"),
        cfg["script"],
        "--raw_pairs",
        cfg["raw_pairs"],
        "--host_catalog",
        cfg["host_catalog"],
        "--taxonomy_parquet",
        cfg["taxonomy_parquet"],
        "--out_dir",
        cfg["out_dir"],
        "--split",
        str(cfg.get("split", "random")),
        "--seed",
        str(cfg.get("seed", 613)),
    ]
    ratios = cfg.get("ratios")
    if ratios is not None:
        if isinstance(ratios, (list, tuple)):
            ratios_value = ",".join(str(item) for item in ratios)
        else:
            ratios_value = str(ratios)
        cmd.extend(["--ratios", ratios_value])
    return cmd


def run_internal(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name != "build_catalogs":
        raise RuntimeError(f"Unsupported internal stage in pairs module: {stage_name}")

    cfg = config["build_catalogs"]
    ensure_dir(cfg["out_dir"])
    host_taxid_map = {}
    if cfg.get("host_metadata_tsv"):
        host_taxid_map = _load_host_taxid_map(cfg["host_metadata_tsv"])
    taxonomy_map = _load_taxonomy_map(cfg["taxonomy_parquet"])

    phage_rows: list[dict[str, Any]] = []
    for source_id, df in _load_dna_frames(cfg["phage_dir"]):
        for _, row in df.iterrows():
            phage_id = str(row.get("phage_id") or row.get("sequence_id") or source_id)
            embedding = row.get("phage_dna_emb", row.get("embedding"))
            phage_rows.append(
                {
                    "phage_id": phage_id,
                    "phage_dna_emb": to_float_list(embedding),
                }
            )
    phage_df = pd.DataFrame(phage_rows).drop_duplicates(subset=["phage_id"]).reset_index(drop=True)
    phage_df.to_parquet(cfg["phage_catalog"], index=False)

    host_rows: list[dict[str, Any]] = []
    for source_id, df in _load_dna_frames(cfg["host_dir"]):
        for _, row in df.iterrows():
            host_gcf = str(row.get("host_gcf") or source_id)
            sequence_id = str(row.get("sequence_id") or source_id)
            taxid = row.get("host_species_taxid")
            if taxid is None or (isinstance(taxid, float) and pd.isna(taxid)):
                taxid = host_taxid_map.get(host_gcf) or host_taxid_map.get(sequence_id)
            taxid_str = str(taxid) if taxid is not None else None
            host_rows.append(
                {
                    "host_gcf": host_gcf,
                    "sequence_id": sequence_id,
                    "host_species_taxid": taxid_str,
                    "host_dna_emb": to_float_list(row.get("host_dna_emb", row.get("embedding"))),
                    "tangent_emb": taxonomy_map.get(taxid_str) if taxid_str else None,
                }
            )
    host_df = pd.DataFrame(host_rows).drop_duplicates(subset=["host_gcf", "sequence_id"]).reset_index(drop=True)
    host_df.to_parquet(cfg["host_catalog"], index=False)

    return {
        "phage_rows": len(phage_df),
        "host_rows": len(host_df),
    }


def post_run(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name != "build_pairs":
        return {}

    cfg = config["pairs"]
    actual = _actual_pair_paths(cfg)
    canonical = _canonical_pair_paths(cfg)

    for key, source_path in actual.items():
        if not source_path.exists():
            raise RuntimeError(f"build_pairs missing expected output: {source_path}")
        target_path = canonical[key]
        if source_path.resolve() != target_path.resolve():
            shutil.copyfile(source_path, target_path)

    counts: dict[str, int] = {}
    for key, target_path in canonical.items():
        if not target_path.exists():
            raise RuntimeError(f"build_pairs missing canonical output: {target_path}")
        if key == "pairs_all":
            counts[key] = len(pd.read_csv(target_path, sep="\t"))
        else:
            counts[key] = len(pd.read_csv(target_path, sep="\t"))
            if counts[key] == 0:
                raise RuntimeError(f"build_pairs produced empty split file: {target_path}")
    return counts

