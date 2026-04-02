from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["NUMEXPR_MAX_THREADS"] = os.environ.get("NUMEXPR_MAX_THREADS", "64")
os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "64")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

from .config import prepare_config
from .execution import DEFAULT_CONDA_BIN, subprocess_env, wrap_command_with_env


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "ragap_phi"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.fullhost_v2.yaml"
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "artifacts" / DEFAULT_DATASET_ID / "manifests" / "train.json"
DEFAULT_TRAIN_OUT_DIR = PROJECT_ROOT / "artifacts" / DEFAULT_DATASET_ID / "train" / "fullhost_v2"
DEFAULT_CHECKPOINT = DEFAULT_TRAIN_OUT_DIR / "best_GAT_attn_fullhost_copymsg_v2.pt"
DEFAULT_GRAPH = PROJECT_ROOT / "artifacts" / DEFAULT_DATASET_ID / "graph" / "hetero_graph.pt"
DEFAULT_NODE_MAPS = PROJECT_ROOT / "artifacts" / DEFAULT_DATASET_ID / "graph" / "node_maps.json"
DEFAULT_HOST_CATALOG = PROJECT_ROOT / "artifacts" / DEFAULT_DATASET_ID / "catalogs" / "host_catalog.parquet"
DEFAULT_TAXONOMY_TREE = PROJECT_ROOT / "data" / "taxonomy" / "taxonomy_with_alias.parquet"
DEFAULT_TAXID2SPECIES = PROJECT_ROOT / "data" / "metadata" / "taxid_species.tsv"
DEFAULT_OUTPUT_MODE = "species"
SUPPORTED_FASTA_SUFFIXES = {".fasta", ".fa", ".fna"}
PHAGE_INTERACTS = ("phage", "interacts", "phage")
PHAGE_ENCODES = ("phage", "encodes", "protein")
PROTEIN_ENCODED_BY_PHAGE = ("protein", "encoded_by_phage", "phage")

# This must match the training script.
EDGE_TYPE_WEIGHT_MAP: dict[tuple[str, str, str], float] = {
    ("phage", "infects", "host"): 3.0,
    ("phage", "interacts", "phage"): 2.0,
    ("host", "belongs_to", "taxonomy"): 3.0,
}


@dataclass(frozen=True)
class TaxonomyNode:
    taxid: int
    parent: int
    name: str
    rank: str


@dataclass(frozen=True)
class InferenceAssets:
    manifest_path: Path
    config_path: Path
    runtime_config: dict[str, Any]
    train_script: Path
    checkpoint: Path
    graph: Path
    node_maps: Path
    host_catalog: Path
    taxonomy_tree: Path
    taxid2species: Path
    conda_bin: str
    dna_embed_script: Path
    dna_model: str
    dna_k: int
    dna_window_tokens: int
    dna_stride_tokens: int
    dna_batch_size: int
    dna_precision: str
    dna_max_windows: int | None
    dna_seed: int
    phanotate_bin: str
    phanotate_extra_args: tuple[str, ...]
    phage_esm_script: Path
    phage_esm_model_name: str
    phage_esm_repr_l: int
    phage_esm_batch_size: int
    phage_esm_workers: int
    sourmash_env: str
    sourmash_bin: str
    phage_similarity_kmer_size: int
    phage_similarity_scaled: int
    phage_similarity_threshold: float
    phage_signatures_dir: Path
    hidden_dim: int
    out_dim: int
    n_layers: int
    n_heads: int
    dropout: float
    relation_aggr: str
    model_seed: int


@dataclass(frozen=True)
class PreparedQuery:
    phage_id: str
    fasta_path: Path
    dna_embedding: torch.Tensor
    protein_embeddings: dict[str, torch.Tensor]
    similarity_rows: list[tuple[str, str, str, float]]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-chain RAGAP inference for a single phage FASTA file."
    )
    parser.add_argument("--input", required=True, help="Input phage FASTA/FA/FNA file")
    parser.add_argument(
        "--mode",
        choices=("species", "genus"),
        default=DEFAULT_OUTPUT_MODE,
        help="Output label mode",
    )
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument("--manifest", default=None, help="Optional train manifest path")
    parser.add_argument("--config", default=None, help="Optional pipeline config path")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--graph", default=None, help="Override hetero_graph.pt path")
    parser.add_argument("--node-maps", default=None, help="Override node_maps.json path")
    parser.add_argument("--host-catalog", default=None, help="Override host catalog parquet path")
    parser.add_argument("--taxonomy-tree", default=None, help="Override taxonomy_with_alias parquet path")
    parser.add_argument("--taxid2species", default=None, help="Override taxid->species TSV path")
    parser.add_argument("--train-script", default=None, help="Override training script path for model loading")
    parser.add_argument(
        "--phage-signatures-dir",
        default=None,
        help="Override sourmash signature directory for cached phage-phage similarities",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for the augmented full-graph forward. Default: cpu",
    )
    parser.add_argument("--work-dir", default=None, help="Override work directory")
    parser.add_argument("--cleanup", action="store_true", help="Remove intermediate files after success")
    return parser.parse_args(argv)


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    return Path(path_value).expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _arg_value(command: list[str], flag: str, default: str | None = None) -> str | None:
    if flag not in command:
        return default
    index = command.index(flag)
    if index + 1 >= len(command):
        return default
    return command[index + 1]


def _checkpoint_from_manifest(manifest: dict[str, Any]) -> Path:
    for output_path in manifest.get("outputs", {}):
        if output_path.endswith(".pt"):
            return Path(output_path).resolve()
    return DEFAULT_CHECKPOINT.resolve()


def _training_module(script_path: Path) -> Any:
    module_name = f"ragap_train_{script_path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import training script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _default_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; falling back to cpu")
        return "cpu"
    return requested


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32, copy=False)


def _tensor_from_embedding(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype).detach().cpu()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("Expected a 1D embedding vector")
    return torch.from_numpy(array).to(dtype=dtype)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _parquet_embedding_rows(path: Path, emb_col: str) -> list[np.ndarray]:
    parquet = pq.ParquetFile(path)
    rows: list[np.ndarray] = []
    for batch in parquet.iter_batches(batch_size=1024, columns=[emb_col], use_threads=False):
        for embedding in batch.column(0).to_pylist():
            if embedding is None:
                continue
            rows.append(np.asarray(embedding, dtype=np.float32))
    if not rows:
        raise RuntimeError(f"No embeddings found in {path}")
    return rows


def aggregate_sequence_embeddings(path: Path, emb_col: str = "embedding") -> torch.Tensor:
    rows = _parquet_embedding_rows(path, emb_col=emb_col)
    mean_vector = np.mean(np.vstack(rows), axis=0)
    return torch.from_numpy(_normalize_vector(mean_vector))


def _taxonomy_nodes(path: Path) -> dict[int, TaxonomyNode]:
    df = pd.read_parquet(path, columns=["taxid", "parent", "name", "rank"])
    result: dict[int, TaxonomyNode] = {}
    for row in df.itertuples(index=False):
        taxid = int(row.taxid)
        parent = int(row.parent) if pd.notna(row.parent) else taxid
        result[taxid] = TaxonomyNode(
            taxid=taxid,
            parent=parent,
            name=str(row.name),
            rank=str(row.rank).strip().lower(),
        )
    return result


def resolve_genus_name(species_taxid: int, taxonomy_nodes: dict[int, TaxonomyNode]) -> str:
    if species_taxid < 0:
        return "NA"
    current = taxonomy_nodes.get(int(species_taxid))
    visited: set[int] = set()
    while current is not None and current.taxid not in visited:
        visited.add(current.taxid)
        if current.rank == "genus":
            return current.name
        if current.parent == current.taxid:
            break
        current = taxonomy_nodes.get(current.parent)
    return "NA"


def resolve_species_name(species_taxid: int, species_lookup: dict[int, str]) -> str:
    return species_lookup.get(int(species_taxid), f"unknown_{species_taxid}")


def build_similarity_edge_rows(
    query_phage_id: str,
    similarities: dict[str, float],
    threshold: float,
) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    for existing_id, score in sorted(similarities.items()):
        if float(score) < threshold:
            continue
        if query_phage_id < existing_id:
            rows.append((query_phage_id, existing_id, "phage-phage", float(score)))
        else:
            rows.append((existing_id, query_phage_id, "phage-phage", float(score)))
    return rows


def _append_node_features(base: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    if extra.numel() == 0:
        return base
    return torch.cat([base, extra.to(dtype=base.dtype)], dim=0).contiguous()


def _existing_edge_index(data: HeteroData, edge_type: tuple[str, str, str]) -> torch.Tensor:
    store = data[edge_type]
    edge_index = getattr(store, "edge_index", None)
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return edge_index.long()


def _append_edge_index(existing: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    if extra.numel() == 0:
        return existing
    if existing.numel() == 0:
        return extra.long().contiguous()
    return torch.cat([existing.long(), extra.long()], dim=1).contiguous()


def augment_graph_with_query(
    data: HeteroData,
    node_maps: dict[str, dict[str, int]],
    phage_id: str,
    phage_embedding: torch.Tensor,
    protein_embeddings: dict[str, torch.Tensor],
    similarity_rows: list[tuple[str, str, str, float]],
) -> tuple[int, dict[str, dict[str, int]], dict[str, int]]:
    phage_map = dict(node_maps.get("phage_map", {}))
    protein_map = dict(node_maps.get("protein_map", {}))

    if phage_id in phage_map:
        raise ValueError(f"phage_id already exists in graph: {phage_id}")

    phage_index = int(data["phage"].x.shape[0])
    phage_map[phage_id] = phage_index
    data["phage"].x = _append_node_features(
        data["phage"].x,
        phage_embedding.view(1, -1),
    )

    sorted_proteins = sorted(protein_embeddings.items())
    new_protein_start = int(data["protein"].x.shape[0])
    if sorted_proteins:
        protein_vectors = []
        for offset, (protein_id, embedding) in enumerate(sorted_proteins):
            if protein_id in protein_map:
                raise ValueError(f"protein_id already exists in graph: {protein_id}")
            protein_map[protein_id] = new_protein_start + offset
            protein_vectors.append(_tensor_from_embedding(embedding, data["protein"].x.dtype))
        protein_matrix = torch.stack(protein_vectors, dim=0)
        data["protein"].x = _append_node_features(data["protein"].x, protein_matrix)

        phage_src = torch.full((len(sorted_proteins),), phage_index, dtype=torch.long)
        protein_dst = torch.arange(
            new_protein_start,
            new_protein_start + len(sorted_proteins),
            dtype=torch.long,
        )
        forward_edges = torch.stack([phage_src, protein_dst], dim=0)
        reverse_edges = torch.stack([protein_dst, phage_src], dim=0)
        data[PHAGE_ENCODES].edge_index = _append_edge_index(
            _existing_edge_index(data, PHAGE_ENCODES),
            forward_edges,
        )
        data[PROTEIN_ENCODED_BY_PHAGE].edge_index = _append_edge_index(
            _existing_edge_index(data, PROTEIN_ENCODED_BY_PHAGE),
            reverse_edges,
        )

    similarity_edges: list[list[int]] = []
    for src_id, dst_id, _edge_type, _weight in similarity_rows:
        src_idx = phage_map.get(src_id)
        dst_idx = phage_map.get(dst_id)
        if src_idx is None or dst_idx is None:
            continue
        similarity_edges.append([src_idx, dst_idx])
    if similarity_edges:
        similarity_tensor = torch.tensor(similarity_edges, dtype=torch.long).t().contiguous()
        data[PHAGE_INTERACTS].edge_index = _append_edge_index(
            _existing_edge_index(data, PHAGE_INTERACTS),
            similarity_tensor,
        )

    updated_maps = dict(node_maps)
    updated_maps["phage_map"] = phage_map
    updated_maps["protein_map"] = protein_map
    counts = {
        "added_phage_nodes": 1,
        "added_protein_nodes": len(sorted_proteins),
        "added_phage_protein_edges": len(sorted_proteins),
        "added_reverse_protein_edges": len(sorted_proteins),
        "added_phage_similarity_edges": len(similarity_rows),
    }
    return phage_index, updated_maps, counts


def _safe_torch_load_graph(path: Path) -> HeteroData:
    try:
        from torch_geometric.data.storage import BaseStorage
        import torch.serialization as torch_serialization

        torch_serialization.add_safe_globals([BaseStorage])
    except Exception:
        pass
    return torch.load(path, map_location="cpu", weights_only=False)


def _prepare_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"execution": config.get("execution", {})}


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_train_manifest_path() -> Path:
    return DEFAULT_TRAIN_MANIFEST.resolve()


def _default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH.resolve()


def load_inference_assets(args: argparse.Namespace) -> InferenceAssets:
    manifest_path = _resolve_path(args.manifest) or _default_train_manifest_path()
    manifest = _read_manifest(manifest_path) if manifest_path.exists() else {}

    config_path = _resolve_path(args.config)
    if config_path is None:
        manifest_config_path = manifest.get("config_path")
        if manifest_config_path:
            candidate = Path(str(manifest_config_path)).expanduser()
            if candidate.exists():
                config_path = candidate.resolve()
    if config_path is None:
        config_path = _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            "Unable to resolve an inference config file. "
            f"Tried manifest={manifest_path} and default={_default_config_path()}."
        )

    config = prepare_config(config_path, [])
    runtime_config = _prepare_runtime_config(config)
    command = manifest.get("command", [])
    params = manifest.get("params", {})

    checkpoint = _resolve_path(args.checkpoint) or _checkpoint_from_manifest(manifest)
    graph = _resolve_path(args.graph) or Path(_arg_value(command, "--data_pt", str(DEFAULT_GRAPH))).resolve()
    node_maps = _resolve_path(args.node_maps) or Path(_arg_value(command, "--node_maps", str(DEFAULT_NODE_MAPS))).resolve()
    host_catalog = _resolve_path(args.host_catalog) or Path(config["build_catalogs"]["host_catalog"]).resolve()
    taxonomy_tree = _resolve_path(args.taxonomy_tree) or Path(config["inputs"]["taxonomy_alias_parquet"]).resolve()
    taxid2species = _resolve_path(args.taxid2species) or Path(
        _arg_value(command, "--taxid2species_tsv", str(DEFAULT_TAXID2SPECIES))
    ).resolve()
    train_script = (
        _resolve_path(args.train_script)
        or _resolve_path(manifest.get("script_path"))
        or Path(config["tools"]["train_script"]).resolve()
    )

    similarity_cfg = config["cluster_assets"]["similarity_edges"]
    phage_similarity_cfg = similarity_cfg["phage"]
    phage_signatures_dir = _resolve_path(args.phage_signatures_dir) or (
        Path(config["cluster_assets"]["sourmash_work_dir"]).resolve() / "phage_phage" / "signatures"
    )

    assets = InferenceAssets(
        manifest_path=manifest_path,
        config_path=config_path,
        runtime_config=runtime_config,
        train_script=train_script,
        checkpoint=checkpoint,
        graph=graph,
        node_maps=node_maps,
        host_catalog=host_catalog,
        taxonomy_tree=taxonomy_tree,
        taxid2species=taxid2species,
        conda_bin=str(config.get("execution", {}).get("conda_bin", DEFAULT_CONDA_BIN)),
        dna_embed_script=Path(config["tools"]["dna_embed_script"]).resolve(),
        dna_model=str(config["dna_embedding"]["phage"]["model"]),
        dna_k=int(config["dna_embedding"]["phage"]["k"]),
        dna_window_tokens=int(config["dna_embedding"]["phage"]["window_tokens"]),
        dna_stride_tokens=int(config["dna_embedding"]["phage"]["stride_tokens"]),
        dna_batch_size=int(config["dna_embedding"]["phage"]["batch_size"]),
        dna_precision=str(config["dna_embedding"]["phage"]["precision"]),
        dna_max_windows=(
            int(config["dna_embedding"]["phage"]["max_windows"])
            if config["dna_embedding"]["phage"].get("max_windows") is not None
            else None
        ),
        dna_seed=int(config["dna_embedding"]["phage"]["seed"]),
        phanotate_bin=str(config["phage_protein_prep"].get("phanotate_bin", "phanotate.py")),
        phanotate_extra_args=tuple(str(arg) for arg in config["phage_protein_prep"].get("extra_args", [])),
        phage_esm_script=Path(config["tools"]["phage_esm_script"]).resolve(),
        phage_esm_model_name=str(config["phage_protein_embedding"]["model_name"]),
        phage_esm_repr_l=int(config["phage_protein_embedding"]["repr_l"]),
        phage_esm_batch_size=int(config["phage_protein_embedding"]["batch_size"]),
        phage_esm_workers=min(int(config["phage_protein_embedding"]["workers"]), 1),
        sourmash_env=str(similarity_cfg.get("sourmash_env", "sourmash_env")),
        sourmash_bin=str(similarity_cfg.get("sourmash_bin", "sourmash")),
        phage_similarity_kmer_size=int(phage_similarity_cfg["kmer_size"]),
        phage_similarity_scaled=int(similarity_cfg.get("scaled", 1000)),
        phage_similarity_threshold=float(phage_similarity_cfg["threshold"]),
        phage_signatures_dir=phage_signatures_dir,
        hidden_dim=int(params.get("hidden_dim", 256)),
        out_dim=int(params.get("out_dim", 256)),
        n_layers=int(params.get("n_layers", 2)),
        n_heads=int(params.get("n_heads", 4)),
        dropout=float(params.get("dropout", 0.15)),
        relation_aggr=str(params.get("relation_aggr", "attention")),
        model_seed=int(params.get("seed", 13)),
    )

    for required_path in (
        assets.checkpoint,
        assets.graph,
        assets.node_maps,
        assets.host_catalog,
        assets.taxonomy_tree,
        assets.taxid2species,
        assets.dna_embed_script,
        assets.phage_esm_script,
        assets.train_script,
        assets.phage_signatures_dir,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required inference asset missing: {required_path}")
    return assets


def _link_single_input(input_path: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target_path = directory / input_path.name
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    target_path.symlink_to(input_path)
    return target_path


def _run_logged_command(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    *,
    stdout_path: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = stdout_path or log_path
    with log_path.open("w", encoding="utf-8") as log_handle:
        if stdout_path == log_path:
            subprocess.run(command, check=True, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            return
        with stdout_path.open("w", encoding="utf-8") as stdout_handle:
            subprocess.run(command, check=True, env=env, stdout=stdout_handle, stderr=log_handle)


def _write_temp_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_dna_embedding(input_path: Path, phage_id: str, work_dir: Path, assets: InferenceAssets) -> torch.Tensor:
    fasta_dir = work_dir / "dna" / "input"
    out_dir = work_dir / "dna" / "out"
    log_path = work_dir / "logs" / "dna_embed.log"
    _link_single_input(input_path, fasta_dir)
    command = [
        "python",
        str(assets.dna_embed_script),
        "--fasta_dir",
        str(fasta_dir),
        "--out_dir",
        str(out_dir),
        "--model",
        assets.dna_model,
        "--k",
        str(assets.dna_k),
        "--window_tokens",
        str(assets.dna_window_tokens),
        "--stride_tokens",
        str(assets.dna_stride_tokens),
        "--batch_size",
        str(assets.dna_batch_size),
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
        "--precision",
        assets.dna_precision,
        "--log",
        str(log_path),
        "--seed",
        str(assets.dna_seed),
    ]
    if assets.dna_max_windows is not None:
        command.extend(["--max_windows", str(assets.dna_max_windows)])
    wrapped = wrap_command_with_env(assets.runtime_config, "dna_embed_phage", command)
    env = subprocess_env(assets.runtime_config, "dna_embed_phage")
    _run_logged_command(wrapped, env, log_path)
    parquet_path = out_dir / f"{input_path.stem}.parquet"
    if not parquet_path.exists():
        raise RuntimeError(f"DNA embedding parquet missing: {parquet_path}")
    embedding = aggregate_sequence_embeddings(parquet_path, emb_col="embedding")
    LOGGER.info("Prepared DNA embedding for %s (dim=%d)", phage_id, embedding.numel())
    return embedding


def run_phanotate(input_path: Path, work_dir: Path, assets: InferenceAssets) -> Path:
    faa_dir = work_dir / "proteins" / "phage_faa"
    faa_dir.mkdir(parents=True, exist_ok=True)
    faa_path = faa_dir / f"{input_path.stem}.faa"
    log_path = work_dir / "logs" / "phanotate.log"
    command = [assets.phanotate_bin, str(input_path), "-f", "faa", *assets.phanotate_extra_args]
    wrapped = wrap_command_with_env(assets.runtime_config, "prepare_phage_proteins", command)
    env = subprocess_env(assets.runtime_config, "prepare_phage_proteins")
    _run_logged_command(wrapped, env, log_path, stdout_path=faa_path)
    if not faa_path.exists() or faa_path.stat().st_size == 0:
        raise RuntimeError(f"Phanotate produced no FAA output: {faa_path}")
    LOGGER.info("Predicted phage proteins to %s", faa_path)
    return faa_path


def run_protein_embedding(faa_path: Path, work_dir: Path, assets: InferenceAssets) -> dict[str, torch.Tensor]:
    out_dir = work_dir / "proteins" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "logs" / "phage_esm.log"
    command = [
        "python",
        str(assets.phage_esm_script),
        "--faa-dir",
        str(faa_path.parent),
        "--out",
        str(out_dir),
        "--model-name",
        assets.phage_esm_model_name,
        "--batch-size",
        str(assets.phage_esm_batch_size),
        "--repr-l",
        str(assets.phage_esm_repr_l),
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
        "--workers",
        str(assets.phage_esm_workers),
    ]
    wrapped = wrap_command_with_env(assets.runtime_config, "embed_phage_proteins", command)
    env = subprocess_env(assets.runtime_config, "embed_phage_proteins")
    _run_logged_command(wrapped, env, log_path)
    pickle_path = out_dir / f"{faa_path.stem}.pkl"
    if not pickle_path.exists():
        raise RuntimeError(f"Phage protein embedding pickle missing: {pickle_path}")
    payload = _load_pickle(pickle_path)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Phage protein embedding pickle is empty: {pickle_path}")
    result: dict[str, torch.Tensor] = {}
    for protein_id, embedding in payload.items():
        result[str(protein_id)] = _tensor_from_embedding(embedding, torch.float32)
    LOGGER.info("Embedded %d phage proteins", len(result))
    return result


def _run_sourmash_command(
    assets: InferenceAssets,
    command: list[str],
    log_path: Path,
) -> None:
    env = dict(os.environ)
    env["NUMEXPR_MAX_THREADS"] = "64"
    env["NUMEXPR_NUM_THREADS"] = "64"
    wrapped = [assets.conda_bin, "run", "--no-capture-output", "-n", assets.sourmash_env, *command]
    _run_logged_command(wrapped, env, log_path)


def run_similarity_search(
    input_path: Path,
    phage_id: str,
    work_dir: Path,
    assets: InferenceAssets,
) -> list[tuple[str, str, str, float]]:
    sourmash_dir = work_dir / "sourmash"
    sourmash_dir.mkdir(parents=True, exist_ok=True)
    query_sig = sourmash_dir / f"{phage_id}.sig"
    sketch_log = work_dir / "logs" / "sourmash_sketch.log"
    _run_sourmash_command(
        assets,
        [
            assets.sourmash_bin,
            "sketch",
            "dna",
            "-p",
            f"k={assets.phage_similarity_kmer_size},scaled={assets.phage_similarity_scaled},abund",
            "-o",
            str(query_sig),
            str(input_path),
        ],
        sketch_log,
    )
    if not query_sig.exists():
        raise RuntimeError(f"Query signature missing after sourmash sketch: {query_sig}")

    compare_script = sourmash_dir / "compare_query_signatures.py"
    compare_output = sourmash_dir / "similarities.tsv"
    compare_log = work_dir / "logs" / "sourmash_compare.log"
    _write_temp_script(
        compare_script,
        """import csv
import sys
from pathlib import Path

from sourmash import load_file_as_signatures

query_path = Path(sys.argv[1])
signatures_dir = Path(sys.argv[2])
output_path = Path(sys.argv[3])

query_sig = next(iter(load_file_as_signatures(str(query_path))))
query_minhash = query_sig.minhash

with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\\t")
    writer.writerow(["existing_id", "score"])
    for sig_path in sorted(signatures_dir.glob("*.sig")):
        sig = next(iter(load_file_as_signatures(str(sig_path))))
        score = query_minhash.jaccard(sig.minhash)
        writer.writerow([sig_path.stem, f"{score:.6f}"])
""",
    )
    _run_sourmash_command(
        assets,
        [
            "python",
            str(compare_script),
            str(query_sig),
            str(assets.phage_signatures_dir),
            str(compare_output),
        ],
        compare_log,
    )
    similarities: dict[str, float] = {}
    with compare_output.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            similarities[str(row["existing_id"])] = float(row["score"])
    rows = build_similarity_edge_rows(
        phage_id,
        similarities,
        threshold=assets.phage_similarity_threshold,
    )
    LOGGER.info("Retained %d phage-phage similarity edges", len(rows))
    return rows


def prepare_query(input_path: Path, work_dir: Path, assets: InferenceAssets) -> PreparedQuery:
    phage_id = input_path.stem
    dna_embedding = run_dna_embedding(input_path, phage_id, work_dir, assets)
    faa_path = run_phanotate(input_path, work_dir, assets)
    protein_embeddings = run_protein_embedding(faa_path, work_dir, assets)
    similarity_rows = run_similarity_search(input_path, phage_id, work_dir, assets)
    return PreparedQuery(
        phage_id=phage_id,
        fasta_path=input_path,
        dna_embedding=dna_embedding,
        protein_embeddings=protein_embeddings,
        similarity_rows=similarity_rows,
    )


def _host_taxid_array(data: HeteroData, host_catalog: Path, host_map: dict[str, int]) -> np.ndarray:
    if hasattr(data["host"], "taxid") and data["host"].taxid is not None:
        taxid_tensor = data["host"].taxid.detach().cpu().numpy()
        if len(taxid_tensor) == len(host_map):
            return taxid_tensor.astype(np.int64, copy=False)
    df = pd.read_parquet(host_catalog, columns=["host_gcf", "host_species_taxid"]).drop_duplicates("host_gcf")
    index_to_taxid = np.full(len(host_map), -1, dtype=np.int64)
    for row in df.itertuples(index=False):
        host_idx = host_map.get(str(row.host_gcf))
        if host_idx is None:
            continue
        if pd.isna(row.host_species_taxid):
            continue
        index_to_taxid[host_idx] = int(row.host_species_taxid)
    return index_to_taxid


def load_model(assets: InferenceAssets, data: HeteroData, device: str) -> torch.nn.Module:
    training_module = _training_module(assets.train_script)
    in_dims = {}
    for node_type in data.node_types:
        if "x" not in data[node_type]:
            raise RuntimeError(f"Node type missing x features: {node_type}")
        in_dims[node_type] = int(data[node_type].x.shape[1])
    model = training_module.GATv2MiniModel(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=assets.hidden_dim,
        out_dim=assets.out_dim,
        n_layers=assets.n_layers,
        n_heads=assets.n_heads,
        dropout=assets.dropout,
        decoder="cosine",
        use_edge_attr=True,
        edge_attr_dim=1,
        rel_init_map=EDGE_TYPE_WEIGHT_MAP,
        relation_aggr=assets.relation_aggr,
    ).to(device)
    checkpoint = torch.load(assets.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def score_hosts(
    model: torch.nn.Module,
    data: HeteroData,
    query_phage_idx: int,
    device: str,
) -> torch.Tensor:
    compute_device = _default_device(device)
    with torch.inference_mode():
        x_dict = {node_type: data[node_type].x.to(compute_device) for node_type in data.node_types}
        edge_index_dict = {}
        for edge_type in data.edge_types:
            edge_index = getattr(data[edge_type], "edge_index", None)
            if edge_index is None:
                continue
            edge_index_dict[edge_type] = edge_index.to(compute_device)
        out = model(x_dict, edge_index_dict, edge_attr_dict=None)
        phage_emb = out["phage"][query_phage_idx].view(1, -1)
        host_emb = out["host"]
        logits = torch.matmul(phage_emb, host_emb.t()).view(-1)
        logits = logits * torch.exp(model.logit_scale)
        probabilities = torch.sigmoid(logits).detach().cpu()
    return probabilities


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def _save_result_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_file",
                "phage_id",
                "mode",
                "top_host_id",
                "top_host_taxid",
                "top_species",
                "top_genus",
                "score",
                "checkpoint",
                "work_dir",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(row)


def _cleanup_work_dir(work_dir: Path, keep_paths: list[Path]) -> None:
    keep_paths = [path.resolve() for path in keep_paths if path.exists()]
    if not keep_paths:
        shutil.rmtree(work_dir, ignore_errors=True)
        return
    for path in sorted(work_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        resolved = path.resolve()
        if any(resolved == keep or keep in resolved.parents for keep in keep_paths):
            continue
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        work_dir.rmdir()
    except OSError:
        pass


def _work_dir_for(args: argparse.Namespace, phage_id: str) -> Path:
    if args.work_dir:
        return Path(args.work_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (PROJECT_ROOT / "artifacts" / "inference" / f"{phage_id}_{timestamp}").resolve()


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA missing: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_FASTA_SUFFIXES:
        raise ValueError(f"Unsupported FASTA suffix: {input_path.suffix}")

    phage_id = input_path.stem
    work_dir = _work_dir_for(args, phage_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    assets = load_inference_assets(args)
    node_maps = _read_json(assets.node_maps)
    if phage_id in node_maps.get("phage_map", {}):
        raise ValueError(
            f"Input stem '{phage_id}' already exists in the training graph. Rename the input file first."
        )

    query = prepare_query(input_path, work_dir, assets)
    data = _safe_torch_load_graph(assets.graph)
    query_phage_idx, updated_maps, counts = augment_graph_with_query(
        data,
        node_maps,
        query.phage_id,
        query.dna_embedding,
        query.protein_embeddings,
        query.similarity_rows,
    )

    model = load_model(assets, data, device=_default_device(args.device))
    host_scores = score_hosts(model, data, query_phage_idx, device=_default_device(args.device))
    top_host_idx = int(torch.argmax(host_scores).item())
    score = float(host_scores[top_host_idx].item())

    host_map = updated_maps["host_map"]
    host_idx_to_id = {int(index): host_id for host_id, index in host_map.items()}
    host_id = host_idx_to_id[top_host_idx]
    host_taxids = _host_taxid_array(data, assets.host_catalog, host_map)
    host_taxid = int(host_taxids[top_host_idx])
    species_lookup = {
        int(row.taxid): str(row.species)
        for row in pd.read_csv(assets.taxid2species, sep="\t").itertuples(index=False)
    }
    taxonomy_nodes = _taxonomy_nodes(assets.taxonomy_tree)
    top_species = resolve_species_name(host_taxid, species_lookup)
    top_genus = resolve_genus_name(host_taxid, taxonomy_nodes)

    _write_json(
        work_dir / "augmented_graph_summary.json",
        {
            "phage_id": query.phage_id,
            "counts": counts,
            "similarity_edges": query.similarity_rows,
            "query_phage_index": query_phage_idx,
        },
    )

    result = {
        "input_file": str(input_path),
        "phage_id": query.phage_id,
        "mode": args.mode,
        "top_host_id": host_id,
        "top_host_taxid": host_taxid,
        "top_species": top_species,
        "top_genus": top_genus,
        "score": round(score, 6),
        "checkpoint": str(assets.checkpoint),
        "work_dir": str(work_dir),
    }
    output_path = Path(args.output).expanduser().resolve()
    _save_result_tsv(output_path, result)
    if args.cleanup:
        _cleanup_work_dir(work_dir, keep_paths=[output_path] if output_path.exists() else [])
        result["work_dir"] = str(work_dir)
    selected_label = top_species if args.mode == "species" else top_genus
    print(
        f"{query.phage_id}\tmode={args.mode}\tlabel={selected_label}\thost={host_id}\tscore={score:.6f}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    run_inference(args)
    return 0


__all__ = [
    "PreparedQuery",
    "augment_graph_with_query",
    "build_similarity_edge_rows",
    "main",
    "resolve_genus_name",
]
