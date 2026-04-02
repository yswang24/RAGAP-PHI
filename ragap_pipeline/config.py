from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .utils import resolve_path_like


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.fullhost_v2.yaml"

STAGE_ORDER = [
    "dna_embed_phage",
    "dna_embed_host",
    "build_catalogs",
    "build_pairs",
    "prepare_phage_proteins",
    "prepare_host_proteins",
    "embed_phage_proteins",
    "embed_host_proteins",
    "build_cluster_assets",
    "build_graph",
    "train",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override '{dotted_key}': '{key}' is not a mapping.")
        cursor = child
    cursor[keys[-1]] = value


def get_nested(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return {}
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise ValueError(f"Config path {'.'.join(path)} must resolve to a mapping.")
    return cursor


def render_templates(obj: Any, variables: dict[str, str], base_dir: Path) -> Any:
    if isinstance(obj, dict):
        return {key: render_templates(value, variables, base_dir) for key, value in obj.items()}
    if isinstance(obj, list):
        return [render_templates(value, variables, base_dir) for value in obj]
    if isinstance(obj, str):
        rendered = obj.format(**variables)
        return resolve_path_like(base_dir, rendered)
    return obj


def _collect_scalar_variables(mapping: dict[str, Any], variables: dict[str, str]) -> None:
    for key, value in mapping.items():
        if isinstance(value, (str, int, float)):
            variables[key] = str(value)


def build_variables(config: dict[str, Any]) -> dict[str, str]:
    project_root = str(Path(config.get("project_root", PROJECT_ROOT)).resolve())
    dataset_id = str(config.get("dataset_id", "ragap_cluster_650"))
    raw_artifact_root = str(config.get("artifact_root", "{project_root}/artifacts/{dataset_id}"))
    artifact_root = raw_artifact_root.format(project_root=project_root, dataset_id=dataset_id)
    artifact_root = str(Path(os.path.expanduser(artifact_root)).resolve())
    variables = {
        "project_root": project_root,
        "dataset_id": dataset_id,
        "artifact_root": artifact_root,
        "manifest_root": os.path.join(artifact_root, "manifests"),
        "dna_dir": os.path.join(artifact_root, "dna"),
        "catalog_dir": os.path.join(artifact_root, "catalogs"),
        "pairs_dir": os.path.join(artifact_root, "pairs"),
        "protein_dir": os.path.join(artifact_root, "proteins"),
        "cluster_dir": os.path.join(artifact_root, "cluster"),
        "graph_dir": os.path.join(artifact_root, "graph"),
        "train_dir": os.path.join(artifact_root, "train"),
        "slurm_dir": os.path.join(artifact_root, "slurm"),
    }
    for section_name in ("inputs", "tools"):
        section = config.get(section_name, {})
        if isinstance(section, dict):
            _collect_scalar_variables(section, variables)
    return variables


def prepare_config(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    raw = load_yaml(config_path)
    raw.setdefault("project_root", str(PROJECT_ROOT))
    raw.setdefault("dataset_id", "ragap_cluster_650")
    raw.setdefault("artifact_root", "{project_root}/artifacts/{dataset_id}")
    raw.setdefault("python_bin", "python")
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --set value '{override}'. Expected key=value.")
        key, raw_value = override.split("=", 1)
        set_nested(raw, key, yaml.safe_load(raw_value))
    variables = build_variables(raw)
    rendered = render_templates(raw, variables, config_path.parent.resolve())
    rendered["_variables"] = variables
    rendered["_config_path"] = str(config_path.resolve())
    rendered["_project_root"] = str(PROJECT_ROOT)
    rendered["_pipeline_entry"] = str(PROJECT_ROOT / "pipeline.py")
    return rendered


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    for key in list(payload):
        if key.startswith("_"):
            payload.pop(key, None)
    return payload
