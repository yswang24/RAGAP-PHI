from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONDA_BIN = os.environ.get("RAGAP_CONDA_BIN") or os.environ.get("CONDA_EXE") or "conda"
DEFAULT_BASE_ENV = os.environ.get("RAGAP_BASE_ENV", "PHPGAT")
DEFAULT_BASE_PYTHON = os.environ.get("RAGAP_BOOTSTRAP_PYTHON", sys.executable)


def _default_envs_root() -> str:
    env_root = os.environ.get("RAGAP_ENVS_ROOT")
    if env_root:
        return env_root
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe:
        conda_path = Path(conda_exe).expanduser().resolve()
        if conda_path.name == "conda" and conda_path.parent.name == "bin":
            return str(conda_path.parents[1] / "envs")
    return str(Path.home() / "miniconda3" / "envs")


DEFAULT_ENVS_ROOT = _default_envs_root()
DEFAULT_STAGE_ENVS = {
    "dna_embed_phage": "dnaberts",
    "dna_embed_host": "dnaberts",
    "build_catalogs": "PHPGAT",
    "build_pairs": "PHPGAT",
    "prepare_phage_proteins": "pharokka_env",
    "prepare_host_proteins": "pharokka_env",
    "embed_phage_proteins": "esm_env",
    "embed_host_proteins": "esm_env",
    "build_cluster_assets": "PHPGAT",
    "build_graph": "PHPGAT",
    "train": "PHPGAT",
}


def execution_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("execution", {})


def conda_bin(config: dict[str, Any]) -> str:
    return str(execution_config(config).get("conda_bin", DEFAULT_CONDA_BIN))


def base_env_name(config: dict[str, Any]) -> str:
    return str(execution_config(config).get("base_env", DEFAULT_BASE_ENV))


def base_python_path(config: dict[str, Any]) -> str:
    return str(execution_config(config).get("base_python", DEFAULT_BASE_PYTHON))


def envs_root(config: dict[str, Any]) -> str:
    return str(execution_config(config).get("envs_root", DEFAULT_ENVS_ROOT))


def _heuristic_stage_env(stage_name: str) -> str:
    if stage_name.startswith("dna_embed_"):
        return "dnaberts"
    if stage_name.startswith("prepare_"):
        return "pharokka_env"
    if stage_name.startswith("embed_") and "proteins" in stage_name:
        return "esm_env"
    return DEFAULT_BASE_ENV


def resolved_stage_env(config: dict[str, Any], stage_name: str, stage_cfg: dict[str, Any] | None = None) -> str:
    if stage_cfg and stage_cfg.get("conda_env"):
        return str(stage_cfg["conda_env"])
    exec_cfg = execution_config(config)
    per_stage = exec_cfg.get("stage_envs", {})
    if isinstance(per_stage, dict) and stage_name in per_stage:
        return str(per_stage[stage_name])
    if stage_name in DEFAULT_STAGE_ENVS:
        return DEFAULT_STAGE_ENVS[stage_name]
    return _heuristic_stage_env(stage_name)


def stage_runtime(config: dict[str, Any], stage_name: str, stage_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    env_name = resolved_stage_env(config, stage_name, stage_cfg=stage_cfg)
    env_root = str(Path(envs_root(config)) / env_name)
    return {
        "conda_bin": conda_bin(config),
        "conda_env": env_name,
        "conda_env_root": env_root,
        "base_env": base_env_name(config),
        "base_python": base_python_path(config),
    }


def wrap_command_with_env(config: dict[str, Any], stage_name: str, command: list[str], stage_cfg: dict[str, Any] | None = None) -> list[str]:
    runtime = stage_runtime(config, stage_name, stage_cfg=stage_cfg)
    env_name = runtime["conda_env"]
    if not env_name:
        return command
    return [runtime["conda_bin"], "run", "--no-capture-output", "-n", env_name, *command]


def should_bootstrap_to_base_python() -> bool:
    if os.environ.get("RAGAP_SKIP_BASE_BOOTSTRAP") == "1":
        return False
    current = Path(os.path.realpath(os.sys.executable))
    target = Path(os.path.realpath(os.environ.get("RAGAP_BASE_PYTHON", DEFAULT_BASE_PYTHON)))
    return target.exists() and current != target


def subprocess_env(config: dict[str, Any], stage_name: str, stage_cfg: dict[str, Any] | None = None) -> dict[str, str]:
    runtime = stage_runtime(config, stage_name, stage_cfg=stage_cfg)
    env = dict(os.environ)
    env_root = runtime["conda_env_root"]
    env["CONDA_PREFIX"] = env_root
    env["PATH"] = f"{env_root}/bin:{env.get('PATH', '')}"
    previous_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{env_root}/lib:{previous_ld}" if previous_ld else f"{env_root}/lib"
    env.setdefault("NUMEXPR_MAX_THREADS", "128")
    env.setdefault("NUMEXPR_NUM_THREADS", "128")
    return env
