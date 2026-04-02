from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FASTA_SUFFIXES = (".fasta", ".fa", ".fna")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_hash(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def looks_like_path(value: str) -> bool:
    if value.startswith("/"):
        return True
    if value.startswith("./") or value.startswith("../") or value.startswith("~/"):
        return True
    return "/" in value


def resolve_path_like(base_dir: Path, value: str) -> str:
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    if looks_like_path(expanded):
        return os.path.abspath(base_dir / expanded)
    return expanded


def iter_fasta_files(path_value: str) -> list[Path]:
    root = Path(path_value)
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    files = [path for path in sorted(root.iterdir()) if path.suffix.lower() in FASTA_SUFFIXES]
    return files


def list_files(path_value: str, suffix: str) -> list[Path]:
    root = Path(path_value)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.name.endswith(suffix))


def to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [float(value)]
    return [float(item) for item in value]


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def dump_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)
