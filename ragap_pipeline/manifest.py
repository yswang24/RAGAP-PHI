from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .utils import ensure_dir, json_hash


def manifest_path(config: dict[str, Any], stage_name: str) -> Path:
    return Path(config["_variables"]["manifest_root"]) / f"{stage_name}.json"


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _directory_signature(path: Path, display_path: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    dir_count = 0
    total_size = 0
    latest_mtime_ns = 0
    samples: list[dict[str, Any]] = []

    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        root_path = Path(root)
        relative_root = root_path.relative_to(path)
        for directory in dirs:
            rel = (relative_root / directory).as_posix()
            digest.update(f"D|{rel}\n".encode("utf-8"))
            dir_count += 1
        for filename in files:
            candidate = root_path / filename
            stat_result = candidate.stat()
            rel = (relative_root / filename).as_posix()
            digest.update(f"F|{rel}|{stat_result.st_size}|{stat_result.st_mtime_ns}\n".encode("utf-8"))
            file_count += 1
            total_size += stat_result.st_size
            latest_mtime_ns = max(latest_mtime_ns, stat_result.st_mtime_ns)
            if len(samples) < 20:
                samples.append(
                    {
                        "path": rel,
                        "size": stat_result.st_size,
                        "mtime_ns": stat_result.st_mtime_ns,
                    }
                )

    return {
        "path": display_path,
        "exists": True,
        "kind": "dir",
        "realpath": str(path.resolve()),
        "dir_count": dir_count,
        "file_count": file_count,
        "total_size": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "tree_hash": digest.hexdigest(),
        "sample_files": samples,
    }


def collect_path_signature(path: str, virtual_paths: set[str] | None = None) -> dict[str, Any]:
    if virtual_paths and path in virtual_paths:
        return {"path": path, "exists": True, "kind": "virtual"}
    candidate = Path(path)
    if not candidate.exists():
        return {"path": path, "exists": False}
    if candidate.is_file():
        stat_result = candidate.stat()
        return {
            "path": path,
            "exists": True,
            "kind": "file",
            "realpath": str(candidate.resolve()),
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        }
    return _directory_signature(candidate, path)


def signature_map(paths: list[str], virtual_paths: set[str] | None = None) -> dict[str, Any]:
    return {path: collect_path_signature(path, virtual_paths=virtual_paths) for path in paths}


def stage_digest_from_manifest(manifest: dict[str, Any]) -> str:
    return json_hash(
        {
            "stage": manifest.get("stage"),
            "script_path": manifest.get("script_path"),
            "runtime": manifest.get("runtime"),
            "inputs": manifest.get("inputs"),
            "params": manifest.get("params"),
            "upstream_digests": manifest.get("upstream_digests"),
            "outputs": manifest.get("outputs"),
            "schema_checks": manifest.get("schema_checks"),
            "status": manifest.get("status"),
        }
    )
