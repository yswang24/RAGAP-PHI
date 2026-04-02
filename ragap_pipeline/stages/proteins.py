from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..execution import subprocess_env, wrap_command_with_env
from ..utils import dump_json, ensure_dir, iter_fasta_files, list_files

PHAGE_BACKEND_PHAROKKA = "pharokka"
PHAGE_BACKEND_PHANOTATE_DIRECT = "phanotate_direct"


def _prep_cfg(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name == "prepare_phage_proteins":
        return config["phage_protein_prep"]
    return config["host_protein_prep"]


def _embed_cfg(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name == "embed_phage_proteins":
        return config["phage_protein_embedding"]
    return config["host_protein_embedding"]


def inputs(config: dict[str, Any], stage_name: str) -> list[str]:
    if stage_name.startswith("prepare_"):
        cfg = _prep_cfg(config, stage_name)
        values = [cfg["fasta_dir"]]
        database = cfg.get("database")
        if stage_name == "prepare_phage_proteins" and _phage_backend(cfg) == PHAGE_BACKEND_PHAROKKA and database:
            values.append(database)
        return values
    cfg = _embed_cfg(config, stage_name)
    return [cfg["faa_dir"]]


def outputs(config: dict[str, Any], stage_name: str) -> list[str]:
    if stage_name == "prepare_phage_proteins":
        cfg = config["phage_protein_prep"]
        paths = [cfg["faa_dir"]]
        if cfg.get("keep_annotation", False):
            paths.append(cfg["annotation_dir"])
        return paths
    if stage_name == "prepare_host_proteins":
        return [config["host_protein_prep"]["faa_dir"]]
    cfg = _embed_cfg(config, stage_name)
    return [cfg["out_dir"], cfg["failure_report"]]


def params(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = _prep_cfg(config, stage_name) if stage_name.startswith("prepare_") else _embed_cfg(config, stage_name)
    ignore = {
        "mode",
        "script",
        "python",
        "fasta_dir",
        "faa_dir",
        "annotation_dir",
        "database",
        "out_dir",
        "failure_report",
        "validate",
        "deps",
    }
    return {key: value for key, value in cfg.items() if key not in ignore}


def script_path(config: dict[str, Any], stage_name: str) -> str:
    if stage_name.startswith("prepare_"):
        cfg = _prep_cfg(config, stage_name)
        if stage_name == "prepare_phage_proteins":
            if _phage_backend(cfg) == PHAGE_BACKEND_PHANOTATE_DIRECT:
                return str(cfg.get("phanotate_bin", "phanotate.py"))
            return str(cfg.get("pharokka_bin", "pharokka.py"))
        return str(cfg.get("prodigal_bin", "prodigal"))
    return str(_embed_cfg(config, stage_name)["script"])


def command(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = _embed_cfg(config, stage_name)
    return [
        cfg.get("python") or config.get("python_bin", "python"),
        cfg["script"],
        "--faa-dir",
        cfg["faa_dir"],
        "--out",
        cfg["out_dir"],
        "--model-name",
        str(cfg["model_name"]),
        "--batch-size",
        str(cfg.get("batch_size", 4)),
        "--repr-l",
        str(cfg.get("repr_l", 32)),
        "--device",
        str(cfg.get("device", "cuda")),
        "--workers",
        str(cfg.get("workers", 4)),
    ]


def _best_faa_candidate(annotation_dir: Path) -> Path | None:
    candidates = sorted(annotation_dir.rglob("*.faa"))
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_size, path.name))
    return candidates[-1]


def _phage_backend(cfg: dict[str, Any]) -> str:
    backend = str(cfg.get("backend", PHAGE_BACKEND_PHANOTATE_DIRECT)).strip().lower()
    aliases = {
        "phanotate": PHAGE_BACKEND_PHANOTATE_DIRECT,
        "direct": PHAGE_BACKEND_PHANOTATE_DIRECT,
    }
    backend = aliases.get(backend, backend)
    if backend not in {PHAGE_BACKEND_PHAROKKA, PHAGE_BACKEND_PHANOTATE_DIRECT}:
        raise RuntimeError(f"Unsupported phage protein prep backend: {backend}")
    return backend


def _phage_annotation_dir(cfg: dict[str, Any], fasta_path: Path) -> Path:
    return Path(cfg["annotation_dir"]) / fasta_path.stem


def _phage_annotation_complete(cfg: dict[str, Any], fasta_path: Path) -> bool:
    if not cfg.get("keep_annotation", False):
        return True
    sample_dir = _phage_annotation_dir(cfg, fasta_path)
    if not sample_dir.is_dir():
        return False
    if _phage_backend(cfg) == PHAGE_BACKEND_PHANOTATE_DIRECT:
        tabular_path = sample_dir / "phanotate_out.txt"
        copied_faa = sample_dir / f"{fasta_path.stem}.faa"
        return (
            tabular_path.exists()
            and tabular_path.stat().st_size > 0
            and copied_faa.exists()
            and copied_faa.stat().st_size > 0
        )
    return any(path.stat().st_size > 0 for path in sample_dir.rglob("*.faa"))


def _run_command_to_file(
    config: dict[str, Any],
    stage_name: str,
    stage_cfg: dict[str, Any],
    command: list[str],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            subprocess.run(
                wrap_command_with_env(config, stage_name, command, stage_cfg=stage_cfg),
                check=True,
                env=subprocess_env(config, stage_name, stage_cfg=stage_cfg),
                stdout=handle,
            )
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Command produced an empty file: {output_path}")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _run_prepare_phage_pharokka(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["phage_protein_prep"]
    ensure_dir(cfg["faa_dir"])
    if cfg.get("keep_annotation", False):
        ensure_dir(cfg["annotation_dir"])
    fasta_files = iter_fasta_files(cfg["fasta_dir"])
    created = 0
    for fasta_path in fasta_files:
        target_faa = Path(cfg["faa_dir"]) / f"{fasta_path.stem}.faa"
        if target_faa.exists() and target_faa.stat().st_size > 0 and _phage_annotation_complete(cfg, fasta_path):
            continue
        work_dir = Path(cfg["annotation_dir"]) / fasta_path.stem
        if work_dir.exists():
            shutil.rmtree(work_dir)
        command = [
            str(cfg.get("pharokka_bin", "pharokka.py")),
            "-i",
            str(fasta_path),
            "-o",
            str(work_dir),
            "-t",
            str(cfg.get("threads", 8)),
        ]
        if cfg.get("database"):
            command.extend(["-d", str(cfg["database"])])
        if cfg.get("gene_predictor"):
            command.extend(["-g", str(cfg["gene_predictor"])])
        for arg in cfg.get("extra_args", []):
            command.append(str(arg))
        subprocess.run(
            wrap_command_with_env(config, "prepare_phage_proteins", command, stage_cfg=cfg),
            check=True,
            env=subprocess_env(config, "prepare_phage_proteins", stage_cfg=cfg),
        )
        source_faa = _best_faa_candidate(work_dir)
        if source_faa is None:
            raise RuntimeError(f"Pharokka did not produce a .faa file for {fasta_path}")
        shutil.copyfile(source_faa, target_faa)
        created += 1
        if not cfg.get("keep_annotation", False):
            shutil.rmtree(work_dir)
    return {
        "backend": PHAGE_BACKEND_PHAROKKA,
        "fasta_files": len(fasta_files),
        "new_faa_files": created,
    }


def _run_prepare_phage_direct_one(config: dict[str, Any], fasta_path: Path) -> tuple[int, int]:
    cfg = config["phage_protein_prep"]
    target_faa = Path(cfg["faa_dir"]) / f"{fasta_path.stem}.faa"
    if target_faa.exists() and target_faa.stat().st_size > 0 and _phage_annotation_complete(cfg, fasta_path):
        return 0, 0

    faa_command = [str(cfg.get("phanotate_bin", "phanotate.py")), str(fasta_path), "-f", "faa"]
    for arg in cfg.get("extra_args", []):
        faa_command.append(str(arg))
    _run_command_to_file(config, "prepare_phage_proteins", cfg, faa_command, target_faa)

    if not cfg.get("keep_annotation", False):
        return 1, 0
    sample_dir = _phage_annotation_dir(cfg, fasta_path)
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    ensure_dir(sample_dir)
    shutil.copyfile(target_faa, sample_dir / f"{fasta_path.stem}.faa")

    tabular_command = [str(cfg.get("phanotate_bin", "phanotate.py")), str(fasta_path), "-f", "tabular"]
    for arg in cfg.get("extra_args", []):
        tabular_command.append(str(arg))
    _run_command_to_file(
        config,
        "prepare_phage_proteins",
        cfg,
        tabular_command,
        sample_dir / "phanotate_out.txt",
    )
    return 1, 1


def _run_prepare_phage_direct(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["phage_protein_prep"]
    ensure_dir(cfg["faa_dir"])
    if cfg.get("keep_annotation", False):
        ensure_dir(cfg["annotation_dir"])
    fasta_files = iter_fasta_files(cfg["fasta_dir"])
    workers = max(1, int(cfg.get("workers", 1)))
    created = 0
    annotation_created = 0
    if workers == 1 or len(fasta_files) <= 1:
        for fasta_path in fasta_files:
            new_faa, new_annotation = _run_prepare_phage_direct_one(config, fasta_path)
            created += new_faa
            annotation_created += new_annotation
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(fasta_files))) as executor:
            for new_faa, new_annotation in executor.map(
                lambda fasta_path: _run_prepare_phage_direct_one(config, fasta_path),
                fasta_files,
            ):
                created += new_faa
                annotation_created += new_annotation
    return {
        "backend": PHAGE_BACKEND_PHANOTATE_DIRECT,
        "fasta_files": len(fasta_files),
        "new_faa_files": created,
        "annotation_dirs": annotation_created,
        "workers": workers,
    }


def _run_prepare_phage(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["phage_protein_prep"]
    backend = _phage_backend(cfg)
    if backend == PHAGE_BACKEND_PHAROKKA:
        return _run_prepare_phage_pharokka(config)
    return _run_prepare_phage_direct(config)


def _run_prepare_host(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["host_protein_prep"]
    ensure_dir(cfg["faa_dir"])
    fasta_files = iter_fasta_files(cfg["fasta_dir"])
    created = 0
    for fasta_path in fasta_files:
        target_faa = Path(cfg["faa_dir"]) / f"{fasta_path.stem}.faa"
        if target_faa.exists() and target_faa.stat().st_size > 0:
            continue
        command = [
            str(cfg.get("prodigal_bin", "prodigal")),
            "-i",
            str(fasta_path),
            "-a",
            str(target_faa),
            "-d",
            "/dev/null",
            "-o",
            "/dev/null",
        ]
        if cfg.get("meta", False):
            command.extend(["-p", "meta"])
        for arg in cfg.get("extra_args", []):
            command.append(str(arg))
        subprocess.run(
            wrap_command_with_env(config, "prepare_host_proteins", command, stage_cfg=cfg),
            check=True,
            env=subprocess_env(config, "prepare_host_proteins", stage_cfg=cfg),
        )
        created += 1
    return {"fasta_files": len(fasta_files), "new_faa_files": created}


def run_internal(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if stage_name == "prepare_phage_proteins":
        return _run_prepare_phage(config)
    if stage_name == "prepare_host_proteins":
        return _run_prepare_host(config)
    raise RuntimeError(f"Unsupported internal protein stage: {stage_name}")


def post_run(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = _embed_cfg(config, stage_name)
    expected = sorted(path.stem for path in list_files(cfg["faa_dir"], ".faa"))
    produced = sorted(path.stem for path in list_files(cfg["out_dir"], ".pkl"))
    missing = sorted(set(expected) - set(produced))
    payload = {
        "stage": stage_name,
        "faa_files": expected,
        "pkl_files": produced,
        "missing": missing,
        "failed": missing,
    }
    dump_json(cfg["failure_report"], payload)
    if missing:
        raise RuntimeError(f"{stage_name} missing embeddings for {len(missing)} FAA files")
    return {"faa_files": len(expected), "pkl_files": len(produced)}
