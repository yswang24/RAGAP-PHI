from __future__ import annotations

import os
import shlex
import stat
import subprocess
from pathlib import Path
from typing import Any

from .config import STAGE_ORDER
from .execution import base_python_path
from .utils import dump_json, ensure_dir, utc_now


SBATCH_OPTION_MAP = {
    "account": "--account",
    "cpus_per_task": "--cpus-per-task",
    "gres": "--gres",
    "mem": "--mem",
    "ntasks": "--ntasks",
    "partition": "--partition",
    "qos": "--qos",
    "time": "--time",
}


def slurm_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("slurm", {})


def sbatch_bin(config: dict[str, Any]) -> str:
    return str(slurm_config(config).get("sbatch_bin", "sbatch"))


def stage_slurm_options(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    slurm = slurm_config(config)
    defaults = dict(slurm.get("defaults", {}))
    per_stage = slurm.get("per_stage", {})
    if isinstance(per_stage, dict):
        defaults.update(per_stage.get(stage_name, {}))
    defaults.setdefault("ntasks", 1)
    return defaults


def resolved_config_path(config: dict[str, Any]) -> Path:
    return Path(config["_variables"]["slurm_dir"]) / "resolved_config.yaml"


def stage_script_path(config: dict[str, Any], stage_name: str) -> Path:
    return Path(config["_variables"]["slurm_dir"]) / f"{stage_name}.sbatch.sh"


def stage_log_paths(config: dict[str, Any], stage_name: str) -> tuple[Path, Path]:
    root = Path(config["_variables"]["slurm_dir"])
    return root / f"{stage_name}.%j.out", root / f"{stage_name}.%j.err"


def job_name(config: dict[str, Any], stage_name: str) -> str:
    dataset_id = str(config.get("dataset_id", "ragap"))
    return f"ragap_{dataset_id}_{stage_name}"


def stage_runner_command(config: dict[str, Any], stage_name: str, force: bool) -> list[str]:
    command = [
        base_python_path(config),
        config["_pipeline_entry"],
        "run",
        "--config",
        str(resolved_config_path(config)),
        "--from",
        stage_name,
        "--to",
        stage_name,
    ]
    if force:
        command.extend(["--force-stage", stage_name])
    return command


def write_stage_script(config: dict[str, Any], stage_name: str, force: bool) -> Path:
    script_path = stage_script_path(config, stage_name)
    ensure_dir(script_path.parent)
    base_python = Path(base_python_path(config)).resolve()
    base_env_root = base_python.parents[1]
    project_root = Path(config["_project_root"]).resolve()
    ld_prefix = f"{base_env_root}/lib"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(project_root))}",
        f"export RAGAP_SKIP_BASE_BOOTSTRAP=1",
        f"export RAGAP_BASE_PYTHON={shlex.quote(str(base_python))}",
        f"export CONDA_PREFIX={shlex.quote(str(base_env_root))}",
        f"export PATH={shlex.quote(str(base_env_root / 'bin'))}:$PATH",
        f'export LD_LIBRARY_PATH={shlex.quote(ld_prefix)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}',
        "export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-128}",
        "export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-128}",
        f"exec {shlex.join(stage_runner_command(config, stage_name, force))}",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    current_mode = script_path.stat().st_mode
    script_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


def sbatch_command(
    config: dict[str, Any],
    stage_name: str,
    force: bool,
    dependency_ids: list[str],
) -> list[str]:
    output_path, error_path = stage_log_paths(config, stage_name)
    command = [
        sbatch_bin(config),
        "--parsable",
        "--job-name",
        job_name(config, stage_name),
        "--output",
        str(output_path),
        "--error",
        str(error_path),
        "--chdir",
        str(Path(config["_project_root"]).resolve()),
    ]
    options = stage_slurm_options(config, stage_name)
    for key, flag in SBATCH_OPTION_MAP.items():
        value = options.get(key)
        if value in (None, ""):
            continue
        command.extend([flag, str(value)])
    if dependency_ids:
        command.extend(["--dependency", "afterok:" + ":".join(dependency_ids)])
    command.append(str(write_stage_script(config, stage_name, force)))
    return command


def submit_stage_jobs(
    config: dict[str, Any],
    targets: list[str],
    statuses: dict[str, Any],
    forced: set[str],
    dry_run: bool,
    stage_deps_fn,
) -> int:
    scheduled: list[str] = []
    for stage_name in targets:
        if statuses[stage_name].state == "valid" and stage_name not in forced:
            print(f"[skip] {stage_name}: cached")
            continue
        invalid_external = [
            dep
            for dep in stage_deps_fn(config, stage_name)
            if dep not in targets and getattr(statuses[dep], "state", None) != "valid"
        ]
        if invalid_external:
            raise RuntimeError(
                f"Cannot submit {stage_name}: upstream invalid outside selection: {', '.join(invalid_external)}"
            )
        scheduled.append(stage_name)

    if not scheduled:
        print("[slurm] no stages need submission")
        return 0

    job_ids: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for stage_name in scheduled:
        dependency_ids = [job_ids[dep] for dep in stage_deps_fn(config, stage_name) if dep in job_ids]
        command = sbatch_command(config, stage_name, stage_name in forced, dependency_ids)
        if dry_run:
            job_ids[stage_name] = f"dryrun-{stage_name}"
            print(f"[dry-run][slurm] {stage_name} {shlex.join(command)}")
            records.append(
                {
                    "stage": stage_name,
                    "command": command,
                    "dependencies": dependency_ids,
                    "submitted_at": utc_now(),
                    "dry_run": True,
                }
            )
            continue

        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        raw_job_id = completed.stdout.strip().splitlines()[-1]
        job_id = raw_job_id.split(";", 1)[0]
        job_ids[stage_name] = job_id
        print(f"[submit] {stage_name}: job_id={job_id}")
        records.append(
            {
                "stage": stage_name,
                "job_id": job_id,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "command": command,
                "dependencies": dependency_ids,
                "submitted_at": utc_now(),
                "dry_run": False,
            }
        )

    latest_path = Path(config["_variables"]["slurm_dir"]) / "latest_submission.json"
    dump_json(
        latest_path,
        {
            "dataset_id": config.get("dataset_id"),
            "targets": targets,
            "scheduled": scheduled,
            "records": records,
        },
    )
    return len(scheduled)
