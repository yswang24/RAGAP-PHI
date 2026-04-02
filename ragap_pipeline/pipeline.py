from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "128")

from .config import DEFAULT_CONFIG, STAGE_ORDER, dump_yaml, get_nested, prepare_config, public_config
from .execution import stage_runtime, subprocess_env, wrap_command_with_env
from .manifest import collect_path_signature, load_manifest, manifest_path, signature_map, stage_digest_from_manifest, write_manifest
from .slurm import submit_stage_jobs
from .stages import STAGES, StageRunResult
from .utils import ensure_dir, json_hash, remove_path, utc_now
from .validators import validate_stage


@dataclass
class StageStatus:
    stage: str
    mode: str
    state: str
    reason: str
    manifest_path: str
    outputs: list[str]
    stage_digest: str | None = None


def stage_config(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    spec = STAGES[stage_name]
    cfg = copy.deepcopy(get_nested(config, spec.section_path))
    cfg.setdefault("mode", spec.default_mode)
    return cfg


def stage_mode(config: dict[str, Any], stage_name: str) -> str:
    return str(stage_config(config, stage_name).get("mode", STAGES[stage_name].default_mode))


def snapshot_entries(config: dict[str, Any], stage_name: str) -> list[dict[str, str]]:
    return list(stage_config(config, stage_name).get("snapshots", []))


def stage_deps(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = stage_config(config, stage_name)
    if cfg.get("skip_upstream", False):
        return []
    if "deps" in cfg:
        return list(cfg["deps"])
    return list(STAGES[stage_name].default_deps)


def stage_inputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = stage_config(config, stage_name)
    if cfg.get("mode") == "snapshot":
        return [entry["source"] for entry in snapshot_entries(config, stage_name)]
    return STAGES[stage_name].module.inputs(config, stage_name)


def stage_outputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = stage_config(config, stage_name)
    if cfg.get("mode") == "snapshot":
        return [entry["target"] for entry in snapshot_entries(config, stage_name)]
    return STAGES[stage_name].module.outputs(config, stage_name)


def stage_params(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = stage_config(config, stage_name)
    if cfg.get("mode") == "snapshot":
        ignore = {"mode", "snapshots", "validate", "deps", "skip_upstream"}
        return {key: value for key, value in cfg.items() if key not in ignore}
    return STAGES[stage_name].module.params(config, stage_name)


def stage_script_path(config: dict[str, Any], stage_name: str) -> str:
    cfg = stage_config(config, stage_name)
    if cfg.get("mode") == "snapshot":
        return "<snapshot>"
    return STAGES[stage_name].module.script_path(config, stage_name)


def stage_schema_config(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    return {
        "built_in": stage_name,
        "custom": stage_config(config, stage_name).get("validate", {}),
    }


def stage_manifest_payload(
    config: dict[str, Any],
    stage_name: str,
    upstream_digests: dict[str, str | None],
    virtual_paths: set[str] | None = None,
) -> dict[str, Any]:
    outputs = stage_outputs(config, stage_name)
    return {
        "stage": stage_name,
        "script_path": stage_script_path(config, stage_name),
        "runtime": stage_runtime(config, stage_name, stage_cfg=stage_config(config, stage_name)),
        "inputs": signature_map(stage_inputs(config, stage_name), virtual_paths=virtual_paths),
        "params": stage_params(config, stage_name),
        "upstream_digests": upstream_digests,
        "outputs": signature_map(outputs, virtual_paths=virtual_paths),
        "schema_checks": {
            "config": stage_schema_config(config, stage_name),
        },
    }


def _status(stage_name: str, mode: str, state: str, reason: str, config: dict[str, Any], stage_digest: str | None = None) -> StageStatus:
    return StageStatus(
        stage=stage_name,
        mode=mode,
        state=state,
        reason=reason,
        manifest_path=str(manifest_path(config, stage_name)),
        outputs=stage_outputs(config, stage_name),
        stage_digest=stage_digest,
    )


def stage_state(
    config: dict[str, Any],
    stage_name: str,
    known: dict[str, StageStatus],
    virtual_paths: set[str] | None = None,
) -> StageStatus:
    mode = stage_mode(config, stage_name)
    deps = stage_deps(config, stage_name)
    invalid_deps = [dep for dep in deps if known[dep].state != "valid"]
    if invalid_deps:
        return _status(stage_name, mode, "blocked", f"upstream invalid: {', '.join(invalid_deps)}", config)

    inputs_sig = signature_map(stage_inputs(config, stage_name), virtual_paths=virtual_paths)
    missing_inputs = [path for path, sig in inputs_sig.items() if not sig["exists"]]
    if missing_inputs:
        return _status(stage_name, mode, "blocked", f"inputs missing: {', '.join(missing_inputs)}", config)

    outputs_sig = signature_map(stage_outputs(config, stage_name), virtual_paths=virtual_paths)
    missing_outputs = [path for path, sig in outputs_sig.items() if not sig["exists"]]
    manifest_file = manifest_path(config, stage_name)
    manifest = load_manifest(manifest_file)
    upstream_digests = {dep: known[dep].stage_digest for dep in deps}
    current_payload = stage_manifest_payload(config, stage_name, upstream_digests, virtual_paths=virtual_paths)

    if missing_outputs:
        reason = f"outputs missing: {', '.join(missing_outputs)}"
        if manifest is None:
            return _status(stage_name, mode, "ready", reason, config)
        return _status(stage_name, mode, "stale", reason, config)

    if virtual_paths and any(path in virtual_paths for path in stage_outputs(config, stage_name)):
        digest = json_hash(current_payload)
        return _status(stage_name, mode, "valid", "dry-run simulated", config, stage_digest=digest)

    validation = validate_stage(stage_name, config, known)
    if not validation["ok"]:
        reason = "; ".join(validation["errors"])
        if manifest is None:
            return _status(stage_name, mode, "ready", reason, config)
        return _status(stage_name, mode, "stale", reason, config)

    if manifest is None:
        return _status(stage_name, mode, "ready", "manifest missing", config)
    if manifest.get("status") != "success":
        return _status(stage_name, mode, "stale", f"previous run status={manifest.get('status')}", config)

    expected = {
        "script_path": current_payload["script_path"],
        "runtime": current_payload["runtime"],
        "inputs": current_payload["inputs"],
        "params": current_payload["params"],
        "upstream_digests": current_payload["upstream_digests"],
        "outputs": current_payload["outputs"],
        "schema_config": stage_schema_config(config, stage_name),
    }
    actual = {
        "script_path": manifest.get("script_path"),
        "runtime": manifest.get("runtime"),
        "inputs": manifest.get("inputs"),
        "params": manifest.get("params"),
        "upstream_digests": manifest.get("upstream_digests"),
        "outputs": manifest.get("outputs"),
        "schema_config": manifest.get("schema_checks", {}).get("config"),
    }
    if expected != actual:
        return _status(stage_name, mode, "stale", "manifest mismatch", config)
    return _status(stage_name, mode, "valid", "cached", config, stage_digest=stage_digest_from_manifest(manifest))


def evaluate_pipeline(config: dict[str, Any], virtual_paths: set[str] | None = None) -> dict[str, StageStatus]:
    statuses: dict[str, StageStatus] = {}
    for stage_name in STAGE_ORDER:
        statuses[stage_name] = stage_state(config, stage_name, statuses, virtual_paths=virtual_paths)
    return statuses


def ensure_output_dirs(paths: list[str]) -> None:
    for path in paths:
        candidate = Path(path)
        if candidate.suffix:
            ensure_dir(candidate.parent)
        else:
            ensure_dir(candidate)


def clear_stage_outputs(config: dict[str, Any], stage_name: str) -> None:
    for path in stage_outputs(config, stage_name):
        candidate = Path(path)
        if candidate.exists() or candidate.is_symlink():
            remove_path(candidate)
    candidate = manifest_path(config, stage_name)
    if candidate.exists() or candidate.is_symlink():
        remove_path(candidate)


def run_snapshot_stage(config: dict[str, Any], stage_name: str) -> StageRunResult:
    entries = snapshot_entries(config, stage_name)
    if not entries:
        raise RuntimeError(f"snapshot stage has no snapshot entries: {stage_name}")
    for entry in entries:
        source = Path(entry["source"])
        target = Path(entry["target"])
        if not source.exists():
            raise FileNotFoundError(f"snapshot source missing: {source}")
        ensure_dir(target.parent)
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            remove_path(target)
        os.symlink(str(source.resolve()), str(target), target_is_directory=source.is_dir())
    return StageRunResult(command=None, notes={"snapshots": len(entries)})


def run_script_stage(config: dict[str, Any], stage_name: str) -> StageRunResult:
    module = STAGES[stage_name].module
    if hasattr(module, "pre_run"):
        module.pre_run(config, stage_name)
    command = wrap_command_with_env(
        config,
        stage_name,
        module.command(config, stage_name),
        stage_cfg=stage_config(config, stage_name),
    )
    ensure_output_dirs(stage_outputs(config, stage_name))
    subprocess.run(command, check=True, env=subprocess_env(config, stage_name, stage_cfg=stage_config(config, stage_name)))
    notes: dict[str, Any] = {}
    if hasattr(module, "post_run"):
        notes = module.post_run(config, stage_name)
    return StageRunResult(command=command, notes=notes)


def run_internal_stage(config: dict[str, Any], stage_name: str) -> StageRunResult:
    module = STAGES[stage_name].module
    ensure_output_dirs(stage_outputs(config, stage_name))
    notes = module.run_internal(config, stage_name)
    return StageRunResult(command=None, notes=notes)


def write_stage_manifest(
    config: dict[str, Any],
    stage_name: str,
    statuses: dict[str, StageStatus],
    result: StageRunResult,
    started_at: float,
    duration_s: float,
    status: str,
    error: str | None = None,
) -> StageStatus:
    upstream_digests = {dep: statuses[dep].stage_digest for dep in stage_deps(config, stage_name)}
    payload = stage_manifest_payload(config, stage_name, upstream_digests)
    validation = validate_stage(stage_name, config, statuses)
    payload.update(
        {
            "created_at": utc_now(),
            "status": status,
            "description": STAGES[stage_name].description,
            "config_path": config["_config_path"],
            "mode": stage_mode(config, stage_name),
            "runtime": stage_runtime(config, stage_name, stage_cfg=stage_config(config, stage_name)),
            "command": result.command,
            "notes": result.notes,
            "duration_s": duration_s,
            "started_at_epoch": started_at,
            "schema_checks": {
                "config": stage_schema_config(config, stage_name),
                "result": validation,
            },
        }
    )
    if error:
        payload["error"] = error
    write_manifest(manifest_path(config, stage_name), payload)
    return evaluate_pipeline(config)[stage_name]


def run_stage(
    config: dict[str, Any],
    stage_name: str,
    statuses: dict[str, StageStatus],
    dry_run: bool,
    force: bool,
    virtual_paths: set[str] | None = None,
) -> StageStatus:
    status = stage_state(config, stage_name, statuses, virtual_paths=virtual_paths)
    if status.state == "valid" and not force:
        print(f"[skip] {stage_name}: cached")
        return status
    if status.state == "blocked":
        raise RuntimeError(f"Cannot run {stage_name}: {status.reason}")

    mode = stage_mode(config, stage_name)
    if dry_run:
        preview = None
        if mode == "script":
            preview = wrap_command_with_env(
                config,
                stage_name,
                STAGES[stage_name].module.command(config, stage_name),
                stage_cfg=stage_config(config, stage_name),
            )
        if preview:
            print(f"[dry-run] {stage_name} ({mode}) {shlex.join(preview)}")
        else:
            print(f"[dry-run] {stage_name} ({mode})")
        return _status(stage_name, mode, "valid", "dry-run simulated", config, stage_digest=json_hash({"stage": stage_name, "mode": mode}))

    if force:
        clear_stage_outputs(config, stage_name)

    print(f"[run] {stage_name} ({mode})")
    started_at = time.time()
    result = StageRunResult()
    try:
        if mode == "snapshot":
            result = run_snapshot_stage(config, stage_name)
        elif mode == "script":
            result = run_script_stage(config, stage_name)
        elif mode == "internal":
            result = run_internal_stage(config, stage_name)
        else:
            raise RuntimeError(f"Unsupported mode '{mode}' for {stage_name}")
        duration_s = time.time() - started_at
        final_status = write_stage_manifest(
            config=config,
            stage_name=stage_name,
            statuses=statuses,
            result=result,
            started_at=started_at,
            duration_s=duration_s,
            status="success",
        )
        if final_status.state != "valid":
            raise RuntimeError(f"{stage_name} finished but validation failed: {final_status.reason}")
        return final_status
    except Exception as exc:
        duration_s = time.time() - started_at
        write_stage_manifest(
            config=config,
            stage_name=stage_name,
            statuses=statuses,
            result=result,
            started_at=started_at,
            duration_s=duration_s,
            status="failed",
            error=str(exc),
        )
        raise


def stage_range(from_stage: str | None, to_stage: str | None) -> list[str]:
    start = STAGE_ORDER.index(from_stage) if from_stage else 0
    end = STAGE_ORDER.index(to_stage) if to_stage else len(STAGE_ORDER) - 1
    if start > end:
        raise ValueError("--from must not come after --to.")
    return STAGE_ORDER[start : end + 1]


def descendants(config: dict[str, Any], stage_name: str) -> set[str]:
    reverse: dict[str, set[str]] = {name: set() for name in STAGE_ORDER}
    for name in STAGE_ORDER:
        for dep in stage_deps(config, name):
            reverse[dep].add(name)
    visited = {stage_name}
    queue = [stage_name]
    while queue:
        current = queue.pop(0)
        for child in sorted(reverse[current]):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return visited


def normalize_forced_stages(values: list[str]) -> set[str]:
    forced: set[str] = set()
    for raw in values:
        for part in [item.strip() for item in raw.split(",") if item.strip()]:
            forced.add(validate_stage_name(part))
    return forced


def expand_forced_stages(config: dict[str, Any], forced: set[str], allowed: list[str]) -> set[str]:
    allowed_set = set(allowed)
    expanded: set[str] = set()
    for stage_name in forced:
        expanded.update(descendants(config, stage_name))
    return expanded & allowed_set


def print_status_table(statuses: dict[str, StageStatus]) -> None:
    headers = ("stage", "state", "mode", "reason")
    rows = [(name, statuses[name].state, statuses[name].mode, statuses[name].reason) for name in STAGE_ORDER]
    widths = [max(len(headers[idx]), max(len(str(row[idx])) for row in rows)) for idx in range(len(headers))]
    print("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))


def validate_stage_name(name: str) -> str:
    if name not in STAGES:
        raise argparse.ArgumentTypeError(f"Unknown stage '{name}'. Allowed: {', '.join(STAGE_ORDER)}")
    return name


def parse_args(argv: list[str]) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to pipeline YAML config.")
    common.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values with key=value.")
    common.add_argument("--submit-slurm", action="store_true", help="Reserved for SLURM submission.")
    common.add_argument("--dry-run", action="store_true", help="Print planned actions without executing them.")
    common.add_argument("--force-stage", action="append", default=[], help="Force rerun of one or more stage names.")

    parser = argparse.ArgumentParser(description="Unified cluster pipeline for RAGAP.", parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a stage range.", parents=[common])
    run_parser.add_argument("--from", dest="from_stage", type=validate_stage_name, help="First stage to run.")
    run_parser.add_argument("--to", dest="to_stage", type=validate_stage_name, help="Last stage to run.")

    stage_parser = subparsers.add_parser("stage", help="Run a stage and its downstream dependents.", parents=[common])
    stage_parser.add_argument("stage_name", type=validate_stage_name)

    train_parser = subparsers.add_parser("train", help="Run the train stage only.", parents=[common])
    train_parser.add_argument("--from-graph", action="store_true", help="Run build_graph and train together.")

    subparsers.add_parser("status", help="Show stage cache state.", parents=[common])
    return parser.parse_args(argv)


def command_targets(config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.command == "run":
        return stage_range(args.from_stage, args.to_stage)
    if args.command == "stage":
        target_set = descendants(config, args.stage_name)
        return [stage for stage in STAGE_ORDER if stage in target_set]
    if args.command == "train":
        return ["build_graph", "train"] if args.from_graph else ["train"]
    return []


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config = prepare_config(Path(args.config).resolve(), args.overrides)
    ensure_dir(config["_variables"]["artifact_root"])
    ensure_dir(config["_variables"]["slurm_dir"])
    dump_yaml(Path(config["_variables"]["slurm_dir"]) / "resolved_config.yaml", public_config(config))

    statuses = evaluate_pipeline(config)
    if args.command == "status":
        print_status_table(statuses)
        return 0

    if args.command == "train" and statuses["build_graph"].state != "valid" and not args.from_graph:
        raise RuntimeError(f"Refusing to train directly because build_graph is {statuses['build_graph'].state}: {statuses['build_graph'].reason}")

    targets = command_targets(config, args)
    forced = expand_forced_stages(config, normalize_forced_stages(args.force_stage), targets)
    if args.submit_slurm:
        submit_stage_jobs(
            config=config,
            targets=targets,
            statuses=statuses,
            forced=forced,
            dry_run=args.dry_run,
            stage_deps_fn=stage_deps,
        )
        print_status_table(statuses)
        return 0

    virtual_paths: set[str] = set()
    for stage_name in targets:
        if not args.dry_run:
            statuses = evaluate_pipeline(config)
        stage_status = run_stage(
            config=config,
            stage_name=stage_name,
            statuses=statuses,
            dry_run=args.dry_run,
            force=stage_name in forced,
            virtual_paths=virtual_paths if args.dry_run else None,
        )
        if args.dry_run:
            statuses[stage_name] = stage_status
            virtual_paths.update(stage_outputs(config, stage_name))
            statuses = evaluate_pipeline(config, virtual_paths=virtual_paths)
        else:
            statuses = evaluate_pipeline(config)

    print_status_table(statuses)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
