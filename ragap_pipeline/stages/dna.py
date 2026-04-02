from __future__ import annotations

from typing import Any


def _cfg(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    side = "phage" if stage_name == "dna_embed_phage" else "host"
    return config["dna_embedding"][side]


def inputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = _cfg(config, stage_name)
    values = [cfg["fasta_dir"]]
    model = str(cfg.get("model", ""))
    if "/" in model:
        values.append(model)
    return values


def outputs(config: dict[str, Any], stage_name: str) -> list[str]:
    return [_cfg(config, stage_name)["out_dir"]]


def params(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = _cfg(config, stage_name)
    ignore = {"mode", "script", "python", "fasta_dir", "out_dir", "log_path", "validate", "deps"}
    return {key: value for key, value in cfg.items() if key not in ignore}


def script_path(config: dict[str, Any], stage_name: str) -> str:
    return str(_cfg(config, stage_name)["script"])


def command(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = _cfg(config, stage_name)
    cmd = [
        cfg.get("python") or config.get("python_bin", "python"),
        cfg["script"],
        "--fasta_dir",
        cfg["fasta_dir"],
        "--out_dir",
        cfg["out_dir"],
        "--model",
        str(cfg["model"]),
        "--k",
        str(cfg.get("k", 6)),
        "--window_tokens",
        str(cfg.get("window_tokens", 510)),
        "--stride_tokens",
        str(cfg.get("stride_tokens", 510)),
        "--batch_size",
        str(cfg.get("batch_size", 8)),
        "--device",
        str(cfg.get("device", "cuda")),
        "--precision",
        str(cfg.get("precision", "fp32")),
        "--log",
        str(cfg.get("log_path", f"{cfg['out_dir']}/dna_embed.log")),
        "--seed",
        str(cfg.get("seed", 42)),
    ]
    if cfg.get("rc", False):
        cmd.append("--rc")
    if cfg.get("max_windows") is not None:
        cmd.extend(["--max_windows", str(cfg["max_windows"])])
    return cmd
