from __future__ import annotations

from pathlib import Path
from typing import Any


def inputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["train"]
    values = [cfg["data_pt"], cfg["node_maps"]]
    if cfg.get("taxid2species_tsv"):
        values.append(cfg["taxid2species_tsv"])
    return values


def outputs(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["train"]
    return [cfg["out_dir"], str(Path(cfg["out_dir"]) / "run.log"), str(Path(cfg["out_dir"]) / Path(str(cfg["save_path"])).name)]


def params(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    cfg = config["train"]
    ignore = {
        "mode",
        "script",
        "python",
        "data_pt",
        "node_maps",
        "taxid2species_tsv",
        "out_dir",
        "save_path",
        "validate",
        "deps",
    }
    return {key: value for key, value in cfg.items() if key not in ignore}


def script_path(config: dict[str, Any], stage_name: str) -> str:
    return str(config["train"]["script"])


def command(config: dict[str, Any], stage_name: str) -> list[str]:
    cfg = config["train"]
    cmd = [
        cfg.get("python") or config.get("python_bin", "python"),
        cfg["script"],
        "--data_pt",
        cfg["data_pt"],
        "--node_maps",
        cfg["node_maps"],
        "--device",
        str(cfg.get("device", "cuda")),
        "--eval_device",
        str(cfg.get("eval_device", "cpu")),
        "--seed",
        str(cfg.get("seed", 613)),
        "--hidden_dim",
        str(cfg.get("hidden_dim", 128)),
        "--out_dim",
        str(cfg.get("out_dim", 128)),
        "--n_layers",
        str(cfg.get("n_layers", 2)),
        "--n_heads",
        str(cfg.get("n_heads", 1)),
        "--dropout",
        str(cfg.get("dropout", 0.2)),
        "--lr",
        str(cfg.get("lr", 1e-3)),
        "--epochs",
        str(cfg.get("epochs", 30)),
        "--batch_size",
        str(cfg.get("batch_size", 256)),
        "--neg_ratio",
        str(cfg.get("neg_ratio", 2)),
        "--eval_neg_ratio",
        str(cfg.get("eval_neg_ratio", 1)),
        "--tau",
        str(cfg.get("tau", 0.1)),
        "--save_path",
        str(cfg.get("save_path", "best_hgt_nb.pt")),
        "--out_dir",
        cfg["out_dir"],
    ]
    if cfg.get("log_every") is not None:
        cmd.extend(["--log_every", str(cfg["log_every"])])
    if cfg.get("train_log_every") is not None:
        cmd.extend(["--train_log_every", str(cfg["train_log_every"])])
    if cfg.get("eval_every") is not None:
        cmd.extend(["--eval_every", str(cfg["eval_every"])])
    if cfg.get("loader_workers") is not None:
        cmd.extend(["--loader_workers", str(cfg["loader_workers"])])
    if cfg.get("pin_memory") is not None:
        cmd.extend(["--pin_memory", str(cfg["pin_memory"]).lower()])
    if cfg.get("save_eval_predictions") is not None:
        cmd.extend(["--save_eval_predictions", str(cfg["save_eval_predictions"]).lower()])
    if cfg.get("save_final_predictions") is not None:
        cmd.extend(["--save_final_predictions", str(cfg["save_final_predictions"]).lower()])
    if cfg.get("enable_tf32") is not None:
        cmd.extend(["--enable_tf32", str(cfg["enable_tf32"]).lower()])
    if cfg.get("deterministic") is not None:
        cmd.extend(["--deterministic", str(cfg["deterministic"]).lower()])
    if cfg.get("cudnn_benchmark") is not None:
        cmd.extend(["--cudnn_benchmark", str(cfg["cudnn_benchmark"]).lower()])
    if cfg.get("hard_negatives") is not None:
        cmd.extend(["--hard_negatives", str(cfg["hard_negatives"])])
    if cfg.get("patience") is not None:
        cmd.extend(["--patience", str(cfg["patience"])])
    if cfg.get("host_cache_refresh_every") is not None:
        cmd.extend(["--host_cache_refresh_every", str(cfg["host_cache_refresh_every"])])
    if cfg.get("relation_aggr") is not None:
        cmd.extend(["--relation_aggr", str(cfg["relation_aggr"])])
    if cfg.get("train_objective") is not None:
        cmd.extend(["--train_objective", str(cfg["train_objective"])])
    if cfg.get("message_passing_relation_scope") is not None:
        cmd.extend(["--message_passing_relation_scope", str(cfg["message_passing_relation_scope"])])
    neighbors = cfg.get("num_neighbors", [15, 10])
    if neighbors:
        cmd.append("--num_neighbors")
        cmd.extend(str(item) for item in neighbors)
    if cfg.get("taxid2species_tsv"):
        cmd.extend(["--taxid2species_tsv", cfg["taxid2species_tsv"]])
    return cmd
