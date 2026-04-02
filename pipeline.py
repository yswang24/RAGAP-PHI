#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_BASE_PYTHON = os.environ.get("RAGAP_BOOTSTRAP_PYTHON", "")


if os.environ.get("RAGAP_SKIP_BASE_BOOTSTRAP") != "1" and DEFAULT_BASE_PYTHON:
    target_python = Path(DEFAULT_BASE_PYTHON).expanduser().resolve()
    current_python = Path(sys.executable).expanduser().resolve()
    if target_python.exists() and current_python != target_python:
        env = dict(os.environ)
        env_root = str(target_python.parents[1])
        env["RAGAP_SKIP_BASE_BOOTSTRAP"] = "1"
        env["RAGAP_BASE_PYTHON"] = str(target_python)
        env["RAGAP_BOOTSTRAP_PYTHON"] = str(target_python)
        env["CONDA_PREFIX"] = env_root
        env["PATH"] = f"{env_root}/bin:{env.get('PATH', '')}"
        previous_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{env_root}/lib:{previous_ld}" if previous_ld else f"{env_root}/lib"
        os.execve(str(target_python), [str(target_python), __file__, *sys.argv[1:]], env)

from ragap_pipeline.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
