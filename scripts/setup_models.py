#!/usr/bin/env python3
"""Download and verify third-party model weights for RAGAP-PHI inference.

Usage:
    python scripts/setup_models.py            # download both
    python scripts/setup_models.py --dnabert   # download DNABERT-4 only
    python scripts/setup_models.py --esm       # download ESM2 only
    python scripts/setup_models.py --verify    # verify only, no download
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DNABERT_DIR = PROJECT_ROOT / "assets" / "models" / "DNA_bert_4"
DNABERT_REPO = "zhihan1996/DNA_bert_4"
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"


def check_dnabert() -> bool:
    required = {"config.json", "pytorch_model.bin", "vocab.txt"}
    if not DNABERT_DIR.exists():
        return False
    existing = {f.name for f in DNABERT_DIR.iterdir() if f.is_file()}
    return required.issubset(existing)


def download_dnabert() -> None:
    if check_dnabert():
        print(f"[OK] DNABERT-4 already present at {DNABERT_DIR}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)
    DNABERT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] DNABERT-4 from {DNABERT_REPO} -> {DNABERT_DIR}")
    snapshot_download(
        repo_id=DNABERT_REPO,
        local_dir=str(DNABERT_DIR),
        local_dir_use_symlinks=False,
    )
    if check_dnabert():
        print("[OK] DNABERT-4 downloaded successfully")
    else:
        print("[ERROR] DNABERT-4 download incomplete")
        sys.exit(1)


def check_esm() -> bool:
    """Check if ESM2 weights are cached. Works without fair-esm installed."""
    import os
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cached = cache_dir / f"{ESM_MODEL_NAME}.pt"
    if cached.exists() and cached.stat().st_size > 100_000_000:
        return True
    # Also check HF cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for p in hf_cache.glob(f"*{ESM_MODEL_NAME}*"):
        if p.is_dir():
            return True
    return False


def download_esm() -> None:
    try:
        import torch
        import esm
    except ImportError:
        print("[ERROR] fair-esm or torch not installed. Activate the esm_env first.")
        sys.exit(1)
    print(f"[DOWNLOAD] ESM2 ({ESM_MODEL_NAME}) weights (cached by fair-esm)")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    del model, alphabet
    print("[OK] ESM2 weights downloaded and cached")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAGAP-PHI model weights")
    parser.add_argument("--dnabert", action="store_true", help="Download DNABERT-4 only")
    parser.add_argument("--esm", action="store_true", help="Download ESM2 only")
    parser.add_argument("--verify", action="store_true", help="Verify only, no download")
    args = parser.parse_args()

    do_all = not args.dnabert and not args.esm and not args.verify

    print("=" * 50)
    print("RAGAP-PHI Model Setup")
    print("=" * 50)

    # DNABERT-4
    if do_all or args.dnabert or args.verify:
        if args.verify:
            status = "OK" if check_dnabert() else "MISSING"
            print(f"[{status}] DNABERT-4 at {DNABERT_DIR}")
        else:
            download_dnabert()

    # ESM2
    if do_all or args.esm or args.verify:
        if args.verify:
            status = "OK" if check_esm() else "MISSING"
            print(f"[{status}] ESM2 ({ESM_MODEL_NAME})")
        else:
            download_esm()

    print("=" * 50)
    dnabert_ok = check_dnabert()
    esm_ok = check_esm()
    if dnabert_ok and esm_ok:
        print("All models ready.")
    else:
        print("Some models missing. Run without --verify to download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
