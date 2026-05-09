#!/usr/bin/env bash
# RAGAP-PHI one-click setup script.
#
# Usage:
#   bash setup.sh                    # full setup (envs + models + bundle)
#   bash setup.sh --envs-only        # create conda environments only
#   bash setup.sh --models-only      # download models only
#   bash setup.sh --bundle-only      # download inference bundle only
#   bash setup.sh --verify           # verify everything is in place
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BIN="${CONDA_EXE:-conda}"
RAGAP_PYTHON=""
ESM_PYTHON=""

# GitHub Release config (edit these if the repo moves)
GITHUB_REPO="yswang24/RAGAP-PHI"
BUNDLE_TAG="v1.0-inference-bundle"

# ── Colors ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

# ── Resolve conda ───────────────────────────────────────
resolve_conda() {
    if command -v conda &>/dev/null; then
        CONDA_BIN="$(command -v conda)"
    elif [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BIN="$CONDA_EXE"
    else
        fail "conda not found. Install Miniconda/Anaconda first."
        exit 1
    fi
    echo "Using conda: $CONDA_BIN"
}

# ── Get env python path ────────────────────────────────
env_python() {
    local env_name="$1"
    local env_path
    env_path="$("$CONDA_BIN" env list 2>/dev/null | awk -v name="$env_name" '$1 == name {print $2; exit}')"
    if [ -n "$env_path" ] && [ -x "$env_path/bin/python" ]; then
        echo "$env_path/bin/python"
        return
    fi
    for base in "$("$CONDA_BIN" info --base 2>/dev/null)" "$HOME/anaconda3" "$HOME/miniconda3"; do
        if [ -x "$base/envs/$env_name/bin/python" ]; then
            echo "$base/envs/$env_name/bin/python"
            return
        fi
    done
    echo ""
}

# ── Step 1: Conda environments ──────────────────────────
setup_envs() {
    echo ""
    echo "════════════════════════════════════════"
    echo " Step 1: Conda environments"
    echo "════════════════════════════════════════"

    if "$CONDA_BIN" env list 2>/dev/null | grep -q "^RAGAP "; then
        ok "RAGAP environment already exists"
    else
        echo "[CREATE] RAGAP environment (this may take a few minutes) ..."
        "$CONDA_BIN" env create -f "$SCRIPT_DIR/envs/RAGAP.yaml"
        ok "RAGAP environment created"
    fi

    if "$CONDA_BIN" env list 2>/dev/null | grep -q "^esm_env "; then
        ok "esm_env environment already exists"
    else
        echo "[CREATE] esm_env environment (this may take a few minutes) ..."
        "$CONDA_BIN" env create -f "$SCRIPT_DIR/envs/esm.yaml"
        ok "esm_env environment created"
    fi

    RAGAP_PYTHON="$(env_python RAGAP)"
    ESM_PYTHON="$(env_python esm_env)"

    echo ""
    echo "[VERIFY] RAGAP packages ..."
    if [ -n "$RAGAP_PYTHON" ]; then
        "$RAGAP_PYTHON" -c "
import torch, transformers, pyarrow, pandas, numpy, yaml
from torch_geometric.data import HeteroData
print('  torch:', torch.__version__)
print('  transformers:', transformers.__version__)
print('  torch_geometric: OK')
" 2>/dev/null && ok "RAGAP ML stack ready" || fail "RAGAP ML packages missing"
        "$RAGAP_PYTHON" -c "from Bio import SeqIO" 2>/dev/null && ok "biopython ready" || fail "biopython missing"
    else
        fail "RAGAP python not found"
    fi

    echo ""
    echo "[VERIFY] esm_env packages ..."
    if [ -n "$ESM_PYTHON" ]; then
        "$ESM_PYTHON" -c "
import torch, esm
print('  torch:', torch.__version__)
print('  fair-esm: OK')
" 2>/dev/null && ok "esm_env stack ready" || fail "esm_env packages missing"
    else
        fail "esm_env python not found"
    fi

    echo ""
    echo "[VERIFY] Bioinformatics tools ..."
    local ragap_bin
    ragap_bin="$(dirname "$RAGAP_PYTHON")"
    "$ragap_bin/phanotate.py" --version &>/dev/null && ok "phanotate ready" || fail "phanotate missing"
    "$ragap_bin/sourmash" --version &>/dev/null && ok "sourmash ready" || fail "sourmash missing"
}

# ── Step 2: Model weights ───────────────────────────────
setup_models() {
    echo ""
    echo "════════════════════════════════════════"
    echo " Step 2: Model weights"
    echo "════════════════════════════════════════"

    RAGAP_PYTHON="${RAGAP_PYTHON:-$(env_python RAGAP)}"
    if [ -z "$RAGAP_PYTHON" ]; then
        fail "RAGAP environment not found. Run with --envs-only first."
        return 1
    fi
    "$RAGAP_PYTHON" "$SCRIPT_DIR/scripts/setup_models.py"
}

# ── Step 3: Inference bundle ────────────────────────────
bundle_complete() {
    test -f "$SCRIPT_DIR/artifacts/ragap_phi/graph/hetero_graph.pt" \
        && test -f "$SCRIPT_DIR/artifacts/ragap_phi/graph/node_maps.json" \
        && test -f "$SCRIPT_DIR/artifacts/ragap_phi/catalogs/host_catalog.parquet" \
        && test -d "$SCRIPT_DIR/artifacts/ragap_phi/cluster/sourmash/phage_phage/signatures" \
        && test -f "$SCRIPT_DIR/artifacts/ragap_phi/train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt"
}

download_bundle() {
    echo ""
    echo "════════════════════════════════════════"
    echo " Step 3: Inference bundle"
    echo "════════════════════════════════════════"

    if bundle_complete; then
        ok "Inference bundle already present"
        return 0
    fi

    local download_dir="$SCRIPT_DIR"
    echo "[DOWNLOAD] Inference bundle from GitHub Release ($BUNDLE_TAG) ..."

    if command -v gh &>/dev/null; then
        echo "  Using gh CLI to download ..."
        gh release download "$BUNDLE_TAG" \
            --repo "$GITHUB_REPO" \
            --pattern "bundle_part_*" \
            --dir "$download_dir" \
            --clobber
        ok "Bundle parts downloaded"
    elif command -v wget &>/dev/null; then
        echo "  gh CLI not found, using wget ..."
        echo "  You may need to download manually from:"
        echo "  https://github.com/$GITHUB_REPO/releases/tag/$BUNDLE_TAG"
        warn "Automatic download requires 'gh' CLI. Install with: conda install -c conda-forge gh"
        return 1
    else
        fail "Neither 'gh' nor 'wget' found."
        echo "  Manual download:"
        echo "    1. Go to https://github.com/$GITHUB_REPO/releases/tag/$BUNDLE_TAG"
        echo "    2. Download all bundle_part_* files to $download_dir/"
        return 1
    fi

    # Merge and extract
    echo "[MERGE] Combining bundle parts ..."
    cat "$download_dir"/bundle_part_* > "$download_dir/ragap_phi_inference_bundle.tar.gz"
    ok "Bundle merged"

    echo "[EXTRACT] Extracting bundle ..."
    tar -xzf "$download_dir/ragap_phi_inference_bundle.tar.gz" -C "$SCRIPT_DIR"
    ok "Bundle extracted to artifacts/ragap_phi/"

    # Cleanup intermediate files
    echo "[CLEANUP] Removing intermediate files ..."
    rm -f "$download_dir"/bundle_part_* "$download_dir/ragap_phi_inference_bundle.tar.gz"
    ok "Cleanup done"

    # Verify
    if bundle_complete; then
        ok "Inference bundle ready"
    else
        fail "Bundle extraction incomplete"
        return 1
    fi
}

check_bundle() {
    echo ""
    echo "════════════════════════════════════════"
    echo " Step 3: Inference bundle"
    echo "════════════════════════════════════════"

    local all_ok=true
    test -f "$SCRIPT_DIR/artifacts/ragap_phi/graph/hetero_graph.pt" \
        && ok "hetero_graph.pt" || { fail "hetero_graph.pt missing"; all_ok=false; }
    test -f "$SCRIPT_DIR/artifacts/ragap_phi/graph/node_maps.json" \
        && ok "node_maps.json" || { fail "node_maps.json missing"; all_ok=false; }
    test -f "$SCRIPT_DIR/artifacts/ragap_phi/catalogs/host_catalog.parquet" \
        && ok "host_catalog.parquet" || { fail "host_catalog.parquet missing"; all_ok=false; }
    test -d "$SCRIPT_DIR/artifacts/ragap_phi/cluster/sourmash/phage_phage/signatures" \
        && ok "phage signatures dir" || { fail "phage signatures missing"; all_ok=false; }
    test -f "$SCRIPT_DIR/artifacts/ragap_phi/train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt" \
        && ok "model checkpoint" || { fail "checkpoint missing"; all_ok=false; }

    if [ "$all_ok" = false ]; then
        echo ""
        warn "Inference bundle incomplete. Run: bash setup.sh --bundle-only"
    fi
}

# ── Verify only ─────────────────────────────────────────
verify_all() {
    echo ""
    echo "════════════════════════════════════════"
    echo " Verification"
    echo "════════════════════════════════════════"

    RAGAP_PYTHON="$(env_python RAGAP)"
    ESM_PYTHON="$(env_python esm_env)"

    [ -n "$RAGAP_PYTHON" ] && ok "RAGAP env" || fail "RAGAP env missing"
    [ -n "$ESM_PYTHON" ]   && ok "esm_env"   || fail "esm_env missing"

    if [ -n "$RAGAP_PYTHON" ]; then
        "$RAGAP_PYTHON" "$SCRIPT_DIR/scripts/setup_models.py" --verify
    fi

    check_bundle
    echo ""
    echo "Done."
}

# ── Main ────────────────────────────────────────────────
main() {
    echo "RAGAP-PHI Setup"
    echo "==============="
    resolve_conda

    case "${1:-}" in
        --envs-only)
            setup_envs
            ;;
        --models-only)
            setup_models
            ;;
        --bundle-only)
            download_bundle
            ;;
        --verify)
            verify_all
            ;;
        *)
            setup_envs
            setup_models
            download_bundle
            echo ""
            echo "════════════════════════════════════════"
            echo " Setup complete!"
            echo "════════════════════════════════════════"
            echo ""
            echo "Run inference:"
            echo "  python infer_phage_host.py --input query.fa --mode species --output result.tsv"
            ;;
    esac
}

main "$@"
