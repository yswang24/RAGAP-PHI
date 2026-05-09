# RAGAP-PHI

RAGAP-PHI is a phage-host interaction prediction system using a heterogeneous graph attention network (GATv2). Given a query phage FASTA file, it predicts which bacterial host the phage can infect by embedding the query, attaching it to a frozen training graph, and scoring all candidate hosts.

## What This Repository Includes

- Inference entrypoint: `infer_phage_host.py`
- Inference engine under `ragap_pipeline/inference.py`
- GATv2 model definition under `ragap_pipeline/model.py`
- Helper scripts under `scripts/` (DNA embedding, protein embedding, bundle packaging)
- Inference config under `configs/pipeline.fullhost_v2.yaml`
- Conda environment files under `envs/`
- Small inference metadata files under `data/`

## Release Model

This repository is published together with a separate inference bundle:

1. GitHub repository: code, configs, small metadata.
2. Inference bundle archive: the minimal cached artifacts needed by inference.

A checkpoint alone is not enough. The inference code attaches a new phage into a frozen training graph and computes phage-phage similarity against a cached training signature library.

## Minimal Inference Bundle

### Download from GitHub Release

Due to GitHub's 2GB file limit, the inference bundle is split into multiple parts. Download all `bundle_part_*` files from the Release page and merge them:

```bash
# Merge split files
cat bundle_part_* > ragap_phi_inference_bundle.tar.gz

# Extract
tar -xzf ragap_phi_inference_bundle.tar.gz
```

Place the extracted `ragap_phi/` directory under `artifacts/`:
```bash
mv ragap_phi artifacts/
```

### Bundle Contents

After extraction, the bundle should provide these paths under `artifacts/ragap_phi/`:

- `graph/hetero_graph.pt`
- `graph/node_maps.json`
- `catalogs/host_catalog.parquet`
- `cluster/sourmash/phage_phage/signatures/`
- `train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt`

Optional:

- `manifests/train.json`

If you are the maintainer and already have an older RAGAP artifact directory, create this bundle with:

```bash
python scripts/package_inference_bundle.py \
  --source-artifacts <OLD_RAGAP_ARTIFACT_ROOT> \
  --output-root /tmp/ragap_phi_release \
  --archive /tmp/ragap_phi_inference_bundle.tar.gz
```

Add `--include-manifest` only if you also want to retain `manifests/train.json` as metadata.

## Quick Start

### One-click setup

```bash
git clone https://github.com/yswang24/RAGAP-PHI.git
cd RAGAP-PHI
bash setup.sh
```

This creates Conda environments, downloads model weights (DNABERT-4 + ESM2), and fetches the inference bundle from GitHub Release — all in one step.

Options:

```bash
bash setup.sh --verify       # check everything without downloading
bash setup.sh --envs-only    # create conda environments only
bash setup.sh --models-only  # download model weights only
bash setup.sh --bundle-only  # download inference bundle only
```

### Run inference

```bash
python infer_phage_host.py \
  --input query.fa \
  --mode species \
  --output result.tsv
```

For genus-level output:

```bash
python infer_phage_host.py \
  --input query.fa \
  --mode genus \
  --output result.tsv
```

For batch processing (multi-record FASTA, each record processed separately):

```bash
python infer_phage_host.py \
  --input multi_phage.fa \
  --mode species \
  --output batch_results.tsv \
  --batch
```

## Repository Layout

```text
RAGAP-PHI/
  setup.sh                     # One-click setup (envs + models + bundle check)
  infer_phage_host.py          # Inference entrypoint
  ragap_pipeline/
    inference.py               # Full inference engine
    model.py                   # GATv2 model definition
    config.py                  # YAML config loading
    execution.py               # Conda env resolution
    utils.py                   # Shared utilities
  scripts/
    setup_models.py            # Download DNABERT-4 and ESM2 weights
    dna_bert_embed.py          # DNABERT DNA embedding
    generate_esm_embeddings_phage.py  # ESM2 phage protein embedding
    package_inference_bundle.py      # Bundle packaging
  configs/
    pipeline.fullhost_v2.yaml  # Inference config
  envs/                        # Conda environment files
  data/                        # Taxonomy metadata
  assets/                      # Model weight placeholders
  tests/                       # Inference tests
  docs/                        # Documentation
```

## Documentation

- [docs/inference.md](docs/inference.md): exact inference reproduction and asset checklist
- [docs/assets.md](docs/assets.md): external assets and recommended distribution strategy

## Notes For GitHub Publication

- Publish the code here.
- Publish the inference bundle as a GitHub Release asset, Zenodo archive, or similar external download.
- Do not publish only the checkpoint unless you also change the inference code path.
- Add a `LICENSE` file before making the repository public if you intend to permit reuse.
