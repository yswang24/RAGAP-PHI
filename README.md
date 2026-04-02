# RAGAP-PHI

RAGAP-PHI is a cleaned, GitHub-ready packaging of the RAGAP phage-host prediction workflow. This repository is prepared for inference-first release: users clone the code, download a minimal inference bundle, place a few third-party assets locally, and run `infer_phage_host.py` on a single phage FASTA file.

## What This Repository Includes

- Inference entrypoint: `infer_phage_host.py`
- Full pipeline entrypoint: `pipeline.py`
- Internal orchestration code under `ragap_pipeline/`
- Vendored helper scripts under `scripts/`
- Main config under `configs/pipeline.fullhost_v2.yaml`
- Conda environment files under `envs/`
- Small taxonomy and metadata files under `data/`
- Small tabular inputs under `inputs/`

## Release Model

This repository is intended to be published together with one separate Release asset:

1. GitHub repository: code, configs, small metadata.
2. Inference bundle archive: the minimal cached artifacts needed by the current inference code path.

For this project, a checkpoint alone is not enough. The inference code attaches a new phage into a frozen training graph and computes phage-phage similarity against a cached training signature library.

## Minimal Inference Bundle

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

## Third-Party Assets Users Still Need

These large external assets are not committed to Git:

- `assets/models/DNA_bert_4/`
- `assets/databases/pharokka_v1.4.0_databases/`
- ESM2 weights, usually downloaded to the local cache on first use

Official sources and setup notes are listed in [docs/assets.md](/home/wangjingyuan/wys/RAGAP-PHI/docs/assets.md) and [docs/inference.md](/home/wangjingyuan/wys/RAGAP-PHI/docs/inference.md).

## Quick Start

Create the required Conda environments:

```bash
conda env create -f envs/PHPGAT.yaml
conda env create -f envs/dnaberts.yaml
conda env create -f envs/esm.yaml
conda env create -f envs/pharokka.yaml
conda env create -f envs/sourmash.yaml
```

If you want the top-level scripts to auto-bootstrap into the base environment, set:

```bash
export RAGAP_BOOTSTRAP_PYTHON=/path/to/envs/PHPGAT/bin/python
```

Then place:

- the inference bundle under `artifacts/ragap_phi/`
- DNABERT under `assets/models/DNA_bert_4/`
- Pharokka databases under `assets/databases/pharokka_v1.4.0_databases/`

Run inference:

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

## Repository Layout

```text
RAGAP-PHI/
  infer_phage_host.py
  pipeline.py
  configs/
  envs/
  ragap_pipeline/
  scripts/
  data/
  inputs/
  assets/
  tests/
```

## What Is Not Committed

These assets are intentionally kept out of normal Git history:

- `assets/models/DNA_bert_4/`
- `assets/databases/pharokka_v1.4.0_databases/`
- `artifacts/ragap_phi/graph/hetero_graph.pt`
- `artifacts/ragap_phi/graph/node_maps.json`
- `artifacts/ragap_phi/catalogs/host_catalog.parquet`
- `artifacts/ragap_phi/cluster/sourmash/phage_phage/signatures/`
- `artifacts/ragap_phi/train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt`
- optional: `artifacts/ragap_phi/manifests/train.json`
- `inputs/phage_fasta/`
- `inputs/host_fasta/`

## Training

Training support is still present, but it is not the primary publication path for this repository.

Default config:

- `configs/pipeline.fullhost_v2.yaml`

Dry-run the stage plan:

```bash
python pipeline.py status
```

Run the full pipeline:

```bash
python pipeline.py run
```

Override paths without editing the YAML:

```bash
python pipeline.py run \
  --set inputs.phage_fasta_dir=/abs/path/to/phage_fasta \
  --set inputs.host_fasta_dir=/abs/path/to/host_fasta
```

## Documentation

- [docs/inference.md](/home/wangjingyuan/wys/RAGAP-PHI/docs/inference.md): exact inference reproduction and asset checklist
- [docs/assets.md](/home/wangjingyuan/wys/RAGAP-PHI/docs/assets.md): external assets and recommended distribution strategy

## Notes For GitHub Publication

- Publish the code here.
- Publish the inference bundle as a GitHub Release asset, Zenodo archive, or similar external download.
- Do not publish only the checkpoint unless you also change the inference code path.
- Add a `LICENSE` file before making the repository public if you intend to permit reuse.
