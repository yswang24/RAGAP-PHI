# Inference Guide

This repository is set up for inference-first distribution. End users do not need to rerun training, but they do need:

1. this code repository
2. a minimal inference bundle published by the maintainer
3. third-party preprocessing assets

## 1. What The Current Inference Code Actually Does

`infer_phage_host.py` is not a standalone classifier wrapper. For each query phage, it runs:

- DNABERT DNA embedding
- PHANOTATE gene calling
- ESM2 protein embedding
- sourmash sketching and phage-phage similarity search
- incremental attachment of the query phage to a frozen training graph
- full-graph GATv2 scoring against all host nodes

Because of that, the code needs more than a checkpoint.

## 2. Minimal Inference Bundle

The maintainer should publish a bundle that extracts into `artifacts/ragap_phi/` with these contents:

- `graph/hetero_graph.pt`
- `graph/node_maps.json`
- `catalogs/host_catalog.parquet`
- `cluster/sourmash/phage_phage/signatures/`
- `train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt`

Optional:

- `manifests/train.json`

Notes:

- The query phage's own sourmash signature is generated at runtime.
- The packaged `signatures/` directory is the cached training phage reference library used for similarity search.
- `host_catalog.parquet` is still needed by the current code as a fallback source of `host_gcf -> host_species_taxid`.

## 3. Third-Party Assets

Users still need these local assets:

### DNABERT 4-mer model

Expected path:

- `assets/models/DNA_bert_4/`

Official sources:

- `https://github.com/jerryji1993/DNABERT`
- `https://huggingface.co/zhihan1996/DNA_bert_4`

Recommended setup:

```bash
git lfs install
git clone https://huggingface.co/zhihan1996/DNA_bert_4 assets/models/DNA_bert_4
```

### Pharokka database bundle

Expected path:

- `assets/databases/pharokka_v1.4.0_databases/`

Official sources:

- `https://github.com/gbouras13/pharokka`
- `https://zenodo.org/record/8276347/files/pharokka_v1.4.0_databases.tar.gz`

Recommended setup:

```bash
conda activate pharokka_env
install_databases.py -o assets/databases/pharokka_v1.4.0_databases
```

### ESM2 weights

Model used:

- `esm2_t33_650M_UR50D`

Official sources:

- `https://github.com/facebookresearch/esm`
- `https://huggingface.co/facebook/esm2_t33_650M_UR50D`

The repository scripts use `fair-esm`, which downloads weights into the local cache on first use.

## 4. Minimal Working Checklist

Before running inference, verify:

```bash
test -f assets/models/DNA_bert_4/config.json
test -d assets/databases/pharokka_v1.4.0_databases
test -f artifacts/ragap_phi/graph/hetero_graph.pt
test -f artifacts/ragap_phi/graph/node_maps.json
test -f artifacts/ragap_phi/catalogs/host_catalog.parquet
test -d artifacts/ragap_phi/cluster/sourmash/phage_phage/signatures
test -f artifacts/ragap_phi/train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt
```

Optional:

```bash
test -f artifacts/ragap_phi/manifests/train.json
```

## 5. Running Inference

Species mode:

```bash
python infer_phage_host.py \
  --input /path/to/query.fa \
  --mode species \
  --output result.tsv
```

Genus mode:

```bash
python infer_phage_host.py \
  --input /path/to/query.fa \
  --mode genus \
  --output result.tsv
```

## 6. Creating The Bundle As The Maintainer

If you already have an older RAGAP artifact directory, export the minimal bundle with:

```bash
python scripts/package_inference_bundle.py \
  --source-artifacts <OLD_RAGAP_ARTIFACT_ROOT> \
  --output-root /tmp/ragap_phi_release \
  --archive /tmp/ragap_phi_inference_bundle.tar.gz
```

If you also want to retain the old training manifest as metadata:

```bash
python scripts/package_inference_bundle.py \
  --source-artifacts <OLD_RAGAP_ARTIFACT_ROOT> \
  --output-root /tmp/ragap_phi_release \
  --archive /tmp/ragap_phi_inference_bundle.tar.gz \
  --include-manifest
```

Typical old checkpoint layouts already handled by the script:

- `train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt`
- `train_attn_fullhost_copymsg_v2/best_GAT_attn_fullhost_copymsg_v2.pt`

## 7. What This Repository Does Not Publish By Itself

This Git repository does not currently host:

- the minimal inference bundle under `artifacts/ragap_phi/`
- the curated training FASTA corpora under `inputs/phage_fasta/` and `inputs/host_fasta/`
- DNABERT weights
- Pharokka databases

If you want the GitHub project to be directly usable by outside users, publish the inference bundle as a Release asset and keep the external download links in this document up to date.
