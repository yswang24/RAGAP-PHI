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

## Minimal Inference Bundle

The inference bundle is available on the GitHub Release page. `setup.sh` downloads it automatically. To download manually:

```bash
# Download all bundle_part_* from the Release page, then:
cat bundle_part_* > ragap_phi_inference_bundle.tar.gz
tar -xzf ragap_phi_inference_bundle.tar.gz
mv ragap_phi artifacts/
```

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

Default (outputs both species and genus):

```bash
python infer_phage_host.py --input query.fa --output result.tsv
```

Species-only or genus-only:

```bash
python infer_phage_host.py --input query.fa --mode species --output result.tsv
python infer_phage_host.py --input query.fa --mode genus --output result.tsv
```

Batch processing (multi-record FASTA):

```bash
python infer_phage_host.py --input multi_phage.fa --output batch_results.tsv --batch
```

### Output format

Default output (both species and genus):

| phage_id | top_host_id | top_host_taxid | top_species | top_genus | score |
|----------|-------------|----------------|-------------|-----------|-------|
| KX266586 | GCF_000005845 | 562 | Escherichia coli | Escherichia | 0.999909 |

With `--mode species`: columns are `phage_id, top_host_id, top_host_taxid, top_species, score`

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
