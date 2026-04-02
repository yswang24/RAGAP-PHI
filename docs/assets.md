# External Assets

This repository keeps code and small metadata in Git, but intentionally excludes large models, databases, FASTA corpora, and graph artifacts.

## Minimal Inference Bundle Paths

These are the release assets the maintainer should publish separately:

| Path | Required For | Bundled In Git |
| --- | --- | --- |
| `artifacts/<dataset_id>/graph/hetero_graph.pt` | Frozen training graph | No |
| `artifacts/<dataset_id>/graph/node_maps.json` | Node id to graph index mapping | No |
| `artifacts/<dataset_id>/catalogs/host_catalog.parquet` | Host taxid fallback during inference | No |
| `artifacts/<dataset_id>/cluster/sourmash/phage_phage/signatures/` | Cached training phage signature library | No |
| `artifacts/<dataset_id>/train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt` | Trained model checkpoint | No |
| `artifacts/<dataset_id>/manifests/train.json` | Optional metadata only | No |

## Third-Party Assets

| Path | Required For | Bundled In Git |
| --- | --- | --- |
| `assets/models/DNA_bert_4/` | DNABERT DNA embedding | No |
| `assets/databases/pharokka_v1.4.0_databases/` | Pharokka / PHANOTATE stage | No |
| local ESM2 cache | Protein embedding | No |

## Bundled Small Files

- `inputs/virus_host_with_GCF.tsv`
- `inputs/edges/*.tsv`
- `data/metadata/taxid_species.tsv`
- `data/taxonomy/taxonomy_with_alias.parquet`
- `data/taxonomy/taxonomy_poincare_tangent.parquet`

## Official External Sources

- DNABERT-4:
  `https://github.com/jerryji1993/DNABERT`
  `https://huggingface.co/zhihan1996/DNA_bert_4`
- Pharokka:
  `https://github.com/gbouras13/pharokka`
  `https://zenodo.org/record/8276347/files/pharokka_v1.4.0_databases.tar.gz`
- sourmash:
  `https://sourmash.readthedocs.io/en/latest/tutorial-install.html`
- ESM2:
  `https://github.com/facebookresearch/esm`
  `https://huggingface.co/facebook/esm2_t33_650M_UR50D`

## Recommended GitHub Distribution Strategy

- Keep code and small metadata in Git.
- Publish the minimal inference bundle as a GitHub Release asset, Zenodo archive, or similar external download.
- Do not publish only the checkpoint unless you also change the inference code path.
- Use [package_inference_bundle.py](../scripts/package_inference_bundle.py) to export the minimal bundle from an older artifact directory.
