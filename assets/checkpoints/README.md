This directory is only a placeholder.

For the current inference workflow, do not publish a checkpoint by itself and expect users to run prediction successfully. The usable release unit is the minimal inference bundle under `artifacts/<dataset_id>/`, which must include:

- `graph/hetero_graph.pt`
- `graph/node_maps.json`
- `catalogs/host_catalog.parquet`
- `cluster/sourmash/phage_phage/signatures/`
- `train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt`
