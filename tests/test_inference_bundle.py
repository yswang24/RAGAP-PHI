from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragap_pipeline import inference


def _load_bundle_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "package_inference_bundle.py"
    spec = importlib.util.spec_from_file_location("package_inference_bundle", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InferenceAssetLoadingTests(unittest.TestCase):
    def test_load_assets_without_manifest_uses_config_and_bundle_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "configs" / "pipeline.fullhost_v2.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("placeholder: true\n", encoding="utf-8")

            graph = root / "artifacts" / "ragap_phi" / "graph" / "hetero_graph.pt"
            node_maps = root / "artifacts" / "ragap_phi" / "graph" / "node_maps.json"
            checkpoint = (
                root
                / "artifacts"
                / "ragap_phi"
                / "train"
                / "fullhost_v2"
                / "best_GAT_attn_fullhost_copymsg_v2.pt"
            )
            host_catalog = root / "artifacts" / "ragap_phi" / "catalogs" / "host_catalog.parquet"
            taxonomy_tree = root / "data" / "taxonomy" / "taxonomy_with_alias.parquet"
            taxid2species = root / "data" / "metadata" / "taxid_species.tsv"
            signatures = (
                root / "artifacts" / "ragap_phi" / "cluster" / "sourmash" / "phage_phage" / "signatures"
            )
            dna_script = root / "scripts" / "dna_bert_embed.py"
            phage_esm_script = root / "scripts" / "generate_esm_embeddings_phage.py"
            train_script = root / "scripts" / "train_hgt_phage_host_weight_RBP_noleak_hard.py"

            for path in (graph, node_maps, checkpoint, host_catalog, taxonomy_tree, taxid2species):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x\n", encoding="utf-8")
            for path in (dna_script, phage_esm_script, train_script):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("print('ok')\n", encoding="utf-8")
            signatures.mkdir(parents=True, exist_ok=True)
            (signatures / "dummy.sig").write_text("sig\n", encoding="utf-8")

            config = {
                "execution": {"conda_bin": "conda"},
                "build_catalogs": {"host_catalog": str(host_catalog)},
                "inputs": {"taxonomy_alias_parquet": str(taxonomy_tree)},
                "tools": {
                    "dna_embed_script": str(dna_script),
                    "phage_esm_script": str(phage_esm_script),
                    "train_script": str(train_script),
                },
                "dna_embedding": {
                    "phage": {
                        "model": "DNA_bert_4",
                        "k": 4,
                        "window_tokens": 510,
                        "stride_tokens": 510,
                        "batch_size": 8,
                        "precision": "fp16",
                        "max_windows": 800,
                        "seed": 13,
                    }
                },
                "phage_protein_prep": {"phanotate_bin": "phanotate.py", "extra_args": []},
                "phage_protein_embedding": {
                    "model_name": "esm2_t33_650M_UR50D",
                    "repr_l": 32,
                    "batch_size": 1,
                    "workers": 1,
                },
                "cluster_assets": {
                    "sourmash_work_dir": str(root / "artifacts" / "ragap_phi" / "cluster" / "sourmash"),
                    "similarity_edges": {
                        "sourmash_env": "sourmash_env",
                        "sourmash_bin": "sourmash",
                        "scaled": 1000,
                        "phage": {"kmer_size": 21, "threshold": 0.0},
                    },
                },
            }

            args = argparse.Namespace(
                input="query.fa",
                mode="species",
                output="result.tsv",
                manifest=None,
                config=str(config_path),
                checkpoint=None,
                graph=None,
                node_maps=None,
                host_catalog=None,
                taxonomy_tree=None,
                taxid2species=None,
                train_script=None,
                phage_signatures_dir=None,
                device="cpu",
                work_dir=None,
                cleanup=False,
            )

            missing_manifest = root / "artifacts" / "ragap_phi" / "manifests" / "train.json"
            with mock.patch.object(inference, "DEFAULT_TRAIN_MANIFEST", missing_manifest), mock.patch.object(
                inference, "DEFAULT_CONFIG_PATH", config_path
            ), mock.patch.object(inference, "DEFAULT_CHECKPOINT", checkpoint), mock.patch.object(
                inference, "DEFAULT_GRAPH", graph
            ), mock.patch.object(
                inference, "DEFAULT_NODE_MAPS", node_maps
            ), mock.patch.object(
                inference, "DEFAULT_TAXID2SPECIES", taxid2species
            ), mock.patch(
                "ragap_pipeline.inference.prepare_config", return_value=config
            ):
                assets = inference.load_inference_assets(args)

            self.assertEqual(assets.config_path, config_path.resolve())
            self.assertEqual(assets.checkpoint, checkpoint.resolve())
            self.assertEqual(assets.graph, graph.resolve())
            self.assertEqual(assets.node_maps, node_maps.resolve())
            self.assertEqual(assets.host_catalog, host_catalog.resolve())
            self.assertEqual(assets.taxonomy_tree, taxonomy_tree.resolve())
            self.assertEqual(assets.taxid2species, taxid2species.resolve())
            self.assertEqual(assets.phage_signatures_dir, signatures.resolve())


class InferenceBundlePackagingTests(unittest.TestCase):
    def test_package_inference_bundle_normalizes_old_checkpoint_layout(self) -> None:
        module = _load_bundle_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source_artifacts"
            output_root = root / "release_root"

            source_files = [
                source_root / "graph" / "hetero_graph.pt",
                source_root / "graph" / "node_maps.json",
                source_root / "catalogs" / "host_catalog.parquet",
                source_root / "train_attn_fullhost_copymsg_v2" / "best_GAT_attn_fullhost_copymsg_v2.pt",
            ]
            for path in source_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x\n", encoding="utf-8")
            manifest_path = source_root / "manifests" / "train.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("x\n", encoding="utf-8")

            signatures = source_root / "cluster" / "sourmash" / "phage_phage" / "signatures"
            signatures.mkdir(parents=True, exist_ok=True)
            (signatures / "query.sig").write_text("sig\n", encoding="utf-8")

            bundle_root = module.package_inference_bundle(source_root, output_root)

            expected_root = output_root / "artifacts" / "ragap_phi"
            self.assertEqual(bundle_root, expected_root.resolve())
            self.assertTrue((expected_root / "graph" / "hetero_graph.pt").exists())
            self.assertTrue((expected_root / "graph" / "node_maps.json").exists())
            self.assertTrue((expected_root / "catalogs" / "host_catalog.parquet").exists())
            self.assertTrue((expected_root / "cluster" / "sourmash" / "phage_phage" / "signatures" / "query.sig").exists())
            self.assertTrue(
                (
                    expected_root
                    / "train"
                    / "fullhost_v2"
                    / "best_GAT_attn_fullhost_copymsg_v2.pt"
                ).exists()
            )
            self.assertFalse((expected_root / "manifests" / "train.json").exists())
            self.assertTrue((expected_root / "INFERENCE_BUNDLE.json").exists())

    def test_package_inference_bundle_can_include_manifest(self) -> None:
        module = _load_bundle_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source_artifacts"
            output_root = root / "release_root"

            for path in [
                source_root / "graph" / "hetero_graph.pt",
                source_root / "graph" / "node_maps.json",
                source_root / "catalogs" / "host_catalog.parquet",
                source_root / "train" / "fullhost_v2" / "best_GAT_attn_fullhost_copymsg_v2.pt",
                source_root / "manifests" / "train.json",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x\n", encoding="utf-8")

            signatures = source_root / "cluster" / "sourmash" / "phage_phage" / "signatures"
            signatures.mkdir(parents=True, exist_ok=True)
            (signatures / "query.sig").write_text("sig\n", encoding="utf-8")

            bundle_root = module.package_inference_bundle(
                source_root,
                output_root,
                include_manifest=True,
            )

            expected_root = output_root / "artifacts" / "ragap_phi"
            self.assertEqual(bundle_root, expected_root.resolve())
            self.assertTrue((expected_root / "manifests" / "train.json").exists())


if __name__ == "__main__":
    unittest.main()
