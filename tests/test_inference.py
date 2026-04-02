from __future__ import annotations

import unittest

import torch
from torch_geometric.data import HeteroData

from ragap_pipeline.inference import (
    augment_graph_with_query,
    build_similarity_edge_rows,
    resolve_genus_name,
)


class InferenceUtilityTests(unittest.TestCase):
    def test_resolve_genus_name_walks_up_parent_chain(self) -> None:
        from ragap_pipeline.inference import TaxonomyNode

        taxonomy = {
            562: TaxonomyNode(taxid=562, parent=561, name="Escherichia coli", rank="species"),
            561: TaxonomyNode(taxid=561, parent=543, name="Escherichia", rank="genus"),
            543: TaxonomyNode(taxid=543, parent=1224, name="Enterobacteriaceae", rank="family"),
        }
        self.assertEqual(resolve_genus_name(562, taxonomy), "Escherichia")
        self.assertEqual(resolve_genus_name(-1, taxonomy), "NA")
        self.assertEqual(resolve_genus_name(999999, taxonomy), "NA")

    def test_build_similarity_edge_rows_matches_full_build_orientation(self) -> None:
        rows = build_similarity_edge_rows(
            "BB_query",
            {"AA_old": 0.91, "CC_old": 0.95, "DD_old": 0.79},
            threshold=0.8,
        )
        self.assertEqual(
            rows,
            [
                ("AA_old", "BB_query", "phage-phage", 0.91),
                ("BB_query", "CC_old", "phage-phage", 0.95),
            ],
        )

    def test_augment_graph_with_query_updates_counts(self) -> None:
        data = HeteroData()
        data["phage"].x = torch.zeros((2, 3), dtype=torch.float32)
        data["host"].x = torch.zeros((1, 3), dtype=torch.float32)
        data["host_sequence"].x = torch.zeros((1, 3), dtype=torch.float32)
        data["protein"].x = torch.zeros((3, 4), dtype=torch.float32)
        data["taxonomy"].x = torch.zeros((1, 2), dtype=torch.float32)
        data[("phage", "interacts", "phage")].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data[("phage", "encodes", "protein")].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        data[("protein", "encoded_by_phage", "phage")].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

        node_maps = {
            "phage_map": {"old_a": 0, "old_b": 1},
            "host_map": {"host_1": 0},
            "host_sequence_map": {"seq_1": 0},
            "protein_map": {"p0": 0, "p1": 1, "p2": 2},
            "tax_map": {"1": 0},
        }
        query_embedding = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        protein_embeddings = {
            "new_p1": torch.ones(4, dtype=torch.float32),
            "new_p2": torch.full((4,), 2.0, dtype=torch.float32),
        }
        similarity_rows = [("old_b", "new_query", "phage-phage", 0.88)]

        query_idx, updated_maps, counts = augment_graph_with_query(
            data,
            node_maps,
            "new_query",
            query_embedding,
            protein_embeddings,
            similarity_rows,
        )

        self.assertEqual(query_idx, 2)
        self.assertEqual(int(data["phage"].x.shape[0]), 3)
        self.assertEqual(int(data["protein"].x.shape[0]), 5)
        self.assertEqual(counts["added_phage_nodes"], 1)
        self.assertEqual(counts["added_protein_nodes"], 2)
        self.assertEqual(counts["added_phage_protein_edges"], 2)
        self.assertEqual(counts["added_reverse_protein_edges"], 2)
        self.assertEqual(counts["added_phage_similarity_edges"], 1)
        self.assertEqual(updated_maps["phage_map"]["new_query"], 2)
        self.assertEqual(updated_maps["protein_map"]["new_p1"], 3)
        self.assertEqual(updated_maps["protein_map"]["new_p2"], 4)
        self.assertEqual(
            data[("phage", "encodes", "protein")].edge_index[:, -2:].tolist(),
            [[2, 2], [3, 4]],
        )
        self.assertEqual(
            data[("protein", "encoded_by_phage", "phage")].edge_index[:, -2:].tolist(),
            [[3, 4], [2, 2]],
        )
        self.assertEqual(
            data[("phage", "interacts", "phage")].edge_index[:, -1:].tolist(),
            [[1], [2]],
        )


if __name__ == "__main__":
    unittest.main()
