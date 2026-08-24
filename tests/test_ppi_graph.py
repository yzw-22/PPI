import unittest

import torch

from src.ppi_graph import PPIGraph


class PPIGraphTest(unittest.TestCase):
    def test_build_graph_uses_only_nodes_from_requested_split(self):
        graph = PPIGraph.__new__(PPIGraph)
        graph.device = torch.device("cpu")
        graph.tensor = torch.arange(5, dtype=torch.float32).reshape(5, 1)
        graph.ppi = torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.long)
        graph.ppi_labels = torch.zeros((3, 7), dtype=torch.float32)
        graph.split_index = {
            "train_index": [1],
            "val_index": [2],
            "test_index": [0],
        }

        split_graph = graph.build_graph("train")

        self.assertEqual(split_graph["node_index"].tolist(), [1, 2])
        self.assertEqual(split_graph["node_feat"].flatten().tolist(), [1.0, 2.0])
        self.assertEqual(split_graph["edge_index"].tolist(), [[0, 1], [1, 0]])
        self.assertEqual(split_graph["edge_label"].shape, (2, 7))
        self.assertTrue(torch.equal(
            split_graph["edge_label"][0], split_graph["edge_label"][1]
        ))

    def test_build_full_graph_covers_all_ppi_pairs(self):
        graph = PPIGraph.__new__(PPIGraph)
        graph.device = torch.device("cpu")
        graph.tensor = torch.arange(5, dtype=torch.float32).reshape(5, 1)
        graph.ppi = torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.long)
        graph.ppi_list = [[0, 1], [1, 2], [3, 4]]
        graph.ppi_labels = torch.zeros((3, 7), dtype=torch.float32)
        graph.split_index = {
            "train_index": [1],
            "val_index": [2],
            "test_index": [0],
        }

        full_graph = graph.build_full_graph()

        self.assertEqual(full_graph["node_index"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(
            full_graph["node_feat"].flatten().tolist(), [0.0, 1.0, 2.0, 3.0, 4.0]
        )
        # Pairs (0,1), (1,2), (3,4) followed by their reverse directions.
        self.assertEqual(
            full_graph["edge_index"].tolist(),
            [[0, 1, 3, 1, 2, 4], [1, 2, 4, 0, 1, 3]],
        )
        self.assertEqual(full_graph["edge_label"].shape, (6, 7))
        for pair in range(3):
            self.assertTrue(torch.equal(
                full_graph["edge_label"][2 * pair],
                full_graph["edge_label"][2 * pair + 1],
            ))

        directed = graph.build_full_graph(undirected=False)
        self.assertEqual(
            directed["edge_index"].tolist(), [[0, 1, 3], [1, 2, 4]]
        )
        self.assertEqual(directed["edge_label"].shape, (3, 7))


if __name__ == "__main__":
    unittest.main()
