import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.ppi_graph import PPIGraph
from src.train_shs27k import _build_training_edge_relations, parse_args


class TrainEntrySamplerArgumentsTest(unittest.TestCase):
    def test_edge_relations_are_opt_in(self):
        with patch.object(sys, "argv", ["train_shs27k"]):
            args = parse_args()
            self.assertFalse(args.use_edge_relations)
            self.assertFalse(args.use_sampler_edge_relations)
        with patch.object(
            sys, "argv", ["train_shs27k", "--use-edge-relations"]
        ):
            self.assertTrue(parse_args().use_edge_relations)
        with patch.object(
            sys, "argv", ["train_shs27k", "--use-sampler-edge-relations"]
        ):
            self.assertTrue(parse_args().use_sampler_edge_relations)

    def test_sampler_relations_require_rl_sampler(self):
        with patch.object(
            sys,
            "argv",
            ["train_shs27k", "--sampler", "static",
             "--use-sampler-edge-relations"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_relation_lookup_contains_only_training_edges(self):
        graph = PPIGraph.__new__(PPIGraph)
        graph.device = torch.device("cpu")
        graph.ppi = torch.tensor([[0, 1], [1, 2], [2, 3]])
        graph.ppi_labels = torch.tensor([
            [1., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0.],
        ])
        graph.split_index = {
            "train_index": [0],
            "val_index": [1],
            "test_index": [2],
        }

        lookup = _build_training_edge_relations(graph, torch.arange(4))

        actual = lookup.lookup([[0, 1], [1, 2], [2, 3]])
        self.assertTrue(torch.equal(actual[0], graph.ppi_labels[0]))
        self.assertTrue(torch.equal(actual[1:], torch.zeros((2, 7))))

    def test_fixed_num_is_rejected_after_removal(self):
        with patch.object(
            sys,
            "argv",
            ["train_shs27k", "--fixed-num", "1"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_k_hops_defaults_to_one_and_accepts_explicit_values(self):
        with patch.object(sys, "argv", ["train_shs27k"]):
            self.assertEqual(parse_args().k_hops, 1)
        with patch.object(sys, "argv", ["train_shs27k", "--k-hops", "2"]):
            self.assertEqual(parse_args().k_hops, 2)

    def test_negative_k_hops_is_rejected(self):
        with patch.object(sys, "argv", ["train_shs27k", "--k-hops", "-1"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_sampler_defaults_to_rl_and_accepts_static(self):
        with patch.object(sys, "argv", ["train_shs27k"]):
            self.assertEqual(parse_args().sampler, "rl")
        with patch.object(sys, "argv", ["train_shs27k", "--sampler", "static"]):
            self.assertEqual(parse_args().sampler, "static")
        with patch.object(sys, "argv", ["train_shs27k", "--sampler", "random-subset"]):
            args = parse_args()
            self.assertEqual(args.sampler, "random-subset")
            self.assertEqual(args.random_subset_min_size, 3)
            self.assertEqual(args.random_subset_max_size, 7)

    def test_invalid_sampler_value_is_rejected(self):
        with patch.object(sys, "argv", ["train_shs27k", "--sampler", "bogus"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_random_subset_size_range_is_validated(self):
        with patch.object(
            sys, "argv", ["train_shs27k", "--random-subset-min-size", "1"]
        ):
            with self.assertRaises(SystemExit):
                parse_args()
        with patch.object(
            sys, "argv",
            ["train_shs27k", "--random-subset-min-size", "5",
             "--random-subset-max-size", "3"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_missing_split_file_is_rejected_at_parse_time(self):
        # bfs is a valid SHS148k split (served by --root dataset_ppisplit), so
        # availability passes but a missing split file must fail cleanly here
        # instead of raising FileNotFoundError during graph loading.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "SHS148k_bfs.json").write_text('{"train_index": []}')
            with patch.object(
                sys, "argv",
                ["train_shs27k", "--root", tmp,
                 "--dataset", "SHS148k", "--split", "bfs"],
            ):
                self.assertEqual(parse_args().split, "bfs")
            with patch.object(
                sys, "argv",
                ["train_shs27k", "--root", tmp,
                 "--dataset", "SHS148k", "--split", "dfs"],
            ):
                with self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
