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
        for sampler in ("static", "random-subset", "heuristic"):
            with patch.object(
                sys,
                "argv",
                ["train_shs27k", "--sampler", sampler,
                 "--use-sampler-edge-relations"],
            ):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_structural_features_flag(self):
        with patch.object(sys, "argv", ["train_shs27k"]):
            self.assertFalse(parse_args().sampler_structural_features)
        with patch.object(
            sys, "argv", ["train_shs27k", "--sampler-structural-features"]
        ):
            self.assertTrue(parse_args().sampler_structural_features)
        with patch.object(
            sys, "argv",
            ["train_shs27k", "--sampler", "static",
             "--sampler-structural-features"],
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
        with patch.object(sys, "argv", ["train_shs27k", "--sampler", "heuristic"]):
            self.assertEqual(parse_args().sampler, "heuristic")

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

    def test_readout_flags(self):
        with patch.object(sys, "argv", ["train_shs27k"]):
            args = parse_args()
            self.assertEqual(args.readout, "mean")
            self.assertAlmostEqual(args.ppr_alpha, 0.15)
            self.assertAlmostEqual(args.ppr_eps, 5e-6)
        with patch.object(
            sys, "argv", ["train_shs27k", "--readout", "attention"]
        ):
            self.assertEqual(parse_args().readout, "attention")
        for argv in (
            ["train_shs27k", "--readout", "bogus"],
            ["train_shs27k", "--ppr-alpha", "0"],
            ["train_shs27k", "--ppr-alpha", "1.5"],
            ["train_shs27k", "--ppr-eps", "0"],
        ):
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_reward_flags(self):
        with patch.object(sys, "argv", ["train_shs27k"]):
            args = parse_args()
            self.assertFalse(args.reward_margin)
            self.assertEqual(args.reward_pos, 2.0)
            self.assertEqual(args.reward_neg, 1.0)
        with patch.object(
            sys, "argv",
            ["train_shs27k", "--reward-margin",
             "--reward-pos", "3.0", "--reward-neg", "0.5"],
        ):
            args = parse_args()
            self.assertTrue(args.reward_margin)
            self.assertEqual(args.reward_pos, 3.0)
            self.assertEqual(args.reward_neg, 0.5)
        for argv in (
            ["train_shs27k", "--reward-pos", "0"],
            ["train_shs27k", "--reward-neg", "-1"],
        ):
            with patch.object(sys, "argv", argv):
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
