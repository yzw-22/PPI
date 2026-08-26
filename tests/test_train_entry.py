import sys
import unittest
from unittest.mock import patch

from src.train_shs27k import parse_args


class TrainEntrySamplerArgumentsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
