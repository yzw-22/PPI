import sys
import unittest
from unittest.mock import patch

from src.train_shs27k import parse_args


class TrainEntryTest(unittest.TestCase):
    def test_sampler_mode_cli_choices(self):
        for mode in (
            "learned", "target_only", "target_proxy", "random_1hop10",
            "random_iterative10",
        ):
            with self.subTest(mode=mode), patch.object(
                sys,
                "argv",
                ["train_shs27k", "--dataset", "SHS27k", "--split", "dfs",
                 "--sampler-mode", mode],
            ):
                args = parse_args()
                self.assertEqual(args.sampler_mode, mode)

    def test_explicit_baseline_modes(self):
        for mode in ("target_only", "target_proxy", "random_1hop10"):
            with self.subTest(mode=mode), patch.object(
                sys,
                "argv",
                ["train_shs27k", "--dataset", "SHS27k", "--split", "dfs",
                 "--sampler-mode", mode],
            ):
                args = parse_args()
                self.assertEqual(args.sampler_mode, mode)

    def test_context_limit_is_derived_from_existing_parameters(self):
        with patch.object(
            sys,
            "argv",
            ["train_shs27k", "--dataset", "SHS27k", "--split", "dfs",
             "--max-steps", "3", "--fixed-num", "2"],
        ):
            args = parse_args()
        self.assertEqual(args.max_steps, 3)
        self.assertEqual(args.fixed_num, 2)


if __name__ == "__main__":
    unittest.main()
