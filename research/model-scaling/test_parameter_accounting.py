#!/usr/bin/env python3
"""Regression checks for the GPT-OSS-shaped parameter ledger and solver."""

from __future__ import annotations

import unittest

from parameter_accounting import GptOssShape, count_parameters, search_shapes


class ParameterAccountingTests(unittest.TestCase):
    def test_gpt_oss_120b_schema_calibration(self) -> None:
        ledger = count_parameters(
            GptOssShape(
                num_hidden_layers=36,
                hidden_size=2880,
                expert_intermediate_size=2880,
                num_routed_experts=128,
                experts_per_token=4,
                num_attention_heads=64,
                num_key_value_heads=8,
                context_length=131072,
            )
        )
        self.assertEqual(ledger.total_parameters, 116_789_341_248)
        self.assertEqual(ledger.active_parameters, 5_131_603_008)
        official_checkpoint_elements = 116_829_156_672
        self.assertLess(
            abs(ledger.total_parameters - official_checkpoint_elements)
            / official_checkpoint_elements,
            0.001,
        )

    def test_top16_candidate_exact_ledger(self) -> None:
        ledger = count_parameters(
            GptOssShape(
                num_hidden_layers=54,
                hidden_size=5632,
                expert_intermediate_size=6656,
                num_routed_experts=328,
                experts_per_token=16,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(ledger.total_parameters, 1_999_859_537_712)
        self.assertEqual(ledger.active_parameters, 104_004_704_048)

    def test_free_topn_candidate_exact_ledger(self) -> None:
        ledger = count_parameters(
            GptOssShape(
                num_hidden_layers=61,
                hidden_size=4096,
                expert_intermediate_size=6400,
                num_routed_experts=416,
                experts_per_token=21,
                num_attention_heads=64,
                num_key_value_heads=8,
            )
        )
        self.assertEqual(ledger.total_parameters, 1_999_704_860_448)
        self.assertEqual(ledger.active_parameters, 103_973_300_000)

    def test_solver_reproduces_published_candidates(self) -> None:
        fixed = search_shapes(
            target_total=2_000_000_000_000,
            target_active=104_000_000_000,
            top_ks=[16],
            limit=1,
        )[0]
        self.assertEqual(fixed.shape.num_hidden_layers, 54)
        self.assertEqual(fixed.shape.hidden_size, 5632)
        self.assertEqual(fixed.shape.expert_intermediate_size, 6656)
        self.assertEqual(fixed.shape.num_routed_experts, 328)

        free = search_shapes(
            target_total=2_000_000_000_000,
            target_active=104_000_000_000,
            top_ks=range(8, 25),
            exclude_top_ks={16},
            limit=1,
        )[0]
        self.assertEqual(free.shape.num_hidden_layers, 61)
        self.assertEqual(free.shape.hidden_size, 4096)
        self.assertEqual(free.shape.expert_intermediate_size, 6400)
        self.assertEqual(free.shape.num_routed_experts, 416)
        self.assertEqual(free.shape.experts_per_token, 21)


if __name__ == "__main__":
    unittest.main()
