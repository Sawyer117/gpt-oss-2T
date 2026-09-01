#!/usr/bin/env python3
"""Regression checks for the GPT-OSS-shaped parameter ledger and solver."""

from __future__ import annotations

import unittest

from parameter_accounting import (
    DEFAULT_TARGET_ACTIVE,
    DEFAULT_TARGET_TOTAL,
    GptOssShape,
    count_parameters,
    search_shapes,
)


class ParameterAccountingTests(unittest.TestCase):
    def test_kimi_ratio_normalized_to_2t(self) -> None:
        kimi_total = 2_779_931_837_184
        kimi_active = 104_000_000_000
        normalized = round(DEFAULT_TARGET_TOTAL * kimi_active / kimi_total)
        self.assertEqual(normalized, DEFAULT_TARGET_ACTIVE)

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
                num_hidden_layers=72,
                hidden_size=6144,
                expert_intermediate_size=3072,
                num_routed_experts=488,
                experts_per_token=16,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(ledger.total_parameters, 2_000_352_059_712)
        self.assertEqual(ledger.active_parameters, 74_837_008_704)

    def test_free_topn_candidate_exact_ledger(self) -> None:
        ledger = count_parameters(
            GptOssShape(
                num_hidden_layers=75,
                hidden_size=6912,
                expert_intermediate_size=4096,
                num_routed_experts=312,
                experts_per_token=10,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(ledger.total_parameters, 1_999_970_034_024)
        self.assertEqual(ledger.active_parameters, 74_810_155_368)

    def test_solver_reproduces_published_candidates(self) -> None:
        fixed = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=DEFAULT_TARGET_ACTIVE,
            top_ks=[16],
            limit=1,
        )[0]
        self.assertEqual(fixed.shape.num_hidden_layers, 72)
        self.assertEqual(fixed.shape.hidden_size, 6144)
        self.assertEqual(fixed.shape.expert_intermediate_size, 3072)
        self.assertEqual(fixed.shape.num_routed_experts, 488)

        free = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=DEFAULT_TARGET_ACTIVE,
            top_ks=range(8, 25),
            exclude_top_ks={16},
            limit=1,
        )[0]
        self.assertEqual(free.shape.num_hidden_layers, 75)
        self.assertEqual(free.shape.hidden_size, 6912)
        self.assertEqual(free.shape.expert_intermediate_size, 4096)
        self.assertEqual(free.shape.num_routed_experts, 312)
        self.assertEqual(free.shape.experts_per_token, 10)


if __name__ == "__main__":
    unittest.main()
