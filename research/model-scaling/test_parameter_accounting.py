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
    def test_industrial_ratios_normalized_to_2t(self) -> None:
        cases = (
            (116_829_156_672, 5_100_000_000, 87_306_972_767),
            (1_598_839_674_782, 49_000_000_000, 61_294_450_936),
            (2_779_931_837_184, 104_000_000_000, DEFAULT_TARGET_ACTIVE),
        )
        for total, active, expected in cases:
            with self.subTest(total=total):
                self.assertEqual(
                    round(DEFAULT_TARGET_TOTAL * active / total), expected
                )

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

    def test_gpt_ratio_candidate_exact_ledger(self) -> None:
        ledger = count_parameters(
            GptOssShape(
                num_hidden_layers=51,
                hidden_size=12288,
                expert_intermediate_size=8192,
                num_routed_experts=128,
                experts_per_token=4,
                num_attention_heads=256,
                num_key_value_heads=32,
            )
        )
        self.assertEqual(ledger.total_parameters, 1_999_517_313_408)
        self.assertEqual(ledger.active_parameters, 87_262_292_352)

    def test_deepseek_ratio_candidates_exact_ledger(self) -> None:
        fixed = count_parameters(
            GptOssShape(
                num_hidden_layers=79,
                hidden_size=5888,
                expert_intermediate_size=6144,
                num_routed_experts=232,
                experts_per_token=6,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(fixed.total_parameters, 2_000_145_983_896)
        self.assertEqual(fixed.active_parameters, 61_309_921_688)

        free = count_parameters(
            GptOssShape(
                num_hidden_layers=94,
                hidden_size=4352,
                expert_intermediate_size=6144,
                num_routed_experts=264,
                experts_per_token=7,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(free.total_parameters, 2_000_042_642_416)
        self.assertEqual(free.active_parameters, 61_307_833_328)

    def test_kimi_top16_candidate_exact_ledger(self) -> None:
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

    def test_kimi_free_topn_candidate_exact_ledger(self) -> None:
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

    def test_solver_reproduces_kimi_candidates(self) -> None:
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

    def test_solver_reproduces_deepseek_candidates(self) -> None:
        target = 61_294_450_936
        fixed = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=target,
            top_ks=[6],
            limit=1,
        )[0]
        self.assertEqual(fixed.shape.num_hidden_layers, 79)
        self.assertEqual(fixed.shape.hidden_size, 5888)
        self.assertEqual(fixed.shape.expert_intermediate_size, 6144)
        self.assertEqual(fixed.shape.num_routed_experts, 232)

        free = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=target,
            top_ks=range(4, 17),
            exclude_top_ks={6},
            limit=1,
        )[0]
        self.assertEqual(free.shape.num_hidden_layers, 94)
        self.assertEqual(free.shape.hidden_size, 4352)
        self.assertEqual(free.shape.expert_intermediate_size, 6144)
        self.assertEqual(free.shape.num_routed_experts, 264)
        self.assertEqual(free.shape.experts_per_token, 7)

    def test_solver_reproduces_gpt_candidate(self) -> None:
        result = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=87_306_972_767,
            top_ks=[4],
            layer_values=range(36, 65),
            hidden_values=range(4096, 12289, 256),
            intermediate_values=range(2048, 12289, 256),
            min_experts=64,
            max_experts=768,
            limit=1,
        )[0]
        self.assertEqual(result.shape.num_hidden_layers, 51)
        self.assertEqual(result.shape.hidden_size, 12288)
        self.assertEqual(result.shape.expert_intermediate_size, 8192)
        self.assertEqual(result.shape.num_routed_experts, 128)
        self.assertEqual(result.shape.experts_per_token, 4)


if __name__ == "__main__":
    unittest.main()
