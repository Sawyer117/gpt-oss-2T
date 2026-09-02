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
from estimate_candidates import pp_layer_balance


class ParameterAccountingTests(unittest.TestCase):
    def test_scaled_shape_defaults_to_figure_11_context(self) -> None:
        shape = GptOssShape(
            num_hidden_layers=48,
            hidden_size=8192,
            expert_intermediate_size=4096,
            num_routed_experts=256,
            experts_per_token=8,
            num_attention_heads=128,
            num_key_value_heads=16,
        )
        self.assertEqual(shape.context_length, 400_000)

    def test_pp_layer_balance_does_not_require_divisible_layers(self) -> None:
        utilization, allocation = pp_layer_balance(61, 16)
        self.assertAlmostEqual(utilization, 61 / 64)
        self.assertEqual(allocation, "13×4 + 3×3")

        utilization, allocation = pp_layer_balance(51, 16)
        self.assertAlmostEqual(utilization, 51 / 64)
        self.assertEqual(allocation, "3×4 + 13×3")

        utilization, allocation = pp_layer_balance(48, 16)
        self.assertEqual(utilization, 1.0)
        self.assertEqual(allocation, "16×3")

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

        topology = count_parameters(
            GptOssShape(
                num_hidden_layers=48,
                hidden_size=12288,
                expert_intermediate_size=8704,
                num_routed_experts=128,
                experts_per_token=4,
                num_attention_heads=256,
                num_key_value_heads=32,
            )
        )
        self.assertEqual(topology.total_parameters, 1_998_153_467_904)
        self.assertEqual(topology.active_parameters, 85_898_446_848)

    def test_deepseek_ratio_candidates_exact_ledger(self) -> None:
        fixed = count_parameters(
            GptOssShape(
                num_hidden_layers=56,
                hidden_size=8192,
                expert_intermediate_size=5632,
                num_routed_experts=256,
                experts_per_token=6,
                num_attention_heads=192,
                num_key_value_heads=24,
            )
        )
        self.assertEqual(fixed.total_parameters, 2_000_372_791_296)
        self.assertEqual(fixed.active_parameters, 60_957_030_400)

        free = count_parameters(
            GptOssShape(
                num_hidden_layers=88,
                hidden_size=5120,
                expert_intermediate_size=7680,
                num_routed_experts=192,
                experts_per_token=5,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(free.total_parameters, 2_003_585_906_176)
        self.assertEqual(free.active_parameters, 61_327_586_816)

    def test_kimi_top16_candidate_exact_ledger(self) -> None:
        ledger = count_parameters(
            GptOssShape(
                num_hidden_layers=84,
                hidden_size=4608,
                expert_intermediate_size=3584,
                num_routed_experts=480,
                experts_per_token=16,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(ledger.total_parameters, 2_006_838_708_096)
        self.assertEqual(ledger.active_parameters, 74_837_755_776)

    def test_kimi_free_topn_candidate_exact_ledger(self) -> None:
        ledger = count_parameters(
            GptOssShape(
                num_hidden_layers=52,
                hidden_size=6656,
                expert_intermediate_size=4608,
                num_routed_experts=416,
                experts_per_token=14,
                num_attention_heads=128,
                num_key_value_heads=16,
            )
        )
        self.assertEqual(ledger.total_parameters, 1_999_617_353_344)
        self.assertEqual(ledger.active_parameters, 74_848_691_840)

    def test_solver_reproduces_kimi_candidates(self) -> None:
        strict_grid = {
            "layer_values": range(48, 97, 4),
            "hidden_values": range(4096, 8193, 512),
            "intermediate_values": range(2048, 8193, 512),
            "expert_multiple": 32,
        }
        fixed = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=DEFAULT_TARGET_ACTIVE,
            top_ks=[16],
            limit=1,
            **strict_grid,
        )[0]
        self.assertEqual(fixed.shape.num_hidden_layers, 84)
        self.assertEqual(fixed.shape.hidden_size, 4608)
        self.assertEqual(fixed.shape.expert_intermediate_size, 3584)
        self.assertEqual(fixed.shape.num_routed_experts, 480)

        free = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=DEFAULT_TARGET_ACTIVE,
            top_ks=range(8, 25),
            exclude_top_ks={16},
            limit=1,
            **strict_grid,
        )[0]
        self.assertEqual(free.shape.num_hidden_layers, 52)
        self.assertEqual(free.shape.hidden_size, 6656)
        self.assertEqual(free.shape.expert_intermediate_size, 4608)
        self.assertEqual(free.shape.num_routed_experts, 416)
        self.assertEqual(free.shape.experts_per_token, 14)

    def test_solver_reproduces_deepseek_candidates(self) -> None:
        target = 61_294_450_936
        strict_grid = {
            "layer_values": range(48, 97, 4),
            "hidden_values": range(4096, 8193, 512),
            "intermediate_values": range(2048, 8193, 512),
            "expert_multiple": 32,
        }
        fixed = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=target,
            top_ks=[6],
            limit=1,
            **strict_grid,
        )[0]
        self.assertEqual(fixed.shape.num_hidden_layers, 56)
        self.assertEqual(fixed.shape.hidden_size, 8192)
        self.assertEqual(fixed.shape.expert_intermediate_size, 5632)
        self.assertEqual(fixed.shape.num_routed_experts, 256)

        free = search_shapes(
            target_total=DEFAULT_TARGET_TOTAL,
            target_active=target,
            top_ks=range(4, 17),
            exclude_top_ks={6},
            required_tp=16,
            required_ep=64,
            required_pp=8,
            limit=1,
            **strict_grid,
        )[0]
        self.assertEqual(free.shape.num_hidden_layers, 88)
        self.assertEqual(free.shape.hidden_size, 5120)
        self.assertEqual(free.shape.expert_intermediate_size, 7680)
        self.assertEqual(free.shape.num_routed_experts, 192)
        self.assertEqual(free.shape.experts_per_token, 5)

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
