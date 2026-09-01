#!/usr/bin/env python3
"""Exact tensor-schema accounting and discrete search for GPT-OSS-shaped proxies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable


DEFAULT_TARGET_TOTAL = 2_000_000_000_000
DEFAULT_TARGET_ACTIVE = 74_821_978_445


@dataclass(frozen=True)
class GptOssShape:
    num_hidden_layers: int
    hidden_size: int
    expert_intermediate_size: int
    num_routed_experts: int
    experts_per_token: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int = 201088
    head_dim: int = 64
    context_length: int = 400000

    def validate(self) -> None:
        positive = (
            self.num_hidden_layers,
            self.hidden_size,
            self.expert_intermediate_size,
            self.num_routed_experts,
            self.experts_per_token,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.vocab_size,
            self.head_dim,
            self.context_length,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all shape fields must be positive")
        if self.experts_per_token > self.num_routed_experts:
            raise ValueError("experts_per_token cannot exceed num_routed_experts")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")


@dataclass(frozen=True)
class ParameterLedger:
    stored: dict[str, int]
    active_per_token: dict[str, int]

    @property
    def total_parameters(self) -> int:
        return sum(self.stored.values())

    @property
    def active_parameters(self) -> int:
        return sum(self.active_per_token.values())

    def as_dict(self) -> dict:
        return {
            "stored": self.stored,
            "active_per_token": self.active_per_token,
            "total_parameters": self.total_parameters,
            "active_parameters": self.active_parameters,
            "active_fraction": self.active_parameters / self.total_parameters,
        }


@dataclass(frozen=True)
class SearchResult:
    shape: GptOssShape
    ledger: ParameterLedger
    total_error: int
    active_error: int
    score: float

    def as_dict(self) -> dict:
        return {
            "shape": asdict(self.shape),
            "total_parameters": self.ledger.total_parameters,
            "active_parameters": self.ledger.active_parameters,
            "total_error": self.total_error,
            "active_error": self.active_error,
            "score": self.score,
        }


def human_parameters(value: int) -> str:
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.3f}T"
    return f"{value / 1_000_000_000:.2f}B"


def shape_from_mapping(mapping: dict, defaults: dict | None = None) -> GptOssShape:
    merged = dict(defaults or {})
    merged.update(mapping)
    return GptOssShape(
        num_hidden_layers=merged["num_hidden_layers"],
        hidden_size=merged["hidden_size"],
        expert_intermediate_size=merged["expert_intermediate_size"],
        num_routed_experts=merged["num_routed_experts"],
        experts_per_token=merged["experts_per_token"],
        num_attention_heads=merged["num_attention_heads"],
        num_key_value_heads=merged["num_key_value_heads"],
        vocab_size=merged.get("vocab_size", 201088),
        head_dim=merged.get("head_dim", 64),
        context_length=merged.get("context_length", 400000),
    )


def count_parameters(shape: GptOssShape) -> ParameterLedger:
    """Count every tensor group in the declared GPT-OSS-shaped logical schema.

    Active parameters include the full LM head, attention, router, norms, and
    selected experts. The full input embedding table is stored but not counted
    as active because a token lookup accesses one row rather than the matrix.
    """

    shape.validate()
    layers = shape.num_hidden_layers
    hidden = shape.hidden_size
    intermediate = shape.expert_intermediate_size
    experts = shape.num_routed_experts
    top_k = shape.experts_per_token
    q_dim = shape.num_attention_heads * shape.head_dim
    kv_dim = shape.num_key_value_heads * shape.head_dim

    attention_weights_per_layer = hidden * (q_dim + 2 * kv_dim) + q_dim * hidden
    attention_biases_per_layer = q_dim + 2 * kv_dim + hidden
    router_per_layer = hidden * experts + experts
    norms_per_layer = 2 * hidden
    expert_per_layer = experts * 3 * hidden * intermediate
    active_expert_per_layer = top_k * 3 * hidden * intermediate

    stored = {
        "token_embeddings": shape.vocab_size * hidden,
        "lm_head": shape.vocab_size * hidden,
        "final_norm": hidden,
        "attention_weights": layers * attention_weights_per_layer,
        "attention_biases": layers * attention_biases_per_layer,
        "layer_norms": layers * norms_per_layer,
        "routers": layers * router_per_layer,
        "routed_expert_weights": layers * expert_per_layer,
    }
    active = {
        "lm_head": shape.vocab_size * hidden,
        "final_norm": hidden,
        "attention_weights": layers * attention_weights_per_layer,
        "attention_biases": layers * attention_biases_per_layer,
        "layer_norms": layers * norms_per_layer,
        "routers": layers * router_per_layer,
        "selected_expert_weights": layers * active_expert_per_layer,
    }
    return ParameterLedger(stored=stored, active_per_token=active)


def nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def gpt_oss_like_heads(hidden_size: int) -> tuple[int, int]:
    # Preserve the baseline Q-projection/hidden ratio, then snap to a topology-
    # friendly 64-head group and retain 8:1 grouped-query attention.
    heads = nearest_multiple(hidden_size * 64 / 2880, 64)
    return heads, heads // 8


def search_shapes(
    *,
    target_total: int,
    target_active: int,
    top_ks: Iterable[int],
    exclude_top_ks: set[int] | None = None,
    layer_values: Iterable[int] = range(48, 97),
    hidden_values: Iterable[int] = range(4096, 8193, 256),
    intermediate_values: Iterable[int] = range(2048, 8193, 256),
    min_experts: int = 192,
    max_experts: int = 768,
    expert_multiple: int = 8,
    required_tp: int | None = None,
    required_ep: int | None = None,
    required_pp: int | None = None,
    limit: int = 10,
) -> list[SearchResult]:
    """Search aligned industrial shapes without brute-forcing expert counts.

    Layer, hidden, and intermediate sizes use fixed aligned grids. The ideal
    expert count is solved analytically and snapped to ``expert_multiple``.
    Optional TP/EP/PP requirements turn hardware topology into a hard filter
    rather than a subjective tie-break. For each remaining shape, all totals
    are recomputed from the tensor ledger.
    """

    excluded = exclude_top_ks or set()
    results: list[SearchResult] = []
    top_k_values = [value for value in top_ks if value not in excluded]

    for layers in layer_values:
        if required_pp is not None and layers % required_pp:
            continue
        for hidden in hidden_values:
            heads, kv_heads = gpt_oss_like_heads(hidden)
            if required_tp is not None and (
                hidden % required_tp
                or heads % required_tp
                or kv_heads % required_tp
            ):
                continue
            for intermediate in intermediate_values:
                if required_tp is not None and intermediate % required_tp:
                    continue
                probe = GptOssShape(
                    num_hidden_layers=layers,
                    hidden_size=hidden,
                    expert_intermediate_size=intermediate,
                    num_routed_experts=1,
                    experts_per_token=1,
                    num_attention_heads=heads,
                    num_key_value_heads=kv_heads,
                )
                probe_ledger = count_parameters(probe)
                per_expert = layers * (3 * hidden * intermediate + hidden + 1)
                base_without_expert = probe_ledger.total_parameters - per_expert
                ideal_experts = (target_total - base_without_expert) / per_expert
                snapped = nearest_multiple(ideal_experts, expert_multiple)

                for experts in (
                    snapped - expert_multiple,
                    snapped,
                    snapped + expert_multiple,
                ):
                    if not min_experts <= experts <= max_experts:
                        continue
                    if required_ep is not None and experts % required_ep:
                        continue
                    for top_k in top_k_values:
                        if not 1 <= top_k <= experts:
                            continue
                        shape = GptOssShape(
                            num_hidden_layers=layers,
                            hidden_size=hidden,
                            expert_intermediate_size=intermediate,
                            num_routed_experts=experts,
                            experts_per_token=top_k,
                            num_attention_heads=heads,
                            num_key_value_heads=kv_heads,
                        )
                        ledger = count_parameters(shape)
                        total_error = ledger.total_parameters - target_total
                        active_error = ledger.active_parameters - target_active
                        score = (
                            abs(total_error) / target_total
                            + abs(active_error) / target_active
                        )
                        results.append(
                            SearchResult(
                                shape=shape,
                                ledger=ledger,
                                total_error=total_error,
                                active_error=active_error,
                                score=score,
                            )
                        )

    results.sort(key=lambda result: result.score)
    return results[:limit]


def format_ledger(shape: GptOssShape, ledger: ParameterLedger) -> str:
    lines = [
        json.dumps(asdict(shape), ensure_ascii=False, indent=2),
        "",
        "Stored tensor groups:",
    ]
    lines.extend(f"  {name}: {value:,}" for name, value in ledger.stored.items())
    lines.append(f"  TOTAL: {ledger.total_parameters:,}")
    lines.append("")
    lines.append("Active-per-token tensor groups:")
    lines.extend(
        f"  {name}: {value:,}" for name, value in ledger.active_per_token.items()
    )
    lines.append(f"  ACTIVE: {ledger.active_parameters:,}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    count = subparsers.add_parser("count", help="count one explicit shape")
    count.add_argument("--layers", type=int, required=True)
    count.add_argument("--hidden", type=int, required=True)
    count.add_argument("--intermediate", type=int, required=True)
    count.add_argument("--experts", type=int, required=True)
    count.add_argument("--top-k", type=int, required=True)
    count.add_argument("--heads", type=int, required=True)
    count.add_argument("--kv-heads", type=int, required=True)
    count.add_argument("--vocab-size", type=int, default=201088)
    count.add_argument("--head-dim", type=int, default=64)
    count.add_argument("--context-length", type=int, default=400000)
    count.add_argument("--json", action="store_true")

    solve = subparsers.add_parser("solve", help="search aligned near-2T shapes")
    solve.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    solve.add_argument("--target-active", type=int, default=DEFAULT_TARGET_ACTIVE)
    solve.add_argument("--fixed-top-k", type=int)
    solve.add_argument("--top-k-min", type=int, default=8)
    solve.add_argument("--top-k-max", type=int, default=24)
    solve.add_argument("--exclude-top-k", type=int, action="append", default=[])
    solve.add_argument("--layer-min", type=int, default=48)
    solve.add_argument("--layer-max", type=int, default=96)
    solve.add_argument("--hidden-min", type=int, default=4096)
    solve.add_argument("--hidden-max", type=int, default=8192)
    solve.add_argument("--intermediate-min", type=int, default=2048)
    solve.add_argument("--intermediate-max", type=int, default=8192)
    solve.add_argument("--dimension-step", type=int, default=256)
    solve.add_argument("--min-experts", type=int, default=192)
    solve.add_argument("--max-experts", type=int, default=768)
    solve.add_argument("--expert-multiple", type=int, default=8)
    solve.add_argument(
        "--require-tp",
        type=int,
        help="require hidden/intermediate/Q-head/KV-head divisibility",
    )
    solve.add_argument(
        "--require-ep", type=int, help="require expert-count divisibility"
    )
    solve.add_argument(
        "--require-pp",
        type=int,
        help="optionally require an exact equal-layer PP split; not required for PP",
    )
    solve.add_argument("--layer-step", type=int, default=1)
    solve.add_argument("--limit", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "count":
        shape = GptOssShape(
            num_hidden_layers=args.layers,
            hidden_size=args.hidden,
            expert_intermediate_size=args.intermediate,
            num_routed_experts=args.experts,
            experts_per_token=args.top_k,
            num_attention_heads=args.heads,
            num_key_value_heads=args.kv_heads,
            vocab_size=args.vocab_size,
            head_dim=args.head_dim,
            context_length=args.context_length,
        )
        ledger = count_parameters(shape)
        if args.json:
            print(json.dumps({"shape": asdict(shape), **ledger.as_dict()}, indent=2))
        else:
            print(format_ledger(shape, ledger))
        return

    top_ks = (
        [args.fixed_top_k]
        if args.fixed_top_k is not None
        else range(args.top_k_min, args.top_k_max + 1)
    )
    results = search_shapes(
        target_total=args.target_total,
        target_active=args.target_active,
        top_ks=top_ks,
        exclude_top_ks=set(args.exclude_top_k),
        layer_values=range(args.layer_min, args.layer_max + 1, args.layer_step),
        hidden_values=range(
            args.hidden_min, args.hidden_max + 1, args.dimension_step
        ),
        intermediate_values=range(
            args.intermediate_min,
            args.intermediate_max + 1,
            args.dimension_step,
        ),
        min_experts=args.min_experts,
        max_experts=args.max_experts,
        expert_multiple=args.expert_multiple,
        required_tp=args.require_tp,
        required_ep=args.require_ep,
        required_pp=args.require_pp,
        limit=args.limit,
    )
    print(json.dumps([result.as_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
