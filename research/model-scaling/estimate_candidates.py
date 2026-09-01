#!/usr/bin/env python3
"""Estimate total and per-token active parameters for GPT-OSS-shaped candidates.

The estimator is a transparent architecture-accounting model, not a training or
performance predictor. It counts the dominant tensors in the public GPT-OSS
configuration and intentionally keeps the formula easy to audit.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def estimate(candidate: dict, defaults: dict) -> tuple[int, int]:
    layers = candidate["num_hidden_layers"]
    hidden = candidate["hidden_size"]
    intermediate = candidate["expert_intermediate_size"]
    experts = candidate["num_routed_experts"]
    top_k = candidate["experts_per_token"]
    heads = candidate["num_attention_heads"]
    kv_heads = candidate["num_key_value_heads"]
    vocab = candidate.get("vocab_size", defaults["vocab_size"])
    head_dim = candidate.get("head_dim", defaults["head_dim"])

    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim

    # Q/K/V/O projections plus their public GPT-OSS attention biases.
    attention = (
        hidden * (q_dim + kv_dim + kv_dim)
        + q_dim * hidden
        + q_dim
        + kv_dim
        + kv_dim
        + hidden
    )

    # SwiGLU gate/up/down weights for every routed expert.
    all_experts = experts * 3 * hidden * intermediate
    active_experts = top_k * 3 * hidden * intermediate
    router = hidden * experts + experts
    norms = 2 * hidden

    # Input embeddings and untied LM head are both stored. For the active count,
    # only the full LM head is counted; an embedding lookup does not touch the
    # entire embedding matrix. This reproduces the published 5.1B baseline.
    stored_fixed = 2 * vocab * hidden + hidden
    active_fixed = vocab * hidden + hidden

    total = stored_fixed + layers * (attention + all_experts + router + norms)
    active = active_fixed + layers * (attention + active_experts + router + norms)
    return total, active


def human_parameters(value: int) -> str:
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.3f}T"
    return f"{value / 1_000_000_000:.2f}B"


def main() -> None:
    document = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    baselines = json.loads((ROOT / "baselines.json").read_text(encoding="utf-8"))
    target = document["target_total_parameters"]

    estimates = []
    for candidate in document["candidates"]:
        if not 0 < candidate["experts_per_token"] <= candidate["num_routed_experts"]:
            raise ValueError(f"invalid top-k for {candidate['id']}")
        if candidate["num_attention_heads"] <= 0 or candidate["num_key_value_heads"] <= 0:
            raise ValueError(f"invalid attention heads for {candidate['id']}")
        estimates.append((candidate, *estimate(candidate, document["defaults"])))

    baseline = baselines["models"]["gpt-oss-120b"]
    o0_total = estimates[0][1]
    o0_active = estimates[0][2]
    total_error = (o0_total - baseline["total_parameters"]) / baseline["total_parameters"]
    active_error = (o0_active - baseline["active_parameters"]) / baseline["active_parameters"]
    if abs(total_error) > 0.01 or abs(active_error) > 0.02:
        raise ValueError("O0 calibration drifted beyond the allowed error")

    print("# GPT-OSS 形状缩放候选（自动生成）\n")
    print("> 由 `estimate_candidates.py` 生成；请修改 `candidates.json`，不要手工编辑本文件。\n")
    print("| ID | 候选 | 主缩放轴 | 总参数 | 每 token 激活 | 激活/总量 | 距 2T |")
    print("|---|---|---|---:|---:|---:|---:|")

    for candidate, total, active in estimates:
        error = (total - target) / target
        print(
            f"| {candidate['id']} | {candidate['name']} | {candidate['axis']} | "
            f"{human_parameters(total)} | {human_parameters(active)} | "
            f"{100 * active / total:.3f}% | {100 * error:+.2f}% |"
        )

    print("\n## 基线校准\n")
    print(
        f"- O0 估算总参数 {human_parameters(o0_total)}，Hugging Face safetensors metadata 为 "
        f"{human_parameters(baseline['total_parameters'])}，误差 {100 * total_error:+.3f}%。"
    )
    print(
        f"- O0 估算激活参数 {human_parameters(o0_active)}，模型卡为 "
        f"{human_parameters(baseline['active_parameters'])}，误差 {100 * active_error:+.3f}%。"
    )

    print("\n## 判读\n")
    for candidate in document["candidates"]:
        print(f"- **{candidate['id']} — {candidate['name']}:** {candidate['assessment']}")

    print("\n## 公式边界\n")
    print(
        "本表只估算 GPT-OSS 形状的 Transformer 与 MoE 张量；不覆盖 Kimi Stable LatentMoE、"
        "KDA/MLA 状态、DeepSeek HCA/CSA 索引、优化器状态、激活显存、KV cache、通信缓冲或模型质量。"
    )


if __name__ == "__main__":
    main()
