#!/usr/bin/env python3
"""Estimate and publish GPT-OSS-shaped scaling candidates.

The estimator is an auditable architecture-accounting model, not a training or
performance predictor. With --update-note it also refreshes the generated
comparison block in the Chinese living research note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BEGIN_MARKER = "<!-- BEGIN GENERATED MODEL COMPARISON -->"
END_MARKER = "<!-- END GENERATED MODEL COMPARISON -->"


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

    # Q/K/V/O projections plus the public GPT-OSS attention biases.
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

    # Both embeddings and the untied LM head are stored. The active count
    # includes the full LM head but not the entire input embedding matrix.
    stored_fixed = 2 * vocab * hidden + hidden
    active_fixed = vocab * hidden + hidden

    total = stored_fixed + layers * (attention + all_experts + router + norms)
    active = active_fixed + layers * (attention + active_experts + router + norms)
    return total, active


def human_parameters(value: int) -> str:
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.3f}T"
    return f"{value / 1_000_000_000:.2f}B"


def human_context(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    return f"{value / 1_000:.0f}K"


def short_attention(value: str) -> str:
    if "KDA" in value:
        return "KDA + Gated MLA"
    if "Compressed Sparse" in value:
        return "CSA + HCA"
    return "Sliding + Full"


def load_and_validate() -> tuple[dict, dict, list[tuple[dict, int, int]]]:
    document = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    baselines = json.loads((ROOT / "baselines.json").read_text(encoding="utf-8"))
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

    return document, baselines, estimates


def render_comparison_table(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> str:
    models = baselines["models"]
    rows = [
        (
            "公开模型",
            "O0 / GPT-OSS-120B",
            models["gpt-oss-120b"]["total_parameters"],
            models["gpt-oss-120b"]["active_parameters"],
            models["gpt-oss-120b"],
            "校准基线",
        ),
        (
            "公开模型",
            "D0 / DeepSeek-V4-Pro",
            models["deepseek-v4-pro"]["total_parameters"],
            models["deepseek-v4-pro"]["active_parameters"],
            models["deepseek-v4-pro"],
            "49B 激活工业锚点",
        ),
        (
            "公开模型",
            "K0 / Kimi-K3",
            models["kimi-k3"]["total_parameters"],
            models["kimi-k3"]["active_parameters"],
            models["kimi-k3"],
            "104B 激活工业锚点",
        ),
    ]

    for candidate, total, active in estimates:
        if candidate["id"] == "O0":
            continue
        model = {
            "num_hidden_layers": candidate["num_hidden_layers"],
            "hidden_size": candidate["hidden_size"],
            "num_routed_experts": candidate["num_routed_experts"],
            "experts_per_token": candidate["experts_per_token"],
            "context_length": candidate.get(
                "context_length", document["defaults"]["context_length"]
            ),
            "attention": "GPT-OSS alternating sliding/full",
        }
        rows.append(
            (
                "缩放候选",
                f"{candidate['id']} / {candidate['name']}",
                total,
                active,
                model,
                candidate["axis"],
            )
        )

    lines = [
        "> 公开模型使用上游仓库精确 metadata/模型卡；O1–O5 使用 GPT-OSS 形状估算器。",
        "",
        "| 性质 | ID / 模型 | 总参数 | 激活/token | 层数 | Hidden | 专家 / Top-k | 上下文 | Attention | 主要轴 / 用途 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for kind, name, total, active, model, purpose in rows:
        lines.append(
            f"| {kind} | {name} | {human_parameters(total)} | "
            f"{human_parameters(active)} | {model['num_hidden_layers']} | "
            f"{model['hidden_size']:,} | {model['num_routed_experts']} / "
            f"{model['experts_per_token']} | {human_context(model['context_length'])} | "
            f"{short_attention(model['attention'])} | {purpose} |"
        )
    return "\n".join(lines)


def render_document(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> str:
    target = document["target_total_parameters"]
    baseline = baselines["models"]["gpt-oss-120b"]
    o0_total = estimates[0][1]
    o0_active = estimates[0][2]
    total_error = (o0_total - baseline["total_parameters"]) / baseline["total_parameters"]
    active_error = (o0_active - baseline["active_parameters"]) / baseline["active_parameters"]

    lines = [
        "# GPT-OSS 形状缩放候选（自动生成）",
        "",
        "> 由 `estimate_candidates.py` 生成；请修改 `candidates.json`，不要手工编辑本文件。",
        "",
        "## 公开工业模型与候选总览",
        "",
        render_comparison_table(document, baselines, estimates),
        "",
        "## GPT-OSS 形状候选的估算误差",
        "",
        "| ID | 候选 | 主缩放轴 | 总参数 | 每 token 激活 | 激活/总量 | 距 2T |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for candidate, total, active in estimates:
        error = (total - target) / target
        lines.append(
            f"| {candidate['id']} | {candidate['name']} | {candidate['axis']} | "
            f"{human_parameters(total)} | {human_parameters(active)} | "
            f"{100 * active / total:.3f}% | {100 * error:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 基线校准",
            "",
            f"- O0 估算总参数 {human_parameters(o0_total)}，Hugging Face safetensors metadata 为 "
            f"{human_parameters(baseline['total_parameters'])}，误差 {100 * total_error:+.3f}%。",
            f"- O0 估算激活参数 {human_parameters(o0_active)}，模型卡为 "
            f"{human_parameters(baseline['active_parameters'])}，误差 {100 * active_error:+.3f}%。",
            "",
            "## 判读",
            "",
        ]
    )
    for candidate in document["candidates"]:
        lines.append(f"- **{candidate['id']} — {candidate['name']}:** {candidate['assessment']}")
    lines.extend(
        [
            "",
            "## 公式边界",
            "",
            "本表只估算 GPT-OSS 形状的 Transformer 与 MoE 张量；不覆盖 Kimi Stable LatentMoE、"
            "KDA/MLA 状态、DeepSeek HCA/CSA 索引、优化器状态、激活显存、KV cache、通信缓冲或模型质量。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_note(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError(f"generated comparison markers are missing or duplicated in {path}")
    start = text.index(BEGIN_MARKER)
    end = text.index(END_MARKER) + len(END_MARKER)
    replacement = f"{BEGIN_MARKER}\n\n{table}\n\n{END_MARKER}"
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-note",
        type=Path,
        help="replace the generated comparison block in the Chinese living note",
    )
    args = parser.parse_args()

    document, baselines, estimates = load_and_validate()
    comparison = render_comparison_table(document, baselines, estimates)
    if args.update_note:
        update_note(args.update_note, comparison)
    print(render_document(document, baselines, estimates), end="")


if __name__ == "__main__":
    main()
