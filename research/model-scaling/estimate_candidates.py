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

from parameter_accounting import count_parameters, human_parameters, shape_from_mapping


ROOT = Path(__file__).resolve().parent
BEGIN_MARKER = "<!-- BEGIN GENERATED MODEL COMPARISON -->"
END_MARKER = "<!-- END GENERATED MODEL COMPARISON -->"
BEGIN_AFFINITY_MARKER = "<!-- BEGIN GENERATED PARALLEL AFFINITY -->"
END_AFFINITY_MARKER = "<!-- END GENERATED PARALLEL AFFINITY -->"


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
        shape = shape_from_mapping(candidate, document["defaults"])
        ledger = count_parameters(shape)
        estimates.append(
            (candidate, ledger.total_parameters, ledger.active_parameters)
        )

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
            "实际 49B；2T 同比目标 61.29B",
        ),
        (
            "公开模型",
            "K0 / Kimi-K3",
            models["kimi-k3"]["total_parameters"],
            models["kimi-k3"]["active_parameters"],
            models["kimi-k3"],
            "实际 104B；2T 同比目标 74.82B",
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
        "> 公开模型使用上游仓库精确 metadata/模型卡；L/G/D/K 候选使用 GPT-OSS 形状参数账本。",
        "",
        "| 性质 | ID / 模型 | 总参数 | 激活/token | 激活/总量 | 层数 | Hidden | 专家 / Top-k | 上下文 | Attention | 主要轴 / 用途 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for kind, name, total, active, model, purpose in rows:
        lines.append(
            f"| {kind} | {name} | {human_parameters(total)} | "
            f"{human_parameters(active)} | {100 * active / total:.3f}% | "
            f"{model['num_hidden_layers']} | "
            f"{model['hidden_size']:,} | {model['num_routed_experts']} / "
            f"{model['experts_per_token']} | {human_context(model['context_length'])} | "
            f"{short_attention(model['attention'])} | {purpose} |"
        )
    return "\n".join(lines)


def parallel_degrees(model: dict) -> tuple[str, str, str, str, str]:
    tp_options = (8, 16, 32)
    ep_options = (8, 16, 32, 64)
    pp_options = (2, 4, 8, 16)
    hidden = model["hidden_size"]
    intermediate = model["expert_intermediate_size"]
    heads = model["num_attention_heads"]
    kv_heads = model["num_key_value_heads"]
    experts = model["num_routed_experts"]
    layers = model["num_hidden_layers"]

    strict_tp = [
        value
        for value in tp_options
        if hidden % value == 0
        and intermediate % value == 0
        and heads % value == 0
        and kv_heads % value == 0
    ]
    replicated_kv_tp = [
        value
        for value in tp_options
        if hidden % value == 0
        and intermediate % value == 0
        and heads % value == 0
    ]
    ep = [value for value in ep_options if experts % value == 0]
    pp = [value for value in pp_options if layers % value == 0]

    def show(values: list[int]) -> str:
        return "/".join(str(value) for value in values) if values else "—"

    limitation = (
        f"最高严格整分：TP{max(strict_tp) if strict_tp else '—'} / "
        f"EP{max(ep) if ep else '—'} / PP{max(pp) if pp else '—'}"
    )
    return show(strict_tp), show(replicated_kv_tp), show(ep), show(pp), limitation


def render_parallel_affinity_table(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> str:
    models = baselines["models"]
    rows = [
        ("O0", "GPT-OSS-120B", models["gpt-oss-120b"]),
        ("D0", "DeepSeek-V4-Pro", models["deepseek-v4-pro"]),
        ("K0", "Kimi-K3", models["kimi-k3"]),
    ]
    for candidate, _, _ in estimates:
        if candidate["id"] == "O0":
            continue
        model = {
            "num_hidden_layers": candidate["num_hidden_layers"],
            "hidden_size": candidate["hidden_size"],
            "expert_intermediate_size": candidate["expert_intermediate_size"],
            "num_routed_experts": candidate["num_routed_experts"],
            "num_attention_heads": candidate["num_attention_heads"],
            "num_key_value_heads": candidate["num_key_value_heads"],
        }
        rows.append((candidate["id"], candidate["name"], model))

    lines = [
        "> 严格 TP 要求 hidden、expert intermediate、Q heads、KV heads 均可整分；“KV复制”列允许 KV heads 复制。PP 只检查层数均分，未计 embedding/LM head 与层类型负载差异。",
        "",
        "| ID / 模型 | 严格 TP | TP（允许KV复制） | EP | PP | 严格整分上限 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for identifier, name, model in rows:
        strict_tp, relaxed_tp, ep, pp, limitation = parallel_degrees(model)
        lines.append(
            f"| {identifier} / {name} | {strict_tp} | {relaxed_tp} | "
            f"{ep} | {pp} | {limitation} |"
        )
    lines.extend(
        [
            "",
            "- TP 候选集合：8/16/32；EP：8/16/32/64；PP：2/4/8/16。",
            "- `—` 表示在上述候选集合中没有完全整分方案，不表示工程上绝对不能运行。",
            "- TP（Tensor Parallel）看单个矩阵与 attention heads 能否切开；EP（Expert Parallel）看专家数能否均匀放置；PP（Pipeline Parallel）看层数能否均匀分 stage。",
            "- 该表只是一级算术门槛；真实亲和性还受单机 GPU 数、NVLink/IB 拓扑、all-to-all、专家负载均衡和 kernel tile 约束影响。",
            "- DeepSeek 的 CSA/HCA 与 Kimi 的 KDA/MLA 是自定义 attention；公开模型行的 TP 结果只是维度整除检查，仍需按真实 kernel 验证。",
            "- PP 可使用不等长 stage，但表中只把完全均分视为严格亲和。",
            "- DP（Data Parallel）通常不要求模型维度整除；CP/SP（Context/Sequence Parallel）对 400K 长上下文很重要，但取决于序列长度、attention pattern 与 kernel，不能只靠本表的模型维度判定。",
        ]
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
        "## TP / EP / PP 亲和性",
        "",
        render_parallel_affinity_table(document, baselines, estimates),
        "",
        "## GPT-OSS 形状候选的估算误差",
        "",
        "| ID | 候选 | 目标族 | 总参数 | 每 token 激活 | 激活/总量 | 距 2T | 目标激活 | 距目标 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, total, active in estimates:
        error = (total - target) / target
        active_target = candidate.get("target_active_parameters")
        active_target_text = (
            human_parameters(active_target) if active_target is not None else "—"
        )
        active_target_error_text = (
            f"{100 * (active - active_target) / active_target:+.2f}%"
            if active_target is not None
            else "—"
        )
        lines.append(
            f"| {candidate['id']} | {candidate['name']} | {candidate['target_family']} | "
            f"{human_parameters(total)} | {human_parameters(active)} | "
            f"{100 * active / total:.3f}% | {100 * error:+.2f}% | "
            f"{active_target_text} | {active_target_error_text} |"
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


def replace_generated_block(text: str, begin: str, end: str, body: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"generated markers are missing or duplicated: {begin}")
    start = text.index(begin)
    finish = text.index(end) + len(end)
    replacement = f"{begin}\n\n{body}\n\n{end}"
    return text[:start] + replacement + text[finish:]


def update_note(path: Path, table: str, affinity: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_generated_block(text, BEGIN_MARKER, END_MARKER, table)
    text = replace_generated_block(
        text, BEGIN_AFFINITY_MARKER, END_AFFINITY_MARKER, affinity
    )
    path.write_text(text, encoding="utf-8")


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
    affinity = render_parallel_affinity_table(document, baselines, estimates)
    if args.update_note:
        update_note(args.update_note, comparison, affinity)
    print(render_document(document, baselines, estimates), end="")


if __name__ == "__main__":
    main()
