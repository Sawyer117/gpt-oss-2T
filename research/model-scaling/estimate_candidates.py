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
BEGIN_WIDE_MARKER = "<!-- BEGIN GENERATED 11-COLUMN TABLE -->"
END_WIDE_MARKER = "<!-- END GENERATED 11-COLUMN TABLE -->"


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


def comparison_rows(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> list[tuple[str, str, int, int, dict, str]]:
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
    return rows


def render_comparison_table(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> str:
    rows = comparison_rows(document, baselines, estimates)

    lines = [
        "> 公开模型使用上游仓库精确 metadata/模型卡；L/G/D/K 候选使用 GPT-OSS 形状参数账本。为避免宽表溢出，参数预算、架构形状与候选定位分表展示。",
        "",
        "**参数预算**",
        "",
        "| 类别 | ID / 模型 | 总参数 | 激活/token | 激活比 |",
        "|---|---|---:|---:|---:|",
    ]
    for kind, name, total, active, model, purpose in rows:
        lines.append(
            f"| {kind} | {name} | {human_parameters(total)} | "
            f"{human_parameters(active)} | {100 * active / total:.3f}% |"
        )
    lines.extend(
        [
            "",
            "**架构形状**",
            "",
            "> `L` = 层数，`d` = hidden size，`E / k` = 路由专家数 / Top-k。",
            "",
            "| ID | L | d | E / k | 上下文 | Attention |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, name, _, _, model, _ in rows:
        identifier = name.split(" / ", 1)[0]
        lines.append(
            f"| {identifier} | {model['num_hidden_layers']} | "
            f"{model['hidden_size']:,} | {model['num_routed_experts']} / "
            f"{model['experts_per_token']} | {human_context(model['context_length'])} | "
            f"{short_attention(model['attention'])} |"
        )
    lines.extend(
        [
            "",
            "**模型与候选定位**",
            "",
            "| ID | 主要轴 / 用途 |",
            "|---|---|",
        ]
    )
    for _, name, _, _, model, purpose in rows:
        identifier = name.split(" / ", 1)[0]
        lines.append(f"| {identifier} | {purpose} |")
    return "\n".join(lines)


def render_wide_comparison_table(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> str:
    rows = comparison_rows(document, baselines, estimates)
    lines = [
        "> 完整字段横向对照；窄屏设备可能需要横向滚动。数据与文首窄表由同一脚本生成。",
        "",
        "| 性质 | ID / 模型 | 总参数 | 激活/token | 激活/总量 | 层数 | Hidden | 专家 / Top-k | 上下文 | Attention | 主要轴 / 用途 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for kind, name, total, active, model, purpose in rows:
        lines.append(
            f"| {kind} | {name} | {human_parameters(total)} | "
            f"{human_parameters(active)} | {100 * active / total:.3f}% | "
            f"{model['num_hidden_layers']} | {model['hidden_size']:,} | "
            f"{model['num_routed_experts']} / {model['experts_per_token']} | "
            f"{human_context(model['context_length'])} | "
            f"{short_attention(model['attention'])} | {purpose} |"
        )
    return "\n".join(lines)


def parallel_degrees(model: dict) -> tuple[str, str, str]:
    tp_options = (8, 16, 32)
    ep_options = (8, 16, 32, 64)
    hidden = model["hidden_size"]
    intermediate = model["expert_intermediate_size"]
    heads = model["num_attention_heads"]
    kv_heads = model["num_key_value_heads"]
    experts = model["num_routed_experts"]

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

    def show(values: list[int]) -> str:
        return "/".join(str(value) for value in values) if values else "—"

    return show(strict_tp), show(replicated_kv_tp), show(ep)


def pp_layer_balance(layers: int, stages: int) -> tuple[float, str]:
    """Return the equal-cost-layer PP utilization proxy and best count split."""

    if layers <= 0 or stages <= 0 or layers < stages:
        raise ValueError("PP layer proxy requires layers >= stages > 0")
    base, extra = divmod(layers, stages)
    max_layers = base + bool(extra)
    utilization = layers / (stages * max_layers)
    if extra:
        allocation = f"{extra}×{base + 1} + {stages - extra}×{base}"
    else:
        allocation = f"{stages}×{base}"
    return utilization, allocation


def render_pp_layer_balance(layers: int, stages: int) -> str:
    utilization, allocation = pp_layer_balance(layers, stages)
    suffix = "等层" if layers % stages == 0 else "不等层"
    return f"{100 * utilization:.1f}%（{allocation}层，{suffix}）"


def affinity_rows(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> list[tuple[str, str, dict]]:
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
            "context_length": candidate.get(
                "context_length", document["defaults"]["context_length"]
            ),
            "attention": "GPT-OSS alternating sliding/full",
        }
        rows.append((candidate["id"], candidate["name"], model))
    return rows


def training_tp_cell(identifier: str, strict_tp: str) -> str:
    if identifier == "D0":
        return "自定义 CSA/HCA；V3 实证 TP1"
    if identifier == "K0":
        return f"自定义 KDA/MLA；维度 {strict_tp}"
    return strict_tp


def training_evidence(identifier: str) -> str:
    if identifier == "D0":
        return "V3：TP1 / EP64 / PP16；V4 延用并调整 DualPipe"
    if identifier in {"O0", "K0"}:
        return "公开训练拓扑未完整披露；本行仅做配置算术筛选"
    return "GPT-OSS 形状代理；尚无训练系统实测"


def render_training_affinity_table(rows: list[tuple[str, str, dict]]) -> str:
    lines = [
        "> 训练维度把 TP/EP 的整除性视为一级门槛；PP 不要求层数整除，改用“所有层等成本”假设下的 stage 利用率代理。",
        "",
        "| ID / 模型 | 训练 TP（严格） | 训练 EP（均匀专家） | PP8 层计数代理 | PP16 层计数代理 | 实证 / 边界 |",
        "|---|---|---:|---|---|---|",
    ]
    for identifier, name, model in rows:
        strict_tp, _, ep = parallel_degrees(model)
        lines.append(
            f"| {identifier} / {name} | {training_tp_cell(identifier, strict_tp)} | "
            f"{ep} | {render_pp_layer_balance(model['num_hidden_layers'], 8)} | "
            f"{render_pp_layer_balance(model['num_hidden_layers'], 16)} | "
            f"{training_evidence(identifier)} |"
        )
    lines.extend(
        [
            "",
            "- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)是直接反例：61 层实际采用 PP16、EP64、TP1；[DeepSeek 官方 DualPipe](https://github.com/deepseek-ai/DualPipe)要求 PP stage 数与 micro-batch 数为偶数，不要求层数整除。",
            "- [DeepSeek-V4 Technical Report](https://arxiv.org/html/2606.19348v1)给出的 V4-Pro 也是 61 层；其训练框架继承 V3，并为 mHC 增加的 stage 间通信调整了 DualPipe 1F1B。",
            "- PP 层计数代理为 `L / (P × ceil(L/P))`。它只回答同成本层的粗粒度均衡，不包含 embedding、head、MTP、不同 attention/MoE 层成本、激活显存或通信。",
            "- TP 候选集合为 8/16/32，严格列不允许复制 KV heads；EP 候选集合为 8/16/32/64。DP/ZeRO 通常不受模型层数整除限制。",
        ]
    )
    return "\n".join(lines)


def inference_attention_tp(identifier: str, strict_tp: str, relaxed_tp: str) -> str:
    if identifier == "D0":
        return "V3 实证 TP4+SP；V4 需 CSA/HCA 专用切分"
    if identifier == "K0":
        return f"维度 {strict_tp}；KDA/MLA 需专用 kernel"
    return f"严格 {strict_tp}；KV复制 {relaxed_tp}"


def context_parallel_cell(identifier: str, model: dict) -> str:
    if identifier == "D0":
        return "1M；CSA/HCA 专用 KV 与压缩边界"
    if identifier == "K0":
        return "1M；KDA/Gated MLA 需专用 kernel"
    degrees = [
        value for value in (8, 16, 32) if model["context_length"] % value == 0
    ]
    return f"{human_context(model['context_length'])}；CP/SP " + "/".join(
        str(value) for value in degrees
    )


def inference_evidence(identifier: str) -> str:
    if identifier == "D0":
        return "V3 官方线上以 TP/SP+DP attention、EP MoE 为主，未采用训练 PP 拓扑"
    if identifier == "K0":
        return "配置整除不等于 KDA/LatentMoE 的实际可用拓扑"
    if identifier == "O0":
        return "静态算术筛选；仍需 serving kernel 与通信实测"
    return "GPT-OSS 形状代理；PP 对吞吐可行但会增加 decode stage 延迟"


def inference_dp_cell(identifier: str) -> str:
    if identifier == "D0":
        return "V3：prefill DP8；decode DP80"
    return "无静态形状约束；取决于 batch / SLO"


def render_inference_affinity_table(rows: list[tuple[str, str, dict]]) -> str:
    lines = [
        "> 推理维度分开看 attention、MoE 与长上下文；允许 KV 复制和冗余专家时，静态整除条件会比训练更宽松。",
        "",
        "| ID / 模型 | Attention TP | MoE EP（均匀放置） | DP / 请求并行 | PP decode 层计数代理 | CP / SP 长上下文 | 实证 / 边界 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for identifier, name, model in rows:
        strict_tp, relaxed_tp, ep = parallel_degrees(model)
        pp8 = render_pp_layer_balance(model["num_hidden_layers"], 8)
        pp16 = render_pp_layer_balance(model["num_hidden_layers"], 16)
        lines.append(
            f"| {identifier} / {name} | "
            f"{inference_attention_tp(identifier, strict_tp, relaxed_tp)} | {ep} | "
            f"{inference_dp_cell(identifier)} | "
            f"PP8 {pp8}；PP16 {pp16} | {context_parallel_cell(identifier, model)} | "
            f"{inference_evidence(identifier)} |"
        )
    lines.extend(
        [
            "",
            "- [DeepSeek-V3 官方线上推理](https://arxiv.org/html/2412.19437)的 attention 使用 TP4+SP，并按 prefill/decode 分别组合 DP；MoE 使用大规模 EP。V4 推理框架大体继承 V3，但 CSA/HCA 改变了 KV 与 kernel 边界。",
            "- 推理 PP 同样允许不等长 stage；是否值得使用取决于最慢 stage、micro-batch/concurrency、跨 stage 激活通信和逐 token 延迟，而不是层数取模。",
            "- EP 列只表示无复制时专家可均匀放置。线上系统可以复制热点/共享专家，所以专家数整除也不是绝对可行性条件。",
            "- CP/SP 对 400K/1M 上下文很重要；表中对 GPT-OSS 形状只检查序列长度整除，真实可用性仍取决于 Sliding/Full attention kernel。",
        ]
    )
    return "\n".join(lines)


def render_parallel_affinity_table(
    document: dict, baselines: dict, estimates: list[tuple[dict, int, int]]
) -> str:
    rows = affinity_rows(document, baselines, estimates)
    return "\n\n".join(
        [
            "#### 训练维度：TP / EP / PP 亲和性\n\n"
            + render_training_affinity_table(rows),
            "#### 推理维度：并行亲和性\n\n"
            + render_inference_affinity_table(rows),
        ]
    )


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
        "## 训练与推理并行亲和性",
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
            "",
            "## 11 列完整宽表",
            "",
            render_wide_comparison_table(document, baselines, estimates),
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


def update_note(path: Path, table: str, affinity: str, wide_table: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_generated_block(text, BEGIN_MARKER, END_MARKER, table)
    text = replace_generated_block(
        text, BEGIN_AFFINITY_MARKER, END_AFFINITY_MARKER, affinity
    )
    text = replace_generated_block(
        text, BEGIN_WIDE_MARKER, END_WIDE_MARKER, wide_table
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
    wide_table = render_wide_comparison_table(document, baselines, estimates)
    if args.update_note:
        update_note(args.update_note, comparison, affinity, wide_table)
    print(render_document(document, baselines, estimates), end="")


if __name__ == "__main__":
    main()
