# GPT-OSS-2T 研究协作仓

本仓库研究 NVIDIA 图表中的 `GPT-OSS-2T` 代理模型，以及不同工业级 MoE 架构缩放到 2T 总参数附近时的合理性。

| 文件 | 说明 |
|---|---|
| [研究笔记.md](研究笔记.md) | 持续维护的中文主文：结论、证据、候选族、争议和实验计划 |
| [X 线索快照](research/scaling01-2094471522228359604/README.md) | 原帖、原图、NVIDIA 原文和讨论整理 |
| [模型基线](research/model-scaling/baselines.json) | 三个官方 Hugging Face 仓库的归一化架构参数 |
| [候选定义](research/model-scaling/candidates.json) | GPT-OSS 形状的多轴缩放候选 |
| [候选发布器](research/model-scaling/estimate_candidates.py) | 从逐张量账本生成候选表并同步主研究笔记 |
| [逐张量参数账本](research/model-scaling/parameter_accounting.py) | 整数级 tensor-schema 计数、breakdown 与离散候选搜索 |
| [参数账本测试](research/model-scaling/test_parameter_accounting.py) | 锁定基线、top-16、自由 top-N 与求解器结果 |
| [生成候选表](research/model-scaling/candidates.generated.md) | 由估算器生成，不手工编辑 |

## 维护约定

- 严格区分：**官方事实、测量结果、投影、工程估算、社区推测**。
- 数字优先记录来源 URL、仓库 revision 和抓取日期。
- 同栈比较优先；跨模型参数量不直接等价于质量或吞吐。
- 生成文件不手改，改 JSON 或脚本后重新生成。
- 浏览器会话、Cookie、HAR、cURL 和认证状态不得进入 Git。

更新候选参数后，用同一条命令刷新独立生成表和主研究笔记中的总览区块：

```bash
python3 research/model-scaling/estimate_candidates.py \
  --update-note 研究笔记.md \
  > research/model-scaling/candidates.generated.md
```

查看某个候选的逐张量整数账本：

```bash
python3 research/model-scaling/parameter_accounting.py count \
  --layers 54 --hidden 5632 --intermediate 6656 \
  --experts 328 --top-k 16 --heads 128 --kv-heads 16
```

重新求解固定 top-16 和自由 top-N 候选：

```bash
python3 research/model-scaling/parameter_accounting.py solve \
  --fixed-top-k 16 --layer-step 4 \
  --dimension-step 512 --expert-multiple 32
python3 research/model-scaling/parameter_accounting.py solve \
  --exclude-top-k 16 --layer-step 4 \
  --dimension-step 512 --expert-multiple 32
```

把 TP/EP/PP 亲和性设为硬约束（例如 D2）而非手工筛选：

```bash
python3 research/model-scaling/parameter_accounting.py solve \
  --target-active 61294450936 --top-k-min 4 --top-k-max 16 \
  --exclude-top-k 6 --layer-step 4 \
  --dimension-step 512 --expert-multiple 32 \
  --require-tp 16 --require-ep 64 --require-pp 8
```

GPT 原生 top-4 候选使用更宽、较少专家的搜索边界：

```bash
python3 research/model-scaling/parameter_accounting.py solve \
  --target-active 87306972767 --fixed-top-k 4 \
  --layer-min 36 --layer-max 64 \
  --hidden-max 12288 --intermediate-max 12288 \
  --min-experts 64
```

运行参数账本与求解器回归测试：

```bash
python3 research/model-scaling/test_parameter_accounting.py
```
