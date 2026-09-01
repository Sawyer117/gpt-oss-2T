# GPT-OSS-2T 研究协作仓

本仓库研究 NVIDIA 图表中的 `GPT-OSS-2T` 代理模型，以及不同工业级 MoE 架构缩放到 2T 总参数附近时的合理性。

| 文件 | 说明 |
|---|---|
| [研究笔记.md](研究笔记.md) | 持续维护的中文主文：结论、证据、候选族、争议和实验计划 |
| [X 线索快照](research/scaling01-2094471522228359604/README.md) | 原帖、原图、NVIDIA 原文和讨论整理 |
| [模型基线](research/model-scaling/baselines.json) | 三个官方 Hugging Face 仓库的归一化架构参数 |
| [候选定义](research/model-scaling/candidates.json) | GPT-OSS 形状的多轴缩放候选 |
| [候选估算器](research/model-scaling/estimate_candidates.py) | 总参数与激活参数的可复算近似公式 |
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
