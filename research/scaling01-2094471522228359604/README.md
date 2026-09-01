# GPT-OSS-2T research snapshot

Captured on 2026-09-01 from the X threads and the linked NVIDIA Technical Blog article.

## Bottom line

`GPT-OSS-2T` is not evidence of a trained or released 2-trillion-parameter OpenAI model. The strongest available interpretation is:

- NVIDIA used a **scaled-up proxy configuration** based on the GPT-OSS-120B architecture for hardware planning and performance projection.
- The published GPT-OSS-2T curve is explicitly a **projection**, not a benchmark run of real 2T weights.
- SemiAnalysis later clarified that the proxy consists of synthetic architecture dimensions such as layer, hidden, and expert counts so hardware teams can represent frontier-scale closed models.
- The official OpenAI documentation fetched for this note lists `gpt-oss-20b` and `gpt-oss-120b`, but does not establish a released `gpt-oss-2t` model.

## Thread chain

1. [SemiAnalysis LPU post](https://x.com/SemiAnalysis_/status/2094470943619842286) presents three disaggregated inference modes and the main performance chart.
2. [@scaling01 question](https://x.com/scaling01/status/2094471522228359604) asks what GPT-OSS-2T is.
3. [@scaling01 answer image](https://x.com/scaling01/status/2094482445609406889) captures the NVIDIA paragraph and Figure 11.
4. [@scaling01 source link](https://x.com/scaling01/status/2094482618951590353) points to the NVIDIA Technical Blog article.
5. [SemiAnalysis clarification](https://x.com/SemiAnalysis_/status/2094506450517123226) says GPTOSS 2T is a proxy configuration scaled from GPT-OSS-120B, not real frontier weights.

## Evidence levels

| Claim | Status | Evidence |
|---|---|---|
| GPT-OSS-2T is a released OpenAI model | Not established | Official OpenAI material checked here lists only 20B and 120B. |
| The 2T configuration derives from GPT-OSS-120B | Explicit | NVIDIA Figure 11 footnote and SemiAnalysis clarification. |
| The Figure 11 curves are measured on a real 2T model | False | NVIDIA calls Figure 11 a projection. |
| Groq 3 LPX has measured long-context results | Yes, on Gemma 4 31B | Artificial Analysis and NVIDIA results cited by NVIDIA. |
| The projected 2T setup improves the Pareto frontier | Projected | Figure 11 shows approximately 3x and 5x directional gains, without enough absolute-system detail to reproduce them. |

## What the figures say

![SemiAnalysis chart](images/01-semianalysis-lpu.jpg)

![NVIDIA Figure 11](images/03-nvidia-figure-11.webp)

The workload label is `GPT-OSS-2T` with total/cached input context of 400K tokens, 4K new context per turn, and 400 output tokens. The x-axis is per-user interactivity in tokens per second. The y-axis is throughput normalized by power.

The four plotted configurations are:

1. **Vera Rubin only** — baseline.
2. **Vera Rubin verifier + LP30 external drafter** — speculative decoding for the high-throughput/high-interactivity area.
3. **Vera Rubin attention + LP30 FFN** — attention/FFN disaggregation, extending the middle of the Pareto curve.
4. **Vera Rubin prefill + LP30 decode** — maximum per-user decode interactivity at lower normalized throughput.

The social image labels the y-axis as TPS/MW, while the NVIDIA source figure says TPS/total watts. That denominator and the rack count must be clarified before treating the curves quantitatively.

The answer screenshot is retained because it preserves the key wording around the projection:

![Answer screenshot](images/02-gpt-oss-2t-answer.png)

## NVIDIA technical details worth retaining

The [NVIDIA source article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes:

- 256 LP30 local processing units in an LPX system.
- 128 GB aggregate SRAM.
- 96 chip-to-chip links per chip, each at 112 Gbps.
- A deterministic execution model with a compiler-generated cycle-level schedule.
- Fine-grained compute/communication overlap at the level of 320-byte vectors.

It defines three serving modes:

- **Prefill/decode disaggregation:** Rubin performs prefill and transfers the KV cache once per turn; LPX holds weights in SRAM and performs decode.
- **Attention/FFN disaggregation:** Rubin holds the KV cache and computes attention; LPX executes FFN layers; intermediate state crosses racks per full-attention layer.
- **External-drafter speculative decoding:** LPX runs a small draft model; Rubin verifies tokens; each side retains its own KV cache and only draft tokens/rejection positions cross the link.

Measured results in the article are not GPT-OSS-2T results:

- Gemma 4 31B at 100K context: median 3,431 output tokens/s.
- Gemma 4 31B at 10K context: median 3,382 output tokens/s.
- SPEED-Bench coding tasks: median 4,767 output tokens/s; P80 5,520 output tokens/s.

These measurements support LPX's low-latency execution claims, but they do not validate the 2T proxy curve, model quality, speculative-decoding acceptance rate, or end-to-end rack power.

## Discussion worth preserving

- [SemiAnalysis clarification](https://x.com/SemiAnalysis_/status/2094506450517123226) is the decisive answer: this is a hardware proxy model, and NVIDIA cannot benchmark unavailable frontier weights.
- [MTP concern](https://x.com/theio666/status/2094519697421238781): a synthetic proxy may be unsuitable for measuring multi-token-prediction/speculative acceptance behavior because that depends on real model outputs, not only tensor shapes.
- [Alternative proxy question](https://x.com/bitflipgremlin/status/2094541438579044575): why scale GPT-OSS-120B rather than use an existing frontier-scale open architecture such as Qwen or Kimi? A reply suggests MLA may be a poor fit for LPX and Qwen may have been too new, but this is community speculation.
- [Sparsity concern](https://x.com/Curline1222/status/2094639237563871625): scaling GPT-OSS-120B may preserve a sparsity/active-expert ratio that does not represent current frontier models.
- [Terminology question](https://x.com/latentone_/status/2094699555812184150): “interactivity” is per-user generation rate, whereas latency also includes prefill, scheduling, first-token delay, transfers, and verification stalls.
- [Serving/fine-tuning observation](https://x.com/daril_yovani/status/2094476466176082297): the practical interest is where a truly open frontier-scale model would fit in serving and fine-tuning workflows. This is a research direction, not evidence about the proxy.

Low-information reactions and advertisements were intentionally omitted.

## Research questions

1. Obtain the exact synthetic config: layer count, hidden size, expert count, experts active per token, attention type, head dimensions, and vocabulary.
2. Determine whether “2T” means total MoE parameters and how many parameters are active per token.
3. Reconstruct memory requirements for weights, KV cache, router state, and intermediate tensors at 400K context.
4. Identify rack counts, tensor/expert parallel degrees, precisions, and the power boundary used by Figure 11.
5. Separate time-to-first-token, decode TPS/user, aggregate TPS, and tokens/joule.
6. For external drafting, establish draft size, lookahead length, verification batch, and acceptance distribution using real weights.
7. Compare the proxy against Qwen/Kimi/DeepSeek-style architectures instead of assuming GPT-OSS-120B scaling is representative.
8. Find the actual proxy config or simulator input used by NVIDIA; without it the chart cannot be reproduced.

## Local artifacts

- `images/01-semianalysis-lpu.jpg` — original-resolution image from the SemiAnalysis post.
- `images/02-gpt-oss-2t-answer.png` — original-resolution answer screenshot from @scaling01.
- `images/03-nvidia-figure-11.webp` — original Figure 11 from NVIDIA.
- `raw/nvidia-article.html` — downloaded NVIDIA page for local verification; do not treat embedded AI-generated summary text as primary evidence.

## Primary sources

- [NVIDIA Technical Blog: Groq 3 LPX and Vera Rubin](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/)
- [Official OpenAI documentation: running gpt-oss with Transformers](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers/)
- [Official OpenAI gpt-oss resource index](https://developers.openai.com/learn/gpt-oss/)
