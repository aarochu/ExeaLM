# Data

The planned data for ExeaLM: how many tokens we train on and why, the pretraining mix, and the post-training sets. This is a proposal. We will tune the mix with small-scale ablations before the main run.

## How many tokens, and why

We train on about **100 billion tokens**, for a model of 0.1 billion (100 million) parameters.

**Why 100 billion.** Extra tokens help less and less. Using the loss fit from [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556), the predicted loss drops by about 0.19 going from 20B to 100B tokens, by about 0.06 from 100B to 200B, and by about 0.05 from 200B to 400B. Past roughly 200B tokens, each doubling buys little, and the effort is better spent on the data mix and on evals. The fit comes from standard transformers, so we treat these numbers as a guide and check the curve on TONN with small runs. Some small models, such as SmolLM2-135M, were trained on 2T tokens, so this is a first-run choice, not a hard limit.

100B tokens on 0.1B parameters is 1,000 tokens per parameter, about 50 times the compute-optimal ratio of about 20. Small models are trained past that ratio on purpose, because they are cheaper to run afterward.

**How long it takes.** Training compute is about 6 x parameters x tokens ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)). At about 300 TFLOPs sustained per MI300X:

- 100M parameters on 100B tokens: about 56 GPU-hours, which is about 7 hours on 8 GPUs (about 2.3 days on one).
- For comparison, 2T tokens would take about 1,100 GPU-hours.

Plan for 30% to 100% more wall-clock time for restarts, evals, and data loading. TONN's real throughput is unmeasured, so we will update these numbers after the first small runs.

We select the 100B tokens from roughly 40T available, so nothing needs to be repeated.

## Pretraining mix (100B tokens)

This is a candidate mix. The team's mix proposal v0 (`exealm/data/sources.py`, [docs/data.md](docs/data.md)) is Japanese-heavier (about 21% Japanese) and was written for a 2T-token pool. Scale it to 100B and compare the two with small ablations before choosing.

| Bucket | Tokens | Share | Datasets |
|---|---|---|---|
| General web, English | 47.5B | 47.5% | [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [K2 Horizon TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) web subsets, English part of the [NII LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4) |
| Reference, academic, books, PDFs | 10B | 10% | [TxT360](https://huggingface.co/datasets/LLM360/TxT360) non-web subsets, [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu), [peS2o](https://huggingface.co/datasets/allenai/peS2o), [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) |
| Code | 15B | 15% | [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu), [The Stack v2 dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup), [K2 Horizon Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning) (no-thinking subsets) |
| Math and science | 10B | 10% | [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [MegaMath](https://huggingface.co/datasets/IFM/MegaMath), [K2 Horizon Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning) (rewrite and dialogue subsets) |
| Reasoning and synthetic | 5B | 5% | [K2 Horizon Pretrain-Behaviors](https://huggingface.co/datasets/IFM/Pretrain-Behaviors), [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) |
| Multilingual | 5B | 5% | Japanese, Chinese, and Korean from the NII LLM-jp Corpus v4, [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| QA and instruction-style text | 2.5B | 2.5% | [K2 Horizon TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) txt360-qa subset, [K2 Horizon SFT-Reasoning](https://huggingface.co/datasets/IFM/SFT-Reasoning) 3efforts-pretrain subset |
| Annealing (last 5% of training) | 5B | 5% | [Dolma 3 Dolmino mix](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125), FineMath-4+, top-scored FineWeb-Edu, K2 Horizon math and code reasoning |
| **Total** | **100B** | **100%** | |

The shares are a first guess. Before the main run we train a small model (about 100M parameters on 20B tokens) on three or four versions of this mix and keep the one that scores best.

Rules: dedupe across every source, decontaminate against every benchmark we report, and repeat no source more than about 4 times (at 100B tokens we should not need to repeat any).

## Post-training

Three stages after pretraining: SFT, preference tuning, then RL with verifiable rewards. A small base model may gain little from long reasoning traces, so we check this at small scale first.

**Stage 1, SFT** (about 1 to 2 billion tokens)

| Bucket | Share | Datasets |
|---|---|---|
| General chat and instruction following | 35% | [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT), [K2 Horizon SFT-Reasoning](https://huggingface.co/datasets/IFM/SFT-Reasoning) |
| Math reasoning | 20% | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), [K2 Horizon Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning) |
| Code | 20% | [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning), OpenThoughts3-1.2M, [K2 Horizon Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning) |
| Science and general reasoning | 10% | OpenThoughts3-1.2M, K2 Horizon SFT-Reasoning |
| Safety and refusals | 5% | Safety subsets inside the Tulu 3 SFT mixture |
| Real user conversations | 5% | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), filtered for toxicity and personal data |
| Tool use and agentic | 5% | Open gap: no verified open set yet |

**Stage 2, preference tuning:** [Tulu 3 preference mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture), [Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO), [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

**Stage 3, RL with verifiable rewards:** [K2 Horizon guru-RL](https://huggingface.co/datasets/IFM/guru-RL-v1.5), [RLVR GSM, MATH, and instruction-following constraints](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints).

## Licenses and caveats

- Most sets are ODC-By, CC-BY, MIT, or Apache 2.0 and need attribution in our data card. Wikipedia is CC-BY-SA. The Stack v2 has per-file code licenses. SmolTalk includes Magpie data generated with Llama, so the Llama license applies to that part. Many Tulu 3 subsets carry their own source terms.
- The base model trains from scratch with no pretrained weights. About half of K2 Horizon's mix is synthetic text from other models, and sets like OpenThoughts3 and OpenMathReasoning are made of other models' outputs.
- We left out NVIDIA's Nemotron-CC-v2 and Nemotron-CC-Math, which are gated under an "other" license.
- License fields were read from each dataset page on November 10, 2026. Confirm them before training.
