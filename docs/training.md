# Training

## Pretraining

| Setting | Starting value | Why |
|---|---|---|
| Model | `100m` preset, 101M parameters | |
| Tokens | about 100B | about 1,000 tokens per parameter, far past "compute-optimal" on purpose. Past roughly 200B, extra data buys little ([DATA.md](../DATA.md)) |
| Context | 2,048 tokens, extended to 8,192 near the end | |
| Batch | about 1M tokens | |
| Learning rate | warmup, stable, then a linear decay over the last 20% (WSD) | during the stable phase we can branch off and try a different final data mix without restarting |
| Optimizer | AdamW (β 0.9/0.95, weight decay 0.1), gradient clip 1.0 | Muon is worth trying ([ideas.md](ideas.md)) |
| Precision | bf16 compute; scans and FFTs in float32 | |
| Architecture search | during a pilot run (about the first 20B tokens), then frozen | a search round runs one forward pass per (layer, circuit type) on each held-out batch |
| Retrieval index | re-encoded every few thousand steps | the model's own encoder keeps changing |
| Agent teams | 4 rows per board | see [memory.md](memory.md) |
| Final decay phase | the highest-quality data (K2 Horizon reasoning, math, code, Wikipedia) | as in SmolLM2 and OLMo 2 |

**Compute.** 6 × 101M parameters × 100B tokens ≈ 6 × 10¹⁹ FLOPs. One H100 does about 10¹⁵ bf16 FLOPs per second at
best. At 40% utilization that's about 42 H100-hours; at 15% (closer to TONN today) about 110, plus the cost of
retrieval. The difference is why fast kernels matter.

What exists: `exealm/train/pretrain.py` runs this loop on one device, with the search, retrieval, teams, WSD,
evaluation and checkpoints.

What's missing:
- multi-GPU (DDP, then FSDP)
- resuming from a checkpoint
- a streaming data loader for the full mix with per-source weights
- context extension
- `torch.compile`
- logging to Weights & Biases

## Post-training

1. **SFT** (`exealm/train/sft.py`): conversations in the chat format, loss on the assistant's tokens only. The loss and
   formatting are written; the data loader and loop are not.
2. **DPO** (`exealm/train/dpo.py`): preferred versus rejected answers. The loss is written; data and loop are not.
3. **GRPO** (`exealm/train/grpo.py`): RL with verifiable rewards (math answers, code tests, format). The loss and group
   advantages are written and tested; the sampling loop and rewards are not.
4. **Multi-agent RL:** teams that share an `AgentBoard` and are rewarded together, for example two agents that each see
   half of a puzzle. Not started, and the part most unique to ExeaLM.

A 0.1B model won't be a strong general assistant. It can be a good small one: fast, cheap, bilingual, and able to use
retrieval and other agents to make up for its size. That's the bet.

## Evaluation during training

Validation bits per token and per byte (`--val`). See `exealm/eval/__init__.py` for the benchmark plan.
