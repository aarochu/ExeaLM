# Training

## Pretraining

| Setting | Starting value | Why |
|---|---|---|
| Model | `100m` preset, 101M parameters | |
| Tokens | about 100B | 1,000 tokens per parameter, far past "compute-optimal" on purpose ([DATA.md](../DATA.md)) |
| Context | 2,048 tokens, extended to 8,192 near the end | |
| Batch | about 1M tokens | |
| Learning rate | warmup, stable, then a linear decay at the end (WSD), where DATA.md's annealing data goes in | during the stable phase we can branch off and try a different final data mix without restarting |
| Optimizer | AdamW (β 0.9/0.95, weight decay 0.1), gradient clip 1.0 | Muon is worth trying ([ideas.md](ideas.md)) |
| Precision | bf16 compute; scans and FFTs in float32 | |
| Architecture search | during a pilot run (about the first 5B tokens), then frozen | a search round runs one forward pass per (layer, circuit type) on each held-out batch |
| Retrieval index | re-encoded every few thousand steps | the model's own encoder keeps changing |
| Swarms | 4 agents per board, many swarms per batch, half split / half random | see [swarms.md](swarms.md) |
| Final decay phase | DATA.md's annealing mix (the last 5%) | as in SmolLM2 and OLMo 2 |

**Compute.** 6 × 101M parameters × 100B tokens ≈ 6 × 10¹⁹ FLOPs. At about 300 TFLOPs sustained per MI300X that is
about 56 GPU-hours. TONN is about 2.7 times slower per step today, and retrieval adds roughly a quarter, so plan for
about 200 GPU-hours until the kernels get faster. The difference is why fast kernels matter. [DATA.md](../DATA.md) has
the full reasoning.

What exists: `exealm/train/pretrain.py` runs this loop on one device, with the search, retrieval, swarms, WSD,
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
4. **Swarm SFT and swarm RL:** swarms that share a board and solve tasks together ([swarms.md](swarms.md)). Not started,
   and the part most unique to ExeaLM.

A 0.1B model won't be a strong general assistant. It can be a good small one: fast, cheap, bilingual, and able to use
retrieval and other agents to make up for its size. That's the bet.

## Evaluation during training

Validation bits per token and per byte (`--val`). See `exealm/eval/__init__.py` for the benchmark plan.
