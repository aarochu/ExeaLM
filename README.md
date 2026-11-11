# ExeaLM

An open language model built by high schoolers, run by [Exea Labs](https://exealabs.org). Code and weights are public under Apache 2.0.

ExeaLM is a 0.1B-parameter (about 100 million) language model, pretrained from scratch on about 100 billion tokens and then post-trained into an assistant. It runs on TONN, with its own RAG and agent-to-agent communication built into the architecture.

No weights are released yet. We are at the start: the model code runs and is tested, and the data pipeline, tokenizer, and first architecture experiments are open tasks. The [to-do list](#to-do-list) below has places to start.

## Join

Fill out the sign-up form so we can list you as a contributor: https://docs.google.com/forms/d/e/1FAIpQLSdPHVZhuXxBDUf-cJM2zDbj1JZPsduu95Y_kTJ_g4gKAXdDVw/viewform

The repo is public, so you can open an issue or pull request without signing up. The form gives us a record of who worked on the project. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to start.

## Architecture

ExeaLM uses TONN, a sequence model that builds its own architecture from data. It starts from a large random network of small circuits (delays, running sums, products, attention) and keeps a circuit only if switching it off raises the error on data the network has not trained on.

Paper: [TONN: Networks That Build Their Own Architecture from Data](https://www.preprints.org/manuscript/202611.0994) (Avneh Singh Bhatia, Preprints.org, November 2026). It is a concept paper and has not been peer reviewed. Its larger-scale results are mixed, and we test TONN against a plain transformer at equal compute on small models before any large run.

### What's new in ExeaLM

Two kinds of circuits are built into every layer and trained end to end from the first step of pretraining:

- **Its own RAG (RG, read-only).** Once every 64 tokens, the model turns what it has read into a search query, looks it up in a read-only datastore of training text using embeddings it learns itself, and reads the best matches with attention. It never retrieves the document it is reading, so it cannot look up the answer.
- **Agent-to-agent communication (AG, read and write).** Copies of the model ("agents") share a message board. Once every 64 tokens each agent posts a message, and at every token it reads what the other agents posted earlier. It can also stay quiet or read nothing.

Every layer starts with one RG and one AG circuit, and TONN's search can never remove the last one. It is free to add more. Details: [docs/memory.md](docs/memory.md).

To our knowledge, ExeaLM is the first language model with agent-to-agent communication built into its architecture. We are building it now, and until an experiment shows agents learning to say something useful, it is a design, not a result.

### Swarms

A **swarm** is a few agents (4 by default) sharing one board. Training runs many swarms at once: each batch is split into swarms, each with a private board.

- **Pretraining: swarms pull data together.** Each agent reads ordinary training text, so swarms cost no extra data. The swarm is chosen so that what one agent reads helps the others: parts of one long document, related documents, or the same topic in Japanese and English. Some swarms are random, so agents also learn when to ignore the board. Each agent also searches the datastore on its own and can share what it finds.
- **Post-training: swarms solve tasks together.** Each agent gets part of the information and one must answer. Because a swarm runs in one forward pass, the answer's loss trains every agent's messages directly. After that, RL rewards the whole swarm for solving split-information questions, divide-and-conquer math, and write-and-test code.

Design, rules and open questions: [docs/swarms.md](docs/swarms.md).

## Start here

```bash
git clone https://github.com/exea-labs/ExeaLM && cd ExeaLM
pip install -e ".[dev,data]"
pytest                                                             # 17 tests, a few seconds on a laptop CPU
python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --datastore_chunks 2000   # tiny TONN, ~10 s
python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --model llama --out runs/llama   # baseline
```

For real data, [docs/data.md](docs/data.md) walks from a downloaded sample to training the 17M `small` preset, which fits on a laptop or a free Colab GPU.

Code map: `exealm/model/` (TONN, RG, AG, search, transformer baseline), `exealm/memory/` (datastore, board), `exealm/data/` (download, filters, tokenizer, packing, swarm batches), `exealm/train/` (pretraining, SFT, DPO, GRPO), `docs/` (architecture, memory, swarms, training, ideas).

## Data

About 100 billion tokens, selected from open corpora: the [NII LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4), the data released with [K2 Horizon](https://ifm.ai/blog/k2), FineWeb-Edu, DCLM-Baseline, and others. That is about 1,000 tokens per parameter for our 0.1B model. The count is set by diminishing returns: past about 200 billion tokens, extra data buys little. [DATA.md](DATA.md) has the full mix, every dataset link, and the math behind the number. Every source, license, and filter gets documented.

## Training

One central run on cloud compute credits arranged by Exea: about 200 MI300X GPU-hours at TONN's current speed ([DATA.md](DATA.md)). Contributors work on data pipelines, architecture experiments, evals, docs, and review. Weights will be released at https://huggingface.co/exea-labs/ExeaLM.

## Roadmap

1. Data pipeline: download, dedupe, filter, and mix
2. Tokenizer
3. Small-scale architecture experiments, including RG, AG and swarms
4. Evaluation harness
5. Full training run
6. Post-training (including swarm training) and release

Each phase gets an owner. Open tasks are in the [to-do list](#to-do-list) below. We need people on data curation, architecture research, training infrastructure, evals, docs, and community.

## To-do list

Starting points. Almost everything here runs on a laptop or a free Google Colab GPU, using the `tiny` preset or the 17M-parameter `small` preset. Items marked **(starter)** are good first tasks. Items marked **(GPU)** need bigger GPUs: maintainers run those on Exea's compute, and you can still write and test the code on a small model. To claim an item, say so in #exealm or open a draft pull request. Anything that changes the model or the training has to win a fair small-scale comparison first ([CONTRIBUTING.md](CONTRIBUTING.md)).

### Swarms and agents (`exealm/model/board.py`, `exealm/data/loader.py`, [docs/swarms.md](docs/swarms.md))
- [ ] First experiment: does a split swarm predict its text better than solo agents reading the same text? (`small` preset)
- [ ] (starter) Measure board use during training: AG worth in the search log, how often agents read nothing, post strength
- [ ] Related swarms: group documents that are near each other in the model's own retrieval space
- [ ] Cross-lingual swarms: pair Japanese and English Wikipedia articles on the same topic (interlanguage links)
- [ ] Swarm sizes that vary within a run (2 to 8)
- [ ] Agent identity: a role or id signal on posts so agents can specialize
- [ ] Retrieval division of labor: do agents in a swarm pull different chunks? Try rewarding it
- [ ] Swarm-aware multi-GPU training, keeping each swarm on one GPU (testable on a CPU with two processes)
- [ ] Swarm SFT data: split-information tasks built from multi-hop QA. This also fills the "tool use and agentic" gap in [DATA.md](DATA.md)
- [ ] Swarm SFT loop: loss on the answering agent, trained through the board (test on the `tiny` preset)
- [ ] Swarm RL (GRPO): whole-swarm attempts with a shared reward (test on the `tiny` preset)
- [ ] Swarm task suite: split-information QA, divide-and-conquer math, write-and-test code, propose-and-check, cross-lingual QA
- [ ] Swarm generation: all agents generate in lockstep in one batch
- [ ] Board server so agents on different machines share one board (HTTP or Redis)
- [ ] Message decoder: turn board messages into text people can read
- [ ] Safety: agents treat other agents' messages as information, not instructions (data and an eval)
- [ ] Swarm eval: the same tasks solo vs swarms of 2, 4 and 8

### Retrieval (`exealm/model/retrieval.py`, `exealm/memory/datastore.py`)
- [ ] FAISS (IVF-PQ) index for the datastore (FAISS runs on a CPU, so start there)
- [ ] Choose what goes in the datastore: a curated few billion tokens of reference text
- [ ] Re-indexing schedule: how stale can the stored keys get? (`small` preset)
- [ ] Cheaper retrieval: cache encodings, share retrievals across layers, tune top-k and chunk length
- [ ] Key size: 32 vs 64 vs 128 numbers (`small` preset)
- [ ] (starter) Script that prints what RG retrieved for a prompt
- [ ] Swap the datastore at inference: can the model use new facts without retraining?

### Architecture (`exealm/model/`)
- [ ] TONN vs Llama with the `small` preset on real tokens, two seeds each (a free Colab GPU is enough)
- [ ] Ablations at `small` size: with and without RG, AG and swarms
- [ ] New circuit types, for example a gated delta-rule memory or sliding-window attention
- [ ] Shape sweep at fixed parameters: depth, width, slots per layer, channels per slot
- [ ] Search schedule: how long to search, how much to replace, how long new circuits are protected
- [ ] Which circuit types survive the search, and does that change from `tiny` to `small`?
- [ ] (starter) Plot the architecture the search chose, layer by layer, over training
- [ ] (GPU) Confirm the winners at the full 0.1B size

### Speed (`exealm/model/scan.py`)
- [ ] Fused Triton kernels for `decay_scan` and `gated_scan` (any NVIDIA GPU works, including a free Colab T4; must also run on AMD MI300X)
- [ ] One scan call for all running-sum circuits in a layer
- [ ] Recurrent state cache, so generating a token doesn't re-read the whole context
- [ ] `torch.compile` support
- [ ] (starter) Benchmark script: training and generation tokens per second, TONN vs Llama
- [ ] (GPU) Run the tests and a short training run on an MI300X and report what breaks

### Data ([DATA.md](DATA.md), `exealm/data/`)
- [ ] Download and stage every DATA.md source: parallel, resumable, checksummed (build and test it on samples)
- [ ] (starter) Count tokens in each K2 Horizon subset (streams, no big download)
- [ ] (starter) Write a formatter for each K2 Horizon subset, since their fields differ
- [ ] Remove near-duplicates across all sources (MinHash LSH, runs on a CPU)
- [ ] Decontaminate against every benchmark we report
- [ ] Filter personal data and toxicity
- [ ] Quality classifier for Japanese web text (small classifier, Colab-sized)
- [ ] Mix ablations: three or four versions of the mix with the `small` preset; the best two get confirmed at 0.1B **(GPU)**
- [ ] Streaming loader with per-source weights
- [ ] (starter) Data card: every source, license, filter and token count

### Tokenizer (`exealm/data/tokenizer.py`)
- [ ] Train 32k, 49k and 64k candidates; compare bytes per token on English, Japanese and code (CPU)
- [ ] Pick by `small`-preset loss and publish on Hugging Face
- [ ] Final special tokens: chat roles, agent roles, board

### Training (`exealm/train/pretrain.py`)
- [ ] Multi-GPU: DDP, then FSDP (DDP is testable on a CPU with two processes)
- [ ] Exact resume from checkpoints
- [ ] Weights & Biases logging
- [ ] Context extension near the end of training
- [ ] Optimizer: Muon vs AdamW (`small` preset)
- [ ] Multi-token prediction heads (`small` preset)
- [ ] Learning-rate transfer from small models (μP): tune on `tiny`/`small`, carry over to 0.1B
- [ ] (starter) Budget sheet: GPU-hours per run, checkpoint cadence, what we can afford

### Post-training (`exealm/train/`)
- [ ] SFT loop and data loader (pack conversations so they can't see each other; test on `tiny`)
- [ ] DPO loop
- [ ] GRPO loop with math and code rewards
- [ ] Chat template
- [ ] Do long reasoning traces help a model this small? Test on an existing small open model such as SmolLM2-135M (Colab)

### Evals (`exealm/eval/`)
- [ ] (starter) lm-evaluation-harness wrapper (test it on a `tiny` checkpoint)
- [ ] Japanese evals (llm-jp-eval)
- [ ] Retrieval eval: questions answered in the datastore but not in the prompt
- [ ] Run evals automatically on every checkpoint
- [ ] Results table in this README

### Release, docs and community
- [ ] Hugging Face model card and a loader, so the model loads with one line
- [ ] Demo: chat with one agent or a swarm (build it on a `tiny` checkpoint now)
- [ ] (starter) Diagrams: a TONN layer, RG, AG, a swarm
- [ ] (starter) Japanese translation of this README
- [ ] (starter) Glossary of terms
- [ ] (starter) Beginner notebook: train the `tiny` model and look inside it
- [ ] Technical report

## Community

Read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
