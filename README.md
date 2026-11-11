# ExeaLM

An open language model built by high schoolers, run by [Exea Labs](https://exealabs.org). Code and weights are public under Apache 2.0.

**The idea:** ExeaLM is a 0.1B-parameter (about 100 million) language model, pretrained from scratch on about 2 trillion tokens and then post-trained into an assistant, like any other LLM. What's different is the architecture, **TONN**, which builds two new abilities into every layer: **its own RAG**, where the model learns its own embeddings, decides what to search for, and pulls text from a read-only datastore, and **agent-to-agent communication**, where copies of the model post messages to a shared board and read each other's. Both are circuits inside the network, trained end to end like everything else, not tools bolted on afterwards.

**Status:** skeleton. No weights are released yet. The model code (TONN, the retrieval circuit, the agent-board circuit, the architecture search) runs and is tested. The data scripts (download, filter, tokenizer, packing) work on small samples. Training at scale, post-training and evals are partly written: those are open tasks.

## Join

Fill out the sign-up form so we can list you as a contributor: https://docs.google.com/forms/d/e/1FAIpQLSdPHVZhuXxBDUf-cJM2zDbj1JZPsduu95Y_kTJ_g4gKAXdDVw/viewform

The repo is public, so you can open an issue or pull request without signing up. The form gives us a record of who worked on the project. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to start. Pick something from [Where we need help](#where-we-need-help) or the [issues](https://github.com/exea-labs/ExeaLM/issues).

## Start in five minutes

```bash
git clone https://github.com/exea-labs/ExeaLM && cd ExeaLM
pip install -e ".[dev,data]"
pytest                                                           # 16 tests, a few seconds, CPU is fine

# train a tiny model on made-up data (no download, about 10 seconds on a laptop CPU)
python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --datastore_chunks 2000

# the fair comparison: same run with the plain transformer baseline
python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --model llama --out runs/llama

# real data: one file of Japanese Wikipedia from the LLM-jp corpus (~320 MB)
python -m exealm.data.download llmjp --subset ja/ja_wiki --max_files 1 --out data/raw
python -m exealm.data.sources                                    # print the planned 2T-token mix with licenses
```

Full data walkthrough (filter, tokenizer, pack, train): [docs/data.md](docs/data.md).

## The plan

### 1. Architecture: TONN

TONN is a sequence model that builds its own architecture from data. A normal transformer layer is attention plus an MLP. A TONN layer instead holds 20 small **circuits**, each one of 14 types:
- memories: delays, leaky running sums, gated memories, a Mamba-like selective state space, an oscillating memory
- local and global mixing: short convolutions, rotary attention
- the two new memory circuits, described in sections 2 and 3

After each TONN layer comes a gated MLP, as in a transformer.

Nobody picks which circuit goes where. While the model trains, TONN switches each kind of circuit off on text it has not trained on and keeps it only if the error goes up. Useless circuits are replaced by new ones, drawn more often from the kinds that have proven useful. Early in training the architecture changes; later it is frozen.

Paper: [TONN: Networks That Build Their Own Architecture from Data](https://www.preprints.org/manuscript/202611.0994) (Avneh Singh Bhatia, Preprints.org, November 2026). It is a concept paper and has not been peer reviewed. Its larger-scale results are mixed, so we test TONN against a plain transformer at equal compute on small models before any large run.

The language-model evidence so far comes from small tests only: 4.85M parameters, byte-level English web text ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), and the same 25 MB of training text for every model.

| Model | Bits per byte (lower is better) |
|---|---|
| Llama-style transformer, same size | 1.813 |
| TONN, seed 0 | **1.738** |
| TONN, seed 1 | 1.766 |
| Transformer, 3× more data (82 MB) | 1.690 |
| TONN, 3× more data (82 MB) | **1.622** |

Caveats:
- TONN currently trains about 2.7 times slower per step, because its kernels aren't optimized yet.
- The seed-to-seed spread is real.
- 4.85M parameters is a long way from 0.1B.

The project's first job is to check that the lead holds at larger scale. Details: [docs/architecture.md](docs/architecture.md).

### 2. Its own RAG (the RG circuit, read-only)

Every layer has at least one **RG** circuit. Once per 64 tokens it:
1. turns what it has read so far into a **query** (each RG circuit learns its own);
2. searches a **read-only datastore** of training-text chunks, using **embeddings made by the model's own encoder** (no outside embedding model);
3. weighs the top results by how well they match;
4. lets each position read the retrieved text with attention.

All of it is learned from the next-token loss. The datastore never returns chunks from the document being read, so the model can't cheat by looking up the answer.

### 3. Agents that talk to each other (the AG circuit, read and write)

Every layer has at least one **AG** circuit, which connects to the **agent board**. Once per 64 tokens, each copy of the model posts a message: what it is about, what it says, and how strongly it wants it read, so it can stay quiet. At every position it reads its teammates' earlier messages with attention, and it can choose to read nothing. The datastore is read-only; the board is read *and* write. During pretraining, the rows of a batch form teams that share a board. At inference, many running agents share a persistent board.

**Guaranteed, then chosen:** every layer starts with one RG and one AG circuit, and the search can never remove the last one, because both need training before they become useful. The search is free to add more wherever they earn it. Design and open questions: [docs/memory.md](docs/memory.md).

### 4. Data: about 2 trillion tokens

Selected from two open corpora:
- **[NII LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4):** 19.5T tokens in total, including 688B Japanese and 17.8T English, plus code, Chinese and Korean. It downloads without an account.
- **The data released with [K2 Horizon](https://ifm.ai/blog/k2)** (Institute of Foundation Models, [collection](https://huggingface.co/collections/IFM/k2-horizon)): [TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) (web and QA), [Pretrain-Behaviors](https://huggingface.co/datasets/IFM/Pretrain-Behaviors), [Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning), [Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning), [SFT-Reasoning](https://huggingface.co/datasets/IFM/SFT-Reasoning).

Every subset, its link and its license are in [`exealm/data/sources.py`](exealm/data/sources.py), with **mix proposal v0**: 1.98T tokens of Japanese, English, math, code and K2 Horizon data. The mix is a starting point to argue with, not a decision. Training a 0.1B model on 2T tokens is far past the "compute-optimal" amount, on purpose: small models keep improving with more data, and [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) was trained on 2T tokens too.

### 5. Training

- **Pretraining:**
  - next-token prediction with a warmup-stable-decay learning rate;
  - the architecture search runs during a pilot run and is then frozen;
  - the retrieval index is refreshed as the model's own encoder learns;
  - the best data is saved for the final decay phase.

  Compute: about 1.2 × 10²¹ FLOPs, or roughly 850 to 2,500 H100-hours depending on how fast we make TONN. [docs/training.md](docs/training.md)
- **Post-training,** in this order:
  1. supervised fine-tuning on conversations ([`sft.py`](exealm/train/sft.py));
  2. preference tuning ([`dpo.py`](exealm/train/dpo.py));
  3. reinforcement learning with **GRPO** ([`grpo.py`](exealm/train/grpo.py), the method behind DeepSeek-R1);
  4. **multi-agent RL**: teams of agents share a board and are rewarded together. This is where the AG circuits really learn to talk.
- **Compute:** one central run on cloud compute arranged by Exea. You don't need a GPU to contribute. Weights will be released at https://huggingface.co/exea-labs/ExeaLM.

## Where we need help

| Area | What | Where |
|---|---|---|
| Data | Count tokens per K2 Horizon subset; improve the mix; near-duplicate removal; a quality classifier for Japanese and English; benchmark decontamination | [`exealm/data/`](exealm/data), [docs/data.md](docs/data.md) |
| Tokenizer | Pick the vocabulary size (32k/49k/64k) by experiment; handle Japanese well | [`tokenizer.py`](exealm/data/tokenizer.py) |
| Architecture | New circuit types; better RG and AG designs; fair TONN vs Llama comparisons at bigger sizes | [`exealm/model/`](exealm/model), [docs/architecture.md](docs/architecture.md) |
| Retrieval | A FAISS index for billions of tokens; what goes in the datastore; cheaper re-indexing | [`datastore.py`](exealm/memory/datastore.py), [docs/memory.md](docs/memory.md) |
| Agents | How to pick teammates in pretraining; multi-agent tasks and rewards; turning board messages into readable text | [`board.py`](exealm/model/board.py), [docs/memory.md](docs/memory.md) |
| Speed | Triton/CUDA kernels for TONN's scans; fast generation with a recurrent state cache; multi-GPU training | [`scan.py`](exealm/model/scan.py), [`pretrain.py`](exealm/train/pretrain.py) |
| Training tricks | Muon optimizer, multi-token prediction (DeepSeek-V3), μP, FP8, distillation from a larger open model | [docs/ideas.md](docs/ideas.md) |
| Post-training | SFT and DPO loops, the GRPO loop and rewards, Japanese instruction data | [`exealm/train/`](exealm/train) |
| Evals | lm-evaluation-harness wrapper; Japanese evals (llm-jp-eval); retrieval and agent evals | [`exealm/eval/`](exealm/eval) |
| Docs | Clearer explanations, diagrams, translations | `docs/` |

The rule for architecture and training changes: **a change has to beat the current default in a fair, small-scale comparison first**. Fair means the same data, the same number of steps and about the same parameter count. Compare against [`exealm/model/baseline.py`](exealm/model/baseline.py), a plain Llama-style transformer, or against the current TONN default.

## Repo layout

```
exealm/
  config.py              model sizes: "100m" (the real one, 101M parameters) and "tiny" (tests)
  model/
    tonn.py              the TONN layer and the full model
    circuits.py          the circuit menu
    retrieval.py         RG: the model's own RAG engine
    board.py             AG: the agent-board circuit
    search.py            worth measurement and circuit replacement
    scan.py              the running-sum recurrences (needs a fast kernel)
    baseline.py          Llama-style transformer to compare against
  memory/
    datastore.py         read-only retrieval datastore
    board.py             persistent read-write agent board
  data/                  sources + licenses, download, quality filters, tokenizer, packing, batches
  train/                 pretrain, sft, dpo, grpo
  eval/                  evaluation plan
docs/                    architecture, memory (RAG + agents), data, training, ideas
tests/                   run with pytest
```

## Roadmap

1. [x] Skeleton: TONN model with RG and AG circuits, architecture search, training script, tests
2. [ ] Data pipeline: download, dedupe, filter, and mix the 2T tokens
3. [ ] Tokenizer
4. [ ] Small-scale architecture experiments: TONN vs Llama from 10M to 100M parameters, with and without RG and AG
5. [ ] Fast kernels and multi-GPU training
6. [ ] Evaluation harness (English and Japanese, retrieval, agents)
7. [ ] Full training run
8. [ ] Post-training (SFT, DPO, GRPO, multi-agent RL) and release

Each phase gets an issue with an owner. Open tasks are in [issues](https://github.com/exea-labs/ExeaLM/issues).

## Community

Read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md). Questions go in #exealm on the Exea Discord.

## License

Apache 2.0. See [LICENSE](LICENSE). Training data keeps its own licenses, listed per source in [`exealm/data/sources.py`](exealm/data/sources.py).
