# Results

Experiments run on a Mac Studio (Apple M3 Ultra) on 2026-11-11. Each section names the script that reproduces it.
Raw numbers are in the JSON files next to this page. Everything here is small-scale: these results test mechanisms,
not the final model.

## 1. Agents learn to talk through the board, during training (`experiments/swarm_relay.py`)

**Task.** The information is split across a swarm. Fact holders read `[FACT, key, value]` chunks. The asker reads
questions `[ASK, key, ?]` about those facts, and only the asker is scored. Values are random for every example, so
the asker can beat chance (1 in 64) only if the others pass the facts over the board. The fact holders get no loss of
their own: what they post is learned entirely from the asker's errors flowing back through the board.

**Model.** TONN, width 128, 4 layers, 8 circuits per layer, one AG circuit per layer, 3,000 to 4,000 steps.

| Setup | Trained as | Accuracy, board shared | Accuracy, board cut |
|---|---|---|---|
| 2 agents, 4 facts, seed 0 | swarm | **99.8%** | 1.5% |
| 2 agents, 4 facts, seed 1 | swarm | **99.8%** | 1.8% |
| 2 agents, 4 facts, seed 2 | swarm | **99.9%** | 1.4% |
| 2 agents, 4 facts, seeds 0-2 | solo (control) | 1.4-2.2% | 1.4-2.2% |
| 4 agents: 3 hold 4 facts each, 1 asks about all 12 | swarm | **99.8%** | 1.5% |
| 4 agents (control) | solo | 1.8% | 1.8% |
| 2 agents, 16 facts | swarm | **99.9%** (step 3,000) | 1.5% |

Chance is 1.6%. All three seeds pass 98% within 500-750 steps.

**What it shows.**
- The AG circuit is a working, trainable communication channel.
- Agents learn to post useful messages and to read the right one, end to end, with ordinary gradients.
- One agent can gather facts from three others at once.
- The board carries 16 separate facts per agent.

**What it doesn't show.** That this helps on real text or real problems. That's the point of the next sections.

## 2. Swarms on real text (`experiments/swarm_text.py`)

**Setup.** FineWeb-Edu as raw bytes (400 MB, document boundaries kept). TONN with 5.2M parameters (width 256, 6
layers). 4,000 steps of 32 sequences × 512 bytes, the same bytes per agent in every arm.

The swarm arms read in swarms of 4. In a **split** swarm, 4 agents read 4 consecutive windows of one stretch of text.
All arms are scored on the same held-out split swarms.

| Trained as | Bits per byte, board shared | Bits per byte, board cut | What the board adds |
|---|---|---|---|
| solo | 1.629 | **1.593** | -0.037 (it never learned to use a board) |
| split swarms | **1.607** | 1.612 | +0.005 |
| mixed (half split, half random) | **1.606** | 1.607 | +0.001 |
| split batches, board cut while training (control) | 1.631 | **1.613** | -0.018 |

**What it shows.** Each model is compared in the mode it was trained in.

| Comparison | Bits per byte |
|---|---|
| Split swarms (board shared) vs the same batches with the board cut while training (control) | 1.607 vs 1.613: **the board gains 0.006** (0.4%) |
| The control vs ordinary solo training | 1.613 vs 1.593: **split batches cost 0.021** |

- On real text at this size, agents gain a little from each other, and the gain is about the same at every position
  in the swarm.
- The overall deficit against solo training comes from the batches, not the board: a split batch holds 4 times fewer
  independent stretches of text.
- In pretraining, swarms should either keep the number of independent stretches up (more rows per batch) or read
  different documents (related swarms) instead of one stretch.

Next steps toward the bet (a 0.1B model that takes on large problems with its swarm):
- swarms of related documents instead of consecutive windows;
- tasks that need collaboration (post-training);
- larger models.

## 3. The model builds its own RAG (`experiments/rag_lookup.py`)

**Task.** A datastore of facts `[FACT, k1, k2, value]` with two-token keys and values drawn fresh every training step,
so they can't be memorized in the weights. The model reads questions `[ASK, k1, k2]`, and in the next chunk must give
the answer `[ANS, value]`, scored only on the value. Chance is 1 in 64. To beat it, the model has to learn all of RAG
on its own:
- its own embeddings for the datastore;
- what to search for;
- which result to trust;
- how to read the answer out of it.

In the "restated" variant the answer chunk repeats the key before the value. In the original variant the model must
also carry the key over from the question chunk.

**Model.** TONN, width 128, 4 layers, 8 circuits per layer, one RG circuit per layer, top-4 retrieval. 4,000 steps.
For the 1,024-fact run the training datastore starts at 16 facts and doubles every 400 steps.

| Datastore at evaluation | Variant | Datastore attached | Datastore removed | Same model trained without a datastore |
|---|---|---|---|---|
| **1,024 facts** | original | **100.0%** | 1.5% | 2.3% |
| 16 facts | original | **100.0%** | 0.9% | 0.8% |
| 16 facts | restated | **100.0%** | 1.6% | 2.1% |
| 4 facts | restated | **100.0%** | 0.1% | 10.2% |
| 4 facts | original | **88.4%** | 4.2% | 5.4% |

Chance is 1.6%. In the 4-fact restated run the control reaches 10%: with only 4 facts, later questions can copy an
earlier answer from the model's own text. The 1,024-fact model reached 93% by step 1,000 and 99.6% by step 2,000.

**What it shows.** RG works as designed. It's a trainable RAG engine the model builds itself, and it finds the right
fact among 1,024 it has never seen.

**How we got here.** The first design never learned: accuracy stayed at chance for 4,000 steps. Diagnosis on small
cases found three separate blockers. Each would have been enough on its own.

1. **The query and the stored keys lived in unrelated spaces.** The search was no better than random. Fix: the query
   for chunk c is the same encoder applied to chunk c-1, plus a learned contribution from the circuit's own stream
   (starting at 0). Both sides go through the same encoder.
2. **An output gate scrambled the retrieved value.** The read was multiplied by a gate computed from the reading
   position. A linear probe recovered the value from the read 68% of the time before the gate and 2.5% after it.
   Fix: no gate.
3. **A zero-initialised matrix swamped the retrieved tokens' identity.** It added the encoder's in-chunk states to each
   retrieved token, and Adam moves every entry of a zero matrix by about the learning rate per step whatever the
   gradient, so within 100 steps it drowned out the token identity the match relies on. Fix: retrieved tokens are
   plain token embeddings, read through the layer's own input projection.

Two smaller choices helped:
- the retrieval keys look one token back, so "find the key, take what follows it" is one attention step;
- the weighting of retrieved chunks starts soft.

Controls along the way:
- The same model learned the same lookup with the facts in its own context (95% in 1,000 steps), so the model could
  do the task. The failure was in retrieval.
- With two fixed datastores the broken design could still fit answers that differ only in the datastore, so
  gradients were flowing.

**What it doesn't show yet.** Whether retrieval helps on real text. That's section 6.

## 4. Tokenizer size (`experiments/tokenizer_compare.py`, `experiments/tokenizer_lm.py`)

Byte-level BPE trained on the same sample: 200 MB English (FineWeb-Edu), 160 MB Japanese (LLM-jp ja_wiki and
ja_fineweb-2) and 80 MB code (LLM-jp code_stack). Measured on held-out files.

| Vocabulary | Embedding parameters (width 640) | Bytes per token: English | Japanese wiki | Japanese web | Code |
|---|---|---|---|---|---|
| 32k | 21.0M | 4.38 | 4.28 | 5.36 | 2.67 |
| 49k | 31.5M | 4.55 | 4.54 | 5.75 | 2.77 |
| 64k | 41.9M | 4.66 | 4.72 | 6.03 | 2.83 |

Small-model check: the `small` preset (width 256, 6 layers), 3,000 steps, the same 49M training tokens each (50%
English, 30% Japanese, 20% code). Bits per byte on held-out text (lower is better):

| Vocabulary | Parameters | English | Japanese wiki | Japanese web | Code |
|---|---|---|---|---|---|
| 32k | 13.3M | 1.489 | **1.575** | **1.433** | **1.949** |
| 49k | 17.5M | **1.484** | 1.586 | 1.434 | 1.982 |
| 64k | 21.7M | 1.487 | 1.613 | 1.446 | 2.062 |

**What it shows.** 32k wins or ties on every kind of text, and it is clearly better on code and Japanese
Wikipedia. Bigger vocabularies fit more text into each token, but at a fixed number of training tokens their extra
embeddings are seen less often, and they cost parameters: at 0.1B, 32k instead of 49k frees 10.5M parameters (10% of
the model) for the rest of the network.

**Recommendation:** 32k, re-checked at larger scale before the main run. The tokenizers themselves are not in git
(they will be published on Hugging Face).

## 5. Speed of the real model (`experiments/speed.py`)

One training step (forward, backward, optimizer) of the `100m` preset. Mac Studio GPU (MPS), float32, 8 sequences
of 2,048 tokens. Median of 5 steps.

| Model | Parameters | Tokens per second | Relative cost per token |
|---|---|---|---|
| Llama baseline | 100.2M | 11,393 | 1.00 |
| TONN, memory circuits idle | 103.4M | 7,417 | 1.54 |
| TONN, retrieval on (100k-chunk datastore) | 103.4M | 4,291 | 2.65 |
| TONN, retrieval on + swarms of 4 | 103.4M | 4,255 | 2.68 |

**What it shows.**
- At full size TONN costs 1.5 times a same-size transformer per token. The 2.7 times measured on 5M models doesn't
  carry over.
- Retrieval adds 74% on top: it re-encodes 4 retrieved chunks for every chunk read, in every layer. This is the
  biggest speed target ("cheaper retrieval" in the to-do list).
- Swarms cost almost nothing (1%).
- Re-encoding a 100k-chunk datastore takes 1.7 seconds.

**After the speed work and the hybrid** (`results/speed_hybrid/`, the new default `100m` preset):

| Model | Parameters | Tokens per second | Relative cost per token |
|---|---|---|---|
| Llama baseline | 100.2M | 12,026 | 1.00 |
| **ExeaLM hybrid**, memory circuits idle | 101.1M | 10,630 | **1.13** |
| ExeaLM hybrid, retrieval on | 101.1M | 7,259 | 1.66 |
| ExeaLM hybrid, retrieval on + swarms of 4 | 101.1M | 7,285 | 1.65 |

Two things made the difference:
- Half the layers are now standard attention.
- The gated recurrences (GI, SM) use a parallel prefix scan: 5 and 3.3 times faster per circuit, measured with
  `experiments/profile_circuits.py`.

**For the compute plan.** If the MI300X shows the same ratios, 100B tokens at DATA.md's 300 TFLOPs per GPU comes to
about 56 GPU-hours (transformer speed) × 2.65, about 150 GPU-hours with retrieval on. Retrieval is not useful yet
(section 3), so this is a cost estimate, not a recommendation.

## 6. Built-in RAG on real text (`experiments/rag_text.py`)

**Setup.** FineWeb-Edu bytes, the 5.2M model, 4,000 steps. The datastore is 200,000 chunks (64 bytes each) of
training documents, and retrieval never returns chunks of the document being read.

**First run: it exposed a bug.** Scores in bits per byte:

| Step | Datastore attached | Datastore removed |
|---|---|---|
| 1,000 | 1.935 | 1.929 |
| 2,000 | 1.785 | 1.783 |
| 3,000 | **6.122** | 1.727 |

What went wrong:
- The search fetched 16 candidates and dropped those from the document being read afterwards.
- The nearest chunks are usually from that same document, so in training the read often came back empty, and the
  circuit's running output scale settled low.
- Held-out documents are not in the datastore at all, so at evaluation every read came back full, and divided by that
  low scale it swamped the model.

**Fix** (`Datastore.search`): the same-document rule is now applied inside the search, so every query gets k results
from other documents. A test covers the case where the document being read dominates the datastore.

**Rerun:** PENDING.

## 7. TONN vs Llama on real tokens (`experiments/tokenizer_lm.py --models ...`)

**Setup.** The `small` preset (width 256, 6 layers, about 13M parameters with the 32k vocabulary) and a Llama-style
transformer of the same size. Same data (50% English, 30% Japanese, 20% code, 32k tokenizer), same 3,000 steps of
32 × 512 tokens, two seeds each. Bits per byte on held-out text (lower is better).

Variants:
- `tonn`: the default, with one retrieval and one board circuit per layer, idle here;
- `tonn_core`: no memory circuits, all 8 per layer model language;
- `tonn_core_search`: the same, with the architecture search on.

`python experiments/leaderboard.py results/arch results/arch2 results/sweep` rebuilds this table:

| Rank | Run | Model | Config | lr | Seed | English | Japanese wiki | Japanese web | Code | Mean | vs Llama |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | arch | llama |  | 0.002 | 0 | 1.461 | 1.534 | 1.413 | 1.775 | 1.546 | -0.008 |
| 2 | arch | llama |  | 0.002 | 1 | 1.473 | 1.545 | 1.426 | 1.803 | 1.562 | +0.008 |
| 3 | arch | tonn |  | 0.002 | 1 | 1.475 | 1.559 | 1.421 | 1.906 | 1.590 | +0.037 |
| 4 | arch2 | tonn_core_search |  | 0.002 | 0 | 1.479 | 1.566 | 1.422 | 1.948 | 1.604 | +0.050 |
| 5 | arch2 | tonn_core |  | 0.002 | 0 | 1.483 | 1.570 | 1.424 | 1.959 | 1.609 | +0.055 |
| 6 | arch2 | tonn_core_search |  | 0.002 | 1 | 1.474 | 1.562 | 1.417 | 1.993 | 1.611 | +0.058 |
| 7 | arch2 | tonn_core |  | 0.002 | 1 | 1.477 | 1.567 | 1.419 | 2.000 | 1.616 | +0.062 |
| 8 | arch | tonn |  | 0.002 | 0 | 1.491 | 1.578 | 1.433 | 1.964 | 1.616 | +0.063 |

Llama baseline (lr 2e-3, 2 seeds): mean 1.554 bits per byte.

**Averages over two seeds:**

| Model | English | Japanese wiki | Japanese web | Code |
|---|---|---|---|---|
| Llama | **1.467** | **1.540** | 1.420 | **1.789** |
| TONN core + search | 1.476 | 1.564 | **1.420** | 1.970 |
| TONN core | 1.480 | 1.568 | 1.422 | 1.979 |
| TONN default | 1.483 | 1.568 | 1.427 | 1.935 |

**What it shows.**
- On prose TONN is close: tied on Japanese web, 0.009 behind on English, 0.024 on Japanese Wikipedia. The deficit is
  almost all code, about 0.18 bits per byte.
- The idle memory circuits cost little, and the search gains 0.004-0.009.
- Code rewards exact copying of names, which attention does well. Llama has full attention in every layer; a random
  TONN layer has on average less than one 32-channel attention circuit.
- The overnight sweep tests this: attention-heavy starting mixes, pinned attention circuits, 64-channel circuits,
  hybrid layers, learning rates. Results land in `results/sweep/`.

## 8. Overnight sweep: making TONN beat Llama (`experiments/run_queue.sh`, `results/sweep/`)

Same setup as section 7, one idea at a time, with the same data and steps for every run.

**Result: the hybrid beats Llama.**
- TONN layers alternate with full attention layers (attention at layers 1, 3 and 5).
- TONN's starting circuits favour short convolutions (`init_weights`: C ×4, G and D ×0.25).

It beat the same-size Llama on every kind of text with both seeds:

| Model | Seed | English | Japanese wiki | Japanese web | Code | Mean |
|---|---|---|---|---|---|---|
| Hybrid, convolution-heavy start | 0 | 1.452 | 1.523 | 1.406 | 1.745 | 1.532 |
| Hybrid, convolution-heavy start | 1 | 1.451 | 1.519 | 1.402 | 1.748 | 1.530 |
| Llama | 0 | 1.461 | 1.534 | 1.413 | 1.775 | 1.546 |
| Llama | 1 | 1.473 | 1.545 | 1.426 | 1.803 | 1.562 |

On average the hybrid is better by 0.015 (English), 0.019 (Japanese wiki), 0.016 (Japanese web) and 0.043 (code)
bits per byte. Its two seeds land within 0.002 of each other. It trains in 651 seconds against Llama's 540 and pure
TONN's 790. It is now the default (`100m`, `small` presets).

**How we got there:**
1. **Hybrid with a random start.** Seed 0 beat Llama (mean 1.525), seed 1 did not: close on prose, but 1.950 on code.
   The seed also picks the TONN layers' starting circuits. Seed 0 drew mostly short convolutions (7); seed 1 drew
   per-position gated units, which duplicate the MLP, and only 3 convolutions.
2. **The search** helped a little on seed 1 (1.594 to 1.585) but did not fix it.
3. **More attention circuits inside pure TONN** (RP ×4) cut the code gap from 0.17 to 0.08 but stayed behind Llama: full
   attention layers do better than more small attention circuits.
4. **A convolution-heavy start** fixed the hybrid's seed 1 (code 1.950 to 1.748).

**Confirmations:**

| Run | English | Japanese wiki | Japanese web | Code | Mean |
|---|---|---|---|---|---|
| Hybrid + retrieval and board circuits (idle), 2 seeds | 1.455 | 1.530 | 1.410 | 1.769 | 1.541 |
| Hybrid + search, 2 seeds | 1.450 | 1.519 | 1.403 | 1.747 | 1.530 |
| Hybrid, 2 seeds (above) | 1.452 | 1.521 | 1.404 | 1.746 | 1.531 |
| Llama, 2 seeds | 1.467 | 1.540 | 1.420 | 1.789 | 1.554 |

- With the memory circuits kept, as in the real model, the hybrid still beats Llama on every kind of text with both
  seeds. The idle circuits (2 of 8 slots in each TONN layer) cost 0.010.
- The search adds about 0.001 on top of a good starting mix; it swapped only a few circuits.

**Longer training (6,000 steps, seed 0; the data is seen about 1.2 times):**

| Model | English | Japanese wiki | Japanese web | Code | Mean | Training time on the Mac |
|---|---|---|---|---|---|---|
| Hybrid | **1.390** | **1.455** | **1.344** | **1.676** | **1.466** | 1,367 s |
| Llama | 1.396 | 1.459 | 1.348 | 1.686 | 1.472 | 1,037 s |

The hybrid is still ahead on every kind of text, but the margin shrank from 0.014 (3,000 steps, seed 0) to 0.006.

**Equal compute is not won yet.** Llama trains 1.3 times faster per step here: TONN's circuits run as plain PyTorch
on the Mac's GPU, while attention uses fused kernels. At equal training time Llama would currently come out ahead. The
hybrid wins at equal size and equal steps. Making it win at equal compute needs:
- faster TONN kernels (the to-do list's top speed item);
- longer runs to see whether the margin keeps shrinking.

**27M parameters** (width 384, 8 layers, attention at layers 1, 3, 5 and 7; 3,000 steps, 2 seeds each):

| Model | English | Japanese wiki | Japanese web | Code | Mean | Training time |
|---|---|---|---|---|---|---|
| **Hybrid** | **1.412** | **1.483** | **1.371** | **1.702** | **1.492** | 1,217 s |
| Llama | 1.446 | 1.523 | 1.402 | 1.778 | 1.537 | 897 s |

The margin doubled with size, from 0.023 at 13M to 0.045 at 27M, with both seeds and every kind of text.

Caveats:
- **Learning rates checked.** Llama does better at 27M with a lower learning rate: 1.523 at 1e-3 against 1.537 at
  2e-3. The hybrid does better at the default: 1.492 at 2e-3 against 1.524 at 1e-3. Comparing each at its best rate:

  | 27M, each at its best rate | English | Japanese wiki | Japanese web | Code | Mean |
  |---|---|---|---|---|---|
  | **Hybrid** (2e-3, 2 seeds) | **1.412** | **1.483** | **1.371** | **1.702** | **1.492** |
  | Llama (1e-3, 2 seeds) | 1.434 | 1.511 | 1.397 | 1.748 | 1.523 |

  The hybrid leads by 0.031 on average, and on every kind of text.
- The Llama runs used the faster rotary code committed mid-sweep; it is the same function class.
- **Speed at this size.** With the new scans the hybrid takes 1.19 times Llama's time per step at 27M (1,064 s against
  897 s for 3,000 steps).

**Equal compute at 27M.** Both models get the same 1,000 seconds on the same GPU, each at its best learning rate
(`--seconds 1000`: the step count is fitted to the budget). Two seeds each:

| Same training time | Steps | English | Japanese wiki | Japanese web | Code | Mean |
|---|---|---|---|---|---|---|
| **Hybrid** (2e-3) | ~2,845 | **1.417** | **1.489** | **1.375** | **1.710** | **1.498** |
| Llama (1e-3) | ~3,370 | 1.418 | 1.492 | 1.380 | 1.725 | 1.504 |

The hybrid wins at equal compute too, narrowly: by 0.006 on average (0.003 and 0.008 on the two seeds). It is ahead on
every kind of text, with English essentially tied. Most of the lead is on code.

**Equal compute at 13M** (600 seconds, 2 seeds each, both at 2e-3, which is Llama's best rate at this size: 1e-3
gave 1.554 and 3e-3 gave 1.547 against 1.546):

| Same training time | Steps | English | Japanese wiki | Japanese web | Code | Mean |
|---|---|---|---|---|---|---|
| Hybrid | ~3,140 | 1.448 | 1.518 | 1.400 | **1.742** | 1.527 |
| Llama | ~3,610 | **1.445** | **1.517** | **1.397** | 1.750 | 1.527 |

A tie at 13M: Llama is slightly ahead on prose, the hybrid on code.

**Equal compute at 48M** (width 512, 10 layers, attention at layers 1, 3, 5, 7, 9; 2,000 seconds; seed 0):

| Same training time | Steps | English | Japanese wiki | Japanese web | Code | Mean |
|---|---|---|---|---|---|---|
| Hybrid (2e-3) | 3,414 | 1.379 | 1.456 | 1.341 | 1.712 | 1.472 |
| **Llama** (1e-3) | 4,183 | **1.370** | **1.439** | **1.332** | **1.668** | **1.452** |

**At 48M Llama wins at equal compute**, by 0.020 on every kind of text. The equal-compute trend did not keep
growing: a tie at 13M, the hybrid by 0.006 at 27M, Llama by 0.020 at 48M. Llama got 23% more steps in the same time.

Not settled yet:
- The hybrid's learning rate was not tuned at this size. Llama's best rate fell from 2e-3 to 1e-3 between 13M and
  27M, so the hybrid's may fall too; 1e-3 and 1.5e-3 are running.
- One seed.
- The equal-steps comparison (no time limit) at 48M isn't run yet.

The honest summary so far: **at equal size and steps the hybrid beats Llama (13M and 27M); at equal compute it ties
or wins up to 27M and loses at 48M with an untuned learning rate.** Faster TONN kernels would move every
equal-compute number toward the hybrid.

**Speed note.** `torch.compile` speeds up the hybrid by 8% and Llama by 20% on this GPU. TONN's per-type branches make
the compiler recompile and fuse only fragments, so both comparisons here are uncompiled. A compile-friendly TONN layer
(one code path for every circuit type) is on the to-do list.

**Full leaderboard** (`python experiments/leaderboard.py results/arch results/arch2 results/sweep`):

| Rank | Run | Model | Config | lr | Seed | English | Japanese wiki | Japanese web | Code | Mean | vs Llama |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | eqt_big_llama | llama |  | 0.001 | 0 | 1.370 | 1.439 | 1.332 | 1.668 | 1.452 | -0.077 |
| 2 | hconv_6k | tonn_core | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.390 | 1.455 | 1.343 | 1.676 | 1.466 | -0.064 |
| 3 | eqt_big_hybrid | tonn_core | {"attn_layers": [1, 3, 5, 7, 9], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.379 | 1.456 | 1.341 | 1.712 | 1.472 | -0.058 |
| 4 | llama_6k | llama |  | 0.002 | 0 | 1.395 | 1.459 | 1.348 | 1.686 | 1.472 | -0.058 |
| 5 | mid_hybrid | tonn_core | {"attn_layers": [1, 3, 5, 7], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 1 | 1.413 | 1.481 | 1.368 | 1.702 | 1.491 | -0.039 |
| 6 | mid_hybrid | tonn_core | {"attn_layers": [1, 3, 5, 7], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.412 | 1.484 | 1.373 | 1.702 | 1.493 | -0.037 |
| 7 | eqt_mid_hybrid | tonn_core | {"attn_layers": [1, 3, 5, 7], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 1 | 1.415 | 1.486 | 1.373 | 1.701 | 1.494 | -0.036 |
| 8 | eqt_mid_hybrid | tonn_core | {"attn_layers": [1, 3, 5, 7], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.418 | 1.492 | 1.376 | 1.719 | 1.502 | -0.028 |
| 9 | eqt_mid_llama | llama |  | 0.001 | 1 | 1.418 | 1.490 | 1.380 | 1.720 | 1.502 | -0.028 |
| 10 | eqt_mid_llama | llama |  | 0.001 | 0 | 1.417 | 1.494 | 1.379 | 1.730 | 1.505 | -0.025 |
| 11 | mid_llama_lr1e3 | llama |  | 0.001 | 1 | 1.435 | 1.508 | 1.397 | 1.741 | 1.520 | -0.010 |
| 12 | eqt_small_llama | llama |  | 0.002 | 0 | 1.440 | 1.511 | 1.394 | 1.745 | 1.522 | -0.007 |
| 13 | eqt_small_hybrid | tonn_core | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 1 | 1.445 | 1.512 | 1.395 | 1.738 | 1.523 | -0.007 |
| 14 | mid_hybrid_lr1e3 | tonn_core | {"attn_layers": [1, 3, 5, 7], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.001 | 0 | 1.426 | 1.512 | 1.390 | 1.768 | 1.524 | -0.006 |
| 15 | hyb135_search | tonn_core_search | {"attn_layers": [1, 3, 5]} | 0.002 | 0 | 1.448 | 1.517 | 1.400 | 1.734 | 1.525 | -0.005 |
| 16 | mid_llama_lr1e3 | llama |  | 0.001 | 0 | 1.434 | 1.514 | 1.398 | 1.754 | 1.525 | -0.005 |
| 17 | hyb135 | tonn_core | {"attn_layers": [1, 3, 5]} | 0.002 | 0 | 1.449 | 1.516 | 1.399 | 1.736 | 1.525 | -0.005 |
| 18 | hyb135_mem | tonn | {"attn_layers": [1, 3, 5]} | 0.002 | 0 | 1.449 | 1.518 | 1.402 | 1.734 | 1.526 | -0.004 |
| 19 | hconv_search | tonn_core_search | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 1 | 1.448 | 1.516 | 1.401 | 1.751 | 1.529 | -0.001 |
| 20 | hyb135_conv | tonn_core | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 1 | 1.451 | 1.519 | 1.402 | 1.748 | 1.530 | +0.000 |
| 21 | eqt_small_hybrid | tonn_core | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.450 | 1.524 | 1.404 | 1.746 | 1.531 | +0.001 |
| 22 | hconv_search | tonn_core_search | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.452 | 1.522 | 1.406 | 1.744 | 1.531 | +0.001 |
| 23 | hyb135_conv | tonn_core | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.452 | 1.523 | 1.406 | 1.745 | 1.532 | +0.002 |
| 24 | eqt_small_llama | llama |  | 0.002 | 1 | 1.449 | 1.523 | 1.401 | 1.754 | 1.532 | +0.002 |
| 25 | mid_llama | llama |  | 0.002 | 1 | 1.447 | 1.522 | 1.401 | 1.777 | 1.537 | +0.007 |
| 26 | mid_llama | llama |  | 0.002 | 0 | 1.444 | 1.524 | 1.402 | 1.778 | 1.537 | +0.007 |
| 27 | hconv_mem | tonn | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 0 | 1.456 | 1.532 | 1.410 | 1.764 | 1.540 | +0.011 |
| 28 | hconv_mem | tonn | {"attn_layers": [1, 3, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25}} | 0.002 | 1 | 1.455 | 1.528 | 1.410 | 1.773 | 1.542 | +0.012 |
| 29 | arch | llama |  | 0.002 | 0 | 1.461 | 1.534 | 1.413 | 1.775 | 1.546 | +0.016 |
| 30 | llama_lr3e3 | llama |  | 0.003 | 0 | 1.465 | 1.538 | 1.415 | 1.769 | 1.547 | +0.017 |
| 31 | llama_lr1e3 | llama |  | 0.001 | 0 | 1.468 | 1.548 | 1.425 | 1.776 | 1.554 | +0.025 |
| 32 | arch | llama |  | 0.002 | 1 | 1.473 | 1.545 | 1.426 | 1.803 | 1.562 | +0.032 |
| 33 | rp4 | tonn_core | {"init_weights": {"RP": 4}} | 0.002 | 0 | 1.475 | 1.555 | 1.424 | 1.869 | 1.581 | +0.051 |
| 34 | hyb135_search | tonn_core_search | {"attn_layers": [1, 3, 5]} | 0.002 | 1 | 1.455 | 1.543 | 1.407 | 1.933 | 1.584 | +0.055 |
| 35 | arch | tonn |  | 0.002 | 1 | 1.475 | 1.559 | 1.421 | 1.906 | 1.590 | +0.061 |
| 36 | hyb135_s1_curve | tonn_core | {"attn_layers": [1, 3, 5]} | 0.002 | 1 | 1.461 | 1.551 | 1.412 | 1.949 | 1.593 | +0.064 |
| 37 | hyb135_s1 | tonn_core | {"attn_layers": [1, 3, 5]} | 0.002 | 1 | 1.461 | 1.551 | 1.412 | 1.950 | 1.594 | +0.064 |
| 38 | arch2 | tonn_core_search |  | 0.002 | 0 | 1.479 | 1.566 | 1.422 | 1.948 | 1.604 | +0.074 |
| 39 | arch2 | tonn_core |  | 0.002 | 0 | 1.483 | 1.570 | 1.424 | 1.959 | 1.609 | +0.079 |
| 40 | arch2 | tonn_core_search |  | 0.002 | 1 | 1.474 | 1.562 | 1.417 | 1.993 | 1.611 | +0.082 |
| 41 | arch2 | tonn_core |  | 0.002 | 1 | 1.477 | 1.567 | 1.419 | 2.000 | 1.616 | +0.086 |
| 42 | arch | tonn |  | 0.002 | 0 | 1.491 | 1.578 | 1.433 | 1.964 | 1.616 | +0.087 |

Llama baseline (lr 2e-3, 7 seeds): mean 1.530 bits per byte.

