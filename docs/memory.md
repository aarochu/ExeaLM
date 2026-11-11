# Memory circuits: its own RAG (RG) and the agent board (AG)

Both are circuit types in TONN's menu, so they live inside the network: they read from the residual stream, write back
to it, and are trained by the same loss as everything else. Every layer starts with one of each, and the search can add
more but can never remove the last one.

|  | RG: retrieval | AG: agent board |
|---|---|---|
| Talks to | the **datastore**: chunks of training text | the **board**: messages from other agents |
| Access | read-only | read and write |
| Embeddings | made by the model's own chunk encoder | the model's own keys and values |
| When | one search per 64 tokens per RG slot | one post per 64 tokens per AG slot; reads at every token |
| Code | `exealm/model/retrieval.py`, `exealm/memory/datastore.py` | `exealm/model/board.py`, `exealm/memory/board.py` |

## RG: the model builds its own RAG engine

A normal RAG system has four parts built by different people: an embedding model, an index, a rule for what to search
for, and a prompt that pastes the results in. In ExeaLM all four are learned by the model:

1. **Embeddings.** `ChunkEncoder` turns a 64-token chunk into per-token states and one 32-number key. It reads the
   model's own token embeddings and is shared by every RG slot, so there is one index.
2. **What to search for.** At the end of each chunk, an RG slot's stream A is its query. Every RG slot learns its own.
3. **Which results to trust.** The top-k results (k = 4) are scored against the query, and the softmax of those scores
   weights them. Gradient flows through the scores to both the query and the encoder.
4. **What to take.** Every position in the next chunk reads the retrieved tokens with attention (query = stream B),
   gated by stream C.

Rules that keep it honest:
- **Causal.** The query for chunk c is made at the last position of chunk c - 1.
- **No looking up the answer.** The datastore never returns chunks from the document being read (`exclude=`). This
  matters: without it, the model would learn to copy the next tokens from the datastore. Near-duplicate documents
  leak the same way, which is one more reason deduplication matters.
- **Stale index, fresh gradients.** The stored keys only *find* candidates. The chosen chunks are re-encoded live so the
  encoder trains, and the whole index is re-encoded every `--refresh_every` steps (the approach REALM used).

To see what a checkpoint actually retrieves, `python -m exealm.eval.show_retrieval --ckpt runs/<run>/model.pt --data
<packed dir> --tokenizer tokenizer --prompt "..."` prints, for each chunk of the prompt, every RG slot's picks with
their weights and text. Random-looking picks mean retrieval hasn't learned much yet.

Open problems:
- **Scale.** Brute-force search is fine for a few million chunks. Billions of tokens need FAISS (IVF-PQ), sharded
  across GPUs, and re-encoding them costs real compute. How often is often enough?
- **What goes in the datastore?** All 100B training tokens is more than one index needs. A curated few billion (encyclopedias, textbooks,
  documentation, in Japanese and English) is the starting idea.
- **Key size.** Keys are 32 numbers (one slot's width). Is that enough to find the right chunk among millions?
- **Cost.** Re-encoding 4 retrieved chunks for each RG slot is about 4× the batch's own tokens per RG slot, through a
  small encoder. Caching, sharing retrievals between layers, or fewer RG slots might be enough.
- **At inference,** the datastore can be swapped for a different one (new facts, private documents) without
  retraining. How well that works is an experiment nobody has run yet.

## AG: agents that talk

Once per 64-token chunk, every AG slot posts one message to its channel of the board:
- **value:** the chunk's average of stream A (what it says)
- **key:** the chunk's average of stream B (what it is about)
- **strength:** sigmoid(w · average of stream C + b), how much it wants the message read. Low strength means staying
  quiet.

At every position it reads the channel with attention (query = stream B at that position) over messages the other agents of its swarm
posted in **earlier** chunks, plus a "nothing" entry so it can ignore the board. The result is gated by stream C.

Where it runs:
- **Training:** the rows of a batch form **swarms** (`swarm_size = 4`), each with its own in-batch board, so posting
  and reading both get gradients. A batch holds many swarms at once, and swarms never hear each other. An agent may
  read its own past posts (`board_read_self`). What each swarm reads, and how swarms are used in post-training, is in
  [swarms.md](swarms.md).
- **Inference:** a persistent `AgentBoard` holds messages from agents that ran before or elsewhere. Posts are keyed by
  (agent, chunk), so re-reading the same text replaces messages instead of duplicating them. An agent never re-reads
  its own old posts from the persistent board, because it gets those from its own context.

Open problems:
- **What should each swarm read in pretraining?** See [swarms.md](swarms.md).
- **Readable messages.** Messages are vectors. A small decoder that turns a message into text would let people see
  what agents tell each other, and would let ExeaLM talk to other models.
- **Infrastructure.** `AgentBoard.save/load` is a file. Agents on different machines need a small board server
  (HTTP or Redis).
- **Safety.** A shared board is an input channel. Agents should treat other agents' messages as information, not as
  instructions. Post-training data should teach that.
