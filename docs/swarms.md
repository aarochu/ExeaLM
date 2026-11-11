# Swarms: how ExeaLM agents learn to work together

A **swarm** is a few copies of the model (agents, 4 by default) that share one agent board through their AG circuits
([memory.md](memory.md)). Training runs **many swarms at the same time**. A batch of 512 sequences is 128 swarms of 4,
each with its own private board, and swarms never hear each other. Each swarm sits on one GPU, so boards need no
traffic between GPUs.

Swarms cost no extra data. Every agent reads ordinary training text and predicts its own next tokens. The only
question is *which* texts end up in the same swarm, and the board adds a little compute.

## Pretraining: the swarm pulls data together

The swarm is chosen so that what one agent reads helps the others:

| Swarm type | Who reads what | What the board teaches | Status |
|---|---|---|---|
| **Split** | one long stretch of text, each agent a different part | pass on context the others haven't seen | built (`--swarm_policy split`) |
| **Random** | unrelated texts | when to ignore the board and stay quiet | built (`random`) |
| **Related** | different documents on the same topic (neighbours in the model's own retrieval space) | share facts across sources | to do |
| **Cross-lingual** | the same topic in Japanese and English (Wikipedia interlanguage links) | share meaning across languages | to do |

The default, `mixed`, makes each swarm *split* or *random* with even odds.

Retrieval multiplies this. Each agent pulls its own chunks from the datastore through its RG circuits and can post what
it found, so a swarm of 4 searches 4 ways at once and shares the results.

Rules that keep it honest:
- An agent hears only what other agents posted in **earlier** 64-token chunks, never the current one (tested).
- Retrieval never returns chunks of a document the agent is reading. Agents of a split swarm share one document id,
  so none of them can look the document up.
- Swarms never hear each other (tested).

## Post-training: the swarm solves tasks together

1. **Swarm SFT.** In these tasks the information is split: each agent gets part of the facts, and one agent must give
   the answer. The whole swarm runs in one forward pass, so the loss on the answer flows back through the board into
   every agent's posts. Collaboration is learned with ordinary gradients, with no RL needed for this step. Example
   tasks can be built from multi-hop question-answering data, where the two supporting documents go to different agents.
2. **Swarm RL (GRPO).** Sample several attempts by the whole swarm for each task. The swarm shares the reward, and
   attempts are compared within their group. Task families:
   - split-information QA
   - divide-and-conquer math (agents take sub-problems)
   - write-and-test code (one agent writes, one runs and checks)
   - propose-and-check (one answers, one verifies)
   - cross-lingual QA (the source is in Japanese, the question in English)
3. **Roles.** Each agent's prompt says who it is ("agent 2 of 4, checker"), so agents can specialize.

## At inference

- **One machine:** run N copies as one batch with the swarm size set to N. The board is in the batch, so a swarm
  costs about what N separate requests cost.
- **Different times or machines:** a persistent `AgentBoard` (`exealm/memory/board.py`). Today it is a file; a small
  board server (HTTP or Redis) is a to-do.

## What we don't know yet

- Does a split swarm predict its text better than solo agents reading the same text? This is the first experiment, and
  until it works, swarms are a design, not a result.
- Which swarm types teach the most, and in what mix?
- Do agents end up specializing, and does accuracy rise with swarm size (2, 4, 8)?
- Can the messages be decoded into text people can read?
- A shared board is an input channel. Agents must treat other agents' messages as information, not instructions,
  and post-training data should teach that.
