"""Batches for training: random windows of packed tokens, with the document id of every position.

Swarms (for the AG agent board): consecutive rows of a batch form one swarm and share one board; a batch holds many
swarms, and swarms never hear each other. What each swarm reads decides what its board can teach (docs/swarms.md):
    "split"   the swarm reads consecutive windows of the token stream (one long document, when there is one), each
              agent its own part, so posting what you read helps the agents reading the other parts
    "random"  unrelated windows: the board has little to offer, so agents learn when to ignore it and stay quiet
    "mixed"   each swarm is "split" with probability p_split, otherwise "random" (the default)

The seam of a split swarm: agent i's last target is the first token agent i + 1 reads, and from its second chunk on,
agent i hears agent i + 1's first post, which was made from that token. So that one target could be read off the board.
batch() sets it to -100 (ignored by the loss) in every split swarm except for the last agent; mask_seams=False keeps it.
  TODO(swarms): "related": documents that are near each other in the model's own retrieval space.
  TODO(swarms): "crosslingual": the same topic in Japanese and English (Wikipedia interlanguage links).
  TODO(swarms): swarm sizes that vary within a run (2 to 8).
"""
import json
import os

import numpy as np
import torch


class TokenShards:
    def __init__(self, path):
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.tokens = np.memmap(os.path.join(path, "tokens.bin"), dtype=self.meta["dtype"], mode="r")
        self.offsets = np.load(os.path.join(path, "offsets.npy"))

    def doc_of(self, pos):
        return np.searchsorted(self.offsets, pos, side="right") - 1

    def batch(self, B, T, rng, swarm_size=1, policy="random", p_split=0.5, mask_seams=True):
        """x, y (B, T) int64 and doc ids (B, T) int64. rng: a numpy Generator. B must be a multiple of swarm_size
        for swarm policies to apply. y is -100 at the seams of split swarms (see the module docstring)."""
        n = len(self.tokens) - T - 1
        seams = []                                                       # rows whose last target the next row reads
        if policy in ("split", "mixed") and swarm_size > 1 and B % swarm_size == 0:
            starts = []
            for _ in range(B // swarm_size):
                if policy == "split" or rng.random() < p_split:
                    s0 = int(rng.integers(0, max(1, n - swarm_size * T)))
                    seams += [len(starts) + j for j in range(swarm_size - 1)]
                    starts += [s0 + j * T for j in range(swarm_size)]
                else:
                    starts += list(rng.integers(0, n, size=swarm_size))
            starts = np.array(starts)
        else:
            starts = rng.integers(0, n, size=B)
        idx = starts[:, None] + np.arange(T + 1)[None]
        win = torch.from_numpy(self.tokens[idx].astype(np.int64))
        docs = torch.from_numpy(self.doc_of(idx[:, :-1]).astype(np.int64))
        x, y = win[:, :-1], win[:, 1:].clone()
        if mask_seams and seams:
            y[seams, -1] = -100
        return x, y, docs


class SyntheticShards:
    """A stand-in corpus for smoke tests: documents from a random sparse Markov chain, so the loss can fall without
    any download. Same interface as TokenShards."""

    def __init__(self, vocab_size=256, n_tokens=200_000, doc_len=500, seed=0):
        rng = np.random.default_rng(seed)
        nxt = rng.integers(0, vocab_size, size=(vocab_size, 4))
        toks = np.empty(n_tokens, dtype=np.int64)
        toks[0] = 0
        for i in range(1, n_tokens):
            toks[i] = nxt[toks[i - 1], rng.integers(0, 4)] if i % doc_len else rng.integers(0, vocab_size)
        self.tokens = toks
        self.offsets = np.arange(0, n_tokens, doc_len)
        self.meta = dict(vocab_size=vocab_size, n_tokens=n_tokens, n_docs=len(self.offsets), bytes_per_token=1.0)

    doc_of = TokenShards.doc_of
    batch = TokenShards.batch


class MultiShards:
    """Several packed sources mixed by weight (default: by token count), e.g. English, Japanese and code packed
    separately. Each swarm (or each row, without swarms) draws all its windows from one source. Document ids stay
    unique across sources. Same batch() interface as TokenShards.
      TODO(data): stream from many shard files per source instead of one file each."""

    def __init__(self, paths, weights=None):
        self.sources = [TokenShards(p) for p in paths]
        sizes = np.array([s.meta["n_tokens"] for s in self.sources], dtype=float)
        w = np.array(weights, dtype=float) if weights is not None else sizes
        self.p = w / w.sum()
        self.doc_base = np.concatenate([[0], np.cumsum([len(s.offsets) for s in self.sources])[:-1]])
        m0 = self.sources[0].meta
        self.meta = dict(vocab_size=m0["vocab_size"], n_tokens=int(sizes.sum()),
                         n_docs=int(sum(len(s.offsets) for s in self.sources)),
                         bytes_per_token=float(sum(s.meta["bytes_per_token"] * s.meta["n_tokens"] for s in self.sources)
                                               / sizes.sum()))

    def batch(self, B, T, rng, swarm_size=1, policy="random", p_split=0.5):
        g = swarm_size if (swarm_size > 1 and B % swarm_size == 0) else 1
        xs, ys, ds = [], [], []
        for _ in range(B // g):
            i = int(rng.choice(len(self.sources), p=self.p))
            x, y, d = self.sources[i].batch(g, T, rng, g, policy, p_split)
            xs.append(x)
            ys.append(y)
            ds.append(d + int(self.doc_base[i]))
        return torch.cat(xs), torch.cat(ys), torch.cat(ds)
