"""Batches for training: random windows of packed tokens, with the document id of every position.

Teams (for the AG agent board): consecutive rows of a batch form a team and share a board. How to choose teammates
is an open question (docs/memory.md). Policies here:
    "random"  unrelated windows (the board carries little useful information; the default for now)
    "doc"     the team reads consecutive windows of one long document, each agent its own part, so posting what
              you read helps the teammates who read the other parts
  TODO(data): "related": windows from documents that are close in the model's own retrieval space.
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

    def batch(self, B, T, rng, team_size=1, policy="random"):
        """x, y (B, T) int64 and doc ids (B, T) int64. rng: a numpy Generator."""
        n = len(self.tokens) - T - 1
        if policy == "doc" and team_size > 1:
            starts = []
            for _ in range(B // team_size):
                s0 = int(rng.integers(0, max(1, n - team_size * T)))
                starts += [s0 + j * T for j in range(team_size)]
            starts = np.array(starts)
        else:
            starts = rng.integers(0, n, size=B)
        idx = starts[:, None] + np.arange(T + 1)[None]
        win = torch.from_numpy(self.tokens[idx].astype(np.int64))
        docs = torch.from_numpy(self.doc_of(idx[:, :-1]).astype(np.int64))
        return win[:, :-1], win[:, 1:], docs


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
