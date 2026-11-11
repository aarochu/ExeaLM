"""The read-only datastore that RG circuits retrieve from.

It holds chunks of training text (token ids), the document each chunk came from, and one key vector per chunk. The
keys are made by the model's OWN chunk encoder (exealm/model/retrieval.py: ChunkEncoder), so the model learns its own
embedding space for retrieval instead of borrowing someone else's. Because that encoder keeps training, the stored keys
go stale; refresh() re-encodes everything every few thousand steps (the approach REALM used).

The stored keys are only used to FIND candidates. The chosen chunks are re-encoded live in the forward pass, so
gradients still reach the encoder.

Scale: brute-force search below is fine up to a few million chunks on one GPU. The real datastore (billions of tokens)
needs an approximate index such as FAISS IVF-PQ, sharded across GPUs. That is an open task (docs/memory.md).
"""
import numpy as np
import torch


class Datastore:
    def __init__(self, tokens, doc_ids):
        """tokens (N, chunk_len) integer ids; doc_ids (N,) the document of each chunk."""
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.doc_ids = torch.as_tensor(doc_ids, dtype=torch.long)
        self.keys = None                                  # (N, key_dim), filled by refresh()
        self.refreshed_at = None

    def __len__(self):
        return self.tokens.shape[0]

    def to(self, device):
        self.tokens, self.doc_ids = self.tokens.to(device), self.doc_ids.to(device)
        if self.keys is not None:
            self.keys = self.keys.to(device)
        return self

    @classmethod
    def from_shards(cls, shards, n_chunks, chunk_len, seed=0):
        """Sample n_chunks chunks that each sit inside one document of a TokenShards corpus (exealm/data/loader.py).
        TODO(data): a curated datastore (encyclopedias, textbooks, code docs) instead of a random sample."""
        rng = np.random.default_rng(seed)
        starts, docs = [], []
        while len(starts) < n_chunks:
            s = rng.integers(0, len(shards.tokens) - chunk_len, size=4 * n_chunks)
            d0, d1 = shards.doc_of(s), shards.doc_of(s + chunk_len - 1)
            ok = d0 == d1
            starts += list(s[ok]); docs += list(d0[ok])
        starts, docs = np.array(starts[:n_chunks]), np.array(docs[:n_chunks])
        toks = np.stack([shards.tokens[s:s + chunk_len] for s in starts]).astype(np.int64)
        return cls(toks, docs)

    @torch.no_grad()
    def refresh(self, encoder, emb, batch=1024, step=None):
        """Re-encode every chunk with the model's current encoder."""
        was = encoder.training
        encoder.eval()
        keys = [encoder(self.tokens[i:i + batch], emb)[1].float() for i in range(0, len(self), batch)]
        encoder.train(was)
        self.keys = torch.cat(keys)
        self.refreshed_at = step

    @torch.no_grad()
    def search(self, q, k, exclude=None, block=262144):
        """Top-k chunks by dot product for each query q (M, key_dim). exclude (M,): a document id per query whose
        chunks must not be returned (the document being read, so the model cannot look up the text it is about to
        predict). Returns indices (M, k') and a validity mask (M, k'), k' = min(k, N)."""
        if self.keys is None:
            raise RuntimeError("Datastore.refresh() must run before search()")
        q = q.float()
        kk = min(len(self), 4 * k if exclude is not None else k)        # fetch extra, then drop excluded ones
        best_s = best_i = None
        for b0 in range(0, len(self), block):                          # blockwise top-k keeps memory bounded
            s, i = (q @ self.keys[b0:b0 + block].T).topk(min(kk, min(block, len(self) - b0)), dim=1)
            i = i + b0
            if best_s is not None:
                s, j = torch.cat([best_s, s], 1).topk(kk, dim=1)
                i = torch.cat([best_i, i], 1).gather(1, j)
            best_s, best_i = s, i
        s, i = best_s, best_i
        if exclude is not None:
            s = s.masked_fill(self.doc_ids[i] == exclude.view(-1, 1), float("-inf"))
            s, j = s.topk(min(k, kk), dim=1)
            i = i.gather(1, j)
        return i, torch.isfinite(s)
