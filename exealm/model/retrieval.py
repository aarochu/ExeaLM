"""RG: the retrieval circuit. The model runs its own RAG engine from scratch.

Nothing here is borrowed from an outside retriever. The model learns all four parts of RAG:

  1. its own embedding model: ChunkEncoder turns a chunk of text into per-token states and one key vector;
  2. what to search for: an RG slot's stream A, read at the end of each chunk, is its query;
  3. which results to trust: a softmax over the top-k results' scores, learned end to end;
  4. what to take from them: each position reads the retrieved tokens with attention (query from stream B), gated by
     stream C.

Retrieval happens once per chunk of `cfg.chunk` tokens. The query for chunk c is made at the last position of chunk
c - 1, so a position never retrieves with information from its own future. The datastore never returns chunks from the
document being read (exealm/memory/datastore.py), so the model cannot look up the answer it is being trained to
predict.

Every layer starts with one RG slot and the search can never remove the last one (exealm/model/search.py). Without a
datastore (for example in unit tests or a plain language-model baseline) RG slots output zeros.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP, RMSNorm


class ChunkEncoder(nn.Module):
    """The model's own retrieval embedding model, shared by every RG slot in every layer, so there is one datastore
    index. It reads the model's token embeddings (passed in, not owned, so they stay tied)."""

    def __init__(self, cfg):
        super().__init__()
        e = cfg.enc_dim
        self.proj = nn.Linear(cfg.d, e, bias=False)
        self.pos = nn.Parameter(torch.zeros(cfg.rag_chunk_len, e))
        self.mlp = MLP(e, 2 * e)
        self.norm = RMSNorm(e)
        self.key = nn.Linear(e, cfg.h, bias=False)

    def forward(self, tokens, emb):
        """tokens (N, Lc) -> per-token states (N, Lc, enc_dim) and unit-length keys (N, h)."""
        z = self.norm(self.mlp(self.proj(emb(tokens)) + self.pos[: tokens.shape[1]]))
        return z, F.normalize(self.key(z.mean(1)), dim=-1)


class RetrievalCircuit(nn.Module):
    """Runs every RG slot of one layer. Its own parameters are shared by the layer's RG slots; the slots differ by their
    input rows (each slot learns its own query)."""

    def __init__(self, cfg):
        super().__init__()
        self.chunk, self.topk, self.h = cfg.chunk, cfg.rag_topk, cfg.h
        self.kv = nn.Linear(cfg.enc_dim, 2 * cfg.h, bias=False)            # retrieved token states -> keys, values
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))           # sharpness of "which result to trust"
        self.record = False         # if True, keep what the last forward pass retrieved in self.last (for inspection)
        self.last = None

    def forward(self, S, mem):
        A, Q, G = S                                                        # each (B, T, g, h)
        B, T, g, h = A.shape
        C = self.chunk
        n = -(-T // C)
        ds = None if mem is None else mem.datastore
        self.last = None
        if ds is None or ds.keys is None or mem.encoder is None or n < 2:
            return A * 0.0                                                 # zeros, still attached to the graph
        ends = torch.arange(1, n, device=A.device) * C - 1                 # last position of chunks 0 .. n-2
        q = F.normalize(A[:, ends].float(), dim=-1)                        # (B, n-1, g, h): one query per chunk
        M = B * (n - 1) * g
        qf = q.reshape(M, h)
        docs = None
        if mem.doc_ids is not None:
            docs = mem.doc_ids[:, ends].unsqueeze(-1).expand(B, n - 1, g).reshape(M)
        idx, valid = ds.search(qf.detach(), self.topk, exclude=docs)       # candidates from the stored index
        k = idx.shape[1]
        states, keys = mem.encoder(ds.tokens[idx].reshape(M * k, -1), mem.emb)    # re-encoded live: gradients flow
        Lc = states.shape[1]
        score = self.log_tau.exp() * (qf.unsqueeze(1) * keys.view(M, k, h).float()).sum(-1)
        logp = torch.log_softmax(score.masked_fill(~valid, -1e4), dim=1)   # (M, k)
        if self.record:                                                    # see exealm/eval/show_retrieval.py
            self.last = dict(idx=idx.view(B, n - 1, g, k).cpu(), weight=logp.detach().exp().view(B, n - 1, g, k).cpu(),
                             valid=valid.view(B, n - 1, g, k).cpu())
        kr, vr = self.kv(states).view(M, k, Lc, 2, h).float().unbind(3)    # (M, k, Lc, h)
        Qc = F.pad(Q, (0, 0, 0, 0, 0, n * C - T))[:, C:]                   # read queries of chunks 1 .. n-1
        Qc = Qc.reshape(B, n - 1, C, g, h).permute(0, 1, 3, 2, 4).reshape(M, C, h).float()
        att = torch.einsum("mch,mkrh->mckr", Qc, kr) / math.sqrt(h) + logp.view(M, 1, k, 1)
        att = att.masked_fill(~valid.view(M, 1, k, 1), -1e4)
        att = att.reshape(M, C, k * Lc).softmax(-1).view(M, C, k, Lc)
        y = torch.einsum("mckr,mkrh->mch", att, vr) * valid.any(1).view(M, 1, 1)
        y = y.view(B, n - 1, g, C, h).permute(0, 1, 3, 2, 4).reshape(B, (n - 1) * C, g, h)
        y = F.pad(y, (0, 0, 0, 0, C, 0))[:, :T]                            # chunk 0 retrieves nothing
        return y.to(A.dtype) * F.silu(G)
