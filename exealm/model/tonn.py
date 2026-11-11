"""ExeaLM: a TONN language model with retrieval (RG) and agent-board (AG) circuits built into every layer.

    tokens -> embedding -> [TONN layer -> gated MLP] x n_layers -> norm -> next-token logits

A TONN layer reads the residual stream into `slots` small circuits through ONE input projection (3 streams of h
channels per slot), runs each circuit, scales every slot's output to unit size, and writes back through ONE output
projection. Which circuit sits in which slot is not fixed by hand: the search (search.py) measures what each kind of
circuit is worth and replaces the useless ones while the model trains.
"""
import math
import random
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..memory import MemoryContext
from . import circuits as Cc
from .board import BoardCircuit
from .common import MLP, RMSNorm
from .retrieval import ChunkEncoder, RetrievalCircuit


# --------------------------------------------------------------------------------------------- choosing circuits
def rand_spec(rng, cfg, allowed=None, weights=None):
    """A random circuit: type (uniform, or in proportion to `weights`), lag and starting decay."""
    allowed = list(allowed or cfg.menu)
    if weights:
        w = [max(0.0, weights.get(t, 0.0)) for t in allowed]
        t = rng.choices(allowed, w)[0] if sum(w) > 0 else rng.choice(allowed)
    else:
        t = rng.choice(allowed)
    return dict(tmpl=t, lag=rng.choice(cfg.lags), lam0=rng.choice([0.2, 0.5, 0.9, -0.5]))


def initial_specs(cfg, seed=None):
    """Every layer starts with its pinned circuits (one RG and one AG by default); the other slots are random draws
    from the rest of the menu. The search may later add more RG or AG slots, never fewer than the pinned number."""
    rng = random.Random(cfg.seed if seed is None else seed)
    fill = [t for t in cfg.menu if t not in cfg.pinned]
    out = []
    for _ in range(cfg.n_layers):
        row = [dict(tmpl=t, lag=0, lam0=0.5) for t, c in cfg.pinned.items() for _ in range(c)]
        if len(row) > cfg.slots:
            raise ValueError("more pinned circuits than slots")
        row += [rand_spec(rng, cfg, fill) for _ in range(cfg.slots - len(row))]
        out.append(row)
    return out


def slot_cost(spec, cfg, T):
    """Rough multiply-adds per token of one slot (its share of the projections plus its circuit). The search compares
    worth per cost, so a cheap circuit that helps a little can beat an expensive one that helps a little more."""
    t, d, h = spec["tmpl"], cfg.d, cfg.h
    proj = 3 * d * h + h * d
    if t == "RP":
        return proj + 2 * h * T
    if t == "RG":
        e, k, Lc = cfg.enc_dim, cfg.rag_topk, cfg.rag_chunk_len
        return proj + k * Lc * (d * e + 4 * e * e + 2 * e * h) / cfg.chunk + 2 * k * Lc * h
    if t == "AG":
        return proj + 2 * h * cfg.swarm_size * T / cfg.chunk
    return proj + 6 * h


# --------------------------------------------------------------------------------------------- the TONN layer
class TONNLayer(nn.Module):
    def __init__(self, cfg, specs, index):
        super().__init__()
        d, K, h = cfg.d, len(specs), cfg.h
        self.cfg, self.index, self.K, self.h = cfg, index, K, h
        self.norm = RMSNorm(d)
        self.inp = nn.Linear(d, K * 3 * h)
        self.out = nn.Linear(K * h, d, bias=False)
        self.lam = nn.Parameter(torch.zeros(K, h))        # decays, through tanh(3 lam)
        self.beta = nn.Parameter(torch.ones(K))           # attention sharpness
        self.register_buffer("mask", torch.ones(K))       # 0 switches a slot off (worth measurement)
        self.register_buffer("scale", torch.zeros(K))     # running output size per slot (0: not measured yet)
        self.rg = RetrievalCircuit(cfg)
        self.ag = BoardCircuit(cfg)
        nn.init.normal_(self.out.weight, std=0.5 / math.sqrt(K * h))
        self.specs = [None] * K
        for k, s in enumerate(specs):
            self.set_spec(k, s, reinit=False)

    def set_spec(self, k, spec, reinit=True, out_scale=0.1):
        """Put a circuit in slot k. reinit: fresh input rows and small output columns for this slot only."""
        self.specs[k] = dict(spec)
        self._groups = None
        with torch.no_grad():
            self.scale[k] = 0.0
            self.lam[k].fill_(math.atanh(max(-0.97, min(0.97, spec["lam0"]))) / 3)
            self.beta[k].fill_(1.0)
            if reinit:
                r = slice(k * 3 * self.h, (k + 1) * 3 * self.h)
                nn.init.kaiming_uniform_(self.inp.weight[r], a=math.sqrt(5))
                self.inp.bias[r].zero_()
                c = slice(k * self.h, (k + 1) * self.h)
                nn.init.normal_(self.out.weight[:, c], std=out_scale / math.sqrt(self.K * self.h))

    def groups(self, device):
        if self._groups is None or self._groups[0] != device:
            g = {}
            for k, s in enumerate(self.specs):
                g.setdefault(s["tmpl"], []).append(k)
            lags = torch.tensor([s["lag"] for s in self.specs])
            self._groups = (device, {t: (torch.tensor(ks, device=device), lags[ks].to(device)) for t, ks in g.items()})
        return self._groups[1]

    def forward(self, x, mem=None):
        B, T, _ = x.shape
        u = self.inp(self.norm(x)).view(B, T, self.K, 3, self.h)
        lam = torch.tanh(self.lam * 3)
        outs, order = [], []
        for t, (idx, lags) in self.groups(x.device).items():
            S = [u[:, :, idx, j] for j in range(3)]
            if t == "RG":
                y = self.rg(S, mem)
            elif t == "AG":
                y = self.ag(S, mem, self.index, idx.tolist())
            else:
                y = Cc.run(t, S, lags, lam[idx], self.beta[idx])
            outs.append(y)
            order.append(idx)
        y = torch.cat(outs, 2).index_select(2, torch.argsort(torch.cat(order)))   # back to slot order
        y = self._unit_scale(y) * self.mask.view(1, 1, -1, 1)
        return x + self.out(y.reshape(B, T, self.K * self.h))

    def _unit_scale(self, y):
        """Divide each slot by the size its output had on PAST batches (a running average), so a prediction never
        depends on other rows of the same batch and no slot drowns out the others."""
        with torch.no_grad():
            rms = y.detach().float().pow(2).mean(dim=(0, 1, 3)).sqrt()
            used = torch.where(self.scale > 0, self.scale, rms)
            if self.training:
                self.scale.copy_(torch.where(self.scale > 0, 0.99 * self.scale + 0.01 * rms, rms))
        return y / (used.view(1, 1, -1, 1).to(y.dtype) + 1e-3)


# --------------------------------------------------------------------------------------------- the model
class ExeaLM(nn.Module):
    def __init__(self, cfg: ModelConfig, specs=None):
        super().__init__()
        self.cfg = cfg
        specs = specs or initial_specs(cfg)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList([TONNLayer(cfg, specs[i], i) for i in range(cfg.n_layers)])
        f = cfg.ffn_width
        self.mlps = nn.ModuleList([MLP(cfg.d, f) for _ in range(cfg.n_layers)]) if f else None
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight
        uses_rg = "RG" in cfg.menu or any(s["tmpl"] == "RG" for row in specs for s in row)
        self.encoder = ChunkEncoder(cfg) if uses_rg else None

    @property
    def specs(self):
        return [[dict(s) for s in lay.specs] for lay in self.layers]

    def forward(self, idx, mem=None, targets=None):
        """idx (B, T) token ids. Returns (logits (B, T, V), loss or None)."""
        if mem is None:
            mem = MemoryContext()
        mem = replace(mem, encoder=self.encoder, emb=self.emb)
        x = self.emb(idx)
        for i, lay in enumerate(self.layers):
            x = lay(x, mem)
            if self.mlps is not None:
                x = self.mlps[i](x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def architecture(self):
        """How many slots of each circuit type each layer holds."""
        rows = []
        for lay in self.layers:
            c = {}
            for s in lay.specs:
                c[s["tmpl"]] = c.get(s["tmpl"], 0) + 1
            rows.append(c)
        return rows

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, mem=None):
        """Plain sampling. It re-reads the whole context for every new token, which is slow.
        TODO(speed): most TONN circuits are recurrences, so generation can carry a small state per slot and cost the
        same for every new token (docs/ideas.md)."""
        for _ in range(max_new_tokens):
            logits = self(idx, mem)[0][:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = logits.topk(top_k)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], 1)
            if mem is not None and mem.doc_ids is not None:
                mem = replace(mem, doc_ids=torch.cat([mem.doc_ids, mem.doc_ids[:, -1:]], 1))
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())
