"""The baseline every architecture change must beat: a plain Llama-style transformer (RoPE attention, SwiGLU MLP,
RMSNorm, tied embeddings) with the same width, depth and MLP as the TONN model, so the two have about the same number
of parameters. Same interface as ExeaLM: forward(idx, mem=None, targets=None) -> (logits, loss)."""
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP, RMSNorm, rotate


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, T, 3, self.heads, d // self.heads).unbind(2)
        q, k = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True)
        return x + self.out(y.transpose(1, 2).reshape(B, T, d))


class Llama(nn.Module):
    def __init__(self, cfg, head_dim=64):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        heads = max(1, cfg.d // head_dim)
        f = cfg.ffn_width or int(8 * cfg.d / 3) // 8 * 8
        self.attn = nn.ModuleList([Attention(cfg.d, heads) for _ in range(cfg.n_layers)])
        self.mlps = nn.ModuleList([MLP(cfg.d, f) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight

    def forward(self, idx, mem=None, targets=None):
        x = self.emb(idx)
        for a, m in zip(self.attn, self.mlps):
            x = m(a(x))
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss
