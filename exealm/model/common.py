"""Building blocks shared by TONN and the baseline transformer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class MLP(nn.Module):
    """Gated MLP (SwiGLU) with its own norm and residual: x + W_o(silu(a) * b), [a, b] = W_i norm(x).

    In our quick tests TONN layers alone tied a same-size transformer on text; with this MLP after every TONN layer
    TONN won, and the MLP was the part whose removal cost the most."""

    def __init__(self, d, hidden):
        super().__init__()
        self.norm = RMSNorm(d)
        self.wi = nn.Linear(d, 2 * hidden, bias=False)
        self.wo = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.wi(self.norm(x)).chunk(2, -1)
        return x + self.wo(F.silu(a) * b)


def rotate(x, base=10000.0):
    """Rotary position embedding (RoPE) on the last dim of x (B, T, g, h): channel i is paired with channel i + h/2 and
    each pair is rotated by an angle proportional to its position, so a query-key dot product depends only on how far
    apart the two positions are. (Pairing halves rather than neighbours avoids a stack and reshape: same function
    class, faster.)"""
    B, T, g, h = x.shape
    half = h // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32).view(T, 1) * freq.view(1, half)
    cos, sin = ang.cos().view(1, T, 1, half).to(x.dtype), ang.sin().view(1, T, 1, half).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
