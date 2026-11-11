"""The two running-sum recurrences most TONN circuits are made of, in plain PyTorch.

    decay_scan:  s_t = lam * s_{t-1} + u_t     (one fixed decay per channel, may be negative)
    gated_scan:  s_t = g_t * s_{t-1} + u_t     (a decay that changes with the input, 0 < g_t < 1)

Both are exact and causal. They are written for clarity, not speed: a fused GPU kernel (Triton or CUDA) for these two
is one of the most useful contributions anyone can make (docs/ideas.md).
"""
import torch


def decay_scan(u, lam):
    """u (B, T, C), lam (C,) with |lam| < 1. One causal FFT convolution with the kernel lam^n, in float32."""
    B, T, C = u.shape
    dtype = u.dtype
    u, lam = u.float(), lam.float()
    n = torch.arange(T, device=u.device, dtype=torch.float32).view(T, 1)
    kern = lam.sign().view(1, C) ** n * lam.abs().clamp_min(1e-6).view(1, C) ** n       # (T, C): lam^n, sign kept
    nfft = 1 << (2 * T - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(u, n=nfft, dim=1) * torch.fft.rfft(kern, n=nfft, dim=0).unsqueeze(0),
                        n=nfft, dim=1)[:, :T]
    return y.to(dtype)


def gated_scan(u, g, chunk=64):
    """u, g (B, T, C), 0 < g < 1. Exact, in chunks: inside a chunk the weight of u_j at time t is exp(L_t - L_j) with L
    the running sum of log g from the chunk start; the state carried in from the previous chunk is scaled by exp(L_t).
    Memory is B * chunk * chunk * C per chunk, not B * T * T * C."""
    B, T, C = u.shape
    dtype = u.dtype
    u, lg = u.float(), torch.log(g.float().clamp_min(1e-6))
    s = u.new_zeros(B, C)
    outs = []
    for c0 in range(0, T, chunk):
        uc, L = u[:, c0:c0 + chunk], torch.cumsum(lg[:, c0:c0 + chunk], 1)
        n = uc.shape[1]
        keep = torch.ones(n, n, device=u.device, dtype=torch.bool).tril().view(1, n, n, 1)
        W = torch.exp((L.unsqueeze(2) - L.unsqueeze(1)).clamp(max=0)) * keep            # (B, t, j, C)
        y = torch.einsum("btjc,bjc->btc", W, uc) + torch.exp(L) * s.unsqueeze(1)
        s = y[:, -1]
        outs.append(y)
    return torch.cat(outs, 1).to(dtype)
