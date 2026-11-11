"""AG: the agent-board circuit. Agents (copies of the model) talk to each other through a shared read-write board.

Once per chunk of `cfg.chunk` tokens, every AG slot POSTS one message to its channel of the board:
    value     = the chunk's average of stream A          (what the message says)
    key       = the chunk's average of stream B, unit length   (what it is about)
    strength  = sigmoid(w . average of stream C + b)      (how much the agent wants it read; the model can stay quiet)
and at every position it READS the channel with attention: query = stream B at that position, over the messages the
other agents of its swarm (and, if cfg.board_read_self, itself) posted in EARLIER chunks. A "nothing" entry is always available, so an
agent can choose to read nothing. The result is gated by silu(stream C).

Training: the rows of a batch are grouped into swarms of cfg.swarm_size consecutive rows. Each swarm has its own in-batch
board, so posting and reading both get gradients, and swarms never hear each other (docs/swarms.md). Inference and multi-agent post-training: a persistent AgentBoard
(exealm/memory/board.py) adds messages from agents that ran earlier or elsewhere, and new posts are written to it.

Every layer starts with one AG slot and the search can never remove the last one (exealm/model/search.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoardCircuit(nn.Module):
    """Runs every AG slot of one layer. Each AG slot is its own channel of the board."""

    def __init__(self, cfg):
        super().__init__()
        self.chunk, self.read_self = cfg.chunk, cfg.board_read_self
        self.post_w = nn.Parameter(torch.zeros(cfg.h))
        self.post_b = nn.Parameter(torch.zeros(()))
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, S, mem, layer, slots):
        A, K, G = S                                                       # each (B, T, g, h)
        B, T, g, h = A.shape
        C = self.chunk
        n = -(-T // C)
        sw = mem.swarm_size if mem is not None else 1                 # agents per swarm
        if sw < 1 or B % sw:
            sw = 1
        ns = B // sw                                                    # swarms in this batch
        chunks = lambda u: F.pad(u, (0, 0, 0, 0, 0, n * C - T)).view(B, n, C, g, h).float()
        val = chunks(A).mean(2)                                           # (B, n, g, h) one post per chunk
        key = F.normalize(chunks(K).mean(2), dim=-1)
        log_w = F.logsigmoid(chunks(G).mean(2) @ self.post_w.float() + self.post_b.float())   # (B, n, g)
        q = F.normalize(K.float(), dim=-1)
        tau = self.log_tau.exp()

        # in-batch board: every swarm member's posts, visible from the next chunk on
        E = sw * n
        keyT, valT = key.view(ns, E, g, h), val.view(ns, E, g, h)
        lin = tau * torch.einsum("atsgh,aegh->atsge", q.view(ns, sw, T, g, h), keyT) \
            + log_w.view(ns, 1, 1, E, g).permute(0, 1, 2, 4, 3)
        e_chunk = torch.arange(n, device=A.device).repeat(sw)
        e_author = torch.arange(sw, device=A.device).repeat_interleave(n)
        r_chunk = torch.arange(T, device=A.device) // C
        vis = e_chunk.view(1, 1, 1, 1, E) < r_chunk.view(1, 1, T, 1, 1)
        if not self.read_self:
            vis = vis & (e_author.view(1, 1, 1, 1, E) != torch.arange(sw, device=A.device).view(1, sw, 1, 1, 1))
        lin = lin.masked_fill(~vis, float("-inf")).reshape(B, T, g, E)
        parts = [lin]

        # persistent board: messages from other agents that ran before (inference / multi-agent post-training)
        ext = None
        if mem is not None and mem.board is not None:
            ext = mem.board.read(layer, slots, A.device)
        if ext is not None:
            eK, eV, eW, eA, ok = ext                                      # (g, E2, h), (g, E2, h), (g, E2) x3
            lex = tau * torch.einsum("btgh,geh->btge", q, eK) + eW.view(1, 1, g, -1)
            seen = ok.view(1, g, -1).expand(B, g, -1)
            if mem.agent_ids is not None:                                 # an agent's own old posts come from context
                seen = seen & (eA.view(1, g, -1) != mem.agent_ids.view(B, 1, 1))
            parts.append(lex.masked_fill(~seen.view(B, 1, g, -1), float("-inf")))
        parts.append(lin.new_zeros(B, T, g, 1))                          # "read nothing"
        att = torch.cat(parts, -1).softmax(-1)
        y = torch.einsum("atsge,aegh->atsgh", att[..., :E].reshape(ns, sw, T, g, E), valT).reshape(B, T, g, h)
        if ext is not None:
            y = y + torch.einsum("btge,geh->btgh", att[..., E:-1], eV)

        if mem is not None and mem.board is not None and mem.write_board:
            authors = mem.agent_ids if mem.agent_ids is not None else torch.arange(B)
            mem.board.post(layer, slots, key, val, log_w, authors, n_complete=T // C)
        return y.to(A.dtype) * F.silu(G)
