"""Where does TONN's training time go? Times forward + backward of one layer made entirely of one circuit type, for each
type, against a standard attention block of the same width, on the chosen device.

python experiments/profile_circuits.py --device mps
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exealm import get_config  # noqa: E402
from exealm.model.baseline import Attention  # noqa: E402
from exealm.model.circuits import CIRCUITS, MEMORY_CIRCUITS  # noqa: E402
from exealm.model.tonn import TONNLayer  # noqa: E402
from exealm.train.pretrain import pick_device  # noqa: E402
from experiments.speed import sync  # noqa: E402


def bench(mod, x, dev, reps):
    for i in range(2 + reps):
        if i == 2:
            sync(dev)
            t0 = time.time()
        y = mod(x)
        y.float().pow(2).mean().backward()
    sync(dev)
    return (time.time() - t0) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    dev = pick_device(a.device)
    cfg = get_config("small-tonn")
    x = torch.randn(a.batch, a.seq, cfg.d, device=dev, requires_grad=True)
    out = {"attention block": round(bench(Attention(cfg.d, cfg.d // 64).to(dev), x, dev, a.reps), 1)}
    for t in CIRCUITS:
        if t in MEMORY_CIRCUITS:
            continue
        specs = [dict(tmpl=t, lag=1, lam0=0.5) for _ in range(cfg.slots)]
        out[t] = round(bench(TONNLayer(cfg, specs, 0).to(dev), x, dev, a.reps), 1)
    print(json.dumps(dict(device=str(dev), ms_per_layer_fwd_bwd=out), indent=1))


if __name__ == "__main__":
    main()
