import numpy as np
import torch

from exealm import ExeaLM, MemoryContext, get_config
from exealm.data.loader import SyntheticShards
from exealm.eval.swarm_vs_solo import compare, eval_bits, fixed_batches, main
from exealm.train.pretrain import main as pretrain


def test_one_agent_swarm_scores_the_same_as_solo():
    torch.manual_seed(0)
    m = ExeaLM(get_config("tiny", swarm_size=1))
    recs = compare(m, SyntheticShards(n_tokens=20_000), batches=2, batch=4, seq=32)
    gains = [r["bits_per_token"] for r in recs if r["condition"] == "swarm_gain_over_solo"]
    assert gains == [0.0, 0.0]


def test_split_swarm_leaks_one_token_to_the_previous_agent_and_the_score_leaves_it_out():
    """Agent 0's last target is agent 1's first input, which agent 0 hears through the board. Changing that token must
    move agent 0's last prediction only, and the default score must not count it."""
    torch.manual_seed(0)
    cfg = get_config("tiny", swarm_size=2)
    m = ExeaLM(cfg)
    m(torch.randint(0, cfg.vocab_size, (2, 32)), MemoryContext(swarm_size=2))   # set the slots' output scales
    m.eval()
    x, y, d = fixed_batches(SyntheticShards(n_tokens=20_000), 1, 2, 32, 2, "split", 0)[0]
    assert y[0, -1] in (x[1, 0], -100)                                          # -100 once the loader hides it
    x2 = x.clone()
    x2[1, 0] = (x2[1, 0] + 1) % cfg.vocab_size
    with torch.no_grad():
        a = m(x, MemoryContext(swarm_size=2))[0]
        b = m(x2, MemoryContext(swarm_size=2))[0]
    assert not torch.allclose(a[0, -1], b[0, -1])                               # the leak
    y2 = y.clone()
    y2[0, -1] = x2[1, 0]
    bits, _ = eval_bits(m, [(x, y, d)], 2, 2)
    bits_changed_target, _ = eval_bits(m, [(x, y2, d)], 2, 2)
    assert bits == bits_changed_target                                          # the last target is not scored


def test_cli_compares_swarm_solo_and_a_baseline(tmp_path):
    common = ["--preset", "tiny", "--data", "synthetic", "--steps", "30", "--eval_every", "30", "--device", "cpu"]
    pretrain(common + ["--out", str(tmp_path / "swarm")])
    pretrain(common + ["--model", "llama", "--out", str(tmp_path / "llama")])
    recs = main(["--ckpt", str(tmp_path / "swarm" / "model.pt"), "--baseline", str(tmp_path / "llama" / "model.pt"),
                 "--batches", "2", "--seq", "32", "--datastore_chunks", "200"])
    conds = {(r["policy"], r["condition"]) for r in recs}
    for p in ("split", "random"):
        assert {(p, "swarm"), (p, "solo"), (p, "baseline"), (p, "swarm_gain_over_solo")} <= conds
    sw = get_config("tiny").swarm_size
    assert all(len(r["bits_per_agent"]) == sw for r in recs if "bits_per_agent" in r)
    assert np.isfinite([r["bits_per_token"] for r in recs]).all()
