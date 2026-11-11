import json

import torch

from exealm.train.dpo import dpo_loss
from exealm.train.grpo import group_advantages, grpo_loss, reward_exact_answer
from exealm.train.pretrain import load_model, main


def test_pretraining_smoke_run_lowers_the_loss_and_saves(tmp_path):
    main(["--preset", "tiny", "--data", "synthetic", "--steps", "120", "--eval_every", "60", "--search_every", "30",
          "--datastore_chunks", "500", "--refresh_every", "50", "--device", "cpu", "--out", str(tmp_path)])
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    evals = [json.loads(l) for l in lines]
    bits = [e["val_bits_per_token"] for e in evals if e.get("event") == "eval"]
    assert bits[-1] < 8.0 and bits[-1] < bits[0] + 1e-6
    assert any(e.get("event") == "search" for e in evals)
    model, ck = load_model(str(tmp_path / "model.pt"))
    assert ck["step"] == 120


def test_grpo_advantages_and_loss():
    adv = group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]), group_size=4)
    assert torch.allclose(adv[4:], torch.zeros(4), atol=1e-4)            # a group that all agree learns nothing
    assert adv[0] > 0 > adv[1]
    logp = torch.zeros(8, 5, requires_grad=True)
    loss = grpo_loss(logp, logp.detach(), logp.detach(), adv, torch.ones(8, 5))
    loss.backward()
    assert logp.grad[0].sum() < 0 < logp.grad[1].sum()                   # raise good answers, lower bad ones
    assert reward_exact_answer("so the answer is 42", 42) == 1.0


def test_dpo_prefers_the_chosen_answer():
    lc = torch.zeros(3, requires_grad=True)
    dpo_loss(lc, torch.zeros(3), torch.zeros(3), torch.zeros(3)).backward()
    assert (lc.grad < 0).all()
