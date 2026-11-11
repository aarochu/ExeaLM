"""Post-training step 3: reinforcement learning with GRPO (Group Relative Policy Optimization, from DeepSeekMath and
used for DeepSeek-R1).

For each prompt, sample a GROUP of answers, score each with a reward (for example: is the final answer right?), and
push up the answers that scored above their group's average. No separate value model is needed, which is what makes
GRPO cheap enough for us.

    advantage_i = (reward_i - mean(group rewards)) / std(group rewards)
    loss = - mean over tokens of min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) + beta * KL(policy || reference)

The loss below is complete and tested. What is left (a good project for a contributor):
  TODO(post-training): the loop: sample groups with model.generate, score them, compute old/reference log-probs,
                       several optimizer steps per batch.
  TODO(post-training): reward functions for math (exact answer), code (unit tests), format, Japanese tasks.
  TODO(post-training): multi-agent rewards: a swarm of agents shares a board and is rewarded together for solving a
                       task none could solve alone. This is how the AG circuits learn to talk (docs/swarms.md).
"""
import re

import torch


def group_advantages(rewards, group_size, eps=1e-6):
    """rewards (N,) with N = prompts * group_size, grouped consecutively."""
    r = rewards.float().view(-1, group_size)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def grpo_loss(logp, logp_old, logp_ref, adv, mask, clip=0.2, beta=0.04):
    """logp, logp_old, logp_ref: (N, L) log-probs of the sampled completion tokens under the current, sampling-time and
    reference (frozen SFT) models; adv (N,); mask (N, L) 1 on completion tokens."""
    ratio = torch.exp(logp - logp_old)
    a = adv.view(-1, 1)
    pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    kl = torch.exp(logp_ref - logp) - (logp_ref - logp) - 1            # unbiased, always >= 0
    per_tok = pg + beta * kl
    mask = mask.float()
    return ((per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()


def token_logprobs(model, ids, mem=None):
    """Log-prob of each token of ids (N, T) given the ones before it: (N, T-1)."""
    logits = model(ids[:, :-1], mem)[0].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def reward_exact_answer(completion, answer):
    """1 if the last number in the completion equals the reference answer."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", completion.replace(",", ""))
    return float(bool(nums) and nums[-1] == str(answer).strip())
