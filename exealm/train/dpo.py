"""Post-training step 2: preference tuning with DPO (Direct Preference Optimization).

Given a prompt, a preferred answer and a rejected one, raise the model's likelihood of the preferred answer relative to
the rejected one, measured against a frozen copy of the SFT model, with no reward model.

    loss = -log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))])

  TODO(post-training): preference data (English and Japanese; check licenses) and the training loop.
"""
import torch.nn.functional as F


def dpo_loss(logp_chosen, logp_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Each argument (N,): summed log-prob of the answer tokens under the policy or the reference model."""
    return -F.logsigmoid(beta * ((logp_chosen - ref_chosen) - (logp_rejected - ref_rejected))).mean()
