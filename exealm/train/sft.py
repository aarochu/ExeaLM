"""Post-training step 1: supervised fine-tuning (SFT) on conversations, so the base model learns to answer.

The loss counts only the assistant's tokens. The chat format uses the special tokens from exealm/data/tokenizer.py:

    <|im_start|><|system|>...<|im_end|><|im_start|><|user|>...<|im_end|><|im_start|><|assistant|>...<|im_end|>

Data to start from (licenses on the cards): IFM/SFT-Reasoning (instruction-following, Apache 2.0); for Japanese,
LLM-jp's instruction datasets (https://llm-jp.nii.ac.jp/en/release) -- check each license.

  TODO(post-training): the data loader (download, format, pack several conversations per window without letting them
                       attend to each other), and the training loop (reuse pretrain.py's optimizer and schedule at a
                       lower learning rate).
"""
import torch
import torch.nn.functional as F

ROLE = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>", "tool": "<|tool|>"}


def format_chat(messages, tok):
    """messages: [{"role": ..., "content": ...}]. Returns token ids and a loss mask (1 on assistant tokens)."""
    ids, mask = [], []
    start, end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    for m in messages:
        head = [start, tok.token_to_id(ROLE[m["role"]])]
        body = tok.encode(m["content"]).ids + [end]
        ids += head + body
        mask += [0] * len(head) + [int(m["role"] == "assistant")] * len(body)
    return ids, mask


def sft_loss(model, ids, mask, mem=None):
    """ids, mask (B, T+1). Mean next-token loss over positions whose TARGET is an assistant token."""
    logits = model(ids[:, :-1], mem)[0]
    tgt = ids[:, 1:].masked_fill(mask[:, 1:] == 0, -100)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
