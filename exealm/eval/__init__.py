"""Evaluation.

Built in: bits per token / per byte on held-out data (exealm/train/pretrain.py reports it during training).

Planned (each is a good first project):
  * English: HellaSwag, ARC-Easy, PIQA, WinoGrande, LAMBADA through EleutherAI's lm-evaluation-harness
    (https://github.com/EleutherAI/lm-evaluation-harness). A 0.1B model scores near chance on MMLU, so don't use it
    to compare small models.
  * Japanese: llm-jp-eval (https://github.com/llm-jp/llm-jp-eval), JCommonsenseQA, JNLI.
  * Retrieval (RG): questions whose answers are in the datastore but not in the prompt; compare with RG switched off.
  * Agents (AG): team tasks where each agent sees part of the information; score with and without the board.
  * Speed: tokens per second for training and generation against the Llama baseline of the same size.

  TODO(evals): an lm-evaluation-harness model wrapper (exealm/eval/harness.py) around ExeaLM.
"""
