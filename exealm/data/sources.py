"""Where the training data comes from, as data the scripts can use.

The plan of record is DATA.md at the repo root: about 100B pretraining tokens for the 0.1B model, the mix, the
post-training sets, and the licenses. This file mirrors it:

  * Two candidate pretraining mixes, both about 100B tokens, to be settled by small ablations:
      MIX     candidate A, DATA.md's mix: English-heavier (5% multilingual), buckets of datasets
      MIX_V0  candidate B, mix proposal v0: Japanese-heavier (about 21% Japanese), mostly LLM-jp and K2 Horizon,
              written for a 2T-token pool (MIX_V0_2T) and scaled down to 100B with every share kept
  * LLMJP and K2: the sub-corpora of the NII LLM-jp Corpus v4 and the subsets of IFM's K2 Horizon release, with their
    licenses, because exealm/data/download.py fetches from these by name.

LLM-jp token counts are from its README (counted with llm-jp-tokenizer v3). Licenses were copied from the cards on
2026-11-10. Re-check a card before training on it.

If you change a mix, change DATA.md (or docs/data.md for v0) and this file together.
"""

LLMJP_REPO = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4"
LLMJP_API = "https://gitlab.llm-jp.nii.ac.jp/api/v4/projects/datasets%2Fllm-jp-corpus-v4/repository/tree"
LLMJP_RAW = LLMJP_REPO + "/-/raw/main/"

CC_TERMS = " + Common Crawl terms of use"

# LLM-jp Corpus v4 sub-corpora: path -> (billions of tokens, license)
LLMJP = {
    "ja/ja_aozorabunko": (0.1, "CC BY 4.0"),
    "ja/ja_cc/level0": (None, "ODC-BY" + CC_TERMS),      # ja_cc is 223B in total over level0..2
    "ja/ja_cc/level1": (None, "ODC-BY" + CC_TERMS),
    "ja/ja_cc/level2": (None, "ODC-BY" + CC_TERMS),
    "ja/ja_e-gov": (0.08, "CC BY 4.0"),
    "ja/ja_fineweb-2": (236, "ODC-BY" + CC_TERMS),
    "ja/ja_kaken": (0.9, "CC BY 4.0"),
    "ja/ja_kokkai_giji": (0.8, "CC BY 4.0"),
    "ja/ja_nwc2010": (16, "CC BY 4.0"),
    "ja/ja_patent": (68, "CC BY 4.0"),
    "ja/ja_sip_comprehensive_html": (11, "CC BY 4.0"),
    "ja/ja_sip_comprehensive_pdf": (30, "CC BY 4.0"),
    "ja/ja_warp_html": (0.8, "CC BY 4.0"),
    "ja/ja_warp_pdf": (58, "CC BY 4.0"),
    "ja/ja_wiki": (1, "CC BY-SA 3.0"),
    "en/en_dolma/books": (5, "ODC-BY"),
    "en/en_dolma/pes2o": (63, "ODC-BY"),
    "en/en_dolma/reddit": (83, "ODC-BY"),
    "en/en_dolma/wiki": (4, "ODC-BY"),
    "en/en_dolmino/stackexchange": (1.5, "CC BY-SA 2.5/3.0/4.0"),
    "en/en_finemath/4plus": (10, "ODC-BY" + CC_TERMS),
    "en/en_fineweb/eduscore2": (6200, "ODC-BY" + CC_TERMS),
    "en/en_fineweb/rest": (11400, "ODC-BY" + CC_TERMS),
    "en/en_gsm8k": (0.003, "MIT"),
    "en/en_mathpile": (9, "CC BY-SA 4.0"),
    "en/en_olmo/algebraicstack": (13, "ODC-BY" + CC_TERMS),
    "en/en_olmo/arxiv": (22, "ODC-BY" + CC_TERMS),
    "en/en_olmo/openwebmath": (13, "ODC-BY" + CC_TERMS),
    "en/en_wiki": (5, "CC BY-SA 3.0"),
    "ko/ko_fineweb2": (52, "ODC-BY" + CC_TERMS),
    "ko/ko_wiki": (0.3, "CC BY-SA 3.0"),
    "zh/zh_fineweb2": (788, "ODC-BY" + CC_TERMS),
    "zh/zh_wiki": (0.8, "CC BY-SA 3.0"),
    "code/code_olmo-starcoder": (104, "ODC-BY" + CC_TERMS),
    "code/code_stack": (114, "each repository's own license"),
}

# K2 Horizon release (IFM): repo -> (license, subsets). Token counts: not published per subset yet -- counting them
# is a good first issue.
K2 = {
    "IFM/TxT360-v2": ("CC BY 4.0", ["web-high-nltk-qa", "web-high-medium", "txt360-qa"]),
    "IFM/Pretrain-Behaviors": ("Apache 2.0", ["reasoning", "general", "planning", "data-science", "games", "other",
                                              "format-rewrites"]),
    "IFM/Math-Reasoning": ("Apache 2.0", ["math-thinking-qwen", "math-thinking-oss", "math-rewrite", "math-dialogue",
                                          "socratic-math-dialogue"]),
    "IFM/Code-Reasoning": ("Apache 2.0", ["code-no-thinking", "code-no-thinking-hard", "code-thinking-v1",
                                          "code-thinking-hard", "code-thinking-v0", "codeio", "code-task-synthesis"]),
    "IFM/SFT-Reasoning": ("Apache 2.0", ["instruction-following", "3efforts-pretrain"]),   # mostly for post-training
}

# Candidate A, DATA.md's pretraining mix: bucket -> (billions of tokens, datasets).
MIX = {
    "General web, English": (47.5, ["mlfoundations/dclm-baseline-1.0", "HuggingFaceFW/fineweb-edu",
                                     "IFM/TxT360-v2 (web subsets)", "LLM-jp Corpus v4 (en)"]),
    "Reference, academic, books, PDFs": (10, ["LLM360/TxT360 (non-web subsets)", "HuggingFaceFW/finepdfs-edu",
                                              "allenai/peS2o", "wikimedia/wikipedia"]),
    "Code": (15, ["HuggingFaceTB/stack-edu", "bigcode/the-stack-v2-dedup", "IFM/Code-Reasoning (no-thinking subsets)"]),
    "Math and science": (10, ["HuggingFaceTB/finemath", "open-web-math/open-web-math", "IFM/MegaMath",
                              "IFM/Math-Reasoning (rewrite and dialogue subsets)"]),
    "Reasoning and synthetic": (5, ["IFM/Pretrain-Behaviors", "HuggingFaceTB/cosmopedia"]),
    "Multilingual": (5, ["LLM-jp Corpus v4 (ja, zh, ko)", "HuggingFaceFW/fineweb-2"]),
    "QA and instruction-style text": (2.5, ["IFM/TxT360-v2 (txt360-qa)", "IFM/SFT-Reasoning (3efforts-pretrain)"]),
    "Annealing (last 5% of training)": (5, ["allenai/dolma3_dolmino_mix-100B-1125", "FineMath-4+",
                                            "top-scored FineWeb-Edu", "K2 Horizon math and code reasoning"]),
}


# Candidate B, mix proposal v0, as written for a 2T-token pool, in billions of tokens. "x2" means two passes over a small,
# high-quality source. Nothing here is final.
MIX_V0_2T = {
    # Japanese (LLM-jp has ~688B Japanese tokens in total; take the cleanest, repeat the best)
    "ja/ja_wiki": "1 x4",
    "ja/ja_aozorabunko": "0.1 x4",
    "ja/ja_kaken": "0.9 x2",
    "ja/ja_kokkai_giji": "0.8 x2",
    "ja/ja_e-gov": "0.08 x2",
    "ja/ja_fineweb-2": "200",
    "ja/ja_cc/level0": "100",
    "ja/ja_warp_pdf": "40",
    "ja/ja_warp_html": "0.8",
    "ja/ja_sip_comprehensive_html": "11",
    "ja/ja_sip_comprehensive_pdf": "20",
    "ja/ja_patent": "20",
    "ja/ja_nwc2010": "10",
    # English: the best-scoring educational web, plus books, papers, encyclopedias
    "en/en_fineweb/eduscore2": "900",
    "en/en_dolma/books": "5 x2",
    "en/en_dolma/pes2o": "40",
    "en/en_dolma/wiki": "4 x2",
    "en/en_wiki": "5 x2",
    "en/en_dolmino/stackexchange": "1.5 x2",
    # math and code
    "en/en_finemath/4plus": "10 x2",
    "en/en_olmo/openwebmath": "13",
    "en/en_olmo/algebraicstack": "13",
    "en/en_mathpile": "9",
    "en/en_olmo/arxiv": "15",
    "code/code_stack": "80",
    "code/code_olmo-starcoder": "60",
    # K2 Horizon: high-quality web and QA, plus behavior and reasoning data (mostly saved for the final decay phase)
    "IFM/TxT360-v2": "250",
    "IFM/Pretrain-Behaviors": "60",
    "IFM/Math-Reasoning": "40",
    "IFM/Code-Reasoning": "40",
}


def _billions(v):
    """'10 x2' -> 20.0 (tokens times passes)."""
    parts = v.split()
    return float(parts[0]) * (int(parts[1][1:]) if len(parts) > 1 else 1)


def scale_v0(total=100.0):
    """Mix v0 scaled to `total` billion tokens, every source keeping its share (passes included)."""
    raw = {k: _billions(v) for k, v in MIX_V0_2T.items()}
    f = total / sum(raw.values())
    return {k: round(b * f, 3) for k, b in raw.items()}


MIX_V0 = scale_v0(100.0)


def planned_tokens(mix=MIX):
    """Billions of tokens a mix adds up to (either format)."""
    return sum(v[0] if isinstance(v, tuple) else v for v in mix.values())


if __name__ == "__main__":
    print(f"candidate A, DATA.md's mix: {planned_tokens(MIX):.0f}B tokens")
    for bucket, (b, sets) in MIX.items():
        print(f"  {bucket:34s} {b:>5}B  {', '.join(sets)}")
    ja = sum(b for k, b in MIX_V0.items() if k.startswith("ja/"))
    print(f"\ncandidate B, mix v0 scaled to {planned_tokens(MIX_V0):.0f}B tokens ({ja:.0f}% Japanese)")
    for k, b in MIX_V0.items():
        lic = LLMJP[k][1] if k in LLMJP else K2[k][0]
        print(f"  {k:32s} {b:>7.2f}B  {lic}")
