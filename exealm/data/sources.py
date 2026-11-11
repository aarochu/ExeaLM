"""Where the training data comes from, as data the scripts can use.

The plan of record is DATA.md at the repo root: about 100B pretraining tokens for the 0.1B model, the mix, the
post-training sets, and the licenses. This file mirrors it:

  * MIX: DATA.md's pretraining buckets, in billions of tokens, with the datasets in each.
  * LLMJP and K2: the sub-corpora of the NII LLM-jp Corpus v4 and the subsets of IFM's K2 Horizon release, with their
    licenses, because exealm/data/download.py fetches from these by name.

LLM-jp token counts are from its README (counted with llm-jp-tokenizer v3). Licenses were copied from the cards on
2026-11-10. Re-check a card before training on it.

If you change the mix, change DATA.md and this file together.
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

# DATA.md's pretraining mix: bucket -> (billions of tokens, datasets). A first guess, to be settled by small ablations.
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


def planned_tokens(mix=MIX):
    """Billions of tokens the mix adds up to."""
    return sum(b for b, _ in mix.values())


if __name__ == "__main__":
    print(f"pretraining mix (DATA.md): {planned_tokens():.0f}B tokens")
    for bucket, (b, sets) in MIX.items():
        print(f"  {bucket:34s} {b:>5}B  {', '.join(sets)}")
