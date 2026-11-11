# Data pipeline

What we train on and why (about 100B tokens for the 0.1B model, the mix, every dataset link and license) is in
[DATA.md](../DATA.md) at the repo root. This page is about the scripts in `exealm/data/`, which all run on a laptop.

| Script | What it does |
|---|---|
| `sources.py` | DATA.md's mix as data, plus the LLM-jp Corpus v4 sub-corpora and K2 Horizon subsets with their licenses. `python -m exealm.data.sources` prints the mix. |
| `download.py` | Fetches a sample: LLM-jp files directly (no account), or any Hugging Face dataset by streaming. |
| `quality.py` | Rule-based filters (with separate rules for Japanese) and exact deduplication. |
| `tokenizer.py` | Trains a byte-level BPE tokenizer on a sample. |
| `pack.py` | Tokenizes documents into one flat token file per split, keeping document boundaries. Retrieval needs them so it can skip the document being read. |
| `loader.py` | Training batches: random windows, the document id of each position, and swarms ([swarms.md](swarms.md)). |

## Try it on a sample

```bash
# 1. download a sample (a few hundred MB)
python -m exealm.data.download llmjp --subset ja/ja_wiki --max_files 1 --out data/raw
python -m exealm.data.download hf --repo HuggingFaceFW/fineweb-edu --config sample-10BT --max_rows 20000 --out data/raw

# 2. filter + exact dedupe
python -m exealm.data.quality data/raw/*.jsonl.gz --out data/filtered

# 3. tokenizer (on a sample of the mix)
python -m exealm.data.tokenizer data/filtered/*.jsonl.gz --vocab 49152 --out tokenizer

# 4. pack into a flat token file with document boundaries (hold out a file as validation)
python -m exealm.data.pack data/filtered/<train files> --tokenizer tokenizer --out data/packed/train
python -m exealm.data.pack data/filtered/<val file>    --tokenizer tokenizer --out data/packed/val

# 5. train the 17M "small" model on it (laptop or a free Colab GPU)
python -m exealm.train.pretrain --preset small --data data/packed/train --val data/packed/val --seq 256 --steps 2000
```

The data tasks (counting tokens, near-duplicate removal, decontamination, a Japanese quality classifier, swarm
construction, the retrieval datastore, scaling the download) are in the README's to-do list.
