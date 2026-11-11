"""Tokenize documents and pack them into one flat token file per split, with document boundaries kept.

Output directory:
    tokens.bin    every token of every document, end to end, <|endoftext|> after each (uint16 if the vocabulary fits)
    offsets.npy   where each document starts in tokens.bin (so retrieval can skip the document being read)
    meta.json     dtype, vocabulary size, token and document counts, bytes per token

python -m exealm.data.pack data/filtered/*.jsonl.gz --tokenizer tokenizer --out data/packed/train
"""
import argparse
import gzip
import json
import os

import numpy as np


def pack(files, tokenizer, out, max_docs=None, batch=1000):
    from .tokenizer import load
    tok = load(tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    V = tok.get_vocab_size()
    dtype = np.uint16 if V <= 65535 else np.uint32
    os.makedirs(out, exist_ok=True)
    offsets, n_tok, n_bytes, buf = [], 0, 0, []
    with open(os.path.join(out, "tokens.bin"), "wb") as fo:
        def flush():
            nonlocal n_tok
            for enc in tok.encode_batch(buf):
                offsets.append(n_tok)
                ids = np.array(enc.ids + [eot], dtype=dtype)
                fo.write(ids.tobytes())
                n_tok += len(ids)
            buf.clear()
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line)["text"]
                    n_bytes += len(text.encode("utf-8"))
                    buf.append(text)
                    if len(buf) >= batch:
                        flush()
                    if max_docs and len(offsets) + len(buf) >= max_docs:
                        break
        if buf:
            flush()
    np.save(os.path.join(out, "offsets.npy"), np.array(offsets, dtype=np.int64))
    meta = dict(dtype=np.dtype(dtype).name, vocab_size=V, n_tokens=n_tok, n_docs=len(offsets),
                bytes_per_token=n_bytes / max(1, n_tok), files=[os.path.basename(f) for f in files])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--out", default="data/packed/train")
    ap.add_argument("--max_docs", type=int, default=None)
    a = ap.parse_args()
    pack(a.files, a.tokenizer, a.out, a.max_docs)


if __name__ == "__main__":
    main()
