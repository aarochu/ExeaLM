import gzip
import json

import numpy as np

from exealm.data.loader import TokenShards
from exealm.data.pack import pack
from exealm.data.quality import check, is_japanese


def test_byte_packing_keeps_documents_and_round_trips(tmp_path):
    docs = ["hello world", "こんにちは", "third document"]
    src = tmp_path / "docs.jsonl.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    meta = pack([str(src)], "bytes", str(tmp_path / "packed"))
    assert meta["vocab_size"] == 257 and meta["n_docs"] == 3
    shards = TokenShards(str(tmp_path / "packed"))
    toks = np.asarray(shards.tokens)
    for i, d in enumerate(docs):
        start = shards.offsets[i]
        end = shards.offsets[i + 1] if i + 1 < len(docs) else len(toks)
        assert bytes(toks[start:end - 1].astype(np.uint8)).decode("utf-8") == d
        assert toks[end - 1] == 256                                   # end-of-text after every document
        assert (shards.doc_of(np.arange(start, end)) == i).all()


def test_quality_rules_treat_japanese_separately():
    ja = "これは日本語の文章です。" * 40
    assert is_japanese(ja) and check(ja) == []
    assert "too short" in check("short")


def test_multishards_mix_sources_and_keep_doc_ids_unique(tmp_path):
    import torch
    from exealm.data.loader import MultiShards
    paths = []
    for name, text in [("a", "aaaa " * 200), ("b", "bbbb " * 200)]:
        src = tmp_path / f"{name}.jsonl.gz"
        with gzip.open(src, "wt", encoding="utf-8") as f:
            for _ in range(5):
                f.write(json.dumps({"text": text}) + "\n")
        pack([str(src)], "bytes", str(tmp_path / name))
        paths.append(str(tmp_path / name))
    ms = MultiShards(paths, weights=[1, 1])
    x, y, d = ms.batch(16, 32, np.random.default_rng(0), swarm_size=4, policy="split")
    assert x.shape == (16, 32)
    for s in range(4):                                    # a swarm reads one source
        rows = x[4 * s:4 * s + 4]
        letters = set(bytes(rows[rows < 256].to(torch.uint8).tolist()).decode().replace(" ", ""))
        assert len(letters) == 1
    assert set(d[x == ord("a")].tolist()).isdisjoint(set(d[x == ord("b")].tolist()))
