import gzip
import json
import random

from exealm.data.dedup import MinHasher, NearDupIndex, main, shingles

WORDS = ("river stone light paper garden window music orange planet engine forest cloud winter market bridge silver "
         "doctor island letter mirror camera pencil rocket valley thunder candle harbor violin desert castle").split()


def article(seed, n=300):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 50)) for _ in range(n))


def edit(text, frac, seed):
    """Replace a fraction of the words."""
    r = random.Random(seed)
    w = text.split()
    for i in r.sample(range(len(w)), int(frac * len(w))):
        w[i] = "changed" + str(i)
    return " ".join(w)


def jaccard(a, b):
    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / len(sa | sb)


def test_signature_agreement_estimates_jaccard():
    mh = MinHasher(num_perm=256)
    a = article(1)
    for frac in (0.0, 0.02, 0.1):
        b = edit(a, frac, 2)
        est = float((mh.signature(shingles(a)) == mh.signature(shingles(b))).mean())
        assert abs(est - jaccard(a, b)) < 0.1


def test_index_drops_near_copies_and_keeps_different_text():
    mh, idx = MinHasher(), NearDupIndex()
    a = article(1)
    assert idx.add(0, mh.signature(shingles(a))) is None
    assert idx.add(1, mh.signature(shingles(a))) == 0                       # exact copy
    assert idx.add(2, mh.signature(shingles(edit(a, 0.01, 3)))) == 0        # a few words changed
    assert idx.add(3, mh.signature(shingles(article(2)))) is None           # different text
    assert idx.add(4, mh.signature(shingles(edit(a, 0.4, 4)))) is None      # heavily rewritten: kept


def test_japanese_uses_character_shingles():
    ja = ("東京は日本の首都であり、人口は約千四百万人です。"
          "江戸時代には江戸と呼ばれ、明治維新の後に首都となりました。") * 4
    near = ja.replace("約千四百万人", "およそ千四百万人", 1)
    assert jaccard(ja, near) > 0.8 and len(shingles(ja)) > 20
    mh, idx = MinHasher(), NearDupIndex()
    assert idx.add(0, mh.signature(shingles(ja))) is None
    assert idx.add(1, mh.signature(shingles(near))) == 0


def write(path, texts, source):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "source": source}, ensure_ascii=False) + "\n")


def test_cli_dedups_across_sources_and_keeps_the_first_copy(tmp_path):
    a, b, c = article(1), article(2), article(3)
    write(tmp_path / "web.jsonl.gz", [a, b], "web")
    write(tmp_path / "k2.jsonl.gz", [edit(a, 0.01, 5), c, c], "k2")
    report = main([str(tmp_path / "web.jsonl.gz"), str(tmp_path / "k2.jsonl.gz"), "--out", str(tmp_path / "out")])
    assert report["web.jsonl.gz"]["kept"] == 2
    assert report["k2.jsonl.gz"] == dict(read=3, kept=1, dropped=2,
                                         duplicates_of={"web.jsonl.gz": 1, "k2.jsonl.gz": 1})
    with gzip.open(tmp_path / "out" / "k2.jsonl.gz", "rt", encoding="utf-8") as f:
        kept = [json.loads(l) for l in f]
    assert kept == [{"text": c, "source": "k2"}]
