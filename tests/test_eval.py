from exealm.eval.speed import run, table


def test_speed_benchmark_times_both_models():
    recs = run("tiny", batch=4, seq=32, steps=2, warmup=1, gen_prompt=16, gen_tokens=3, datastore_chunks=50, swarm=2,
               device="cpu")
    assert [r["model"] for r in recs] == ["tonn", "llama"]
    tonn, llama = recs
    assert tonn["retrieval"] and tonn["swarm"] == 2 and not llama["retrieval"] and llama["swarm"] == 1
    for r in recs:
        assert r["train_tokens_per_sec"] > 0 and r["gen_tokens_per_sec"] > 0
    assert "slower than llama" in table(recs)
