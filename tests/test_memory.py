"""The retrieval (RG) and agent-board (AG) circuits: no peeking at the future, no looking up the document being read,
agents only hear what their swarm posted earlier, and gradients reach the model's own retrieval encoder."""
import torch

from exealm import AgentBoard, Datastore, ExeaLM, MemoryContext, get_config


def setup(swarm_size=1, B=2, T=40, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = get_config("tiny", swarm_size=swarm_size, **kw)
    m = ExeaLM(cfg)
    ds = Datastore(torch.randint(0, cfg.vocab_size, (50, cfg.rag_chunk_len)), torch.arange(50) % 7)
    ds.refresh(m.encoder, m.emb)
    x = torch.randint(0, cfg.vocab_size, (B, T))
    docs = torch.full((B, T), 3)
    mem = MemoryContext(datastore=ds, doc_ids=docs, swarm_size=swarm_size)
    m.train()
    m(x, mem)                                    # set each slot's running output scale
    m.eval()
    return cfg, m, x, mem


import pytest  # noqa: E402


@pytest.mark.parametrize("query", ["last", "mean"])
def test_retrieval_and_board_are_causal_for_one_agent(query):
    cfg, m, x, mem = setup(rag_query=query)
    t = 21
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5)
    assert not torch.allclose(a[0, t:], b[0, t:])


def test_swarm_members_only_hear_posts_from_earlier_chunks():
    cfg, m, x, mem = setup(swarm_size=2)
    t = 13                                       # inside chunk 1 (chunks of 8)
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    c_end = (t // cfg.chunk + 1) * cfg.chunk     # the post made in t's chunk is readable from here on
    assert torch.allclose(a[0, :t], b[0, :t], atol=1e-5)
    assert torch.allclose(a[1, :c_end], b[1, :c_end], atol=1e-5)
    assert not torch.allclose(a[1, c_end:], b[1, c_end:])      # the other agent does hear it later


def test_separate_swarms_do_not_hear_each_other():
    cfg, m, x, mem = setup(swarm_size=1)
    x2 = x.clone()
    x2[0] = (x2[0] + 1) % cfg.vocab_size
    with torch.no_grad():
        assert torch.allclose(m(x, mem)[0][1], m(x2, mem)[0][1], atol=1e-5)


def test_datastore_never_returns_the_document_being_read():
    keys = torch.nn.functional.normalize(torch.randn(100, 8), dim=-1)
    ds = Datastore(torch.zeros(100, 4, dtype=torch.long), torch.arange(100) % 10)
    ds.keys = keys
    q = keys[:20]                                # each query's best match is itself, in document i % 10
    idx, ok = ds.search(q, 3, exclude=torch.arange(20) % 10)
    assert ok.all()
    assert not (ds.doc_ids[idx] == (torch.arange(20) % 10).view(-1, 1)).any()


def test_gradients_reach_the_models_own_retrieval_encoder():
    cfg, m, x, mem = setup()
    m.train()
    m(x, mem, x)[1].backward()
    assert m.encoder.key.weight.grad is not None and m.encoder.key.weight.grad.abs().sum() > 0
    assert m.layers[0].rg.alpha.grad is not None and m.layers[0].inp.weight.grad.abs().sum() > 0


def test_persistent_board_shares_messages_between_agents():
    cfg, m, x, mem = setup()
    board = AgentBoard()
    with torch.no_grad():
        alone = m(x[1:], MemoryContext(board=board, agent_ids=torch.tensor([1])))[0]
        m(x[:1], MemoryContext(board=board, agent_ids=torch.tensor([0]), write_board=True))    # agent 0 posts
        n = sum(len(c) for c in board.channels.values())
        m(x[:1], MemoryContext(board=board, agent_ids=torch.tensor([0]), write_board=True))    # same text again
        assert sum(len(c) for c in board.channels.values()) == n                              # replaced, not duplicated
        heard = m(x[1:], MemoryContext(board=board, agent_ids=torch.tensor([1])))[0]        # agent 1 reads
        own = m(x[:1], MemoryContext(board=board, agent_ids=torch.tensor([0])))[0]
        own_alone = m(x[:1], MemoryContext())[0]
    assert n == cfg.n_layers * (x.shape[1] // cfg.chunk)     # one AG slot per layer, one post per full chunk
    assert not torch.allclose(alone, heard)                   # agent 1 heard agent 0
    assert torch.allclose(own, own_alone, atol=1e-5)          # an agent does not re-read its own old posts


def test_swarm_batches_split_and_mix():
    import numpy as np
    from exealm.data.loader import SyntheticShards
    data = SyntheticShards(n_tokens=20_000)
    x, y, d = data.batch(8, 16, np.random.default_rng(0), swarm_size=4, policy="split")
    for s in range(2):                                    # each swarm reads one stretch of text, in order
        rows = x[4 * s:4 * s + 4]
        assert torch.equal(rows[1:, 0], y[4 * s:4 * s + 3, -1])
    x, y, d = data.batch(8, 16, np.random.default_rng(0), swarm_size=4, policy="mixed")
    assert x.shape == (8, 16) and d.shape == (8, 16)


def test_datastore_returns_k_results_even_when_the_own_document_dominates():
    keys = torch.nn.functional.normalize(torch.randn(200, 8), dim=-1)
    doc = torch.zeros(200, dtype=torch.long)
    doc[190:] = torch.arange(1, 11)                       # 190 chunks of document 0, 10 chunks of other documents
    ds = Datastore(torch.zeros(200, 4, dtype=torch.long), doc)
    ds.keys = keys
    idx, ok = ds.search(keys[:5], 4, exclude=torch.zeros(5, dtype=torch.long))
    assert ok.all() and (ds.doc_ids[idx] != 0).all()


def test_larger_retrieval_keys_work_and_stay_causal():
    cfg, m, x, mem = setup(rag_key_dim=24)
    assert m.encoder.key.out_features == 24 and mem.datastore.keys.shape[1] == 24
    x2 = x.clone()
    x2[0, 21] = (x2[0, 21] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    assert torch.allclose(a[:, :21], b[:, :21], atol=1e-5)


def test_lexical_keys_match_chunks_that_share_rare_words():
    cfg = get_config("tiny", rag_lexical=16)
    m = ExeaLM(cfg)
    common = torch.full((8,), 5)
    a = common.clone(); a[2], a[5] = 200, 201                 # two chunks sharing two rare tokens
    b = common.clone(); b[1], b[6] = 200, 201
    c = common.clone(); c[3], c[4] = 150, 151                 # a chunk with different rare tokens
    store = torch.stack([a, b, c] + [torch.full((8,), 5)] * 20)
    m.encoder.set_idf(store)
    assert m.encoder.idf[200] > m.encoder.idf[5]
    lex = lambda t: torch.nn.functional.normalize((m.encoder.lex[t] * m.encoder.idf[t].unsqueeze(-1)).sum(0), dim=-1)
    assert float(lex(a) @ lex(b)) > float(lex(a) @ lex(c))
    ds = Datastore(store, torch.arange(len(store)))
    ds.refresh(m.encoder, m.emb)
    assert ds.keys.shape[1] == cfg.h + 16
