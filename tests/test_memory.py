"""The retrieval (RG) and agent-board (AG) circuits: no peeking at the future, no looking up the document being read,
agents only hear what teammates posted earlier, and gradients reach the model's own retrieval encoder."""
import torch

from exealm import AgentBoard, Datastore, ExeaLM, MemoryContext, get_config


def setup(team_size=1, B=2, T=40, seed=0):
    torch.manual_seed(seed)
    cfg = get_config("tiny", team_size=team_size)
    m = ExeaLM(cfg)
    ds = Datastore(torch.randint(0, cfg.vocab_size, (50, cfg.rag_chunk_len)), torch.arange(50) % 7)
    ds.refresh(m.encoder, m.emb)
    x = torch.randint(0, cfg.vocab_size, (B, T))
    docs = torch.full((B, T), 3)
    mem = MemoryContext(datastore=ds, doc_ids=docs, team_size=team_size)
    m.train()
    m(x, mem)                                    # set each slot's running output scale
    m.eval()
    return cfg, m, x, mem


def test_retrieval_and_board_are_causal_for_one_agent():
    cfg, m, x, mem = setup()
    t = 21
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5)
    assert not torch.allclose(a[0, t:], b[0, t:])


def test_teammates_only_hear_posts_from_earlier_chunks():
    cfg, m, x, mem = setup(team_size=2)
    t = 13                                       # inside chunk 1 (chunks of 8)
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    c_end = (t // cfg.chunk + 1) * cfg.chunk     # the post made in t's chunk is readable from here on
    assert torch.allclose(a[0, :t], b[0, :t], atol=1e-5)
    assert torch.allclose(a[1, :c_end], b[1, :c_end], atol=1e-5)
    assert not torch.allclose(a[1, c_end:], b[1, c_end:])      # the teammate does hear it later


def test_separate_teams_do_not_hear_each_other():
    cfg, m, x, mem = setup(team_size=1)
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
    assert m.layers[0].rg.kv.weight.grad.abs().sum() > 0


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
