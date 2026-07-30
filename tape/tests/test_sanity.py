"""Sanity checks. Run: python -m tape.tests.test_sanity  (from repo root)."""

import numpy as np
import torch

from tape import caching, controls, data as tdata, model_loader, probes


def test_generator_invariants():
    tokens, latents = tdata.generate_split(seed=7, n_steps=2000)
    assert tokens.shape == (2001,) and latents.shape == (2001, 5)
    assert tokens.min() >= 0 and tokens.max() < tdata.VOCAB_SIZE
    spread, imb = latents[:, 0], latents[:, 1]
    assert (spread >= 1).all(), "spread must be >= 1 tick (uncrossed book)"
    assert (np.abs(imb) <= 1.0 + 1e-6).all(), "imbalance in [-1, 1]"
    assert (latents[:, 3] >= 0).all(), "dist_to_touch >= 0"
    assert (latents[:, 4] >= 0).all(), "realized vol >= 0"
    # determinism
    t2, l2 = tdata.generate_split(seed=7, n_steps=2000)
    assert (tokens == t2).all() and np.allclose(latents, l2)
    # token roundtrip
    for tid in np.unique(tokens):
        assert tdata.VOCAB[tdata.ID_TO_TOKEN[int(tid)]] == int(tid)
    print("ok generator")


def test_model_shapes_and_hooks():
    cfg = model_loader.ModelConfig(n_layers=2, d_model=16, n_heads=2,
                                   d_mlp=32, n_ctx=32)
    model = model_loader.MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 32))
    logits, cache = model.run_with_cache(x)
    assert logits.shape == (3, 32, cfg.vocab_size)
    for l in range(cfg.n_layers):
        for h in ("pre", "mid", "post"):
            assert cache[f"blocks.{l}.hook_resid_{h}"].shape == (3, 32, 16)
    # residual bookkeeping: resid_pre[l+1] == resid_post[l]
    assert torch.allclose(cache["blocks.1.hook_resid_pre"],
                          cache["blocks.0.hook_resid_post"])
    # causality: changing a future token must not change past activations
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size
    _, cache2 = model.run_with_cache(x2)
    assert torch.allclose(cache["blocks.1.hook_resid_post"][:, :-1],
                          cache2["blocks.1.hook_resid_post"][:, :-1],
                          atol=1e-5), "attention mask leaks the future"
    print("ok model")


def test_cache_alignment(tmp="cache/_test"):
    cfg = model_loader.ModelConfig(n_layers=2, d_model=16, n_heads=2,
                                   d_mlp=32, n_ctx=32)
    model = model_loader.MiniGPT(cfg).eval()
    tokens, latents = tdata.generate_split(seed=3, n_steps=300)
    meta = caching.cache_activations(model, cfg, tokens, latents, tmp,
                                     batch=4, warmup=8)
    meta2, acts, ktok, klab = caching.load_cache(tmp)
    n = meta["n_tokens"]
    assert all(np.asarray(a).shape == (n, 16) for a in acts.values())
    assert ktok.shape == (n,) and klab.shape == (n, tdata.N_LATENTS)
    # spot-check alignment: recompute activations for the first window
    with torch.no_grad():
        _, c = model.run_with_cache(
            torch.from_numpy(tokens[:32])[None],
            names_filter={"blocks.1.hook_resid_post"}.__contains__)
    got = np.asarray(acts["blocks.1.hook_resid_post"][:32])
    assert np.allclose(got, c["blocks.1.hook_resid_post"][0].numpy(),
                       atol=1e-5)
    assert (ktok[:32] == tokens[:32]).all()
    print("ok caching")


def test_probe_recovers_planted_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 16)).astype(np.float32)
    w = rng.normal(size=16)
    y = (X @ w + 0.1 * rng.normal(size=4000)).astype(np.float32)
    r = probes.probe(X, y)
    assert r.r2 > 0.95, f"probe failed to recover linear signal: {r.r2}"
    # shuffled baseline near zero
    shuf = probes.probe(X, controls.shuffled_targets(y, 0))
    assert abs(shuf.r2) < 0.02, f"shuffled baseline leaked: {shuf.r2}"
    print("ok probes")


def test_factored_generator():
    n_ev = 1500
    s = tdata.generate_split_ex(seed=7, n_steps=n_ev, tokenization="factored")
    n_tok = 1 + tdata.SPAN_LEN * n_ev
    assert s.span_len == tdata.SPAN_LEN == 5
    assert s.tokens.shape == s.span_index.shape == s.span_pos.shape \
        == s.event_type_id.shape == (n_tok,)
    assert s.latents.shape == (n_tok, tdata.N_LATENTS)
    assert s.tokens.min() >= 0 and s.tokens.max() < tdata.FACTORED_VOCAB_SIZE

    # BOS: its own span 0
    assert s.tokens[0] == tdata.FACTORED_VOCAB[tdata.BOS_TOKEN]
    assert s.span_index[0] == 0 and s.span_pos[0] == 0
    assert s.event_type_id[0] == tdata.VOCAB[tdata.BOS_TOKEN]

    # slot structure: every span is exactly [TYPE][SIDE][OFF][SIZE][DT]
    body = s.tokens[1:].reshape(n_ev, 5)
    slot_of = {tid: name.split("_")[0]
               for name, tid in tdata.FACTORED_VOCAB.items()
               if name != tdata.BOS_TOKEN}
    for col, slot in enumerate(["TYPE", "SIDE", "OFF", "SIZE", "DT"]):
        assert all(slot_of[int(t)] == slot for t in body[:, col]), \
            f"slot {col} is not all {slot}"

    # span bookkeeping: 5 tokens per event, indices nondecreasing
    assert (s.span_index[1:].reshape(n_ev, 5)
            == np.arange(1, n_ev + 1)[:, None]).all()
    assert (s.span_pos[1:].reshape(n_ev, 5) == np.arange(5)).all()
    assert (np.diff(s.span_index) >= 0).all()

    # latents: constant within span, equal to combined-mode event latents
    tc, lc = tdata.generate_split(seed=7, n_steps=n_ev)
    body_lat = s.latents[1:].reshape(n_ev, 5, tdata.N_LATENTS)
    assert np.allclose(body_lat, body_lat[:, :1]), "latents vary within span"
    assert np.allclose(body_lat[:, 0], lc[1:]), "factored/combined latents drift"
    assert np.allclose(s.latents[0], lc[0])

    # alignment killer-check: reconstruct the combined token from the slot
    # tokens (DT excluded -- it does not exist in combined) and compare with
    # event_type_id AND the combined-mode stream itself
    etid = s.event_type_id[1:].reshape(n_ev, 5)
    assert (etid == etid[:, :1]).all()
    for k in range(0, n_ev, 97):   # sample; full loop is slow in python
        names = [tdata.FACTORED_ID_TO_TOKEN[int(t)] for t in body[k]]
        combined = "_".join(nm.split("_", 1)[1] for nm in names[:4])
        assert tdata.VOCAB[combined] == etid[k, 0] == tc[k + 1], \
            f"event {k}: factored span does not reconstruct combined token"

    # determinism (event process AND the dedicated DT rng)
    s2 = tdata.generate_split_ex(seed=7, n_steps=n_ev, tokenization="factored")
    assert (s.tokens == s2.tokens).all()

    # combined mode via the ex API is the degenerate case
    sc = tdata.generate_split_ex(seed=7, n_steps=n_ev)
    assert sc.span_len == 1
    assert (sc.tokens == tc).all()
    assert (sc.span_index == np.arange(len(tc))).all()
    assert (sc.span_pos == 0).all()
    assert (sc.event_type_id == tc).all()
    print("ok factored generator")


def test_factored_cache_spans(tmp="cache/_test_factored"):
    cfg = model_loader.ModelConfig(vocab_size=tdata.FACTORED_VOCAB_SIZE,
                                   n_layers=2, d_model=16, n_heads=2,
                                   d_mlp=32, n_ctx=32)
    model = model_loader.MiniGPT(cfg).eval()
    s = tdata.generate_split_ex(seed=3, n_steps=200, tokenization="factored")
    meta = caching.cache_activations(
        model, cfg, s.tokens, s.latents, tmp, batch=4, warmup=2,
        span_index=s.span_index, span_pos=s.span_pos,
        event_type_id=s.event_type_id, span_len=s.span_len,
        tokenization="factored")
    assert meta["warmup_tokens"] == 2 * 5 + 4
    meta2, acts, ktok, klab, spans = caching.load_cache_ex(tmp)
    n = meta["n_tokens"]
    # kept positions are a contiguous prefix of the stream, so every kept
    # array must equal its source prefix -- including across window edges
    # where spans are split
    assert (ktok == s.tokens[:n]).all()
    assert np.allclose(klab, s.latents[:n])
    assert (spans["span_index"] == s.span_index[:n]).all()
    assert (spans["span_pos"] == s.span_pos[:n]).all()
    assert (spans["event_type_id"] == s.event_type_id[:n]).all()
    assert meta2["tokenization"] == "factored" and meta2["span_len"] == 5
    print("ok factored caching")


def test_position_selection():
    import run_probe
    s = tdata.generate_split_ex(seed=5, n_steps=50, tokenization="factored")
    last = run_probe.select_positions(s.span_index, s.span_pos, "last")
    first = run_probe.select_positions(s.span_index, s.span_pos, "first")
    slot2 = run_probe.select_positions(s.span_index, s.span_pos, "2")
    assert (s.span_pos[last[1:]] == 4).all() and last[0] == 0   # BOS span
    assert (s.span_pos[first] == 0).all()
    assert (s.span_pos[slot2] == 2).all() and len(slot2) == 50
    assert len(last) == len(first) == 51
    # combined mode: every position is selected under "last"
    sc = tdata.generate_split_ex(seed=5, n_steps=50)
    all_pos = run_probe.select_positions(sc.span_index, sc.span_pos, "last")
    assert (all_pos == np.arange(len(sc.tokens))).all()
    print("ok position selection")


def test_control_task_consistency():
    tokens = np.array([1, 2, 1, 3, 2, 1] * 500)
    y = np.random.default_rng(1).normal(size=len(tokens)).astype(np.float32)
    c = controls.control_targets(tokens, y, seed=0)
    for t in (1, 2, 3):
        vals = c[tokens == t]
        assert np.allclose(vals, vals[0]), "control label not type-consistent"
    print("ok controls")


if __name__ == "__main__":
    test_generator_invariants()
    test_model_shapes_and_hooks()
    test_cache_alignment()
    test_factored_generator()
    test_factored_cache_spans()
    test_position_selection()
    test_probe_recovers_planted_signal()
    test_control_task_consistency()
    print("ALL SANITY CHECKS PASS")
