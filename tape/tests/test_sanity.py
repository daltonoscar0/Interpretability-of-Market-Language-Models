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


def _write_fixture_bin(path, n_events, seed=0):
    """A tiny stream in the real OFTK v2 format: BOS SESSION_OPEN <n_events
    5-tuples> SESSION_CLOSE EOS, each slot a valid id in its range."""
    import struct
    rng = np.random.default_rng(seed)
    toks = [tdata.R_BOS, tdata.R_SESSION_OPEN]
    for _ in range(n_events):
        toks += [int(rng.integers(tdata.R_TYPE_LO, tdata.R_TYPE_HI + 1)),
                 int(rng.integers(tdata.R_SIDE_LO, tdata.R_SIDE_HI + 1)),
                 int(rng.integers(tdata.R_PX_LO, tdata.R_PX_HI + 1)),
                 int(rng.integers(tdata.R_SZ_LO, tdata.R_SZ_HI + 1)),
                 int(rng.integers(tdata.R_DT_LO, tdata.R_DT_HI + 1))]
    toks += [tdata.R_SESSION_CLOSE, tdata.R_EOS]
    arr = np.array(toks, dtype="<u2")
    with open(path, "wb") as f:
        f.write(b"OFTK")
        f.write(struct.pack("<I", 2))
        f.write(b"SPY".ljust(12, b"\x00"))
        f.write(struct.pack("<Q", len(arr)))
        f.write(arr.tobytes())


def test_real_token_stream(tmp="cache/_test_real"):
    import copy
    import json
    import os
    os.makedirs(tmp, exist_ok=True)
    bin_path = os.path.join(tmp, "SPY_fixture.tokens.bin")
    man_path = os.path.join(tmp, "manifest.json")
    _write_fixture_bin(bin_path, n_events=1000)

    # fixture manifest: real vocab (52 ids, orderflow-factored-v2), SPY split
    # boundary placed inside the fixture so train/eval are both non-empty
    real = tdata.load_manifest("pipeline/data/tokens/manifest.json")
    man = copy.deepcopy(real)
    man["tickers"] = {"SPY": copy.deepcopy(real["tickers"]["SPY"])}
    man["tickers"]["SPY"]["fit"]["train_end_index"] = 700
    with open(man_path, "w") as f:
        json.dump(man, f)

    ticker, ids = tdata.read_token_bin(bin_path)
    assert ticker == "SPY"
    assert ids.min() >= 0 and ids.max() < 52

    tr = tdata.load_real_token_stream(bin_path, man_path, split="train")
    ev = tdata.load_real_token_stream(bin_path, man_path, split="eval")
    al = tdata.load_real_token_stream(bin_path, man_path, split="all")
    # split honors the manifest ROW boundary; whole tuples kept intact
    assert (tr.n_events, ev.n_events, al.n_events) == (700, 300, 1000)
    assert tr.vocab_size == 52 and tr.train_end_index == 700
    assert tr.manifest_format == "orderflow-factored-v2"
    for s in (tr, ev, al):
        assert s.tokens[0] == tdata.R_BOS and s.tokens[1] == tdata.R_SESSION_OPEN
        assert s.tokens[-2] == tdata.R_SESSION_CLOSE and s.tokens[-1] == tdata.R_EOS
        assert int(s.tokens.max()) < s.vocab_size
        # body is whole 5-tuples -> (len - 4 wrapper) divisible by 5
        assert (len(s.tokens) - 4) % tdata.R_EVENT_TOKENS == 0
    # max_events caps the loaded events
    capped = tdata.load_real_token_stream(bin_path, man_path, split="all",
                                          max_events=120)
    assert capped.n_events == 120
    # a corrupt tuple (SIDE id where PRICE_OFF must be) is rejected loudly
    _, raw = tdata.read_token_bin(bin_path)
    bad = raw.copy()
    bad[2 + 2] = tdata.R_SIDE_LO   # PRICE_OFF slot of event 0 -> a SIDE id
    try:
        tdata._token_event_rows(bad)
        assert False, "malformed tuple not rejected"
    except ValueError:
        pass
    print("ok real token stream")


def _write_fixture_csvs(dirpath, n_rows, seed=1):
    """Tiny LOBSTER-shaped orderbook + message CSVs on a 100-tick grid.
    Orderbook lines carry 8 columns so the reader's only-first-four
    streaming is actually exercised."""
    import os
    rng = np.random.default_rng(seed)
    ob = os.path.join(dirpath, "fixture_orderbook.csv")
    ms = os.path.join(dirpath, "fixture_message.csv")
    bid, ask = 1356400, 1356500
    with open(ob, "w") as fo, open(ms, "w") as fm:
        for i in range(n_rows):
            if rng.random() < 0.2:
                bid += 100 * int(rng.integers(-1, 2))
            if bid >= ask:
                ask = bid + 100
            elif rng.random() < 0.1:
                ask = bid + 100 * int(rng.integers(1, 4))
            asz = int(rng.integers(100, 5000))
            bsz = int(rng.integers(100, 5000))
            fo.write(f"{ask},{asz},{bid},{bsz},"
                     f"{ask + 100},{asz},{bid - 100},{bsz}\n")
            mtype = int(rng.choice([1, 2, 3, 4]))
            mdir = int(rng.choice([-1, 1]))
            off = 100 * int(rng.integers(-1, 4))
            price = bid - off if mdir == 1 else ask + off
            fm.write(f"34200.{i:06d},{mtype},{1000 + i},100,{price},{mdir}\n")
    return ob, ms


def _naive_lobster_latents(ob, ms, tick, start, n):
    """Deque-based mirror of the SYNTHETIC latent extractor, applied to the
    fixture files -- the vectorized reader must reproduce it exactly."""
    import collections
    pad = min(start, tdata.W_VOL)
    touch = tdata.read_orderbook_touch(ob, start - pad, start + n)
    msg = tdata.read_message_rows(ms, start - pad, start + n)
    mids = collections.deque(maxlen=tdata.W_VOL + 1)
    ofi_terms = collections.deque(maxlen=tdata.W_OFI)
    prev = None
    out = []
    for (a, asz, b, bsz), (mtype, price, mdir) in zip(touch, msg):
        if prev is not None:
            pa, pasz, pb, pbsz = prev
            e_bid = bsz if b > pb else (bsz - pbsz if b == pb else -pbsz)
            e_ask = asz if a < pa else (asz - pasz if a == pa else -pasz)
            ofi_terms.append(float(e_bid - e_ask))
            dist = ((pb - price) if mdir == 1 else (price - pa)) / tick
        else:
            dist = ((b - price) if mdir == 1 else (price - a)) / tick
        dist = 0.0 if mtype == 7 else max(dist, 0.0)
        mids.append((a + b) / (2.0 * tick))
        rets = np.diff(np.asarray(mids))
        rvol = float(np.std(rets)) if len(mids) >= 3 else 0.0
        tot = asz + bsz
        out.append([(a - b) / tick,
                    (bsz - asz) / tot if tot > 0 else 0.0,
                    float(sum(ofi_terms)) if ofi_terms else 0.0,
                    dist, rvol])
        prev = (a, asz, b, bsz)
    return np.array(out[pad:], dtype=np.float32)


def test_lobster_latents(tmp="cache/_test_real"):
    import os
    os.makedirs(tmp, exist_ok=True)
    ob, ms = _write_fixture_csvs(tmp, 400)
    for start, n in ((0, 400), (200, 150), (37, 100)):
        got = tdata.load_lobster_latents(ob, ms, 100, start, n)
        want = _naive_lobster_latents(ob, ms, 100, start, n)
        assert got.shape == (n, tdata.N_LATENTS)
        assert np.allclose(got, want, atol=1e-4), (start, n)
    # streamed reader: shape, deeper columns ignored, bounds enforced
    assert tdata.read_orderbook_touch(ob, 5, 10).shape == (5, 4)
    try:
        tdata.read_orderbook_touch(ob, 390, 405)
        assert False, "read past EOF not rejected"
    except ValueError:
        pass
    print("ok lobster latents")


def test_real_eval_stream(tmp="cache/_test_real"):
    import copy
    import json
    import os
    os.makedirs(tmp, exist_ok=True)
    bin_path = os.path.join(tmp, "SPY_fixture.tokens.bin")
    man_path = os.path.join(tmp, "manifest.json")
    _write_fixture_bin(bin_path, n_events=1000)
    real = tdata.load_manifest("pipeline/data/tokens/manifest.json")
    man = copy.deepcopy(real)
    man["tickers"] = {"SPY": copy.deepcopy(real["tickers"]["SPY"])}
    man["tickers"]["SPY"]["fit"]["train_end_index"] = 700
    with open(man_path, "w") as f:
        json.dump(man, f)
    ob, ms = _write_fixture_csvs(tmp, 1000)

    st = tdata.load_real_eval_stream(bin_path, man_path, ob, ms, split="eval")
    n_ev = 300
    assert st.span_len == tdata.R_EVENT_TOKENS
    assert len(st.tokens) == 4 + 5 * n_ev
    assert st.tokens[0] == tdata.R_BOS and st.tokens[-1] == tdata.R_EOS
    # wrapper specials: own 1-token spans, distinct negative control ids
    assert st.event_type_id[0] == -1 - tdata.R_BOS
    assert st.event_type_id[1] == -1 - tdata.R_SESSION_OPEN
    assert st.span_index[0] != st.span_index[1] != st.span_index[2]
    # control id constant across each span, DT-EXCLUDED by construction
    et = st.event_type_id[2:-2].reshape(n_ev, 5)
    assert (et == et[:, :1]).all() and (et >= 0).all()
    a = np.array([7, 13, 16, 28, 37], dtype=np.int64)
    b = a.copy()
    b[4] = 50                                    # different DT slot only
    assert (tdata._real_span_arrays(a)[2][0]
            == tdata._real_span_arrays(b)[2][0]), "DT leaked into control id"
    c = a.copy()
    c[3] = 29                                    # different SIZE slot
    assert tdata._real_span_arrays(a)[2][0] != tdata._real_span_arrays(c)[2][0]
    # probe row selection picks exactly the event tuples, none of the specials
    import run_probe
    rows = run_probe.select_event_rows(st.span_index, st.span_pos, 5)
    assert len(rows) == n_ev and rows[0, 0] == 2
    # latents: constant within span, aligned to the split's message rows
    lat = st.latents[2:-2].reshape(n_ev, 5, tdata.N_LATENTS)
    assert (lat == lat[:, :1]).all()
    direct = tdata.load_lobster_latents(ob, ms, 100, 700, n_ev)
    assert np.array_equal(lat[:, 0], direct)
    # max_events truncates on an event boundary
    capped = tdata.load_real_eval_stream(bin_path, man_path, ob, ms,
                                         split="eval", max_events=50)
    assert len(capped.tokens) == 4 + 5 * 50
    # halt/resume: 1-token spans owning a message row (AAPL/MSFT may have them)
    h = np.array([1, 3, 7, 13, 16, 28, 37, 5, 6, 7, 13, 16, 28, 37, 4, 2],
                 dtype=np.int64)
    si, sp, eh = tdata._real_span_arrays(h)
    assert eh[7] == -1 - tdata.R_HALT and eh[8] == -1 - tdata.R_RESUME
    assert si[7] != si[8] != si[9]
    assert (si[9:14] == si[9]).all() and (sp[9:14] == np.arange(5)).all()
    assert tdata._token_event_rows(h)[7] == 1    # halt consumes a message row
    print("ok real eval stream")


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
    test_real_token_stream()
    test_lobster_latents()
    test_real_eval_stream()
    print("ALL SANITY CHECKS PASS")
