"""Factored-suite machinery check + combined/factored comparison table.

Run from the repo root:  python -m tape.tests.test_factored_parity

This test asserts that the PROBING MACHINERY is sound in both tokenizations:

  * span alignment / contiguous coverage: the event rows the probe actually
    uses tile the cache with no gaps and no partial spans (only BOS and a
    possibly truncated final span may be excluded);
  * shuffled-label baselines pass their sample-size-scaled tolerance;
  * the Hewitt-Liang control is NON-VACUOUS: event identity is linearly
    recoverable from the probed features at the embed layer (control R^2
    decisively above shuffled noise), so selectivity has a real bar to
    clear. In event mode this holds by construction -- the concatenated
    span contains the identity tokens -- and this test is the tripwire for
    regressions that break it (as last-of-span slot probing silently did,
    control R^2 ~0.00). Note the bar is mode-dependent in level: combined
    embeddings support near-free-form functions of joint identity (~0.93)
    while factored concat features only reach the slot-additive component
    of the random joint-identity pseudo-label (~0.2); see
    CONTROL_EMBED_MIN below.

It deliberately does NOT assert that the two tokenizations produce matching
probe results. They are different models over differently tokenized streams;
asserting numeric or qualitative parity bakes in a wrong assumption. The
side-by-side table is printed for the human to read, not asserted.

Preserved finding (surfaced by run_probe as `dt_slot_realized_vol`, and
reproducible via probe.mode: slot): in factored mode the DT token alone
supports realized_vol R^2 ~0.67 -- inter-arrival time is itself a
volatility proxy. That is a real per-slot result, not an artifact.

Reuses cache/*/probe_results.json when present with current metadata
(probe_mode == event); otherwise runs run_probe on the missing config, which
may train the factored stand-in first (one-time, ~10-20 min CPU).
"""

import json
import os
import sys

import numpy as np

import run_probe
from tape import caching

RUNS = [
    ("combined", "configs/synthetic.yaml", "cache/synthetic"),
    ("factored", "configs/synthetic_factored.yaml", "cache/synthetic_factored"),
]

# Non-vacuity bar for the Hewitt-Liang control at embed. The control target
# is a random function of JOINT event identity; a linear probe on factored
# event-concat features can only reach its slot-ADDITIVE component (~0.2
# observed), whereas one combined-mode embedding per joint type is nearly
# free-form (~0.93 observed). So the bar is "decisively above shuffled
# noise", not "near 1": vacuous means the probe cannot decode identity AT
# ALL (as at the DT slot, where control R^2 was ~0.00).
CONTROL_EMBED_MIN = 0.10
CONTROL_NOISE_MULT = 3.0  # ... and >= this multiple of the shuffled tolerance


def ensure_results(tokenization: str, cfg_path: str, cache_dir: str) -> dict:
    json_path = os.path.join(cache_dir, "probe_results.json")
    res = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            res = json.load(f)
        m = res.get("_meta", {})
        if (m.get("tokenization") != tokenization
                or m.get("probe_mode") != "event"):
            print(f"[suite] {json_path} is stale (pre-event-mode); "
                  f"rerunning {cfg_path}")
            res = None
    if res is None:
        run_probe.main(cfg_path)
        with open(json_path) as f:
            res = json.load(f)
    return res


def check_span_machinery(tokenization: str, cache_dir: str) -> None:
    """Recompute the event rows from the cached span arrays and assert they
    tile the cache contiguously with intact slot structure."""
    meta, _, ktok, klab, spans = caching.load_cache_ex(cache_dir)
    span_len = meta["span_len"]
    si, sp = spans["span_index"], spans["span_pos"]
    rows = run_probe.select_event_rows(si, sp, span_len)
    assert len(rows), "no complete event spans in cache"

    # slot structure intact within every probed row
    assert (sp[rows] == np.arange(span_len)).all(), "slot order broken"
    assert (si[rows] == si[rows[:, :1]]).all(), "row spans a span boundary"

    # coverage contiguous: consecutive rows are adjacent spans, and the only
    # unprobed positions are BOS (span_len > 1) and a truncated tail
    assert (np.diff(rows[:, 0]) == span_len).all(), "gap between event rows"
    assert (np.diff(si[rows[:, 0]]) == 1).all(), "skipped an event index"
    first = rows[0, 0]
    assert first == (0 if span_len == 1 else 1), \
        f"coverage starts at {first}, expected BOS handling only"
    tail = len(si) - (rows[-1, 0] + span_len)
    assert 0 <= tail < span_len, f"{tail} unprobed tail positions"

    # labels constant within each span
    assert np.allclose(klab[rows[:, 0]], klab[rows[:, -1]]), \
        "latents vary within span"
    print(f"[{tokenization}] machinery ok: {len(rows)} contiguous event "
          f"rows x {span_len} slots, {tail} tail positions dropped")


def check_probe_hygiene(tokenization: str, res: dict) -> list[str]:
    m = res["_meta"]
    hooks, latents = m["hook_names"], m["latent_names"]
    problems = []
    bad = [(h, lat) for h in hooks for lat in latents
           if not res[h][lat]["pass_shuffled"]]
    if bad:
        problems.append(
            f"{tokenization}: shuffled baseline above tolerance "
            f"{m['shuffled_tolerance']:.4f} for {bad}")
    embed = hooks[0]
    bar = max(CONTROL_EMBED_MIN, CONTROL_NOISE_MULT * m["shuffled_tolerance"])
    vac = {lat: res[embed][lat]["r2_control"] for lat in latents
           if res[embed][lat]["r2_control"] < bar}
    if vac:
        problems.append(
            f"{tokenization}: Hewitt-Liang control VACUOUS at {embed} "
            f"(R^2 < {bar:.2f}): "
            + ", ".join(f"{k}={v:+.2f}" for k, v in vac.items()))
    return problems


def summarize(res: dict) -> dict:
    m = res["_meta"]
    out = {}
    for lat in m["latent_names"]:
        pts = [res[h][lat] for h in m["hook_names"]]
        out[lat] = {
            "best_r2": max(p["r2_task"] for p in pts),
            "best_ctx": max(p["ctx_gain"] for p in pts),
            "best_sel": max(p["selectivity"] for p in pts),
            "encoded": any(p["pass_shuffled"] and p["pass_selectivity"]
                           for p in pts),
        }
    return out


def print_table(summaries: dict) -> None:
    c, f = summaries["combined"], summaries["factored"]
    header = (f"{'latent':16s} {'combined r2/ctx/sel/enc':>26s} "
              f"{'factored r2/ctx/sel/enc':>26s}")
    print(header)
    print("-" * len(header))
    for lat in c:
        def cell(s):
            return (f"{s['best_r2']:+.2f} {s['best_ctx']:+.2f} "
                    f"{s['best_sel']:+.2f} {'Y' if s['encoded'] else 'n':>2s}")
        print(f"{lat:16s} {cell(c[lat]):>26s} {cell(f[lat]):>26s}")


def main():
    summaries, problems = {}, []
    for tok, cfg_path, cache_dir in RUNS:
        res = ensure_results(tok, cfg_path, cache_dir)
        check_span_machinery(tok, cache_dir)
        problems += check_probe_hygiene(tok, res)
        summaries[tok] = summarize(res)
        dt = res["_meta"].get("dt_slot_realized_vol")
        if dt:
            print(f"[{tok}] DT-slot diagnostic: realized_vol from the DT "
                  f"token alone, best R^2 {max(dt.values()):+.2f} "
                  "(inter-arrival time is a volatility proxy)")

    print()
    print_table(summaries)
    print("\n(comparison is informational: the modes are different models "
          "over different token streams and are NOT asserted to match)")

    if problems:
        raise AssertionError(
            "PROBING MACHINERY UNSOUND:\n  - " + "\n  - ".join(problems))
    print("\nMACHINERY OK: spans aligned, coverage contiguous, shuffled "
          "baselines pass, controls non-vacuous in both tokenizations")


if __name__ == "__main__":
    sys.exit(main())
