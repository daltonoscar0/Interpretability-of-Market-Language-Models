#!/usr/bin/env python3
"""Phase-1 driver: `python run_probe.py configs/synthetic.yaml`

Pipeline: load config -> load (or train) model -> generate/load eval stream
-> cache residual activations (memmap, all positions) -> build probe rows ->
layerwise probes for each latent with Hewitt-Liang controls + shuffled
baselines -> "what is encoded where" figure -> append RESULTS.md row.

The unit of analysis is the EVENT, not the token. Default probe mode
("event") concatenates the residual activations across the event's whole
span into one [span_len * d_model] feature row -- in combined mode
(span_len=1) that degenerates to per-token probing, so both tokenizations
are probed at the same unit. This also keeps the Hewitt-Liang control
non-vacuous in factored mode: event identity is linearly recoverable from
the full span (the TYPE/SIDE/OFF/SIZE tokens are in it), exactly as it is
from a combined token. Per-slot probing (probe.mode: slot +
probe.label_position) is kept as a diagnostic, not the default.

Boring-explanation gate (mandatory): a (layer, latent) probe result is only
reported as evidence of encoding if (a) its shuffled-label baseline R^2 is
within noise of zero -- tolerance scales as coeff/sqrt(n_probed) -- and
(b) selectivity clears the configured threshold. Failures are printed and
marked on the figure, never silently dropped.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time

import numpy as np
import yaml  # justification: config files; stdlib has no yaml parser

import matplotlib  # justification: Phase-1 deliverable is a figure
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tape import caching, controls, data as tdata, model_loader, probes


def main(cfg_path: str) -> None:
    t_start = time.time()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run_name = cfg.get("run_name", "run")
    seed = int(cfg.get("seed", 0))

    dcfg = cfg["data"]
    tokenization = dcfg.get("tokenization", "combined")

    # 1. model ------------------------------------------------------------
    model, mcfg = model_loader.load_model(cfg["model"], tokenization)

    # 2. eval stream + ground-truth latents -------------------------------
    if dcfg.get("source", "synthetic") != "synthetic":
        raise NotImplementedError(
            "wire the LOBSTER-derived eval stream in tape/data.py and add a "
            "branch here")
    if "eval_events" in dcfg:
        n_events = int(dcfg["eval_events"])
    elif tokenization == "combined":
        n_events = int(dcfg.get("eval_tokens", 40_000))
    else:
        n_events = int(dcfg.get("eval_tokens", 50_000)) // tdata.SPAN_LEN
    print("generating eval stream ...")
    stream = tdata.generate_split_ex(
        seed=int(dcfg.get("eval_seed", seed + 1)),  # disjoint from train seed
        n_steps=n_events, tokenization=tokenization)

    # 3. cache activations -------------------------------------------------
    cache_dir = cfg.get("cache_dir", "cache/" + run_name)
    warmup_events = int(cfg.get("warmup_events", cfg.get("warmup", 50)))
    print("caching residual activations ...")
    caching.cache_activations(model, mcfg, stream.tokens, stream.latents,
                              cache_dir,
                              batch=int(cfg.get("cache_batch", 8)),
                              warmup=warmup_events,
                              span_index=stream.span_index,
                              span_pos=stream.span_pos,
                              event_type_id=stream.event_type_id,
                              span_len=stream.span_len,
                              tokenization=tokenization)
    meta, acts, ktokens, klabels, spans = caching.load_cache_ex(cache_dir)
    n = meta["n_tokens"]
    print(f"cached {n} positions x {len(meta['hook_names'])} hook points "
          f"(warmup {warmup_events} events)")

    # 4. probes + controls --------------------------------------------------
    # One probe row per EVENT. The control sees the event's COMBINED identity
    # (event_type_id), not raw slot tokens, so the Hewitt-Liang bar is
    # comparable across tokenizations.
    pcfg = cfg.get("probe", {})
    mode = str(pcfg.get("mode", "event"))
    span_len = int(meta["span_len"])
    if mode == "event":
        rows = select_event_rows(spans["span_index"], spans["span_pos"],
                                 span_len)
        pos0, etypes = rows[:, 0], spans["event_type_id"][rows[:, 0]]
        assert np.allclose(klabels[rows[:, 0]], klabels[rows[:, -1]]), \
            "latents not constant within span -- alignment broken"
        policy_desc = ("per-token (1:1)" if span_len == 1
                       else f"event-concat ({span_len}x{meta['d_model']})")

        def features(name):
            return np.asarray(acts[name])[rows].reshape(len(rows), -1)
    elif mode == "slot":
        policy = str(pcfg.get("label_position", "last"))
        sel = select_positions(spans["span_index"], spans["span_pos"], policy)
        pos0, etypes = sel, spans["event_type_id"][sel]
        policy_desc = ("per-token (1:1)" if span_len == 1
                       else f"slot:{policy}-of-span")

        def features(name):
            return np.asarray(acts[name])[sel]
    else:
        raise ValueError(f"unknown probe.mode '{mode}' (event | slot)")
    n_probed = len(pos0)
    print(f"probing {n_probed} events from {n} cached positions "
          f"(features @ {policy_desc})")
    ctrl_diag = control_diagnostics(etypes)

    # |R^2_shuffled| tolerance shrinks with sample size like a standard
    # error: coeff / sqrt(n). coeff=4.0 reproduces the old 0.02 at n=40k.
    shuf_coeff = float(cfg.get("shuffled_tolerance_coeff", 4.0))
    shuf_tol = shuf_coeff / math.sqrt(n_probed)
    print(f"shuffled-baseline tolerance: {shuf_coeff}/sqrt({n_probed}) "
          f"= {shuf_tol:.4f}")
    sel_thresh = float(cfg.get("selectivity_threshold", 0.10))
    results: dict[str, dict[str, dict]] = {}
    for name in meta["hook_names"]:
        X = features(name)
        results[name] = {}
        for li, latent in enumerate(meta["latent_names"]):
            r = controls.probe_with_controls(X, klabels[pos0, li], etypes,
                                             seed=seed)
            r["pass_shuffled"] = abs(r["r2_shuffled"]) <= shuf_tol
            r["pass_selectivity"] = r["selectivity"] >= sel_thresh
            # contextual gain: information beyond the current token, measured
            # against the embed layer (a pure function of token identity)
            r["ctx_gain"] = (r["r2_task"]
                             - results[meta["hook_names"][0]][latent]["r2_task"]
                             if name != meta["hook_names"][0] else 0.0)
            results[name][latent] = r
            flag = "" if r["pass_shuffled"] else "  <-- SHUFFLED BASELINE " \
                                                 "NONZERO: DO NOT REPORT"
            print(f"  {name:28s} {latent:16s} "
                  f"R2={r['r2_task']:+.3f} ctrl={r['r2_control']:+.3f} "
                  f"sel={r['selectivity']:+.3f} ctx={r['ctx_gain']:+.3f} "
                  f"shuf={r['r2_shuffled']:+.3f}{flag}")

    # 4b. DT-slot diagnostic ------------------------------------------------
    # Preserved per-slot finding, not an artifact: the DT token alone
    # supports realized_vol R^2 ~0.67 -- inter-arrival time is itself a
    # volatility proxy. Surfaced explicitly here so switching the default to
    # event-level probing does not discard it.
    dt_diag = None
    if mode == "event" and span_len > 1:
        li = meta["latent_names"].index("realized_vol")
        dt_pos = rows[:, -1]
        dt_diag = {}
        for name in meta["hook_names"]:
            r = probes.probe(np.asarray(acts[name])[dt_pos],
                             klabels[dt_pos, li])
            dt_diag[name] = round(r.r2, 4)
        print("DT-slot diagnostic (realized_vol from the DT token alone): "
              + ", ".join(f"{h.replace('blocks.', 'L').replace('.hook_resid_post', '')}"
                          f"={r2:+.2f}" for h, r2 in dt_diag.items()))

    # 5. figure -------------------------------------------------------------
    fig_path = os.path.join(cfg.get("fig_dir", "figures"),
                            f"encoded_where_{run_name}.png")
    plot(results, meta, fig_path, sel_thresh,
         subtitle=f"{tokenization} tokens, features @ {policy_desc}")
    print(f"figure -> {fig_path}")

    # 6. persist ------------------------------------------------------------
    results["_meta"] = {
        "tokenization": tokenization, "probe_mode": mode,
        "policy_desc": policy_desc, "n_probed": int(n_probed),
        "n_cached": int(n), "span_len": span_len,
        "warmup_events": warmup_events,
        "shuffled_tolerance_coeff": shuf_coeff,
        "shuffled_tolerance": shuf_tol,
        "hook_names": meta["hook_names"],
        "latent_names": meta["latent_names"],
        "control_diagnostics": ctrl_diag,
        "dt_slot_realized_vol": dt_diag,
    }
    with open(os.path.join(cache_dir, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    append_results_md(cfg, results, meta, fig_path, policy_desc, ctrl_diag)
    print(f"done in {time.time() - t_start:.0f}s")


def select_event_rows(span_index: np.ndarray, span_pos: np.ndarray,
                      span_len: int) -> np.ndarray:
    """Row-index matrix [n_events, span_len]: the cached token positions of
    every FULLY cached event span, in slot order. BOS (a 1-token span when
    span_len > 1) and a span truncated at the cache tail are dropped; the
    cache itself is a contiguous prefix of the stream (asserted in tests),
    so nothing else can be partial."""
    starts = np.flatnonzero(span_pos == 0)
    starts = starts[starts + span_len <= len(span_index)]
    rows = starts[:, None] + np.arange(span_len)
    ok = ((span_index[rows] == span_index[rows[:, :1]]).all(axis=1)
          & (span_pos[rows] == np.arange(span_len)).all(axis=1))
    return rows[ok]


def select_positions(span_index: np.ndarray, span_pos: np.ndarray,
                     policy: str) -> np.ndarray:
    """Positions to probe. Defined via span_index boundaries (not span_pos
    arithmetic) so 'last'/'first' are correct for BOS and for spans split
    across cache-window edges; in combined mode every position qualifies."""
    if policy == "last":
        return np.flatnonzero(np.r_[span_index[1:] != span_index[:-1], True])
    if policy == "first":
        return np.flatnonzero(np.r_[True, span_index[1:] != span_index[:-1]])
    return np.flatnonzero(span_pos == int(policy))


def control_diagnostics(etypes: np.ndarray) -> dict:
    """The Hewitt-Liang control is only a meaningful bar if event types recur
    enough for the probe to learn type->pseudo-label on train and be tested
    on SEEN types. With a huge factored type space (real orderflow-lm:
    22x2x3x8x14 combos) most types occur once or twice, the control degrades
    and selectivity is unreliable -- so report the distribution loudly."""
    from tape import probes
    tr, _, te = probes.chrono_split(len(etypes))
    types, counts = np.unique(etypes, return_counts=True)
    train_types = np.unique(etypes[tr])
    unseen = ~np.isin(etypes[te], train_types)
    diag = {
        "n_positions": int(len(etypes)),
        "n_distinct_types": int(len(types)),
        "events_per_type": {
            "min": int(counts.min()), "p25": int(np.percentile(counts, 25)),
            "median": int(np.median(counts)),
            "p75": int(np.percentile(counts, 75)), "max": int(counts.max()),
        },
        "frac_types_lt5": float((counts < 5).mean()),
        "frac_test_unseen_type": float(unseen.mean()),
    }
    e = diag["events_per_type"]
    print(f"control types: {diag['n_distinct_types']} distinct; events/type "
          f"min={e['min']} p25={e['p25']} median={e['median']} "
          f"p75={e['p75']} max={e['max']}; "
          f"{diag['frac_types_lt5']:.1%} of types occur <5x; "
          f"{diag['frac_test_unseen_type']:.1%} of test positions have a "
          f"type unseen in train")
    if e["median"] < 5 or diag["frac_test_unseen_type"] > 0.05:
        print("  WARNING: control-task type space is sparse -- the "
              "Hewitt-Liang control degrades and selectivity may collapse "
              "even where encoding is real. Do not read low selectivity as "
              "'nothing encoded' without checking this distribution.")
    return diag


def plot(results, meta, fig_path, sel_thresh, subtitle=""):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    layers = meta["hook_names"]          # x axis: embed, L0..L{n-1}
    xticks = ["embed"] + [f"L{i}" for i in range(len(layers) - 1)]
    xs = np.arange(len(layers))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    cmap = plt.get_cmap("tab10")
    for li, latent in enumerate(meta["latent_names"]):
        c = cmap(li)
        task = [results[h][latent]["r2_task"] for h in layers]
        ctrl = [results[h][latent]["r2_control"] for h in layers]
        shuf = [results[h][latent]["r2_shuffled"] for h in layers]
        sel = [results[h][latent]["selectivity"] for h in layers]
        ctx = [results[h][latent]["ctx_gain"] for h in layers]
        ax1.plot(xs, task, "-o", color=c, label=latent, ms=4)
        ax1.plot(xs, ctrl, "--", color=c, alpha=0.55)
        ax1.plot(xs, shuf, ":", color=c, alpha=0.35)
        ax2.plot(xs, sel, "-o", color=c, ms=4)
        ax3.plot(xs, ctx, "-o", color=c, ms=4)
        for x, r in zip(xs, (results[h][latent] for h in layers)):
            if not r["pass_shuffled"]:
                ax1.plot([x], [r["r2_task"]], "x", color="red", ms=10)
    ax1.set_title("probe R\u00b2 by layer  (solid=task, dashed=control, "
                  "dotted=shuffled)")
    ax2.set_title("selectivity (task \u2212 control)")
    ax2.axhline(sel_thresh, color="gray", lw=0.8, ls="--")
    ax3.set_title("contextual gain (R\u00b2 \u2212 R\u00b2 at embed)")
    ax3.axhline(0.0, color="gray", lw=0.8)
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(xs, xticks)
        ax.set_xlabel("residual stream after")
        ax.grid(alpha=0.25)
    ax1.set_ylabel("held-out R\u00b2")
    ax1.set_ylim(-0.1, 1.0)
    ax1.legend(fontsize=8, loc="upper left")
    title = "Tape: what is encoded where"
    if subtitle:
        title += f"  ({subtitle})"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def append_results_md(cfg, results, meta, fig_path, policy_desc, ctrl_diag):
    path = cfg.get("results_md", "RESULTS.md")
    best = {}
    for latent in meta["latent_names"]:
        b = max(meta["hook_names"],
                key=lambda h: results[h][latent]["ctx_gain"])
        best[latent] = (b, results[b][latent])
    summary = "; ".join(
        f"{lat}: sel={r['selectivity']:.2f}, "
        f"ctx={r['ctx_gain']:.2f}@"
        f"{h.replace('blocks.', 'L').replace('.hook_resid_post', '')}"
        for lat, (h, r) in best.items())
    ok = all(results[h][lat]["pass_shuffled"]
             for h in meta["hook_names"] for lat in meta["latent_names"])
    dt = results["_meta"].get("dt_slot_realized_vol")
    dt_note = (" DT-slot diag: realized_vol R\u00b2="
               f"{max(dt.values()):+.2f}." if dt else "")
    row = (f"| {datetime.date.today()} | {cfg.get('run_name')} "
           f"({cfg['model'].get('arch')}, {len(meta['hook_names'])-1}L, "
           f"{meta['tokenization']}, features@{policy_desc}, "
           f"n={results['_meta']['n_probed']} events of {meta['n_tokens']} "
           f"cached) "
           f"| held-out R\u00b2 / selectivity | {summary}. "
           f"Shuffled baselines {'all ~0 (pass)' if ok else 'NONZERO (fail)'}"
           f" (tol {results['_meta']['shuffled_tolerance']:.4f})."
           f" Ctrl types={ctrl_diag['n_distinct_types']} "
           f"(median {ctrl_diag['events_per_type']['median']}/type)."
           f"{dt_note} Fig: {fig_path} |\n")
    with open(path, "a") as f:
        f.write(row)
    print(f"RESULTS.md row appended -> {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python run_probe.py configs/synthetic.yaml")
    main(sys.argv[1])
