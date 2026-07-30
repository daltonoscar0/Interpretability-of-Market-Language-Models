# PLAN — Tape

Goal: determine what an order-flow LM encodes about latent book state
(spread, queue imbalance, OFI, distance-to-touch, realized vol) — and then
whether it *uses* those representations. Methods and hygiene rules live in
README.md; run history in RESULTS.md.

## Phase 1 — probing suite

Machinery done and validated on synthetic stand-ins: residual-stream
caching, event-level ridge probes (both tokenizations), Hewitt–Liang
controls + sample-size-scaled shuffled baselines, contextual gain,
"what is encoded where" figures.

Remaining:
- [x] Real orderflow-lm checkpoint + LOBSTER eval stream (2026-07-30:
      `configs/spy.yaml`, lobster branch in run_probe, orderbook-based
      latents in tape/data.py; see README "What the probes found on SPY")
- [ ] Probe error bars: multi-seed control/shuffled draws + block bootstrap
      over the chronological test split. Concrete motivator: queue_imbalance
      on SPY is regime-persistent (test-block mean +0.383 vs train +0.044)
      and unmeasurable with a single chronological split.
- [ ] Per-layer shuffled tolerance. On the SPY run OFI's shuffled baselines
      drift monotonically with depth (-0.002, -0.002, -0.014, -0.015,
      -0.028 across embed..L3). That is not noise: deeper layers have more
      correlated features, so ridge overfits shuffled labels slightly more.
      The L3 do-not-report is therefore systematic rather than a coin flip,
      and the drift will bite other latents in deeper models. The fixed
      coefficient 4.0/sqrt(n) may need to become per-layer.

## Phase 2 — causal analysis

- [ ] Activation patching: does the imbalance representation drive
      next-message predictions? Clean/corrupted book-state pairs (prototype
      via forced-imbalance interventions in SyntheticStream)
- [ ] Logit-lens pass as a cheap cross-check

Milestone: at least one causal claim with effect sizes + controls.

(A Phase-3 SAE/feature-atlas stage was cut on 2026-07-30 — out of scope.)

## Status — 2026-07-30

Phase 1 machinery complete; headline numbers blocked on the real
checkpoint. Synthetic results (see RESULTS.md): combined mode encodes
queue_imbalance and realized_vol with clear selectivity; factored mode is
probed at the event level (concatenated spans), with the DT-slot
realized_vol ≈ 0.67 result preserved as an explicit diagnostic.
