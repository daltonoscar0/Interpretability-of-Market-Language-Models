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
- [ ] Real orderflow-lm checkpoint + LOBSTER eval stream
      (`model.checkpoint` in config; `EventSource` branch in tape/data.py)
- [ ] Probe error bars: multi-seed control/shuffled draws + block bootstrap
      over the chronological test split

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
