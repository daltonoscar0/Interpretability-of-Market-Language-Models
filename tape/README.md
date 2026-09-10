# Tape

What does an order-flow language model represent about latent book state?

Tape is a probing suite for LMs trained on limit-order-book event streams.
Given a model and an eval stream, it caches residual-stream activations,
fits layerwise linear probes for microstructure latents, spread, queue
imbalance, order-flow imbalance (OFI), distance-to-touch, realized
volatility, and reports which are actually encoded, guarded by controls
designed to kill boring explanations before they become claims.

Currently validated end-to-end on a synthetic queue-reactive stand-in
(4-layer miniGPT); the drop-in target is the real orderflow-lm checkpoint
with a LOBSTER-derived eval stream.

## Quickstart

```
pip install torch pyyaml matplotlib          # numpy assumed; CPU torch is fine
python -m tape.tests.test_sanity             # ~30s
python run_probe.py configs/synthetic.yaml            # combined tokenization
python run_probe.py configs/synthetic_factored.yaml   # factored tokenization
python -m tape.tests.test_factored_parity    # machinery check + comparison
```

First run of each config trains its stand-in model (~4-20 min CPU) and
caches it under `checkpoints/`; subsequent runs reuse it (~25s). Each run
writes `figures/encoded_where_<run>.png`, appends a row to `RESULTS.md`,
and dumps per-probe JSON to `cache/<run>/probe_results.json`.

## Two tokenizations, one unit of analysis

The same event process is tokenized two ways, mirroring orderflow-lm:

- **combined**, one token per event, `{TYPE}_{SIDE}_{LEVEL}_{SIZE}`
  (121 ids);
- **factored**, five tokens per event, `[TYPE][SIDE][PRICE_OFF][SIZE][DT]`
  (19 ids).

Probing always operates at the **event** level. In factored mode the
default probe concatenates the residual activations across the event's
5-token span into one `5 × d_model` feature row; combined mode is the
degenerate `span_len = 1` case. This keeps the two modes comparable and -
critically, keeps the Hewitt-Liang control non-vacuous in factored mode:
event identity is linearly recoverable from the full span, as it is from a
combined token. (The control's *level* still differs by mode: a linear
probe on concatenated slot embeddings only reaches the slot-additive
component of a random joint-identity pseudo-label, ~0.2, while a combined
embedding supports near-free-form functions of joint identity, ~0.93.
Non-vacuous means decisively above shuffled noise, not near 1.) Probing a
single slot instead (available as `probe.mode: slot` +
`probe.label_position`) is a useful diagnostic but a broken default: at the
DT slot the control collapses to ~0 and selectivity loses its bar.

One per-slot finding is preserved explicitly rather than discarded with the
default switch: the DT token alone supports realized_vol R² ≈ 0.67 -
inter-arrival time is itself a volatility proxy. `run_probe` surfaces this
as a `DT-slot diagnostic` line and as `dt_slot_realized_vol` in the results
JSON whenever it probes a factored stream in event mode.

## Probe hygiene

Every (layer, latent) probe is a closed-form ridge regression on
standardized activations, with **chronological** 70/15/15 splits (random
splits leak: order-flow latents are strongly autocorrelated) and lambda
chosen on the validation block. Reported alongside held-out task R²:

- **Control task** (Hewitt-Liang, regression adaptation): a fixed random
  pseudo-label per event type, drawn from the empirical latent marginal.
  `selectivity = R²_task − R²_control`, high control R² means the probe
  could be reading mere token identity.
- **Shuffled-label baseline**: must sit within noise of zero or the run is
  flagged DO NOT REPORT. The tolerance scales with sample size:
  `shuffled_tolerance_coeff / sqrt(n_probed)` (coefficient 4.0 ≡ the old
  0.02 at n = 40k).
- **Contextual gain**: `R²_task(layer) − R²_task(embed)`. The embed layer
  is a pure function of the current event's tokens, so this isolates state
  integrated from context and refuses to call token-local information
  "state encoding".

Gate failures are printed loudly and marked on the figure, never silently
dropped.

## Swapping in the real orderflow-lm checkpoint

1. `configs/*.yaml`: set `model.checkpoint: /path/to/orderflow_lm.pt`. If
   its format isn't a MiniGPT state_dict, register a loader under its
   `arch` name in `tape/model_loader.py`.
2. Set `data.source: lobster` and implement the LOBSTER `EventSource`
   branch in `tape/data.py`, label extraction is deliberately isolated
   there.

## Layout

- `tape/data.py`, synthetic queue-reactive generator, both tokenizers,
  ground-truth latent extraction (the only module that knows data
  provenance)
- `tape/model_loader.py`, MiniGPT with TransformerLens-style hook names
  (`blocks.{l}.hook_resid_post`, …), `run_with_cache`, stand-in trainer
- `tape/caching.py`, memmapped residual-stream cache with event-based
  context warmup and per-token span bookkeeping
- `tape/probes.py`, closed-form ridge probes, chronological splits
- `tape/controls.py`, Hewitt-Liang control tasks, shuffled baselines,
  selectivity
- `run_probe.py`, end-to-end driver: event-row selection, probes, gates,
  figure, RESULTS.md logging
- `tape/tests/`, sanity checks (generator invariants, hook shapes, cache
  alignment, span structure) and the factored machinery suite
- `PLAN.md`, roadmap; `RESULTS.md`, run log

## License

MIT, see `LICENSE`.
