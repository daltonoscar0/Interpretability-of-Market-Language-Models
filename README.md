# Tape

[![tests](https://github.com/daltonoscar0/Interpretability-of-Market-Language-Models/actions/workflows/tests.yml/badge.svg)](https://github.com/daltonoscar0/Interpretability-of-Market-Language-Models/actions/workflows/tests.yml)

This repository turns raw limit order book data into a trained language model
and then interrogates the model. The pipeline parses LOBSTER message files,
reconstructs the order book and validates the reconstruction row by row
against LOBSTER's own orderbook files, then tokenizes each market event into a
five-token factored sequence: type, side, price offset, size, inter-event
time. A small autoregressive transformer trains on that stream. The probing
suite asks the question the project exists for: does the model encode latent
book state, meaning spread, queue imbalance, order flow imbalance, distance to
touch, and realized volatility, or does it predict the next event from surface
statistics of the tape? Layerwise linear probes answer it, with controls built
to separate the two.

## Layout

`pipeline/` is C++17 with CMake and no dependencies beyond the standard
library. It owns everything up to and including the binary token stream:
parser, book reconstruction, validation tools, and the tokenizer with its
fit/apply split that freezes quantile bins on training data. `tape/` and
`run_probe.py` are Python (numpy, torch, pyyaml, matplotlib) and own the
analysis: model training, activation caching, probes, controls, figures.

They are separate because they are different kinds of code with different
lifetimes. The pipeline is data engineering: it has to be exact, it changes
rarely, and it is useful to someone who never runs a probe. The analysis half
is research code that changes with every experiment and has no business
reimplementing CSV parsing or book logic. The two halves meet at three files:
`SPY_50.tokens.bin`, `manifest.json` (vocabulary, frozen bin edges, and the
train/eval row boundary), and the LOBSTER orderbook CSV, which the Python side
reads for ground-truth latents instead of reconstructing book state itself.

## The LOBSTER censoring finding

LOBSTER's message file reports only the events that update the book inside the
sample's requested price range. Reconstruct the book from messages alone and
it drifts from the companion orderbook file: worst at the window edge, and by
an amount that depends on the ticker even on the identical day and window.

Measured with `pipeline/build/validate_book` on the 50-level one-hour samples
(2012-06-21, 09:30 to 10:30), cumulative fatal mismatch rate at depth K:

| ticker (rows) | K=1 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| MSFT (142k) | 0.00% | 0.00% | 0.00% | 15.85% |
| AAPL (92k) | 0.10% | 1.16% | 2.51% | 9.18% |
| SPY (1.15M) | 0.26% | 2.83% | 9.98% | 35.50% |

MSFT's top 11 levels are exact at every one of its 142k message indices; its
first mismatch anywhere in the file is at level 12. SPY is 9.98% wrong by
depth 10 on the same day and window. Narrow windows are far worse than shallow
depth: on the 10-level full-day AAPL file, 99.21% of rows mismatch at level 1,
contamination reaches top of book by message 440, and there are 5,423 phantom
reductions, meaning deletes or cancels against orders the message stream never
showed. The 50-level hour files have 5 (MSFT), 36 (SPY), and 41 (AAPL). A
separate censoring applies at the file's depth limit: AAPL's occupied-level
distribution reaches level 49, the deepest a 50-level file can express, with
19.2% of its events beyond level +10, so its deep tail is truncated by the
file, not by the market.

Two decisions follow, recorded with evidence in `pipeline/SPEC.md`. Book-state
features come from the orderbook file, never from our reconstruction;
reconstruction stays in the tree for ITCH data, where no orderbook file
exists. And the training corpus uses 50-level samples only, without pooling
the 30-level samples, which cover the same ticker-day-window and would
duplicate events across the train/eval split.

## What the probing suite measures

For each residual-stream layer and each of the five latents, the suite fits a
closed-form ridge probe on standardized activations with chronological
70/15/15 splits. The unit of analysis is the event: in factored mode the
activations of the event's five token positions concatenate into one feature
row. Three numbers guard every reading:

- Selectivity: task R² minus the R² of the same probe fit on a Hewitt-Liang
  control, a fixed random pseudo-label per event type. The control's notion of
  event type is (TYPE, SIDE, PRICE_OFF, SIZE), with DT deliberately excluded
  so the control cannot absorb inter-arrival information.
- Shuffled baseline: must sit within noise of zero (tolerance 4.0/sqrt(n)), or
  the result is marked do-not-report.
- Contextual gain: R² at the layer minus R² at the embedding layer. The
  embedding is a pure function of the current event's tokens, so this isolates
  what the model integrated from context.

dist_to_touch is the case that shows why the gates exist. On the SPY run it
probes at R² 0.905 at the embedding layer, before any attention has run, and
its contextual gain never exceeds +0.007 at any layer. The token stream
literally contains the answer: PRICE_OFF is a bucketed distance to touch, so a
linear readout of token identity recovers the latent while the model computes
nothing. On real data it is contextual gain that catches this, not
selectivity: the factored control, with 312 event types in this eval window,
only reaches R² 0.145, so selectivity sits at a healthy-looking 0.76.
Selectivity alone would have called this encoding; the two gates catch
different failure modes.

## What the probes found on SPY

From `cache/spy/probe_results.json`: 19,995 eval-split events, probed against
the 4-layer model trained to held-out loss 0.863 (train 0.807).

- spread: encoded. R² 0.143 at the embedding rising to 0.843 at L2,
  contextual gain +0.70, selectivity +0.53. One caveat travels with it: SPY's
  spread is effectively two-valued this hour, so contextual gain establishes
  that the information came from context, not that the mechanism is book
  tracking rather than memory of recent inside-spread tokens.
- realized_vol: encoded. R² 0.288 at the embedding rising to 0.826 at L1,
  contextual gain +0.54. The DT-slot diagnostic shows the DT token's residual
  stream alone reaches 0.76 by L2: inter-arrival time is itself a volatility
  proxy, and the suite prints that line on every factored run rather than
  letting event-level probing bury it.
- ofi: moderate, rising with depth. R² 0.259 at L2 with contextual gain
  +0.25. The L3 reading (0.348) is auto-flagged do-not-report because its
  shuffled baseline reached the tolerance.
- dist_to_touch: token-local, refused. The case above.
- queue_imbalance: not measurable with this split, which is not the same as
  not encoded. The test block's latent mean is +0.383 against a train mean of
  +0.044; the probe's validation R² at L2 is +0.094 while its test R² is
  -0.530; and refitting entirely within the test block also fails. A
  regime-persistent latent defeats a chronological split. The block-bootstrap
  item in `PLAN.md` exists for this.

## Scope

Everything here runs on the free LOBSTER academic samples: one day,
2012-06-21, one hour, 09:30 to 10:30, 50 levels of book, for AAPL, MSFT, and
SPY. The trained model sees SPY only: 1.15M events, about 5.8M tokens, split
80/20 chronologically. Every probing result inherits that scope; it is a
statement about this model trained on this hour of this ticker. SPY in 2012 is
also a tick-constrained ETF whose spread sits at one tick most of the hour,
which shapes what "spread" even varies over. AAPL and MSFT are tokenized with
the same scheme and reserved as held-out generalization checks. The
price-window censoring above bounds the model's input as well: it only ever
observes events inside the sample's window.

## Running it

The C++ half:

```sh
cd pipeline
cmake -B build && cmake --build build -j
ctest --test-dir build
```

Token streams and the manifest are committed under `pipeline/data/tokens/`
(the three `.tokens.bin` files total about 14 MB), so the Python half trains
without any download. They are regenerable — steps 2-4 of
[`pipeline/README.md`](pipeline/README.md) fit the bins on the training split,
tokenize, and round-trip the result — but doing so needs the raw LOBSTER CSVs,
so they are kept in the repo rather than regenerated:

```sh
pipeline/scripts/fetch_lobster_sample.sh                      # ~1.7 GB
cd pipeline && build/tokenizer_fit ... && build/tokenize ... && build/token_roundtrip ...
``` The raw CSVs are needed only for probing latents and for
reproducing the corpus; they total about 1.7 GB and are fetched with
confirmation:

```sh
pipeline/scripts/fetch_lobster_sample.sh
```

The full corpus reproduction sequence (validate, fit bins, tokenize,
round-trip) is in `pipeline/README.md`.

The Python half, from the repo root:

```sh
pip install torch pyyaml matplotlib
python -m tape.tests.test_sanity                   # ~30 s
python run_probe.py configs/synthetic.yaml         # synthetic stand-in, no data needed
python -c "
from tape.model_loader import ModelConfig, train_spy
train_spy(ModelConfig(vocab_size=52, n_ctx=320),
          'checkpoints/spy_minigpt.pt',
          'pipeline/data/tokens/SPY_50.tokens.bin',
          'pipeline/data/tokens/manifest.json', steps=8000)"   # ~35 min CPU
python run_probe.py configs/spy.yaml               # needs the orderbook CSV
```

Each probe run writes `figures/encoded_where_<run>.png`, appends a row to
`RESULTS.md`, and dumps per-probe JSON under `cache/<run>/`. Probe method
detail lives in `tape/README.md`, run history in `RESULTS.md`, the roadmap in
`PLAN.md`.

MIT license, see `LICENSE`.

## License

MIT. See [LICENSE](LICENSE).
