# orderflow-lm

Language-model analysis toolkit pointed at limit order book message streams.
Treat market events as a discrete symbol system: tokenize, train an
autoregressive model, then analyze it with surprisal and probing methods.

## Owner context
- Author is experienced in C++17, from-scratch transformers (miniGPT), and
  FST-based tokenization (Scalpel). Prefer that style: zero/minimal
  dependencies, typed structs, explicit ownership, no framework bloat.
- Downstream model training reuses miniGPT (separate repo). This repo owns
  everything up to and including the binary token stream.

## Architecture (pipeline stages)
1. `src/parser/`, LOBSTER message-file parser (CSV) first; ITCH 5.0 binary later.
2. `src/book/`, order book reconstruction (per-side price-level book),
                      validated against LOBSTER's precomputed orderbook files.
3. `src/tokenizer/`, normalized event -> factored token sequence
                      [TYPE][SIDE][PRICE_OFF][SIZE][DT], FST-style transduction.
4. `src/bpe/`, BPE over the factored stream; vocab-size sweep (raw, 1k, 4k, 16k).
5. `analysis/`, corpus statistics: token frequency (Zipf plot), bigram
                      stats, round-trip validation. Python is fine here.

## Tokenization spec (source of truth: SPEC.md)
- Price: signed occupied-level index on the event's side (-1 = inside
  spread, 0 = at best, +k = k occupied levels better), buckets -1..+10
  plus one positive tail.
- Size: quantile bins (8) fit on TRAINING data only, never full-dataset stats.
- Time: per-event delta-t token, log-bucketed (~14 buckets, sub-ms to multi-sec).
- Identity: v1 drops order reference identity; document as a limitation.
- Boundaries: never cross day boundaries; special tokens for session open/close,
  halt/resume, BOS/EOS, UNK.

## Conventions
- C++17, CMake, no external deps beyond the standard library for pipeline code.
- Every stage reads stdin-or-file, writes a well-defined binary or CSV format;
  stages must be runnable and testable independently.
- Tests in `tests/` (assert-based is fine; no gtest dependency unless asked).
- Correctness before speed, but keep the book hot path allocation-free.

## Validation invariants (do not skip)
- Book reconstruction must match LOBSTER's orderbook file exactly at every
  message index for the sample data. Mismatch = bug, full stop.
- Tokenizer must round-trip: tokens -> approximate event -> same buckets.
