"""Synthetic order-flow stream + tokenizer + ground-truth latent extraction.

This module is the ONLY place that knows how (token stream, per-step latents)
are produced. To swap in the real orderflow-lm eval stream later, implement
another `EventSource` that yields the same `Event` records from LOBSTER-derived
messages and point the config's `data.source` at it. Everything downstream
(caching, probes, controls) consumes StreamData arrays and is agnostic.

Two tokenizations of the same event process:
  combined : one token per event, f"{TYPE}_{SIDE}_{LEVEL}_{SIZE}" (121 ids)
  factored : five tokens per event, [TYPE][SIDE][PRICE_OFF][SIZE][DT],
             matching orderflow-lm's factored-span scheme (19 ids).
Both return a StreamData with per-token span bookkeeping; combined mode is
the degenerate span_len=1 case, so downstream code has one uniform shape.

Generator: a Cont-Stoikov-Talreja-flavored queue-reactive process.
Poisson-ish arrivals of limit adds / cancels / market orders on a tick grid
around a moving mid. Event intensities are modulated by queue imbalance and
recent signed flow so the stream has genuinely predictable structure for the
LM to learn (and hence something for the probes to find).

Latents extracted per event step (AFTER the event is applied):
  0 spread            : best_ask - best_bid, in ticks
  1 queue_imbalance   : (q_bid - q_ask) / (q_bid + q_ask) at the touch
  2 ofi               : Cont-Kukanov-Stoikov order-flow imbalance, rolling W_OFI
  3 dist_to_touch     : |event price - same-side touch| in ticks (MO = 0)
  4 realized_vol      : std of 1-step mid returns over rolling W_VOL
"""

from __future__ import annotations

import collections
import dataclasses
import math
import random

import numpy as np

# ---------------------------------------------------------------- tokenizer

EVENT_TYPES = ["ADD", "CAN", "MO"]          # limit add, cancel, market order
SIDES = ["B", "A"]
LEVEL_BUCKETS = ["INSIDE", "L0", "L1", "L2", "L3P"]   # dist-to-touch buckets
SIZE_BUCKETS = ["S", "M", "L", "XL"]                  # size quantiles

LATENT_NAMES = ["spread", "queue_imbalance", "ofi", "dist_to_touch",
                "realized_vol"]
N_LATENTS = len(LATENT_NAMES)

W_OFI = 20      # rolling window (events) for OFI
W_VOL = 50      # rolling window (events) for realized vol

BOS_TOKEN = "BOS"


def build_vocab() -> dict[str, int]:
    vocab = {BOS_TOKEN: 0}
    for t in EVENT_TYPES:
        for s in SIDES:
            for lb in LEVEL_BUCKETS:
                for sb in SIZE_BUCKETS:
                    vocab[f"{t}_{s}_{lb}_{sb}"] = len(vocab)
    return vocab


VOCAB = build_vocab()
VOCAB_SIZE = len(VOCAB)          # 1 + 3*2*5*4 = 121
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

# ------------------------------------------------- factored tokenization

SPAN_LEN = 5                     # [TYPE][SIDE][PRICE_OFF][SIZE][DT]
N_DT_BUCKETS = 4
DT_THRESHOLDS = (0.5, 1.5, 3.0)  # exponential(1) inter-arrival -> 4 buckets


def build_factored_vocab() -> dict[str, int]:
    """Slot-tagged factored vocab: one token per event attribute, five tokens
    per event, mirroring orderflow-lm's factored-span scheme. Slot prefixes
    keep the sub-vocabularies disjoint."""
    vocab = {BOS_TOKEN: 0}
    for t in EVENT_TYPES:
        vocab[f"TYPE_{t}"] = len(vocab)
    for s in SIDES:
        vocab[f"SIDE_{s}"] = len(vocab)
    for lb in LEVEL_BUCKETS:
        vocab[f"OFF_{lb}"] = len(vocab)
    for sb in SIZE_BUCKETS:
        vocab[f"SIZE_{sb}"] = len(vocab)
    for d in range(N_DT_BUCKETS):
        vocab[f"DT_{d}"] = len(vocab)
    return vocab


FACTORED_VOCAB = build_factored_vocab()
FACTORED_VOCAB_SIZE = len(FACTORED_VOCAB)   # 1 + 3+2+5+4+4 = 19
FACTORED_ID_TO_TOKEN = {v: k for k, v in FACTORED_VOCAB.items()}


def vocab_size(tokenization: str) -> int:
    return {"combined": VOCAB_SIZE, "factored": FACTORED_VOCAB_SIZE}[
        tokenization]


def dt_bucket(dt: float) -> int:
    for i, th in enumerate(DT_THRESHOLDS):
        if dt < th:
            return i
    return len(DT_THRESHOLDS)


def size_bucket(size: int) -> str:
    if size <= 2:
        return "S"
    if size <= 5:
        return "M"
    if size <= 10:
        return "L"
    return "XL"


def level_bucket(dist: int) -> str:
    # dist in ticks from same-side touch; -1 == inside the spread
    if dist < 0:
        return "INSIDE"
    if dist == 0:
        return "L0"
    if dist == 1:
        return "L1"
    if dist == 2:
        return "L2"
    return "L3P"


@dataclasses.dataclass
class Event:
    etype: str      # ADD | CAN | MO
    side: str       # B | A
    dist: int       # ticks from same-side touch at event time (-1 = inside)
    size: int

    def token(self) -> str:
        return (f"{self.etype}_{self.side}_"
                f"{level_bucket(self.dist)}_{size_bucket(self.size)}")


# ---------------------------------------------------------------- book state


class Book:
    """Minimal L2 book on an integer tick grid."""

    K = 5  # levels tracked per side (beyond that liquidity is 'regenerated')

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.best_bid = 1000
        self.best_ask = 1002
        # price -> size
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        for i in range(self.K):
            self.bids[self.best_bid - i] = rng.randint(3, 12)
            self.asks[self.best_ask + i] = rng.randint(3, 12)

    # -- helpers ----------------------------------------------------------
    @property
    def spread(self) -> int:
        return self.best_ask - self.best_bid

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    def touch_sizes(self) -> tuple[int, int]:
        return (self.bids.get(self.best_bid, 0),
                self.asks.get(self.best_ask, 0))

    def imbalance(self) -> float:
        qb, qa = self.touch_sizes()
        tot = qb + qa
        return (qb - qa) / tot if tot > 0 else 0.0

    def _regen(self, side: str) -> None:
        """Keep K levels populated behind the touch."""
        book = self.bids if side == "B" else self.asks
        sign = -1 if side == "B" else +1
        touch = self.best_bid if side == "B" else self.best_ask
        for i in range(self.K):
            p = touch + sign * i
            if book.get(p, 0) <= 0:
                book[p] = self.rng.randint(2, 8)
        # drop stale far levels
        for p in list(book.keys()):
            if abs(p - touch) >= self.K + 2:
                del book[p]

    # -- event application ------------------------------------------------
    def apply(self, ev: Event) -> None:
        if ev.etype == "ADD":
            self._apply_add(ev)
        elif ev.etype == "CAN":
            self._apply_cancel(ev)
        else:
            self._apply_mo(ev)
        # invariant: bid < ask, both sides populated
        assert self.best_bid < self.best_ask, "crossed book"
        self._regen("B")
        self._regen("A")

    def _apply_add(self, ev: Event) -> None:
        if ev.side == "B":
            price = self.best_bid + (1 if ev.dist < 0 else -ev.dist)
            if ev.dist < 0:                      # improves the bid
                self.best_bid = price
            self.bids[price] = self.bids.get(price, 0) + ev.size
        else:
            price = self.best_ask + (-1 if ev.dist < 0 else ev.dist)
            if ev.dist < 0:
                self.best_ask = price
            self.asks[price] = self.asks.get(price, 0) + ev.size

    def _apply_cancel(self, ev: Event) -> None:
        book = self.bids if ev.side == "B" else self.asks
        sign = -1 if ev.side == "B" else +1
        touch = self.best_bid if ev.side == "B" else self.best_ask
        price = touch + sign * ev.dist
        if price in book:
            book[price] = max(0, book[price] - ev.size)
            if book[price] == 0 and ev.dist == 0:
                self._retreat(ev.side)

    def _apply_mo(self, ev: Event) -> None:
        # MO side B == buy market order -> consumes the ASK touch
        opp = "A" if ev.side == "B" else "B"
        book = self.asks if opp == "A" else self.bids
        touch = self.best_ask if opp == "A" else self.best_bid
        remaining = ev.size
        while remaining > 0:
            avail = book.get(touch, 0)
            take = min(avail, remaining)
            book[touch] = avail - take
            remaining -= take
            if book[touch] == 0:
                self._retreat(opp)
                touch = self.best_ask if opp == "A" else self.best_bid
                if remaining > 0 and touch not in book:
                    book[touch] = self.rng.randint(2, 8)
            else:
                break

    def _retreat(self, side: str) -> None:
        """Touch depleted: move touch one tick away, widening the spread."""
        if side == "B":
            self.bids.pop(self.best_bid, None)
            self.best_bid -= 1
            if self.best_bid not in self.bids:
                self.bids[self.best_bid] = self.rng.randint(2, 8)
        else:
            self.asks.pop(self.best_ask, None)
            self.best_ask += 1
            if self.best_ask not in self.asks:
                self.asks[self.best_ask] = self.rng.randint(2, 8)


# ---------------------------------------------------------------- generator


class SyntheticStream:
    """Queue-reactive event stream with ground-truth latents.

    Predictable structure (what the LM can learn):
      * P(market buy) increases with positive queue imbalance (and vice versa)
        -- the classic queue-reactive stylized fact.
      * Flow momentum: recent signed OFI tilts add/MO side probabilities.
      * Spread > 1 raises the probability of an INSIDE add (spread-closing).
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.book = Book(self.rng)
        self.mids = collections.deque(maxlen=W_VOL + 1)
        self.ofi_terms = collections.deque(maxlen=W_OFI)
        self.mids.append(self.book.mid)
        self._prev_touch = self._touch_state()

    def _touch_state(self):
        qb, qa = self.book.touch_sizes()
        return (self.book.best_bid, qb, self.book.best_ask, qa)

    # -- event sampling ---------------------------------------------------
    def _sample_event(self) -> Event:
        rng = self.rng
        imb = self.book.imbalance()
        ofi = float(np.sum(self.ofi_terms)) if self.ofi_terms else 0.0
        ofi_n = math.tanh(ofi / 10.0)

        # base intensities
        p_add, p_can, p_mo = 0.50, 0.30, 0.20
        if self.book.spread > 1:
            p_add += 0.15
            p_mo = max(0.05, p_mo - 0.10)
        r = rng.random() * (p_add + p_can + p_mo)
        if r < p_add:
            etype = "ADD"
        elif r < p_add + p_can:
            etype = "CAN"
        else:
            etype = "MO"

        # side choice modulated by imbalance + flow momentum
        drift = 0.5 + 0.45 * (0.8 * imb + 0.2 * ofi_n)
        drift = min(0.95, max(0.05, drift))
        if etype == "MO":
            side = "B" if rng.random() < drift else "A"       # buy MO w/ +imb
        elif etype == "ADD":
            side = "B" if rng.random() < 0.5 - 0.15 * imb else "A"
        else:  # CAN: cancel the crowded side slightly more
            side = "B" if rng.random() < 0.5 + 0.15 * imb else "A"

        if etype == "ADD":
            if self.book.spread > 1 and rng.random() < 0.35:
                dist = -1                                     # inside spread
            else:
                dist = min(int(rng.expovariate(0.9)), 4)
        elif etype == "CAN":
            dist = min(int(rng.expovariate(1.2)), 4)
        else:
            dist = 0

        size = max(1, min(20, int(rng.expovariate(0.25))))
        return Event(etype, side, dist, size)

    # -- OFI (Cont-Kukanov-Stoikov e_n) ----------------------------------
    def _ofi_term(self, prev, cur) -> float:
        pb, qb0, pa, qa0 = prev
        b, qb1, a, qa1 = cur
        if b > pb:
            e_bid = qb1
        elif b == pb:
            e_bid = qb1 - qb0
        else:
            e_bid = -qb0
        if a < pa:
            e_ask = qa1
        elif a == pa:
            e_ask = qa1 - qa0
        else:
            e_ask = -qa0
        return float(e_bid - e_ask)

    # -- main loop --------------------------------------------------------
    def generate_events(self, n_steps: int
                        ) -> tuple[list[Event], np.ndarray]:
        """Runs the process; returns (events [n_steps],
        latents [n_steps+1, N_LATENTS]). latents[0] describes the initial
        book (BOS); latents[k] the state AFTER event k was applied."""
        events: list[Event] = []
        latents = np.empty((n_steps + 1, N_LATENTS), dtype=np.float32)
        latents[0] = self._latents(dist=0)
        for i in range(1, n_steps + 1):
            ev = self._sample_event()
            self.book.apply(ev)
            cur = self._touch_state()
            self.ofi_terms.append(self._ofi_term(self._prev_touch, cur))
            self._prev_touch = cur
            self.mids.append(self.book.mid)
            events.append(ev)
            latents[i] = self._latents(dist=max(ev.dist, 0))
        return events, latents

    def generate(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Combined tokenization: (tokens [n_steps+1], latents [n_steps+1, L]).

        Position 0 is BOS with latents describing the initial book, so that
        tokens[i] aligns with latents[i] = state AFTER event i was applied.
        """
        events, latents = self.generate_events(n_steps)
        tokens = np.empty(n_steps + 1, dtype=np.int64)
        tokens[0] = VOCAB[BOS_TOKEN]
        for i, ev in enumerate(events, start=1):
            tokens[i] = VOCAB[ev.token()]
        return tokens, latents

    def _latents(self, dist: int) -> np.ndarray:
        mids = np.asarray(self.mids, dtype=np.float64)
        if len(mids) >= 3:
            rets = np.diff(mids)
            rvol = float(np.std(rets))
        else:
            rvol = 0.0
        ofi = float(np.sum(self.ofi_terms)) if self.ofi_terms else 0.0
        return np.array([
            float(self.book.spread),
            float(self.book.imbalance()),
            ofi,
            float(dist),
            rvol,
        ], dtype=np.float32)


@dataclasses.dataclass
class StreamData:
    """Per-token aligned arrays. Combined mode is the degenerate span_len=1
    case (span_index=arange, span_pos=0, event_type_id=tokens)."""
    tokenization: str
    span_len: int
    tokens: np.ndarray         # [n_tok] int64
    latents: np.ndarray        # [n_tok, N_LATENTS]; owning event's post-state
    span_index: np.ndarray     # [n_tok] token position -> event idx (BOS = 0)
    span_pos: np.ndarray       # [n_tok] position within span, 0..span_len-1
    event_type_id: np.ndarray  # [n_tok] combined-vocab id of the owning event


def generate_split_ex(seed: int, n_steps: int,
                      tokenization: str = "combined") -> StreamData:
    """Public entry point used by training and caching."""
    events, ev_latents = SyntheticStream(seed).generate_events(n_steps)

    if tokenization == "combined":
        n_tok = n_steps + 1
        tokens = np.empty(n_tok, dtype=np.int64)
        tokens[0] = VOCAB[BOS_TOKEN]
        for i, ev in enumerate(events, start=1):
            tokens[i] = VOCAB[ev.token()]
        return StreamData("combined", 1, tokens, ev_latents,
                          np.arange(n_tok, dtype=np.int64),
                          np.zeros(n_tok, dtype=np.int64), tokens.copy())

    if tokenization != "factored":
        raise ValueError(f"unknown tokenization '{tokenization}'")

    # DT is sampled at tokenization time from a DEDICATED rng: drawing it
    # inside the event loop would shift every subsequent event sample and
    # silently change combined-mode streams (invalidating the committed
    # checkpoint, caches, and determinism tests).
    dt_rng = random.Random(f"dt-{seed}")
    n_tok = 1 + SPAN_LEN * n_steps
    tokens = np.empty(n_tok, dtype=np.int64)
    latents = np.empty((n_tok, N_LATENTS), dtype=np.float32)
    span_index = np.empty(n_tok, dtype=np.int64)
    span_pos = np.empty(n_tok, dtype=np.int64)
    event_type_id = np.empty(n_tok, dtype=np.int64)
    tokens[0] = FACTORED_VOCAB[BOS_TOKEN]
    latents[0] = ev_latents[0]
    span_index[0], span_pos[0] = 0, 0
    event_type_id[0] = VOCAB[BOS_TOKEN]
    for k, ev in enumerate(events, start=1):
        ids = (FACTORED_VOCAB[f"TYPE_{ev.etype}"],
               FACTORED_VOCAB[f"SIDE_{ev.side}"],
               FACTORED_VOCAB[f"OFF_{level_bucket(ev.dist)}"],
               FACTORED_VOCAB[f"SIZE_{size_bucket(ev.size)}"],
               FACTORED_VOCAB[f"DT_{dt_bucket(dt_rng.expovariate(1.0))}"])
        sl = slice(1 + SPAN_LEN * (k - 1), 1 + SPAN_LEN * k)
        tokens[sl] = ids
        latents[sl] = ev_latents[k]      # post-event state, broadcast to span
        span_index[sl] = k
        span_pos[sl] = np.arange(SPAN_LEN)
        event_type_id[sl] = VOCAB[ev.token()]
    return StreamData("factored", SPAN_LEN, tokens, latents,
                      span_index, span_pos, event_type_id)


def generate_split(seed: int, n_steps: int):
    """Combined-mode entry point, kept for existing callers/tests."""
    s = generate_split_ex(seed, n_steps, "combined")
    return s.tokens, s.latents
