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
import json
import math
import random
import struct

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


# ==================================================================== real data
# Reader for the orderflow-lm factored token stream (pipeline/src/tokenizer).
# This half of the module produces token ids from the committed .tokens.bin +
# manifest.json; the synthetic half above is untouched. Book-state LATENTS for
# probing come from the companion LOBSTER orderbook file (SPEC Decision A) and
# live in a separate reader; THIS reader only needs tokens + the manifest, so
# training can run without the (large) orderbook file present.
#
# Stream layout (orderflow::tok::tokenize_stream):
#   BOS SESSION_OPEN <body...> SESSION_CLOSE EOS
# where <body> is, per LOBSTER message row in order, either a 5-token event
# tuple [TYPE][SIDE][PRICE_OFF][SIZE][DT] (MsgType 1..6) or a single HALT/RESUME
# token (MsgType 7). It is NOT a flat 5-tokens-per-event array: halts are one
# token, and the four session/BOS/EOS specials carry no message row.

# Real vocabulary id ranges (must match include/orderflow/tokenizer.hpp).
R_UNK, R_BOS, R_EOS = 0, 1, 2
R_SESSION_OPEN, R_SESSION_CLOSE, R_HALT, R_RESUME = 3, 4, 5, 6
R_TYPE_LO, R_TYPE_HI = 7, 12       # MsgType 1..6
R_SIDE_LO, R_SIDE_HI = 13, 14      # BID, ASK
R_PX_LO, R_PX_HI = 15, 27          # PX-1..PX+10, PX_TAIL (UNK when no level)
R_SZ_LO, R_SZ_HI = 28, 35          # 8 size bins
R_DT_LO, R_DT_HI = 36, 51          # DT_ZERO, DT0..DT13, DT_TAIL
R_EVENT_TOKENS = 5
_BIN_MAGIC = b"OFTK"
_BIN_VERSION = 2
_BIN_HEADER = 28                   # 4 magic + 4 version + 12 ticker + 8 count


def read_token_bin(path: str) -> tuple[str, np.ndarray]:
    """Parse an OFTK v2 token file -> (ticker, ids[int64]). Rejects other
    versions rather than reinterpreting them (v1 was a 62-id tick-offset
    vocab)."""
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < _BIN_HEADER:
        raise ValueError(f"{path}: shorter than a token-file header")
    if buf[:4] != _BIN_MAGIC:
        raise ValueError(f"{path}: bad magic (not an OFTK token file)")
    version = struct.unpack_from("<I", buf, 4)[0]
    if version != _BIN_VERSION:
        raise ValueError(f"{path}: unsupported OFTK version {version}")
    ticker_field = buf[8:20]
    if ticker_field[-1] != 0:
        raise ValueError(f"{path}: unterminated ticker field")
    ticker = ticker_field.split(b"\x00", 1)[0].decode("ascii")
    count = struct.unpack_from("<Q", buf, 20)[0]
    ids = np.frombuffer(buf, dtype="<u2", count=count, offset=_BIN_HEADER)
    if len(ids) != count:
        raise ValueError(f"{path}: truncated token payload")
    return ticker, ids.astype(np.int64)


def load_manifest(path: str) -> dict:
    """The tokenizer's vocab + per-ticker frozen-bin/fit-provenance manifest."""
    with open(path) as f:
        return json.load(f)


def _token_event_rows(ids: np.ndarray) -> np.ndarray:
    """Message-row index owning each token: event tuples and HALT/RESUME tokens
    advance the row counter (they consume one LOBSTER message/orderbook row);
    BOS/SESSION_OPEN/SESSION_CLOSE/EOS get -1. Validates that every TYPE token
    starts a well-formed [TYPE][SIDE][PRICE_OFF][SIZE][DT] tuple so a corrupt or
    misaligned stream fails loudly instead of training on garbage."""
    rows = np.full(len(ids), -1, dtype=np.int64)
    i, row, n = 0, 0, len(ids)
    while i < n:
        t = int(ids[i])
        if R_TYPE_LO <= t <= R_TYPE_HI:
            if i + R_EVENT_TOKENS > n:
                raise ValueError(f"truncated event tuple at token {i}")
            s, px, sz, dt = (int(ids[i + 1]), int(ids[i + 2]),
                             int(ids[i + 3]), int(ids[i + 4]))
            if not (R_SIDE_LO <= s <= R_SIDE_HI
                    and (px == R_UNK or R_PX_LO <= px <= R_PX_HI)
                    and R_SZ_LO <= sz <= R_SZ_HI
                    and R_DT_LO <= dt <= R_DT_HI):
                raise ValueError(f"malformed event tuple at token {i}: "
                                 f"{ids[i:i + R_EVENT_TOKENS].tolist()}")
            rows[i:i + R_EVENT_TOKENS] = row
            row += 1
            i += R_EVENT_TOKENS
        elif t in (R_HALT, R_RESUME):
            rows[i] = row
            row += 1
            i += 1
        elif t in (R_BOS, R_SESSION_OPEN, R_SESSION_CLOSE, R_EOS):
            i += 1
        else:
            raise ValueError(f"unexpected token {t} at position {i}")
    return rows


@dataclasses.dataclass
class RealTokenStream:
    """A split of the real token stream, wrapped BOS SESSION_OPEN ... EOS.

    `tokens` is ready to feed the LM. `train_end_index` is the manifest's
    message-ROW boundary the size/dt bins were fit on -- honored by `split`, and
    carried so a checkpoint can be pinned to the exact split it was trained on.
    """
    tokens: np.ndarray          # [n_tok] int64
    vocab_size: int
    train_end_index: int        # manifest row boundary (first NON-train row)
    n_events: int               # 5-token event tuples in this split
    ticker: str
    manifest_format: str


def load_real_token_stream(bin_path: str, manifest_path: str,
                           ticker: str | None = None, split: str = "train",
                           max_events: int | None = None) -> RealTokenStream:
    """Load the token stream for one split ('train' | 'eval' | 'all').

    Split is by the manifest's train_end_index (a message-ROW index): 'train'
    keeps rows [0, train_end_index) -- the exact prefix the quantile bins were
    fit on -- so training never sees eval-split events. Whole event tuples are
    kept intact (all five slots share a row). The selected body is re-wrapped as
    BOS SESSION_OPEN <body> SESSION_CLOSE EOS; original interior HALT/RESUME
    tokens inside the split are preserved, the original outer specials dropped.
    """
    tk, ids = read_token_bin(bin_path)
    ticker = ticker or tk
    if ticker != tk:
        raise ValueError(f"requested ticker {ticker!r} but bin holds {tk!r}")
    man = load_manifest(manifest_path)
    if ticker not in man.get("tickers", {}):
        raise ValueError(f"ticker {ticker!r} not in manifest")
    te = int(man["tickers"][ticker]["fit"]["train_end_index"])
    vocab_size = int(man["vocab_size"])
    if ids.size and int(ids.max()) >= vocab_size:
        raise ValueError(f"token id {int(ids.max())} >= manifest vocab_size "
                         f"{vocab_size}")

    rows = _token_event_rows(ids)
    body, _ = _select_body(ids, rows, te, split, max_events)

    tokens = np.concatenate((
        np.array([R_BOS, R_SESSION_OPEN], dtype=np.int64),
        body,
        np.array([R_SESSION_CLOSE, R_EOS], dtype=np.int64),
    ))
    n_events = int(((body >= R_TYPE_LO) & (body <= R_TYPE_HI)).sum())
    return RealTokenStream(tokens=tokens, vocab_size=vocab_size,
                           train_end_index=te, n_events=n_events,
                           ticker=ticker,
                           manifest_format=str(man.get("format", "")))


def _select_body(ids: np.ndarray, rows: np.ndarray, train_end: int,
                 split: str, max_events: int | None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(body tokens, owning message row per body token) for one split.
    `max_events` truncates at an event-tuple boundary, never mid-span."""
    if split == "train":
        mask = (rows >= 0) & (rows < train_end)
    elif split == "eval":
        mask = rows >= train_end
    elif split == "all":
        mask = rows >= 0
    else:
        raise ValueError(f"unknown split {split!r} (train | eval | all)")
    body, brows = ids[mask], rows[mask]
    if max_events is not None:
        type_pos = np.flatnonzero((body >= R_TYPE_LO) & (body <= R_TYPE_HI))
        if len(type_pos) > max_events:
            cut = type_pos[max_events]
            body, brows = body[:cut], brows[:cut]
    return body, brows


# ------------------------------------------------- orderbook-based latents
# Ground truth for probing comes from LOBSTER's precomputed orderbook file
# (SPEC Decision A), NOT from our book reconstruction. Only the level-1
# columns are needed; the 50-level files are ~1.5 GB x 200 columns, so both
# readers stream lines and keep O(rows requested) in memory.


def read_orderbook_touch(path: str, start_row: int,
                         stop_row: int) -> np.ndarray:
    """Rows [start_row, stop_row) of a LOBSTER orderbook file, level-1
    columns only: [ask_px, ask_sz, bid_px, bid_sz] int64. Splits off just
    the first four fields of each line; deeper levels are never parsed."""
    out = np.empty((stop_row - start_row, 4), dtype=np.int64)
    k = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start_row:
                continue
            if i >= stop_row:
                break
            a, asz, b, bsz = line.split(",", 4)[:4]
            out[k] = (int(a), int(asz), int(b), int(bsz))
            k += 1
    if k != stop_row - start_row:
        raise ValueError(f"{path}: wanted rows [{start_row}, {stop_row}) "
                         f"but file ends at row {start_row + k}")
    return out


def read_message_rows(path: str, start_row: int, stop_row: int) -> np.ndarray:
    """Rows [start_row, stop_row) of a LOBSTER message file as
    [type, price, direction] int64 (time and order id are not needed)."""
    out = np.empty((stop_row - start_row, 3), dtype=np.int64)
    k = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start_row:
                continue
            if i >= stop_row:
                break
            c = line.split(",")
            out[k] = (int(c[1]), int(c[4]), int(c[5]))
            k += 1
    if k != stop_row - start_row:
        raise ValueError(f"{path}: wanted rows [{start_row}, {stop_row}) "
                         f"but file ends at row {start_row + k}")
    return out


def load_lobster_latents(orderbook_path: str, message_path: str,
                         tick_size: int, start_row: int,
                         n_rows: int) -> np.ndarray:
    """Latents for message rows [start_row, start_row + n_rows): the same
    five quantities as the synthetic stream (LATENT_NAMES, same order and
    units-in-ticks) so probe results are directly comparable across sources.

    Conventions match the synthetic extractor exactly: spread / imbalance /
    ofi / realized_vol describe the book AFTER the row's event (orderbook
    row i); dist_to_touch is |event price - same-side touch| in ticks
    against the PRE-event book (row i-1, the tokenizer's PRICE_OFF
    convention), clamped at 0 so inside-spread orders read 0, and halt rows
    (type 7) get 0. Rolling windows (OFI over W_OFI, vol over W_VOL events)
    warm up on up to W_VOL rows BEFORE start_row, so latents at a split
    boundary carry real history instead of a cold start."""
    pad = min(start_row, W_VOL)
    lo, hi = start_row - pad, start_row + n_rows
    touch = read_orderbook_touch(orderbook_path, lo, hi).astype(np.float64)
    msg = read_message_rows(message_path, lo, hi)
    ask_p, ask_sz, bid_p, bid_sz = touch.T
    n = len(touch)
    i = np.arange(n)

    spread = (ask_p - bid_p) / tick_size
    tot = ask_sz + bid_sz
    imb = np.divide(bid_sz - ask_sz, tot, out=np.zeros(n), where=tot > 0)
    mid = (ask_p + bid_p) / (2.0 * tick_size)

    # OFI e_n (Cont-Kukanov-Stoikov): term j compares rows j-1 -> j; the
    # latent is the rolling sum of the last W_OFI terms including j.
    e = np.zeros(n)
    if n > 1:
        b0, b1 = bid_p[:-1], bid_p[1:]
        a0, a1 = ask_p[:-1], ask_p[1:]
        qb0, qb1 = bid_sz[:-1], bid_sz[1:]
        qa0, qa1 = ask_sz[:-1], ask_sz[1:]
        e_bid = np.where(b1 > b0, qb1, np.where(b1 == b0, qb1 - qb0, -qb0))
        e_ask = np.where(a1 < a0, qa1, np.where(a1 == a0, qa1 - qa0, -qa0))
        e[1:] = e_bid - e_ask
    cs = np.concatenate(([0.0], np.cumsum(e)))
    ofi = cs[i + 1] - cs[np.maximum(0, i + 1 - W_OFI)]

    # realized vol: population std of the last <= W_VOL 1-step mid diffs
    # (rets[first:i]), 0 with fewer than 2 diffs -- mirrors the deque logic.
    rets = np.diff(mid)
    c1 = np.concatenate(([0.0], np.cumsum(rets)))
    c2 = np.concatenate(([0.0], np.cumsum(rets * rets)))
    first = np.maximum(0, i - W_VOL)
    cnt = (i - first).astype(np.float64)
    safe = np.maximum(cnt, 1.0)
    m = (c1[i] - c1[first]) / safe
    var = (c2[i] - c2[first]) / safe - m * m
    rvol = np.where(cnt >= 2, np.sqrt(np.maximum(var, 0.0)), 0.0)

    mtype, price, mdir = msg.T
    prev_bid = np.concatenate(([bid_p[0]], bid_p[:-1]))
    prev_ask = np.concatenate(([ask_p[0]], ask_p[:-1]))
    dist = np.where(mdir == 1, prev_bid - price,
                    price - prev_ask) / tick_size
    dist = np.where(mtype == 7, 0.0, np.maximum(dist, 0.0))

    lat = np.stack([spread, imb, ofi, dist, rvol], axis=1)[pad:]
    return np.ascontiguousarray(lat, dtype=np.float32)


def _real_span_arrays(tokens: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Span bookkeeping for a wrapped real stream: each event tuple is a
    5-token span; each special (BOS/EOS/session/halt/resume) is its own
    1-token span, which select_event_rows() then never picks up.

    event_type_id is the event's COMBINED identity over (TYPE, SIDE,
    PRICE_OFF, SIZE). DT is deliberately EXCLUDED: the Hewitt-Liang control
    must not be able to absorb the inter-arrival signal (the DT-slot
    realized_vol ~0.67 diagnostic), and the synthetic combined id has no DT
    either, so the control stays semantically identical across sources.
    Specials get distinct negative ids (-1 - token_id)."""
    n = len(tokens)
    span_index = np.empty(n, dtype=np.int64)
    span_pos = np.empty(n, dtype=np.int64)
    etype = np.empty(n, dtype=np.int64)
    n_px_codes = R_PX_HI - R_PX_LO + 2       # 13 buckets + UNK
    i, si = 0, 0
    while i < n:
        t = int(tokens[i])
        if R_TYPE_LO <= t <= R_TYPE_HI:
            px = int(tokens[i + 2])
            px_code = 0 if px == R_UNK else px - R_PX_LO + 1
            eid = ((((t - R_TYPE_LO) * 2 + int(tokens[i + 1]) - R_SIDE_LO)
                    * n_px_codes + px_code)
                   * (R_SZ_HI - R_SZ_LO + 1)
                   + int(tokens[i + 3]) - R_SZ_LO)
            span_index[i:i + R_EVENT_TOKENS] = si
            span_pos[i:i + R_EVENT_TOKENS] = np.arange(R_EVENT_TOKENS)
            etype[i:i + R_EVENT_TOKENS] = eid
            i += R_EVENT_TOKENS
        else:
            span_index[i] = si
            span_pos[i] = 0
            etype[i] = -1 - t
            i += 1
        si += 1
    return span_index, span_pos, etype


def load_real_eval_stream(bin_path: str, manifest_path: str,
                          orderbook_path: str, message_path: str,
                          ticker: str | None = None, split: str = "eval",
                          max_events: int | None = None) -> StreamData:
    """Real-data analogue of generate_split_ex for PROBING: one split's
    token stream plus per-token ground-truth latents from the orderbook
    file, packaged as StreamData so caching/probes stay source-agnostic.

    Latents are the owning message row's values broadcast across the
    event's whole 5-token span (the run_probe span-constancy assertion
    holds by construction). Wrapper specials get the nearest row's values;
    they are never selected for probing."""
    tk, ids = read_token_bin(bin_path)
    ticker = ticker or tk
    if ticker != tk:
        raise ValueError(f"requested ticker {ticker!r} but bin holds {tk!r}")
    man = load_manifest(manifest_path)
    if ticker not in man.get("tickers", {}):
        raise ValueError(f"ticker {ticker!r} not in manifest")
    tinfo = man["tickers"][ticker]
    te = int(tinfo["fit"]["train_end_index"])

    rows = _token_event_rows(ids)
    body, brows = _select_body(ids, rows, te, split, max_events)
    if not len(body):
        raise ValueError(f"{split!r} split of {bin_path} is empty")
    tokens = np.concatenate((
        np.array([R_BOS, R_SESSION_OPEN], dtype=np.int64),
        body,
        np.array([R_SESSION_CLOSE, R_EOS], dtype=np.int64),
    ))

    row_lo, row_hi = int(brows[0]), int(brows[-1]) + 1
    lat = load_lobster_latents(orderbook_path, message_path,
                               int(tinfo["tick_size"]), row_lo,
                               row_hi - row_lo)
    tok_rows = np.concatenate((
        [brows[0], brows[0]], brows, [brows[-1], brows[-1]]))
    latents = lat[tok_rows - row_lo]

    span_index, span_pos, etype = _real_span_arrays(tokens)
    return StreamData("factored", R_EVENT_TOKENS, tokens, latents,
                      span_index, span_pos, etype)
