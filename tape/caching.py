"""Run the model over the eval stream; cache residual activations to disk.

One float32 memmap per hook point, shape [n_tokens, d_model], plus
labels.npy [n_tokens, N_LATENTS], tokens.npy [n_tokens], span bookkeeping
(span_index / span_pos / event_type_id .npy) and a meta.json.

Warmup is EVENT-based: every cached position (outside the very first window)
has at least `warmup` COMPLETE events of left context within its window.
The token margin is warmup * span_len + (span_len - 1); the +(span_len - 1)
covers the worst case of a window starting mid-span, so the guarantee holds
regardless of span phase while windows stay token-strided and coverage stays
contiguous with no duplicates. All positions are cached — probes select by
span position downstream; nothing is filtered here.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import data as tdata


def hook_names(n_layers: int) -> list[str]:
    return ["hook_embed"] + [f"blocks.{l}.hook_resid_post"
                             for l in range(n_layers)]


SPAN_FILES = ("span_index", "span_pos", "event_type_id")


def cache_activations(model, model_cfg, tokens: np.ndarray,
                      latents: np.ndarray, cache_dir: str,
                      batch: int = 8, warmup: int = 32,
                      span_index: np.ndarray | None = None,
                      span_pos: np.ndarray | None = None,
                      event_type_id: np.ndarray | None = None,
                      span_len: int = 1,
                      tokenization: str = "combined") -> dict:
    """`warmup` counts EVENTS (== tokens when span_len == 1)."""
    os.makedirs(cache_dir, exist_ok=True)
    n = len(tokens)
    if span_index is None:
        span_index = np.arange(n, dtype=np.int64)
    if span_pos is None:
        span_pos = np.zeros(n, dtype=np.int64)
    if event_type_id is None:
        event_type_id = tokens.copy()

    T = model_cfg.n_ctx
    warmup_tokens = warmup * span_len + (span_len - 1)
    stride = T - warmup_tokens
    assert stride >= 1, (
        f"n_ctx={T} too small for warmup of {warmup} events x span_len "
        f"{span_len} (= {warmup_tokens} tokens)")
    names = hook_names(model_cfg.n_layers)

    # window start offsets
    starts = list(range(0, max(1, n - T + 1), stride))

    # how many positions will be kept
    keep_total = 0
    for w, s in enumerate(starts):
        lo = 0 if w == 0 else warmup_tokens
        keep_total += T - lo

    mms = {name: np.lib.format.open_memmap(
        os.path.join(cache_dir, name.replace(".", "_") + ".npy"),
        mode="w+", dtype=np.float32, shape=(keep_total, model_cfg.d_model))
        for name in names}
    kept_tokens = np.empty(keep_total, dtype=np.int64)
    kept_latents = np.empty((keep_total, tdata.N_LATENTS), dtype=np.float32)
    kept_spans = {k: np.empty(keep_total, dtype=np.int64)
                  for k in SPAN_FILES}
    span_src = {"span_index": span_index, "span_pos": span_pos,
                "event_type_id": event_type_id}

    filt = set(names)
    write = 0
    with torch.no_grad():
        for b0 in range(0, len(starts), batch):
            wstarts = starts[b0:b0 + batch]
            x = torch.stack([torch.from_numpy(tokens[s:s + T])
                             for s in wstarts])
            _, cache = model.run_with_cache(x, names_filter=filt.__contains__)
            for j, s in enumerate(wstarts):
                w = b0 + j
                lo = 0 if w == 0 else warmup_tokens
                span = slice(write, write + (T - lo))
                for name in names:
                    mms[name][span] = cache[name][j, lo:].numpy()
                kept_tokens[span] = tokens[s + lo:s + T]
                kept_latents[span] = latents[s + lo:s + T]
                for k, src in span_src.items():
                    kept_spans[k][span] = src[s + lo:s + T]
                write += T - lo
    assert write == keep_total, (write, keep_total)
    for mm in mms.values():
        mm.flush()
    np.save(os.path.join(cache_dir, "tokens.npy"), kept_tokens)
    np.save(os.path.join(cache_dir, "labels.npy"), kept_latents)
    for k in SPAN_FILES:
        np.save(os.path.join(cache_dir, k + ".npy"), kept_spans[k])
    meta = {"hook_names": names, "n_tokens": int(keep_total),
            "d_model": model_cfg.d_model,
            "warmup_events": warmup, "warmup_tokens": warmup_tokens,
            "span_len": span_len, "tokenization": tokenization,
            "latent_names": tdata.LATENT_NAMES}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_cache_ex(cache_dir: str):
    """Returns (meta, acts, tokens, labels, spans) where spans is a dict with
    span_index / span_pos / event_type_id. Legacy caches without span files
    get the degenerate combined-mode defaults synthesized."""
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    meta.setdefault("span_len", 1)
    meta.setdefault("tokenization", "combined")
    acts = {name: np.load(os.path.join(cache_dir,
                                       name.replace(".", "_") + ".npy"),
                          mmap_mode="r")
            for name in meta["hook_names"]}
    tokens = np.load(os.path.join(cache_dir, "tokens.npy"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"))
    spans = {}
    for k in SPAN_FILES:
        path = os.path.join(cache_dir, k + ".npy")
        if os.path.exists(path):
            spans[k] = np.load(path)
        elif k == "span_index":
            spans[k] = np.arange(len(tokens), dtype=np.int64)
        elif k == "span_pos":
            spans[k] = np.zeros(len(tokens), dtype=np.int64)
        else:
            spans[k] = tokens.copy()
    return meta, acts, tokens, labels, spans


def load_cache(cache_dir: str):
    return load_cache_ex(cache_dir)[:4]
