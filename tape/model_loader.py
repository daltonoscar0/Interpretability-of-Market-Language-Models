"""MiniGPT with TransformerLens-style hook points + checkpoint load/train.

Swapping in the real orderflow-lm checkpoint later is a one-line config
change: set `model.checkpoint` in the yaml to its path (and, if it is not a
MiniGPT state_dict, register a loader under its `model.arch` name below).

Hook naming follows TransformerLens conventions so writeups are legible:
  hook_embed, hook_pos_embed,
  blocks.{l}.hook_resid_pre, blocks.{l}.hook_resid_mid,
  blocks.{l}.hook_resid_post, ln_final.hook_normalized
`run_with_cache(tokens, names_filter=...)` mirrors the TL API surface we use.
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as tdata


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int = tdata.VOCAB_SIZE
    n_layers: int = 4
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    n_ctx: int = 128


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x, attn_mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=attn_mask, need_weights=False)
        mid = x + a
        out = mid + self.mlp(self.ln2(mid))
        return mid, out


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _mask(self, T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), 1)

    def forward(self, tokens):
        logits, _ = self.run_with_cache(tokens, names_filter=lambda n: False)
        return logits

    def run_with_cache(self, tokens, names_filter=None):
        """tokens: [B, T] int64. Returns (logits, cache dict name->tensor)."""
        if names_filter is None:
            names_filter = lambda n: True
        cache: dict[str, torch.Tensor] = {}

        def save(name, tensor):
            if names_filter(name):
                cache[name] = tensor.detach()

        B, T = tokens.shape
        assert T <= self.cfg.n_ctx, f"seq len {T} > n_ctx {self.cfg.n_ctx}"
        emb = self.embed(tokens)
        pos = self.pos_embed(torch.arange(T, device=tokens.device))[None]
        save("hook_embed", emb)
        save("hook_pos_embed", pos.expand(B, -1, -1))
        x = emb + pos
        mask = self._mask(T, tokens.device)
        for l, block in enumerate(self.blocks):
            save(f"blocks.{l}.hook_resid_pre", x)
            mid, x = block(x, mask)
            save(f"blocks.{l}.hook_resid_mid", mid)
            save(f"blocks.{l}.hook_resid_post", x)
        normed = self.ln_final(x)
        save("ln_final.hook_normalized", normed)
        return self.unembed(normed), cache


# ------------------------------------------------------------ train / load


def train_standin(cfg: ModelConfig, ckpt_path: str, seed: int = 0,
                  steps: int = 900, batch: int = 16, lr: float = 3e-3,
                  n_train_tokens: int = 200_000, log_every: int = 100,
                  tokenization: str = "combined"):
    """Train the tiny stand-in LM on the synthetic stream (CPU).

    `n_train_tokens` counts TOKENS in both modes; factored mode generates
    n_train_tokens // SPAN_LEN events."""
    torch.manual_seed(seed)
    n_steps = (n_train_tokens if tokenization == "combined"
               else max(1, n_train_tokens // tdata.SPAN_LEN))
    stream = tdata.generate_split_ex(seed=seed, n_steps=n_steps,
                                     tokenization=tokenization)
    tokens = torch.from_numpy(stream.tokens)
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1)
    T = cfg.n_ctx
    n_windows = len(tokens) - T - 1
    t0 = time.time()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n_windows, (batch,))
        x = torch.stack([tokens[i:i + T] for i in idx])
        y = torch.stack([tokens[i + 1:i + T + 1] for i in idx])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"  train step {step+1}/{steps} "
                  f"loss {sum(losses[-log_every:])/log_every:.4f} "
                  f"({time.time()-t0:.0f}s)")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"arch": "minigpt", "cfg": dataclasses.asdict(cfg),
                "tokenization": tokenization,
                "state_dict": model.state_dict(),
                "final_loss": losses[-1]}, ckpt_path)
    return model, losses[-1]


def split_loss(model, cfg: ModelConfig, tokens: torch.Tensor,
               batch: int = 16, max_windows: int | None = None) -> float:
    """Deterministic per-token cross-entropy over NON-overlapping n_ctx
    windows of `tokens` (evenly subsampled to max_windows if capped).
    Token-weighted, forward-only -- comparable across splits and runs,
    unlike the sampled-window running training loss."""
    T = cfg.n_ctx
    starts = np.arange(0, len(tokens) - T - 1, T)
    if len(starts) == 0:
        raise ValueError(f"split has {len(tokens)} tokens < n_ctx {T} + 1")
    if max_windows is not None and len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1,
                                    max_windows).astype(int)]
    total, count = 0.0, 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(starts), batch):
            bs = starts[b0:b0 + batch]
            x = torch.stack([tokens[s:s + T] for s in bs])
            y = torch.stack([tokens[s + 1:s + T + 1] for s in bs])
            logits = model(x)
            total += F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                                     y.reshape(-1),
                                     reduction="sum").item()
            count += y.numel()
    if was_training:
        model.train()
    return total / count


def train_spy(cfg: ModelConfig, ckpt_path: str, bin_path: str,
              manifest_path: str, ticker: str = "SPY", seed: int = 0,
              steps: int = 900, batch: int = 16, lr: float = 3e-3,
              log_every: int = 100, max_events: int | None = None,
              eval_every: int = 1000, eval_windows: int = 256):
    """Train the LM on the real SPY factored token stream.

    Same structure as `train_standin`, but the synthetic generator is swapped
    for the .tokens.bin reader and training is restricted to the manifest's
    TRAIN split: the size/dt quantile bins were fit on rows [0, train_end_index)
    only, so any window drawn past that boundary would leak eval-split
    statistics into the model. Vocab size comes from the manifest, not the
    synthetic vocab. The checkpoint records the manifest format string and
    train_end_index so `load_model` can refuse a checkpoint that was fit on a
    different split/format, exactly as it already does for tokenization mode.

    Held-out loss: every `eval_every` steps, split_loss on `eval_windows`
    evenly-spaced windows of the EVAL split (rows >= train_end_index) --
    the divergence between the two curves makes memorization visible
    during the run, not just after it. The final checkpoint stores both
    curves plus full-split final losses.

    Returns (model, loss_curve, tokens_per_sec).
    """
    torch.manual_seed(seed)
    rt = tdata.load_real_token_stream(bin_path, manifest_path, ticker=ticker,
                                      split="train", max_events=max_events)
    if cfg.vocab_size != rt.vocab_size:
        raise ValueError(f"cfg.vocab_size {cfg.vocab_size} != manifest "
                         f"vocab_size {rt.vocab_size}")
    ev = tdata.load_real_token_stream(bin_path, manifest_path, ticker=ticker,
                                      split="eval")
    tokens = torch.from_numpy(rt.tokens)
    eval_tokens = torch.from_numpy(ev.tokens)
    T = cfg.n_ctx
    n_windows = len(tokens) - T - 1
    if n_windows < 1:
        raise ValueError(f"train split has {len(tokens)} tokens, < n_ctx "
                         f"{T} + 1; nothing to train on")
    print(f"SPY train split: {rt.n_events} events, {len(tokens)} tokens "
          f"(< row {rt.train_end_index}); {n_windows} windows, vocab "
          f"{rt.vocab_size}, n_ctx {T}")
    print(f"SPY eval split:  {ev.n_events} events, {len(eval_tokens)} tokens "
          f"(rows >= {ev.train_end_index})")

    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1)
    t0 = time.time()
    losses = []
    eval_curve: list[tuple[int, float]] = []
    for step in range(steps):
        idx = torch.randint(0, n_windows, (batch,))
        x = torch.stack([tokens[i:i + T] for i in idx])
        y = torch.stack([tokens[i + 1:i + T + 1] for i in idx])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"  train step {step+1}/{steps} "
                  f"loss {sum(losses[-log_every:])/log_every:.4f} "
                  f"({time.time()-t0:.0f}s)")
        if eval_every and (step + 1) % eval_every == 0:
            el = split_loss(model, cfg, eval_tokens, batch=batch,
                            max_windows=eval_windows)
            eval_curve.append((step + 1, el))
            tl = sum(losses[-log_every:]) / min(log_every, len(losses))
            print(f"  eval  step {step+1}/{steps} held-out loss {el:.4f} "
                  f"(train {tl:.4f}, gap {el - tl:+.4f})")
    elapsed = time.time() - t0
    tokens_per_sec = steps * batch * T / elapsed if elapsed > 0 else float("nan")

    # final apples-to-apples numbers: the same deterministic window sweep on
    # both FULL splits (the running training loss is sampled + pre-update)
    final_train = split_loss(model, cfg, tokens, batch=batch)
    final_eval = split_loss(model, cfg, eval_tokens, batch=batch)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"arch": "minigpt", "cfg": dataclasses.asdict(cfg),
                "tokenization": "factored", "source": "lobster",
                "manifest_format": rt.manifest_format,
                "train_end_index": rt.train_end_index, "ticker": rt.ticker,
                "state_dict": model.state_dict(),
                "final_loss": losses[-1] if losses else float("nan"),
                "final_train_loss": final_train,
                "final_eval_loss": final_eval,
                "loss_curve": losses, "eval_curve": eval_curve}, ckpt_path)
    print(f"trained {steps} steps in {elapsed:.0f}s "
          f"({tokens_per_sec:,.0f} tokens/sec), saved to {ckpt_path}")
    print(f"FINAL full-split loss: train {final_train:.4f}  "
          f"held-out {final_eval:.4f}  gap {final_eval - final_train:+.4f}")
    return model, losses, tokens_per_sec


def benchmark_step_rate(cfg: ModelConfig, batch: int = 16, steps: int = 20,
                        warmup: int = 3, seed: int = 0):
    """Time the real train step (fwd+bwd+opt) at `cfg` on this machine.

    Uses random tokens of the target shape -- the step cost depends on
    (batch, n_ctx, vocab, model dims), not on token content -- so it gives a
    faithful CPU tokens/sec without needing the real bin. Returns
    (steps_per_sec, tokens_per_sec)."""
    torch.manual_seed(seed)
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    T = cfg.n_ctx

    def one():
        x = torch.randint(0, cfg.vocab_size, (batch, T))
        y = torch.randint(0, cfg.vocab_size, (batch, T))
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one()
    t0 = time.time()
    for _ in range(steps):
        one()
    dt = time.time() - t0
    steps_per_sec = steps / dt
    return steps_per_sec, steps_per_sec * batch * T


def load_model(model_cfg: dict, tokenization: str = "combined"):
    """Load from `checkpoint`; if absent, train the stand-in and cache it.

    model_cfg keys: arch, checkpoint, and any ModelConfig overrides.
    """
    arch = model_cfg.get("arch", "minigpt")
    if arch != "minigpt":
        raise NotImplementedError(
            f"arch '{arch}': register a loader here when the real "
            f"orderflow-lm checkpoint format is known")
    source = model_cfg.get("source", "synthetic")
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in model_cfg.items() if k in fields})

    # Real (LOBSTER) source: vocab + split provenance come from the manifest,
    # and the checkpoint is pinned to the exact manifest format + train_end_index
    # it was fit on. A missing checkpoint is NOT auto-trained here -- a full SPY
    # run is a long CPU job; it must be launched deliberately via train_spy after
    # reviewing the wall-clock estimate.
    man_expect = None
    if source == "lobster":
        man = tdata.load_manifest(model_cfg["manifest"])
        ticker = model_cfg.get("ticker")
        if ticker not in man.get("tickers", {}):
            raise ValueError(f"ticker {ticker!r} not in manifest "
                             f"{model_cfg['manifest']}")
        cfg.vocab_size = int(man["vocab_size"])
        man_expect = (str(man.get("format", "")),
                      int(man["tickers"][ticker]["fit"]["train_end_index"]))
    elif "vocab_size" not in model_cfg:
        cfg.vocab_size = tdata.vocab_size(tokenization)

    ckpt_path = model_cfg["checkpoint"]
    if os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_tok = blob.get("tokenization", "combined")
        assert ckpt_tok == tokenization, (
            f"checkpoint {ckpt_path} was trained on '{ckpt_tok}' tokens but "
            f"this run uses '{tokenization}' -- probing it would produce "
            f"garbage, not results")
        if man_expect is not None:
            got = (blob.get("manifest_format"), blob.get("train_end_index"))
            assert got == man_expect, (
                f"checkpoint {ckpt_path} was fit on (format, train_end_index)="
                f"{got} but the manifest now says {man_expect} -- the bins/split "
                f"moved under the model; retrain before probing")
        cfg = ModelConfig(**blob["cfg"])
        model = MiniGPT(cfg)
        model.load_state_dict(blob["state_dict"])
        print(f"loaded checkpoint {ckpt_path} "
              f"(train loss {blob.get('final_loss', float('nan')):.4f})")
    elif source == "lobster":
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path}. A full SPY train is a long CPU job "
            f"and is not auto-launched: run model_loader.train_spy(...) after "
            f"reviewing the wall-clock estimate, then re-run.")
    else:
        print(f"no checkpoint at {ckpt_path}; training stand-in ...")
        model, final_loss = train_standin(
            cfg, ckpt_path,
            seed=int(model_cfg.get("train_seed", 0)),
            steps=int(model_cfg.get("train_steps", 900)),
            batch=int(model_cfg.get("train_batch", 16)),
            n_train_tokens=int(model_cfg.get("train_tokens", 200_000)),
            tokenization=tokenization)
        print(f"stand-in trained, final loss {final_loss:.4f}, "
              f"saved to {ckpt_path}")
    model.eval()
    return model, cfg
