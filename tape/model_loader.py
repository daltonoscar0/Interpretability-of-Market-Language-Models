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


def load_model(model_cfg: dict, tokenization: str = "combined"):
    """Load from `checkpoint`; if absent, train the stand-in and cache it.

    model_cfg keys: arch, checkpoint, and any ModelConfig overrides.
    """
    arch = model_cfg.get("arch", "minigpt")
    if arch != "minigpt":
        raise NotImplementedError(
            f"arch '{arch}': register a loader here when the real "
            f"orderflow-lm checkpoint format is known")
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in model_cfg.items() if k in fields})
    if "vocab_size" not in model_cfg:
        cfg.vocab_size = tdata.vocab_size(tokenization)
    ckpt_path = model_cfg["checkpoint"]
    if os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_tok = blob.get("tokenization", "combined")
        assert ckpt_tok == tokenization, (
            f"checkpoint {ckpt_path} was trained on '{ckpt_tok}' tokens but "
            f"this run uses '{tokenization}' -- probing it would produce "
            f"garbage, not results")
        cfg = ModelConfig(**blob["cfg"])
        model = MiniGPT(cfg)
        model.load_state_dict(blob["state_dict"])
        print(f"loaded checkpoint {ckpt_path} "
              f"(train loss {blob.get('final_loss', float('nan')):.4f})")
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
