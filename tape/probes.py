"""Layerwise linear probes (closed-form ridge regression).

Latents are continuous, so probes are ridge regressions and the headline
metric is held-out R^2. Splits are CHRONOLOGICAL (first 70% train / next 15%
val / last 15% test) -- a random split would leak: order-flow latents are
strongly autocorrelated, so temporally adjacent tokens in train and test
would make every probe look good. Ridge lambda is selected per (layer,
latent) on the val block from a small log grid; test R^2 is reported.
"""

from __future__ import annotations

import dataclasses

import numpy as np

LAMBDA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclasses.dataclass
class ProbeResult:
    r2: float          # held-out test R^2
    lam: float         # chosen ridge lambda
    r2_val: float


def chrono_split(n: int, train=0.70, val=0.15):
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return slice(0, i1), slice(i1, i2), slice(i2, n)


def _fit_ridge(X, y, lam):
    # X [n, d] (bias appended by caller), y [n]
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float64)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def probe(X: np.ndarray, y: np.ndarray) -> ProbeResult:
    """X [n, d] activations, y [n] latent. Returns held-out ridge R^2."""
    assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y)
    tr, va, te = chrono_split(len(X))
    # standardize using TRAIN stats only
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-8

    def prep(sl):
        Z = ((X[sl] - mu) / sd).astype(np.float64)
        return np.concatenate([Z, np.ones((len(Z), 1))], axis=1)

    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr = ((y[tr] - ymu) / ysd).astype(np.float64)
    yva = ((y[va] - ymu) / ysd).astype(np.float64)
    yte = ((y[te] - ymu) / ysd).astype(np.float64)

    best = None
    for lam in LAMBDA_GRID:
        w = _fit_ridge(Xtr, ytr, lam)
        r2v = _r2(yva, Xva @ w)
        if best is None or r2v > best[1]:
            best = (lam, r2v, w)
    lam, r2v, w = best
    return ProbeResult(r2=_r2(yte, Xte @ w), lam=lam, r2_val=r2v)
