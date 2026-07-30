"""Probe hygiene: Hewitt-Liang control tasks + shuffled-label baselines.

Hewitt-Liang (EMNLP 2019) define a control task for classification probes:
each word TYPE is assigned a random but consistent label, so the control
measures the probe's capacity to memorize type->label mappings from the
representation. Selectivity = task accuracy - control accuracy.

Regression adaptation (logged as a design decision): each token TYPE t is
assigned a fixed pseudo-label c(t) sampled i.i.d. from the EMPIRICAL
distribution of the true latent (so the control target has the same marginal
as the real one), and control targets are y_ctrl[i] = c(token[i]). A probe
can score well on this only by decoding token identity, which the residual
stream trivially contains -- exactly the confound Hewitt-Liang target.

  selectivity = R^2_task - R^2_control

Shuffled-label baseline: labels randomly permuted across positions; held-out
R^2 should be within noise of 0 (tolerance coeff/sqrt(n_probed), applied by
the driver). If it is not, the probe/split machinery leaks and no result
from that run should be reported.
"""

from __future__ import annotations

import numpy as np

from . import probes


def control_targets(tokens: np.ndarray, y: np.ndarray,
                    seed: int) -> np.ndarray:
    """Hewitt-Liang-style control: fixed random pseudo-label per token type,
    drawn from the empirical distribution of y (train-portion marginal)."""
    rng = np.random.default_rng(seed)
    tr, _, _ = probes.chrono_split(len(y))
    pool = y[tr]
    types = np.unique(tokens)
    mapping = {int(t): float(rng.choice(pool)) for t in types}
    return np.array([mapping[int(t)] for t in tokens], dtype=y.dtype)


def shuffled_targets(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(y)


def probe_with_controls(X: np.ndarray, y: np.ndarray, tokens: np.ndarray,
                        seed: int) -> dict:
    """Runs task probe, control-task probe, shuffled baseline on one
    (layer, latent) pair. Returns dict with r2s + selectivity."""
    task = probes.probe(X, y)
    ctrl = probes.probe(X, control_targets(tokens, y, seed))
    shuf = probes.probe(X, shuffled_targets(y, seed))
    return {
        "r2_task": task.r2, "lam_task": task.lam,
        "r2_control": ctrl.r2,
        "r2_shuffled": shuf.r2,
        "selectivity": task.r2 - ctrl.r2,
    }
