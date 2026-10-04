"""Statistics: physician agreement rate, and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def agreement_rate(f):
    """Majority agreement pi = max(f, 1-f) in [0.5,1] from frac-malignant f."""
    f = np.asarray(f, float)
    out = np.maximum(f, 1.0 - f)
    return float(out) if out.ndim == 0 else out


def pearson_ci(x, y, n_boot: int = 5000, seed: int = 20260719):
    """Return (r, ci_lo, ci_hi, p) for Pearson(x, y) with a bootstrap 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = pearsonr(x[idx], y[idx])[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(r), float(lo), float(hi), float(p)


def slope_ci(x, y, n_boot: int = 5000, seed: int = 20260720):
    """OLS slope of y on x with a bootstrap 95% CI (capability-inversion trend)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = np.polyfit(x[idx], y[idx], 1)[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, float(lo), float(hi)


def mean_ci(values, n_boot: int = 5000, seed: int = 20260719):
    """Return (mean, ci_lo, ci_hi) with a bootstrap 95% CI."""
    values = np.asarray(values, float)
    m = float(np.mean(values))
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return m, float(lo), float(hi)


def scan_cluster_indices(groups, n_boot: int, seed: int):
    """Yield n_boot index arrays resampling whole scans (clusters) with replacement.

    Nodules on one scan share anatomy/readers, so case-level iid resampling
    understates uncertainty; resampling scans keeps each scan's nodules together.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    g = len(uniq)
    for _ in range(n_boot):
        pick = rng.integers(0, g, g)
        yield np.concatenate([members[k] for k in pick])


def nested_cluster_indices(calls, scans, n_boot: int, seed: int):
    """Two-stage cluster bootstrap: resample calls (chunks) with replacement, then resample the
    scans inside each drawn call with replacement. Captures call-level drift and scan clustering.
    With very few calls (e.g. 3) the first stage is nearly degenerate; callers should say so.
    """
    calls, scans = np.asarray(calls), np.asarray(scans)
    rng = np.random.default_rng(seed)
    call_ids = np.unique(calls)
    within = {}
    for c in call_ids:
        idx = np.flatnonzero(calls == c)
        s = scans[idx]
        us = np.unique(s)
        within[c] = [idx[s == u] for u in us]
    for _ in range(n_boot):
        parts = []
        for c in rng.choice(call_ids, len(call_ids)):
            blocks = within[c]
            pick = rng.integers(0, len(blocks), len(blocks))
            parts.extend(blocks[j] for j in pick)
        yield np.concatenate(parts)
