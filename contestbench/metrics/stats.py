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


def mean_ci(values, n_boot: int = 5000, seed: int = 20260719):
    """Return (mean, ci_lo, ci_hi) with a bootstrap 95% CI."""
    values = np.asarray(values, float)
    m = float(np.mean(values))
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return m, float(lo), float(hi)
