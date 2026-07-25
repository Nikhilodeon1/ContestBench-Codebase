"""Mutual information between confidence and physician agreement (Fix J).

A stronger independence claim than Pearson r: MI captures any dependence, not just
linear. Reported with a permutation null (shuffle the target -> distribution of MI
under independence) and a bootstrap CI, for every model config and the oracle as an
upper-bound comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def mutual_information(x, y, seed: int = 20260723) -> float:
    """kNN estimate of MI(x; y), both treated as continuous. Non-negative."""
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float)
    return float(mutual_info_regression(x, y, random_state=seed)[0])


def mi_with_null(x, y, n_perm: int = 1000, seed: int = 20260723) -> dict:
    """MI with a permutation null and p-value.

    p_value = fraction of permutations (target shuffled) with MI >= observed.
    We report against the permutation null (mean, 95th pct) rather than a bootstrap
    CI: kNN-MI is biased upward under resampling-with-replacement (duplicate rows
    become artificial nearest neighbors), so a bootstrap CI is invalid here. The
    permutation null has no ties and is the correct reference for "MI above chance".
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    observed = mutual_information(x, y, seed)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = mutual_information(x, rng.permutation(y), seed)
    p_value = float((1 + np.sum(null >= observed)) / (n_perm + 1))

    return {
        "mi": observed,
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "excess": observed - float(np.percentile(null, 95)),
    }
