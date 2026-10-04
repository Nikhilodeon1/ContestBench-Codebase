"""Tests for mutual information with a permutation null (Fix J)."""

import numpy as np

from contestbench.metrics import mutual_info as mi


def test_mi_detects_dependence_and_is_significant():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 400)
    y = x + rng.normal(0, 0.02, 400)          # y strongly depends on x
    res = mi.mi_with_null(x, y, n_perm=200, seed=1)
    assert res["mi"] > 0.3
    assert res["p_value"] < 0.05
    assert res["excess"] > 0                    # MI above the null 95th percentile


def test_mi_independent_is_not_significant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    y = rng.normal(0, 1, 400)                  # independent
    res = mi.mi_with_null(x, y, n_perm=200, seed=1)
    assert res["p_value"] > 0.05               # observed MI within the null band


def test_mi_nonnegative():
    rng = np.random.default_rng(2)
    assert mi.mutual_information(rng.normal(0, 1, 100), rng.normal(0, 1, 100), seed=1) >= 0
