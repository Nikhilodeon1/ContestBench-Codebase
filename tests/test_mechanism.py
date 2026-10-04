"""Tests for the mechanism analysis (what drives confidence, if not agreement)."""

import numpy as np
import pandas as pd

from contestbench.analysis import mechanism


def _frame(conf, feat):
    return pd.DataFrame({"id": [f"c{i}" for i in range(len(conf))],
                         "confidence": conf, "featA": feat})


def test_feature_correlation_detects_dependence():
    feat = np.linspace(0, 1, 60)
    conf = 0.5 + 0.3 * feat                      # confidence tracks featA
    out = mechanism.feature_correlations(_frame(conf, feat), ["featA"])
    assert out["featA"] > 0.9


def test_feature_r2_high_when_explained():
    feat = np.linspace(0, 1, 60)
    conf = 0.5 + 0.3 * feat
    r2 = mechanism.feature_r2(_frame(conf, feat), ["featA"])
    assert r2 > 0.95


def test_feature_r2_near_zero_for_noise():
    rng = np.random.default_rng(0)
    df = _frame(rng.normal(0.6, 0.1, 200), rng.normal(0, 1, 200))
    assert mechanism.feature_r2(df, ["featA"]) < 0.1


def test_cross_model_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(1)
    ids = [f"c{i}" for i in range(50)]
    panel = {
        "m1": pd.DataFrame({"id": ids, "confidence": rng.normal(0.6, 0.1, 50)}),
        "m2": pd.DataFrame({"id": ids, "confidence": rng.normal(0.7, 0.1, 50)}),
    }
    m = mechanism.cross_model_confidence_corr(panel)
    assert abs(m.loc["m1", "m1"] - 1.0) < 1e-9
    assert abs(m.loc["m1", "m2"] - m.loc["m2", "m1"]) < 1e-9
