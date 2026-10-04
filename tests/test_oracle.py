"""Tests for the oracle/upper-bound baseline (predict agreement from features)."""

import numpy as np
import pandas as pd

from contestbench.analysis import oracle


def _corpus(pi_agree, **feats):
    n = len(pi_agree)
    df = pd.DataFrame({"id": [f"c{i}" for i in range(n)], "pi_agree": pi_agree})
    for k, v in feats.items():
        df[k] = v
    return df


def test_cv_predictions_cover_every_row_out_of_fold():
    rng = np.random.default_rng(0)
    df = _corpus(rng.uniform(0.5, 1, 100), f1=rng.normal(0, 1, 100))
    preds = oracle.cv_predictions(df, ["f1"], k=5, seed=1)
    assert len(preds) == 100
    assert not np.isnan(preds).any()


def test_oracle_recovers_learnable_signal():
    # agreement is a clean function of the feature -> oracle PAD should be tiny
    rng = np.random.default_rng(1)
    f1 = rng.uniform(0, 1, 300)
    pi = 0.5 + 0.5 * f1                       # perfectly learnable
    df = _corpus(pi, f1=f1)
    res = oracle.evaluate(df, ["f1"], k=5, seed=1)
    assert res["oracle_pad"] < res["constant_pad"]
    assert res["oracle_pad"] < 0.05


def test_ridge_estimator_recovers_linear_signal():
    rng = np.random.default_rng(7)
    f1 = rng.uniform(0, 1, 300)
    df = _corpus(0.5 + 0.5 * f1, f1=f1)
    res = oracle.evaluate(df, ["f1"], estimator="ridge")
    assert res["oracle_pad"] < 0.05


def test_robustness_reports_all_estimators():
    rng = np.random.default_rng(8)
    f1 = rng.uniform(0, 1, 300)
    df = _corpus(0.5 + 0.5 * f1, f1=f1)
    out = oracle.robustness(df, ["f1"], estimators=["gbt", "ridge", "linear"])
    assert set(out) == {"gbt", "ridge", "linear"}
    assert all(v < 0.05 for v in out.values())      # all recover the signal


def test_oracle_no_leakage_on_pure_noise():
    # target independent of features -> held-out oracle must NOT beat the constant
    rng = np.random.default_rng(2)
    df = _corpus(rng.uniform(0.5, 1, 300), f1=rng.normal(0, 1, 300))
    res = oracle.evaluate(df, ["f1"], k=5, seed=1)
    # allow a hair of slack; the point is it can't meaningfully beat the constant
    assert res["oracle_pad"] >= res["constant_pad"] - 0.01
