"""Tests for the post-hoc recalibrator (Fix H)."""

import numpy as np
import pandas as pd

from contestbench.analysis import recalibration


def _data(pi_agree, confidence, **feats):
    n = len(pi_agree)
    df = pd.DataFrame({"id": [f"c{i}" for i in range(n)], "pi_agree": pi_agree,
                       "confidence": confidence})
    for k, v in feats.items():
        df[k] = v
    return df


def test_recalibrated_predictions_out_of_fold_cover_all_rows():
    rng = np.random.default_rng(0)
    n = 120
    df = _data(rng.uniform(0.5, 1, n), rng.uniform(0.5, 1, n), f1=rng.normal(0, 1, n))
    preds = recalibration.recalibrate(df, ["f1"], k=5, seed=1)
    assert len(preds) == n and not np.isnan(preds).any()


def test_recalibration_improves_when_features_carry_signal():
    # model confidence is garbage, but a feature predicts pi -> recalibrator helps
    rng = np.random.default_rng(1)
    f1 = rng.uniform(0, 1, 300)
    pi = 0.5 + 0.5 * f1
    conf = rng.uniform(0.5, 1, 300)          # uninformative confidence
    df = _data(pi, conf, f1=f1)
    res = recalibration.evaluate(df, ["f1"], oracle_pad=0.02, k=5, seed=1)
    assert res["pad_after"] < res["pad_before"]


def test_confidence_importance_high_when_informative():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0, 1, 300)
    df = _data(0.5 + 0.5 * conf, conf, f1=rng.normal(0, 1, 300))  # pi tracks confidence
    imp = recalibration.confidence_importance(df, ["confidence", "f1"], k=5, seed=1)
    assert imp["permutation"] > 0.1          # confidence clearly matters


def test_confidence_importance_near_zero_when_noise():
    rng = np.random.default_rng(4)
    f1 = rng.uniform(0, 1, 300)
    df = _data(0.5 + 0.5 * f1, rng.uniform(0, 1, 300), f1=f1)  # pi tracks f1, not conf
    imp = recalibration.confidence_importance(df, ["confidence", "f1"], k=5, seed=1)
    assert imp["permutation"] < 0.05


def test_ablation_gap_positive_when_confidence_adds_signal():
    # pi tracks confidence, features are noise -> adding confidence must help (gap>0)
    rng = np.random.default_rng(5)
    conf = rng.uniform(0, 1, 300)
    df = _data(0.5 + 0.5 * conf, conf, f1=rng.normal(0, 1, 300))
    res = recalibration.ablation_gap_ci(df, ["f1"], n_boot=300, seed=1)
    assert res["gap"] > 0
    assert res["lo"] > 0                       # confidence's contribution is significant


def test_ablation_gap_brackets_zero_when_confidence_is_noise():
    rng = np.random.default_rng(6)
    f1 = rng.uniform(0, 1, 300)
    df = _data(0.5 + 0.5 * f1, rng.uniform(0, 1, 300), f1=f1)   # pi tracks f1, not conf
    res = recalibration.ablation_gap_ci(df, ["f1"], n_boot=300, seed=1)
    assert res["lo"] <= 0 <= res["hi"]          # confidence adds nothing significant


def test_gap_closed_fraction_reported():
    rng = np.random.default_rng(2)
    f1 = rng.uniform(0, 1, 300)
    df = _data(0.5 + 0.5 * f1, rng.uniform(0.5, 1, 300), f1=f1)
    res = recalibration.evaluate(df, ["f1"], oracle_pad=0.02, k=5, seed=1)
    assert "gap_closed_frac" in res
    assert 0.0 <= res["gap_closed_frac"] <= 1.5
