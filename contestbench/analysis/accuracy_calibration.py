"""Accuracy-calibrated baseline (weekend fix #2).

Converts the paper's asserted ECE-vs-PAD distinction into a demonstrated one:
calibrate model confidence to its own ACCURACY (does the model's answer match the
majority malignancy label) via Platt scaling, cross-validated (no leakage), then
measure PAD against physician agreement pi. If accuracy-calibration doesn't drive
PAD to zero, calibrating to a gold label is provably not the same as calibrating
to the physician-agreement distribution.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from contestbench import config
from contestbench.metrics import stats


def platt_cv(confidence, correct, k: int = 5, seed: int = 20260726) -> np.ndarray:
    """Out-of-fold Platt-scaled P(correct) from a 1-D confidence feature."""
    x = np.asarray(confidence, float).reshape(-1, 1)
    y = np.asarray(correct, int)
    out = np.full(len(x), np.nan)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in kf.split(x):
        if len(np.unique(y[tr])) < 2:          # degenerate fold -> base rate
            out[te] = y[tr].mean()
        else:
            out[te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return out


def evaluate(df, k: int = 5, seed: int = 20260726) -> dict:
    """PAD of accuracy-calibrated confidence vs pi. df needs pi, answer, confidence.

    Correctness is defined vs the majority malignancy label; ambiguous cases
    (pi==0.5, no majority) are excluded from the accuracy fit but the calibrator is
    applied to every case and PAD is measured over all.
    """
    d = df.dropna(subset=["confidence"]).copy()
    d["pi_agree"] = stats.agreement_rate(d["pi"].values)
    fit = d[d["pi"] != 0.5].copy()
    fit["correct"] = (fit["answer"] == "malignant").values == (fit["pi"] > 0.5).values
    # calibrate on the labelled subset, apply to all cases
    model = LogisticRegression().fit(fit["confidence"].values.reshape(-1, 1),
                                     fit["correct"].astype(int).values)
    c_acc = model.predict_proba(d["confidence"].values.reshape(-1, 1))[:, 1]
    pad_raw = float(np.abs(d["confidence"].values - d["pi_agree"].values).mean())
    pad_acc = float(np.abs(c_acc - d["pi_agree"].values).mean())
    # out-of-fold accuracy-calibration quality on the labelled subset (sanity)
    oof = platt_cv(fit["confidence"].values, fit["correct"].astype(int).values, k, seed)
    return {"n": len(d), "pad_raw": pad_raw, "pad_accuracy_calibrated": pad_acc,
            "mean_accuracy": float(fit["correct"].mean()),
            "oof_calibrated_mean": float(np.nanmean(oof))}
