"""Phases 3-4: honest oracle and post-hoc recalibrator under PAD-B.

Target is the vote fraction f_t (fraction of raters rating malignancy >= t), the
quantity PAD-B scores against. All CV is 5-fold GroupKFold by scan (a scan's
nodules never straddle train/test) and all CIs are scan-cluster bootstraps.

Feature sets (what the model actually sees is subtlety/spiculation/margin only):
  vignette  : subtlety, spiculation, margin   -- the vignette-optimal oracle
  extended  : vignette + extent               -- UPPER BOUND, extent is NOT in the
                                                 prompt, so not a same-information
                                                 comparison
  geometry  : extent only                     -- zero subjective input
n_raters / malignancy_extremity are excluded: derived from the same ratings as f.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

from contestbench import config
from contestbench.analysis import panel_b
from contestbench.data import ratings as rt
from contestbench.metrics import brier, pad, stats

VIGNETTE = ["subtlety", "spiculation", "margin"]
EXTENDED = VIGNETTE + ["extent"]
GEOMETRY = ["extent"]
FEATURE_SETS = {"vignette": VIGNETTE, "extended": EXTENDED, "geometry": GEOMETRY}
ESTIMATORS = ["gbt", "ridge", "linear"]
SEED = 20260723


def _model(estimator: str):
    if estimator == "gbt":
        return GradientBoostingRegressor(random_state=SEED, n_estimators=200,
                                         max_depth=3, learning_rate=0.05)
    if estimator == "ridge":
        return Ridge(alpha=1.0)
    if estimator == "linear":
        return LinearRegression()
    raise ValueError(estimator)


def cv_predict(X, y, groups, estimator: str = "gbt", k: int = 5, seed: int = SEED) -> np.ndarray:
    """Out-of-fold predictions of f, clipped to [0,1], folds grouped by scan."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    out = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in gkf.split(X, y, np.asarray(groups)):
        out[te] = np.clip(_model(estimator).fit(X[tr], y[tr]).predict(X[te]), 0.0, 1.0)
    return out


def _l1_from_forecast(p_hat, pi) -> float:
    """PAD-L1 of the answer/confidence an oracle forecast implies (a=1 iff p>.5, c=max(p,1-p))."""
    p_hat = np.asarray(p_hat, float)
    return pad.pad_l1(np.maximum(p_hat, 1.0 - p_hat), pi)


def corpus_with_f(threshold: int) -> pd.DataFrame:
    c = pd.read_parquet(config.CORPUS_PARQUET)
    r = rt.load_ratings().set_index("id")["ratings"]
    c["f"] = rt.vote_fraction(r.loc[c["id"]], threshold)
    c["pi_agree"] = stats.agreement_rate(c["f"].values)
    return c


def _cluster_ci(arr_fn, groups, n_boot, seed, calls=None):
    gen = (stats.nested_cluster_indices(calls, groups, n_boot, seed) if calls is not None
           else stats.scan_cluster_indices(groups, n_boot, seed))
    vals = np.array([arr_fn(idx) for idx in gen])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def oracle_table(threshold: int, n_boot: int = 1000) -> pd.DataFrame:
    c = corpus_with_f(threshold)
    f, pi, g = c["f"].values, c["pi_agree"].values, c["scan_idx"].values
    p_star = brier.optimal_constant_p(f)
    const_sq = (p_star - f) ** 2
    rows = []
    for name, feats in FEATURE_SETS.items():
        for est in ESTIMATORS:
            pred = cv_predict(c[feats].values, f, g, est)
            sq = (pred - f) ** 2
            lo, hi = _cluster_ci(lambda i: sq[i].mean() - const_sq[i].mean(), g, n_boot, 1)
            rows.append({"threshold": threshold, "oracle": name, "estimator": est, "n": len(c),
                         "PAD_B": sq.mean(), "const_B": const_sq.mean(),
                         "margin_vs_const": sq.mean() - const_sq.mean(),
                         "margin_lo": lo, "margin_hi": hi,
                         "PAD_L1": _l1_from_forecast(pred, pi),
                         "const_L1": float(np.abs(0.75 - pi).mean())})
    return pd.DataFrame(rows)


def recal_table(threshold: int, n_boot: int = 1000, estimators=("gbt", "ridge")) -> pd.DataFrame:
    """Per-config recalibration + features-only vs features+p ablation (paired, cluster CI)."""
    rat = rt.load_ratings().set_index("id")["ratings"]
    corpus = pd.read_parquet(config.CORPUS_PARQUET)[["id", "scan_idx"] + EXTENDED]
    rows = []
    for label, d in panel_b.load_panel_b().items():
        d = d.drop(columns=["scan_idx"]).merge(corpus, on="id")
        d["f"] = rt.vote_fraction(rat.loc[d["id"]], threshold)
        d["pi_agree"] = stats.agreement_rate(d["f"].values)
        f, g, pi = d["f"].values, d["scan_idx"].values, d["pi_agree"].values
        sq_raw = (d["p"].values - f) ** 2
        for fs_name, feats in (("vignette", VIGNETTE), ("extended", EXTENDED)):
            for est in estimators:
                sq = {}
                sq["features_only"] = (cv_predict(d[feats].values, f, g, est) - f) ** 2
                sq["features+p"] = (cv_predict(d[feats + ["p"]].values, f, g, est) - f) ** 2
                sq["p_only"] = (cv_predict(d[["p"]].values, f, g, est) - f) ** 2
                gap = sq["features_only"] - sq["features+p"]
                glo, ghi = _cluster_ci(lambda i: gap[i].mean(), g, n_boot, 2, calls=d["chunk"].values)
                red = sq_raw - sq["features+p"]
                rlo, rhi = _cluster_ci(lambda i: red[i].mean(), g, n_boot, 3, calls=d["chunk"].values)
                before, oracle_pad = sq_raw.mean(), sq["features_only"].mean()
                rows.append({"threshold": threshold, "config": label, "feature_set": fs_name,
                             "estimator": est, "n": len(d),
                             "PAD_B_raw": before, "PAD_B_p_only": sq["p_only"].mean(),
                             "PAD_B_features_only": oracle_pad,
                             "PAD_B_features+p": sq["features+p"].mean(),
                             "ablation_gap": gap.mean(), "ablation_lo": glo, "ablation_hi": ghi,
                             "recal_gain_vs_raw": red.mean(), "gain_lo": rlo, "gain_hi": rhi,
                             "gap_closed_frac": (before - sq["features+p"].mean()) / (before - oracle_pad)
                             if abs(before - oracle_pad) > 1e-9 else float("nan")})
    return pd.DataFrame(rows)
