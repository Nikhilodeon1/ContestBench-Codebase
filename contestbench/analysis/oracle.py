"""Oracle / upper-bound baseline (strategy directive A, 2026-07-23).

A lightweight supervised model predicts the physician agreement rate pi directly
from the case-level features already extracted for the mechanism analysis, using
k-fold cross-validation (out-of-fold predictions only -> no train/test leakage).

This is a DIAGNOSTIC INSTRUMENT, not a modeling contribution and NOT a novel
"disagreement prediction method" (predicting agreement from features is
established: LIDC panel-opinion work 2009; multi-rater disagreement work 2019).
Its PAD is an oracle/upper-bound: it shows how much of the confidence-vs-agreement
gap is closable from cheap features alone, and therefore how far short the LLMs
fall of even a simple feature-based predictor.

Caveat (report honestly): n_raters and malignancy_extremity are derived from the
same annotations that define pi, so they share source with the target; imaging-only
features (subtlety/spiculation/margin/extent) are the fully independent variant and
are reported alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from contestbench import config
from contestbench.metrics import baselines, stats

# Three levels of independence from the ratings that define pi:
#   geometry-only : ROI-extent = pure polygon geometry, ZERO subjective input
#                   (fully rater-independent upper bound)
#   rating-indep. : the imaging Likert calls, independent of the MALIGNANCY rating
#                   specifically (but still same-session reader judgments)
#   all-features  : adds annotation-derived features that share source with pi
GEOMETRY_FEATURES = ["extent"]
RATING_INDEPENDENT_FEATURES = ["subtlety", "spiculation", "margin", "extent"]
ALL_FEATURES = RATING_INDEPENDENT_FEATURES + ["n_raters", "malignancy_extremity"]


def _make_model(seed: int, estimator: str = "gbt"):
    if estimator == "gbt":
        return GradientBoostingRegressor(random_state=seed, n_estimators=200,
                                         max_depth=3, learning_rate=0.05)
    if estimator == "ridge":
        return Ridge(alpha=1.0)
    if estimator == "linear":
        return LinearRegression()
    raise ValueError(f"unknown estimator {estimator}")


def cv_predictions(df: pd.DataFrame, features: list[str], target: str = "pi_agree",
                   k: int = 5, seed: int = 20260723, estimator: str = "gbt") -> np.ndarray:
    """Out-of-fold predictions of ``target`` from ``features`` (no leakage)."""
    X = df[features].to_numpy(float)
    y = df[target].to_numpy(float)
    preds = np.full(len(df), np.nan)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        model = _make_model(seed, estimator)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = np.clip(model.predict(X[test_idx]), 0.5, 1.0)
    return preds


def robustness(df: pd.DataFrame, features: list[str], target: str = "pi_agree",
               estimators: list[str] | None = None, k: int = 5,
               seed: int = 20260723) -> dict:
    """Oracle PAD under each estimator class -- guards against a GBT-specific artifact."""
    estimators = estimators or ["gbt", "ridge", "linear"]
    pi = df[target].to_numpy(float)
    return {e: float(np.abs(cv_predictions(df, features, target, k, seed, e) - pi).mean())
            for e in estimators}


def evaluate(df: pd.DataFrame, features: list[str], target: str = "pi_agree",
             k: int = 5, seed: int = 20260723, estimator: str = "gbt") -> dict:
    """Oracle PAD (out-of-fold) vs the case-blind constant, on the same cases."""
    pi = df[target].to_numpy(float)
    preds = cv_predictions(df, features, target, k, seed, estimator)
    c_star = baselines.optimal_constant(pi)
    return {
        "features": features,
        "n": len(df),
        "oracle_pad": float(np.abs(preds - pi).mean()),
        "constant_pad": baselines.constant_pad(pi, c_star),
        "c_star": c_star,
    }


def load_corpus_with_target() -> pd.DataFrame:
    df = pd.read_parquet(config.CORPUS_PARQUET).copy()
    df["pi_agree"] = stats.agreement_rate(df["pi"].values)
    return df
