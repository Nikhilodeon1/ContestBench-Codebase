"""Mechanism analysis (Fix 5): if confidence doesn't track agreement, what does?

Feature list is LOCKED (no additions after seeing results) to avoid fishing:
subtlety, spiculation, margin, extent, n_raters, malignancy_extremity.
Also tests whether models covary with *each other*, which would indicate a shared
latent difficulty signal even absent any tie to physician agreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from contestbench import config

FEATURES = ["subtlety", "spiculation", "margin", "extent",
            "n_raters", "malignancy_extremity"]


def load_corpus_features(features: list[str] | None = None) -> pd.DataFrame:
    feats = features or FEATURES
    return pd.read_parquet(config.CORPUS_PARQUET)[["id"] + feats]


def feature_correlations(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Pearson r between confidence and each feature."""
    return {f: float(pearsonr(df[f].values, df["confidence"].values)[0]) for f in features}


def feature_r2(df: pd.DataFrame, features: list[str]) -> float:
    """R^2 of an OLS fit of confidence on all features (how much is explained)."""
    X = np.column_stack([df[f].values for f in features] + [np.ones(len(df))])
    y = df["confidence"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = ((y - X @ beta) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return float(1 - resid / total) if total > 0 else float("nan")


def cross_model_confidence_corr(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise correlation of confidence across models on their shared cases."""
    series = {k: v.set_index("id")["confidence"] for k, v in panel.items()}
    common = sorted(set.intersection(*[set(s.index) for s in series.values()]))
    labels = list(series)
    m = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            m.loc[a, b] = pearsonr(series[a].reindex(common), series[b].reindex(common))[0]
    return m


def mechanism_table(panel: dict[str, pd.DataFrame],
                    features: list[str] | None = None) -> pd.DataFrame:
    """Per-model feature correlations + multivariate R^2."""
    feats = features or FEATURES
    corpus = load_corpus_features(feats)
    rows = []
    for label, d in panel.items():
        merged = d.merge(corpus, on="id")
        row = {"config": label, **feature_correlations(merged, feats),
               "R2_all": feature_r2(merged, feats)}
        rows.append(row)
    return pd.DataFrame(rows)
