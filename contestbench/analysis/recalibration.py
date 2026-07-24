"""Post-hoc recalibrator (Fix H) -- the paper's proposed intervention.

Distinct from the oracle: the oracle ignores the model entirely, this maps the
model's OWN confidence plus rating-independent case features to the physician
agreement rate. A per-config recalibrator (never one shared model, so each
model's fixability is measured separately) trained with the same 5-fold
out-of-fold protocol as the oracle. Reports PAD before vs after and how much of
the gap between the raw model and the feature-only oracle it closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contestbench.analysis import oracle
from contestbench.metrics import stats

# model confidence + the rating-independent imaging features (per directive)
RECAL_FEATURES = ["confidence"] + oracle.RATING_INDEPENDENT_FEATURES


def recalibrate(df: pd.DataFrame, features: list[str], target: str = "pi_agree",
                k: int = 5, seed: int = 20260723) -> np.ndarray:
    """Out-of-fold recalibrated confidence from [model confidence + features]."""
    return oracle.cv_predictions(df, features, target=target, k=k, seed=seed)


def evaluate(df: pd.DataFrame, features: list[str], oracle_pad: float,
             target: str = "pi_agree", k: int = 5, seed: int = 20260723) -> dict:
    """PAD before (raw model) vs after (recalibrated), and gap-to-oracle closed."""
    pi = df[target].to_numpy(float)
    pad_before = float(np.abs(df["confidence"].to_numpy(float) - pi).mean())
    preds = recalibrate(df, features, target, k, seed)
    pad_after = float(np.abs(preds - pi).mean())
    gap = pad_before - oracle_pad
    closed = (pad_before - pad_after) / gap if abs(gap) > 1e-9 else float("nan")
    return {
        "n": len(df),
        "pad_before": pad_before,
        "pad_after": pad_after,
        "oracle_pad": oracle_pad,
        "gap_closed_frac": closed,
    }


def evaluate_panel(panel: dict[str, pd.DataFrame], oracle_pad: float,
                   features: list[str] | None = None) -> pd.DataFrame:
    """Run the per-config recalibrator across the whole panel."""
    feats = features or RECAL_FEATURES
    corpus = oracle.load_corpus_with_target()[["id"] + oracle.RATING_INDEPENDENT_FEATURES]
    rows = []
    for label, d in panel.items():
        merged = d.merge(corpus, on="id")
        merged["pi_agree"] = stats.agreement_rate(merged["pi"].values)
        res = evaluate(merged, feats, oracle_pad=oracle_pad)
        rows.append({"config": label, **res})
    return pd.DataFrame(rows)
