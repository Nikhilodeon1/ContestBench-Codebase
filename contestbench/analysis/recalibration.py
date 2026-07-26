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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

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


def confidence_importance(df: pd.DataFrame, features: list[str],
                          target: str = "pi_agree", k: int = 5,
                          seed: int = 20260723) -> dict:
    """Importance of the 'confidence' feature (impurity + held-out permutation).

    Answers: does model confidence carry independent signal, or is the GBT just
    pruning it in favor of the cleaner features? Permutation importance on the
    held-out fold is the honest test (impurity can reward overfit splits).
    """
    ci = features.index("confidence")
    X = df[features].to_numpy(float)
    y = df[target].to_numpy(float)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    impurity, perm = [], []
    for tr, te in kf.split(X):
        model = oracle._make_model(seed).fit(X[tr], y[tr])
        impurity.append(model.feature_importances_[ci])
        pi = permutation_importance(model, X[te], y[te], n_repeats=5, random_state=seed)
        perm.append(pi.importances_mean[ci])
    return {"impurity": float(np.mean(impurity)),
            "permutation": float(np.mean(perm)),
            "permutation_std": float(np.std(perm))}


def importance_panel(panel: dict[str, pd.DataFrame],
                     features: list[str] | None = None) -> pd.DataFrame:
    feats = features or RECAL_FEATURES
    corpus = oracle.load_corpus_with_target()[["id"] + oracle.RATING_INDEPENDENT_FEATURES]
    rows = []
    for label, d in panel.items():
        m = d.merge(corpus, on="id")
        m["pi_agree"] = stats.agreement_rate(m["pi"].values)
        rows.append({"config": label, **confidence_importance(m, feats)})
    return pd.DataFrame(rows)


def ablation_gap_ci(df: pd.DataFrame, features: list[str], target: str = "pi_agree",
                    k: int = 5, seed: int = 20260726, n_boot: int = 3000) -> dict:
    """Bootstrap CI on how much model confidence adds over features alone.

    gap = PAD(features-only oracle) - PAD(features+confidence recalibrator), both
    out-of-fold on the SAME cases. Positive gap = confidence lowers PAD. Paired
    bootstrap over cases (same resample applied to both) isolates confidence's
    marginal contribution. CI bracketing 0 => confidence adds nothing significant.
    """
    pi = df[target].to_numpy(float)
    feat_only = oracle.cv_predictions(df, features, target, k, seed)
    with_conf = oracle.cv_predictions(df, ["confidence"] + features, target, k, seed)
    e_feat = np.abs(feat_only - pi)          # per-case abs error, features only
    e_conf = np.abs(with_conf - pi)          # per-case abs error, + confidence
    gap = float(e_feat.mean() - e_conf.mean())
    rng = np.random.default_rng(seed)
    n = len(pi)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = e_feat[idx].mean() - e_conf[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"gap": gap, "lo": float(lo), "hi": float(hi),
            "pad_features_only": float(e_feat.mean()),
            "pad_with_confidence": float(e_conf.mean())}


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
