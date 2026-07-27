"""Vignette-template audit (weekend fix #5).

Checks whether the feature-based vignette structurally limits the models, which
would make some decoupling unavoidable regardless of model behavior:

- Collision analysis: distinct nodules whose *presented* features (subtlety,
  spiculation, margin, at the 2-dp precision shown in the prompt) are identical
  but whose physician agreement pi differs. The model sees an identical prompt
  for these, so it CANNOT track pi for them -- any decoupling there is forced by
  the template, not a model failure. If this fraction is small, the decoupling is
  a genuine model property, not a template artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRESENTED = ["subtlety", "spiculation", "margin"]


def _bucket_key(df: pd.DataFrame, ndigits: int) -> list:
    return [df[c].round(ndigits) for c in PRESENTED]


def bucket_mean_pad(df: pd.DataFrame, ndigits: int = 2,
                    leave_one_out: bool = False) -> float:
    """PAD of the trivial no-ML predictor: pi = mean pi within identical-vignette bucket.

    In-sample (default) is the vignette-optimal (L2) floor but is leaky for small
    buckets. leave_one_out=True excludes each case from its own bucket mean (and
    falls back to the global pi mean for singletons) -- the fair, no-leakage
    comparator for the CV oracle.
    """
    d = df.copy()
    g = d.groupby(_bucket_key(d, ndigits))["pi_agree"]
    if not leave_one_out:
        pred = g.transform("mean")
    else:
        bsum, bcnt = g.transform("sum"), g.transform("count")
        gmean = d["pi_agree"].mean()
        pred = np.where(bcnt > 1, (bsum - d["pi_agree"]) / (bcnt - 1), gmean)
    return float(np.abs(np.asarray(pred) - d["pi_agree"].to_numpy()).mean())


def non_colliding_mask(df: pd.DataFrame, ndigits: int = 2) -> pd.Series:
    """True for cases whose vignette bucket maps to a single pi (no forced decoupling)."""
    d = df.copy()
    nun = d.groupby(_bucket_key(d, ndigits))["pi_agree"].transform("nunique")
    return (nun == 1).reset_index(drop=True)


def collision_report(df: pd.DataFrame, ndigits: int = 2) -> dict:
    d = df.copy()
    key = [f"_{c}" for c in PRESENTED]
    for c, k in zip(PRESENTED, key):
        d[k] = d[c].round(ndigits)
    groups = d.groupby(key)
    n_unique = groups.ngroups
    pi_var = groups["pi_agree"].transform("nunique")   # distinct pi per vignette
    forced = d[pi_var > 1]
    # mean pi-spread within vignettes that map to >1 pi
    spread = (groups["pi_agree"].agg(lambda s: s.max() - s.min()))
    spread = float(spread[spread > 0].mean()) if (spread > 0).any() else 0.0
    return {
        "n_cases": len(d),
        "n_unique_vignettes": int(n_unique),
        "cases_in_pi_varying_collisions": int(len(forced)),
        "frac_forced": len(forced) / len(d) if len(d) else 0.0,
        "mean_pi_spread_in_collisions": spread,
    }
