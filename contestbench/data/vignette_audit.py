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
