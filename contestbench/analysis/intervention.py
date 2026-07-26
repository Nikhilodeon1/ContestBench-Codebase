"""Prompting-intervention analysis (Fix I).

Tests whether in-context exemplars describing the physician-agreement pattern per
tier shift a model's verbalized confidence toward pi -- no external recalibration
layer, just prompting. Exemplars are held out of the scored corpus; this module
verifies that and compares intervention vs baseline on the shared eval ids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contestbench.eval import batch
from contestbench.metrics import pad, stats


def leaked_exemplars(eval_ids, exemplars) -> set:
    """Exemplar ids that leaked into the evaluated set (should be empty)."""
    return set(eval_ids) & set(exemplars)


def load_intervention(paths: list[str], corpus: pd.DataFrame, label: str) -> pd.DataFrame:
    parts = [batch.parse_batch_file(p) for p in paths]
    inv = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    out = corpus[["id", "pi", "pi_agree", "tier"]].merge(inv, on="id", how="inner")
    out["label"] = label
    return out


def _row(df: pd.DataFrame) -> dict:
    from scipy.stats import pearsonr
    c, pa = df["confidence"].values, df["pi_agree"].values
    signed = {t: pad.pad_signed(df[df.tier == t]["confidence"], df[df.tier == t]["pi_agree"])
              for t in ["high", "contested", "ambiguous"]}
    return {"conf": float(c.mean()), "pad": pad.pad(c, pa),
            "r": float(pearsonr(pa, c)[0]),
            "signed_high": signed["high"], "signed_contested": signed["contested"],
            "signed_ambiguous": signed["ambiguous"]}


def compare(baseline: pd.DataFrame, inv: pd.DataFrame) -> dict:
    """Baseline vs intervention on the SHARED ids only."""
    ids = set(baseline["id"]) & set(inv["id"])
    b = baseline[baseline["id"].isin(ids)]
    i = inv[inv["id"].isin(ids)]
    rb, ri = _row(b), _row(i)
    return {"n": len(ids), "pad_before": rb["pad"], "pad_after": ri["pad"],
            "conf_before": rb["conf"], "conf_after": ri["conf"],
            "r_before": rb["r"], "r_after": ri["r"],
            "amb_before": rb["signed_ambiguous"], "amb_after": ri["signed_ambiguous"],
            "high_before": rb["signed_high"], "high_after": ri["signed_high"]}
