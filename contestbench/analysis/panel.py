"""Panel-level analysis: the cross-vendor comparison behind the paper's claims.

Everything the results section reports is computed here (not in ad-hoc scripts)
so `reproduce.sh` regenerates every number:
  - per-config metrics table (r, PAD, gap vs the case-blind constant, signed PAD)
  - reasoning effect (thinking - standard) per model family
  - capability trend (signed-PAD-ambiguous vs capability rank, cluster bootstrap)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contestbench import config
from contestbench.metrics import ece, pad, stats

# label -> responses parquet stem. The primary panel is same-family Claude
# (capability x reasoning); gemini/deepseek are the cross-vendor robustness panel.
PANEL: dict[str, str] = {
    "haiku:standard": "responses_haiku-standard",
    "haiku:thinking": "responses_haiku-thinking",
    "sonnet:standard": "responses_sonnet-standard",
    "sonnet:thinking": "responses_sonnet-thinking",
    "opus:standard": "responses_opus-standard",
    "opus:thinking": "responses_opus-thinking",
    "gemini:standard": "responses_gemini-m-standard",
    "gemini:thinking": "responses_gemini-m-thinking",
    "deepseek:standard": "responses_deepseek-standard",
    "deepseek:thinking": "responses_deepseek-thinking",
}

CAPABILITY_RANK = {"haiku": 1, "sonnet": 2, "opus": 3}
C_STAR = 0.75  # reported case-blind constant (median agreement rate)


def load_panel(panel: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Load each config's responses, adding the agreement-rate column."""
    panel = panel or PANEL
    out = {}
    for label, stem in panel.items():
        path = config.DATA_DIR / f"{stem}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path).dropna(subset=["confidence"]).copy()
        d["pi_agree"] = stats.agreement_rate(d["pi"].values)
        out[label] = d
    return out


def gap_vs_constant(conf, pi_agree, c_star: float = C_STAR,
                    n_boot: int = 3000, seed: int = 20260720):
    """(gap, lo, hi) for PAD_model - PAD_constant. Positive => worse than case-blind."""
    conf = np.asarray(conf, float)
    pi_agree = np.asarray(pi_agree, float)
    gap = float(np.abs(conf - pi_agree).mean() - np.abs(c_star - pi_agree).mean())
    rng = np.random.default_rng(seed)
    n = len(conf)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = (np.abs(conf[idx] - pi_agree[idx]).mean()
                    - np.abs(c_star - pi_agree[idx]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return gap, float(lo), float(hi)


def reasoning_effect(pads: dict[str, float]) -> dict[str, float]:
    """thinking - standard PAD per family, for families with both settings."""
    out = {}
    for label, value in pads.items():
        family, _, setting = label.partition(":")
        if setting == "thinking":
            std = pads.get(f"{family}:standard")
            if std is not None:
                out[family] = value - std
    return out


def capability_slope(per_rank: dict[int, np.ndarray], n_boot: int = 2000,
                     seed: int = 20260720):
    """OLS slope of per-rank mean signed deviation vs rank, cluster-bootstrapped.

    ``per_rank`` maps capability rank -> per-case signed deviations (same cases
    across ranks), so resampling case indices propagates case-level uncertainty.
    """
    ranks = sorted(per_rank)
    arrays = [np.asarray(per_rank[r], float) for r in ranks]
    n = len(arrays[0])
    slope = float(np.polyfit(ranks, [a.mean() for a in arrays], 1)[0])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = np.polyfit(ranks, [a[idx].mean() for a in arrays], 1)[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, float(lo), float(hi)


def panel_table(panel_data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Per-config metrics: the paper's main results table."""
    data = panel_data if panel_data is not None else load_panel()
    rows = []
    for label, d in data.items():
        c = d["confidence"].values
        pi = d["pi_agree"].values
        r, rlo, rhi, _ = stats.pearson_ci(pi, c, n_boot=1000)
        gap, glo, ghi = gap_vs_constant(c, pi, n_boot=1500)
        signed = {t: pad.pad_signed(d[d.tier == t]["confidence"], d[d.tier == t]["pi_agree"])
                  for t in ["high", "contested", "ambiguous"]}
        nz = d[d["pi"] != 0.5]
        correct = ((nz["answer"] == "malignant").values == (nz["pi"] > 0.5).values).astype(float)
        rows.append({
            "config": label, "n": len(d),
            "conf_mean": c.mean(), "conf_sd": c.std(),
            "r": r, "r_lo": rlo, "r_hi": rhi,
            "PAD": pad.pad(c, pi),
            "gap": gap, "gap_lo": glo, "gap_hi": ghi,
            "signed_high": signed["high"], "signed_contested": signed["contested"],
            "signed_ambiguous": signed["ambiguous"],
            "ECE": ece.ece(nz["confidence"].values, correct, n_bins=10),
        })
    return pd.DataFrame(rows)
