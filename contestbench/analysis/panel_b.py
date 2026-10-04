"""Phase 2: ten-config panel under PAD-B (primary) with PAD-L1 for comparison.

CIs use a scan-cluster bootstrap (resample the 709 scans, keeping each scan's
nodules together); the case-level iid CI is reported alongside for gap-vs-constant
so the effect of clustering is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from contestbench import config
from contestbench.analysis import panel
from contestbench.metrics import brier, pad, stats


def load_panel_b() -> dict[str, pd.DataFrame]:
    """Per-config frames with f, pi_agree, scan_idx, a (binary), p, s."""
    corpus = pd.read_parquet(config.CORPUS_PARQUET)[["id", "scan_idx"]].reset_index(drop=True)
    corpus["order"] = np.arange(len(corpus))
    out = {}
    for label, d in panel.load_panel().items():
        d = d.merge(corpus, on="id", how="left").reset_index(drop=True)
        d["chunk"] = d["order"] // (471 if label.startswith("deepseek") else 100)  # one chunk = one model call
        d["f"] = d["pi"].astype(float)
        d["a"] = brier.answer_to_binary(d["answer"])
        d["p"] = brier.fold_forecast(d["a"], d["confidence"])
        d["s"] = brier.support(d["a"], d["f"])
        out[label] = d
    return out


def majority_match(d: pd.DataFrame) -> dict:
    """Share of answers on the physician-majority side (ties f==.5 excluded)."""
    nz = d[d["f"] != 0.5]
    maj = (nz["f"] > 0.5).astype(float)
    return {"match_majority": float((nz["a"] == maj).mean()),
            "n_ties_excl": int((d["f"] == 0.5).sum()),
            "frac_answer_malignant": float(d["a"].mean())}


def _boot(d: pd.DataFrame, stat, n_boot: int, seed: int):
    """95% CI. Two-stage (call -> scan) cluster bootstrap when the frame carries a call id ('chunk'),
    scan-only otherwise."""
    if "chunk" in d.columns:
        gen = stats.nested_cluster_indices(d["chunk"].values, d["scan_idx"].values, n_boot, seed)
    else:
        gen = stats.scan_cluster_indices(d["scan_idx"].values, n_boot, seed)
    vals = np.array([stat(d.iloc[idx]) for idx in gen])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def config_row(label: str, d: pd.DataFrame, p_star: float, n_boot: int = 2000,
               seed: int = 20260801) -> dict:
    f, p, c = d["f"].values, d["p"].values, d["confidence"].values
    pi = d["pi_agree"].values
    padb = brier.pad_b(p, f)
    const_b = brier.constant_pad_b(f, p_star)
    gap_stat = lambda g: (brier.pad_b(g["p"], g["f"]) - brier.constant_pad_b(g["f"], p_star))
    glo, ghi = _boot(d, gap_stat, n_boot, seed)
    rng = np.random.default_rng(seed)  # iid comparison
    iid = np.array([gap_stat(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n_boot // 2)])
    rcf, rpf = pearsonr(c, f)[0], pearsonr(p, f)[0]
    r_lo, r_hi = _boot(d, lambda g: pearsonr(g["p"], g["f"])[0], n_boot, seed + 1)
    rc_lo, rc_hi = _boot(d, lambda g: pearsonr(g["confidence"], g["f"])[0], n_boot, seed + 2)
    l1 = pad.pad_l1(c, pi)
    l1_const = float(np.abs(0.75 - pi).mean())
    l1_lo, l1_hi = _boot(d, lambda g: pad.pad_l1(g["confidence"], g["pi_agree"]) -
                         float(np.abs(0.75 - g["pi_agree"]).mean()), n_boot, seed + 3)
    b_lo, b_hi = _boot(d, lambda g: brier.pad_b(g["p"], g["f"]), n_boot, seed + 4)
    nested = [gap_stat(d.iloc[i]) for i in stats.nested_cluster_indices(d["chunk"].values, d["scan_idx"].values,
                                                                       n_boot // 2, seed + 9)]
    row = {"config": label, "n": len(d),
           "gap_B_nested_lo": float(np.percentile(nested, 2.5)), "gap_B_nested_hi": float(np.percentile(nested, 97.5)),
           "PAD_B": padb, "PAD_B_lo": b_lo, "PAD_B_hi": b_hi,
           "const_B": const_b, "gap_B": padb - const_b, "gap_B_lo": glo, "gap_B_hi": ghi,
           "gap_B_iid_lo": float(np.percentile(iid, 2.5)),
           "gap_B_iid_hi": float(np.percentile(iid, 97.5)),
           "PAD_L1": l1, "const_L1_0.75": l1_const, "gap_L1": l1 - l1_const,
           "gap_L1_lo": l1_lo, "gap_L1_hi": l1_hi,
           "r_c_f": rcf, "r_c_f_lo": rc_lo, "r_c_f_hi": rc_hi,
           "r_p_f": rpf, "r_p_f_lo": r_lo, "r_p_f_hi": r_hi,
           "r_c_pi_old": pearsonr(c, pi)[0]}
    row.update(majority_match(d))
    return row


def panel_table_b(n_boot: int = 2000) -> pd.DataFrame:
    data = load_panel_b()
    ref = next(iter(data.values())).sort_values("id")
    corpus_f = pd.read_parquet(config.CORPUS_PARQUET)["pi"].values
    p_star = brier.optimal_constant_p(corpus_f)  # full-corpus mean malignant rate
    rows = [config_row(lb, d, p_star, n_boot) for lb, d in data.items()]
    t = pd.DataFrame(rows)
    t.attrs["p_star"] = p_star
    t.attrs["const_B_full_corpus"] = brier.constant_pad_b(corpus_f, p_star)
    return t


def threshold_sensitivity(thresholds=(3, 4), n_boot: int = 1000, seed: int = 20260802) -> pd.DataFrame:
    """Re-score the panel with f_t = fraction of raters rating malignancy >= t.

    The constant baseline is re-derived per threshold (p* = mean f_t). The model
    answers/confidences are fixed; only the physician target moves.
    """
    from contestbench.data import ratings as rt
    rat = rt.load_ratings().set_index("id")["ratings"]
    data = load_panel_b()
    rows = []
    for t in thresholds:
        f_all = rt.vote_fraction(rat, t)
        p_star = brier.optimal_constant_p(f_all)
        for label, d in data.items():
            d = d.copy()
            d["f"] = rt.vote_fraction(rat.loc[d["id"]], t)
            d["s"] = brier.support(d["a"], d["f"])
            gap = lambda g: brier.pad_b(g["p"], g["f"]) - brier.constant_pad_b(g["f"], p_star)
            lo, hi = _boot(d, gap, n_boot, seed)
            q = np.where(d["a"] == 1, d.loc[d["a"] == 1, "f"].mean(), d.loc[d["a"] == 0, "f"].mean())
            row = {"threshold": t, "config": label, "n": len(d), "p_star": p_star,
                   "const_B": brier.constant_pad_b(d["f"], p_star),
                   "PAD_B": brier.pad_b(d["p"], d["f"]),
                   "gap_B": gap(d), "gap_lo": lo, "gap_hi": hi,
                   "PAD_B_answer_only": brier.pad_b(q, d["f"]),
                   "r_p_f": pearsonr(d["p"], d["f"])[0], "r_a_f": pearsonr(d["a"], d["f"])[0]}
            row.update(majority_match(d))
            rows.append(row)
    return pd.DataFrame(rows)


def load_panel_t(threshold: int) -> dict[str, pd.DataFrame]:
    """Panel frames with f = vote fraction at ``threshold`` plus f2..f5 and mean rating.

    f_t = fraction of raters rating malignancy >= t; mean over t=2..5 of f_t equals
    (mean rating - 1)/4, so the t-grid spans the whole ordinal scale.
    """
    from contestbench.data import ratings as rt
    rat = rt.load_ratings().set_index("id")["ratings"]
    out = {}
    for label, d in load_panel_b().items():
        r = rat.loc[d["id"]]
        d = d.copy()
        for t in (2, 3, 4, 5):
            d[f"f{t}"] = rt.vote_fraction(r, t)
        d["mean_rating"] = r.map(np.mean).to_numpy()
        d["f"] = d[f"f{threshold}"]
        d["s"] = brier.support(d["a"], d["f"])
        out[label] = d
    return out
