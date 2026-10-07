"""Analysis of the informed-prompt conditions vs the original prompt (same configs, same cases).

python -m contestbench.analysis.informed  -> results/tables/informed_b.csv
Gap = PAD-B(model) - PAD-B(case-blind constant); positive = worse than the constant.
delta = gap(informed) - gap(original), paired over the same cases with the nested call->scan bootstrap.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from contestbench import config
from contestbench.analysis import panel
from contestbench.data import ratings as rt
from contestbench.eval.batch import parse_batch_text
from contestbench.metrics import brier, stats

RAW = Path("results/informed/raw")
PRED_RULES = {
    "P1_scales_still_worse": "cond=scales, t=3: gap CI lower bound > 0 for every standard config",
    "P2_rate_halves_gap": "cond=scales_rate, t=3: gap <= 0.5 * original gap for every standard config",
    "P3_prompt_dependent": "cond=scales_rate, t=3: any gap CI upper bound < 0 (headline becomes prompt-dependent)",
}


def parse_raw(cfg: str, cond: str, corpus: pd.DataFrame) -> pd.DataFrame:
    fam, mode = cfg.split(":")
    rows = []
    for f in sorted(glob.glob(str(RAW / f"{fam}-{mode}__{cond}_*.txt"))):
        rows.append(parse_batch_text(Path(f).read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame()
    r = pd.concat(rows).drop_duplicates("id")
    d = corpus[["id", "pi", "tier"]].merge(r, on="id", how="inner")
    d["label"] = f"{cfg}|{cond}"
    return d.dropna(subset=["confidence"])


def _prep(d: pd.DataFrame, corpus: pd.DataFrame, rat: pd.Series, t: int) -> pd.DataFrame:
    c = corpus[["id", "scan_idx"]].copy()
    c["order"] = np.arange(len(c))
    d = d.drop(columns=[x for x in ("scan_idx", "order") if x in d.columns]).merge(c, on="id", how="left").reset_index(drop=True)
    d["chunk"] = d["order"] // 100
    d["f"] = rt.vote_fraction(rat.loc[d["id"]], t)
    d["a"] = brier.answer_to_binary(d["answer"])
    d["p"] = brier.fold_forecast(d["a"], d["confidence"])
    return d


def _ci(vals):
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def informed_table(n_boot: int = 1000, seed: int = 20261006) -> pd.DataFrame:
    corpus = pd.read_parquet(config.CORPUS_PARQUET).reset_index(drop=True)
    rat = rt.load_ratings().set_index("id")["ratings"]
    orig = panel.load_panel()
    rows = []
    for t in (3, 4):
        p_star = brier.optimal_constant_p(rt.vote_fraction(rat, t))
        for key in sorted({Path(f).name.split("__")[0] for f in glob.glob(str(RAW / "*.txt"))}):
            fam, mode = key.split("-")
            cfg = f"{fam}:{mode}"
            if cfg not in orig:
                continue
            o = _prep(orig[cfg][["id", "pi", "tier", "answer", "confidence"]], corpus, rat, t)
            for cond in ("scales", "scales_rate"):
                n_ = parse_raw(cfg, cond, corpus)
                if n_.empty:
                    continue
                n_ = _prep(n_, corpus, rat, t)
                m = n_.merge(o[["id", "p", "a"]], on="id", suffixes=("", "_o"))  # paired cases
                gap = lambda g, col="p": brier.pad_b(g[col], g["f"]) - brier.constant_pad_b(g["f"], p_star)
                idx = list(stats.nested_cluster_indices(m["chunk"].values, m["scan_idx"].values, n_boot, seed + t))
                g_new = [gap(m.iloc[i]) for i in idx]
                g_old = [gap(m.iloc[i], "p_o") for i in idx]
                dlt = np.array(g_new) - np.array(g_old)
                rows.append({"threshold": t, "config": cfg, "cond": cond, "n": len(m),
                             "gap_new": gap(m), "gap_new_lo": _ci(g_new)[0], "gap_new_hi": _ci(g_new)[1],
                             "gap_orig": gap(m, "p_o"), "gap_orig_lo": _ci(g_old)[0], "gap_orig_hi": _ci(g_old)[1],
                             "delta": gap(m) - gap(m, "p_o"), "delta_lo": _ci(dlt)[0], "delta_hi": _ci(dlt)[1],
                             "share_gap_closed": 1 - gap(m) / gap(m, "p_o") if gap(m, "p_o") > 0 else np.nan,
                             "PAD_B_new": brier.pad_b(m["p"], m["f"]), "const_B": brier.constant_pad_b(m["f"], p_star),
                             "mean_p_new": float(m["p"].mean()), "mean_p_orig": float(m["p_o"].mean()),
                             "frac_malignant_new": float(m["a"].mean()), "frac_malignant_orig": float(m["a_o"].mean()),
                             "r_p_f_new": float(np.corrcoef(m["p"], m["f"])[0, 1])})
    return pd.DataFrame(rows)


def score_predictions(t: pd.DataFrame) -> pd.DataFrame:
    x = t[(t.threshold == 3) & t.config.str.endswith(":standard")]
    out = []
    a = x[x.cond == "scales"]
    out.append(("P1_scales_still_worse", len(a) > 0 and bool((a.gap_new_lo > 0).all())))
    b = x[x.cond == "scales_rate"]
    out.append(("P2_rate_halves_gap", len(b) > 0 and bool((b.gap_new <= 0.5 * b.gap_orig).all())))
    out.append(("P3_prompt_dependent", len(b) > 0 and bool((b.gap_new_hi < 0).any())))
    return pd.DataFrame(out, columns=["prediction", "held"]).assign(rule=lambda d: d.prediction.map(PRED_RULES))


if __name__ == "__main__":
    tdir = Path("results/tables")
    tbl = informed_table()
    tbl.to_csv(tdir / "informed_b.csv", index=False)
    sc = score_predictions(tbl)
    sc.to_csv(tdir / "informed_predictions_b.csv", index=False)
    pd.set_option("display.width", 220)
    print(tbl[["threshold", "config", "cond", "n", "gap_orig", "gap_new", "gap_new_lo", "gap_new_hi", "delta",
               "delta_lo", "delta_hi", "frac_malignant_orig", "frac_malignant_new"]].round(3).to_string(index=False))
    print(sc.to_string(index=False))
