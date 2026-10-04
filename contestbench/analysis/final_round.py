"""Final methodology round (plan: results/analysis_plan_final.md, frozen before running).

A. residualized partial correlations of p / c / a with f given the three vignette features,
B. irreducible rater-noise floor for PAD-B and excess over it,
C. disjoint-reader oracle check.
All exploratory / post-hoc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contestbench import config
from contestbench.analysis import oracle_b, panel_b
from contestbench.data import ratings as rt
from contestbench.metrics import brier, stats

SEEN = ["subtlety", "spiculation", "margin"]
THRESHOLDS = (3, 4)
HEADLINE = ["sonnet:standard", "sonnet:thinking", "opus:standard", "opus:thinking"]


# ---------------- Task A: residualized test ----------------
def _r2(y, yhat):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / tot) if tot > 0 else float("nan")


def residualized_table(n_boot: int = 1000, seed: int = 20261008) -> pd.DataFrame:
    feats = pd.read_parquet(config.CORPUS_PARQUET)[["id"] + SEEN]
    rows = []
    for t in THRESHOLDS:
        for label, d in panel_b.load_panel_t(t).items():
            d = d.merge(feats, on="id").reset_index(drop=True)
            X, g = d[SEEN].values, d["scan_idx"].values
            for est in ("linear", "gbt"):
                fhat = oracle_b.cv_predict(X, d["f"].values, g, est)
                rf = d["f"].values - fhat
                for var, col in (("p", "p"), ("c", "confidence"), ("a", "a")):
                    y = d[col].values.astype(float)
                    yhat = oracle_b.cv_predict(X, y, g, est)
                    rv = y - yhat
                    work = pd.DataFrame({"rv": rv, "rf": rf, "scan_idx": g, "chunk": d["chunk"].values})
                    partial = float(np.corrcoef(rv, rf)[0, 1]) if rv.std() > 0 and rf.std() > 0 else np.nan
                    lo, hi = panel_b._boot(work, lambda x: np.corrcoef(x["rv"], x["rf"])[0, 1], n_boot, seed)
                    rows.append({"threshold": t, "config": label, "headline": label in HEADLINE, "estimator": est,
                                 "variable": var, "n": len(d),
                                 "R2_var_on_features": _r2(y, yhat), "R2_f_on_features": _r2(d["f"].values, fhat),
                                 "raw_r_var_f": float(np.corrcoef(y, d["f"].values)[0, 1]),
                                 "partial_r": partial, "partial_lo": lo, "partial_hi": hi})
    return pd.DataFrame(rows)


# ---------------- Task B: irreducible-noise floor ----------------
def case_floor(f, n_raters):
    f, n = np.asarray(f, float), np.asarray(n_raters, float)
    return f * (1 - f) / (n - 1)


def floor_tables(n_boot: int = 1000, seed: int = 20261009):
    corpus = pd.read_parquet(config.CORPUS_PARQUET)[["id", "scan_idx", "n_raters", "tier"]]
    rat = rt.load_ratings().set_index("id")["ratings"]
    rows, per_cfg = [], {}
    for t in THRESHOLDS:
        c = corpus.copy()
        c["f"] = rt.vote_fraction(rat.loc[c["id"]], t)
        c["floor"] = case_floor(c["f"], c["n_raters"])
        for scope, sub in [("all", c)] + [(f"tier:{k}", c[c["tier"] == k]) for k in ("high", "contested", "ambiguous")]:
            lo, hi = panel_b._boot(sub, lambda x: x["floor"].mean(), n_boot, seed)
            rows.append({"threshold": t, "scope": scope, "n": len(sub), "floor": float(sub["floor"].mean()),
                         "floor_lo": lo, "floor_hi": hi})
        for label, d in panel_b.load_panel_t(t).items():
            m = d[["id"]].merge(c[["id", "floor"]], on="id")
            per_cfg[(t, label)] = float(m["floor"].mean())
    floors = pd.DataFrame(rows)

    # simulation: unbiasedness and invariance of forecaster differences
    rng = np.random.default_rng(seed)
    N = 2_000_000
    rho = rng.beta(0.6, 0.33, N)  # U-shaped, mean about 0.645
    n = np.where(rng.random(N) < 506 / 1413, 3, 4)
    votes = rng.binomial(n, rho)
    fh = votes / n
    est = fh * (1 - fh) / (n - 1)
    truth = rho * (1 - rho) / n
    pA, pB = np.full(N, 0.647), np.clip(0.55 * rho + 0.29 + 0.1 * rng.standard_normal(N), 0, 1)
    d_obs = ((pA - fh) ** 2 - (pB - fh) ** 2).mean()
    d_true = ((pA - rho) ** 2 - (pB - rho) ** 2).mean()
    sim = pd.DataFrame([{"draws": N, "mean_rho": float(rho.mean()), "mean_estimator": float(est.mean()),
                         "mean_true_noise_var": float(truth.mean()),
                         "estimator_bias": float(est.mean() - truth.mean()),
                         "naive_plugin_bias": float((fh * (1 - fh) / n).mean() - truth.mean()),
                         "forecaster_diff_observed": float(d_obs), "forecaster_diff_noise_free": float(d_true),
                         "forecaster_diff_gap": float(d_obs - d_true)}])

    # excess over floor from the frozen tables
    T = config.RESULTS_DIR / "tables"
    orc, rec = pd.read_csv(T / "oracle_b.csv"), pd.read_csv(T / "recalibration_b.csv")
    base, panel3 = pd.read_csv(T / "base_rate_check_b.csv"), pd.read_csv(T / "panel_b.csv")
    ex = []
    for t in THRESHOLDS:
        allf = float(floors[(floors.threshold == t) & (floors.scope == "all")]["floor"].iloc[0])
        const = float(orc[(orc.threshold == t)]["const_B"].iloc[0])
        ex.append({"threshold": t, "forecaster": "constant (case-blind)", "PAD_B": const, "floor": allf,
                   "excess": const - allf})
        for name in ("vignette", "extended"):
            for est_ in ("gbt", "ridge"):
                r = orc[(orc.threshold == t) & (orc.oracle == name) & (orc.estimator == est_)].iloc[0]
                ex.append({"threshold": t, "forecaster": f"oracle {name} ({est_})", "PAD_B": float(r["PAD_B"]),
                           "floor": allf, "excess": float(r["PAD_B"]) - allf})
        for label in HEADLINE:
            fl = per_cfg[(t, label)]
            b = base[(base.threshold == t) & (base.config == label)].iloc[0]
            rc = rec[(rec.threshold == t) & (rec.config == label) & (rec.feature_set == "vignette")
                     & (rec.estimator == "gbt")].iloc[0]
            for name, val in (("raw p", b["raw"]), ("shift-corrected p", b["shift"]),
                              ("recalibrated (features + p, gbt)", rc["PAD_B_features+p"])):
                ex.append({"threshold": t, "forecaster": f"{label}: {name}", "PAD_B": float(val),
                           "floor": fl, "excess": float(val) - fl})
    return floors, sim, pd.DataFrame(ex)


# ---------------- Task C: disjoint-reader oracle ----------------
PAIR_SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def disjoint_table(n_boot: int = 1000, seed: int = 20261010):
    rf = pd.read_parquet(config.DATA_DIR / "corpus_reader_features.parquet")
    corpus = pd.read_parquet(config.CORPUS_PARQUET)[["id", "scan_idx", "n_raters"]]
    four = corpus[corpus["n_raters"] == 4]["id"]
    rf = rf[rf["id"].isin(four)].sort_values(["id", "member"])
    ids = rf["id"].unique()
    mal = rf.pivot(index="id", columns="member", values="malignancy").loc[ids].values.astype(float)
    feat = np.stack([rf.pivot(index="id", columns="member", values=c).loc[ids].values.astype(float)
                     for c in SEEN], axis=2)  # (n, 4, 3)
    scan = corpus.set_index("id").loc[ids, "scan_idx"].values
    rows = []
    for t in THRESHOLDS:
        for (pa, pb) in PAIR_SPLITS:
            for k, (p1, p2) in enumerate([(pa, pb), (pb, pa)]):
                X = np.nanmean(feat[:, list(p1), :], axis=1)
                ok = ~np.isnan(X).any(axis=1)
                y_dis = (mal[:, list(p2)] >= t).mean(axis=1)
                y_same = (mal[:, list(p1)] >= t).mean(axis=1)
                for kind, y in (("disjoint", y_dis), ("same_pair_control", y_same)):
                    for est in ("gbt", "ridge"):
                        pred = oracle_b.cv_predict(X[ok], y[ok], scan[ok], est)
                        sq_o = (pred - y[ok]) ** 2
                        sq_c = (y[ok].mean() - y[ok]) ** 2
                        diff = sq_o - sq_c
                        lo, hi = oracle_b._cluster_ci(lambda i: diff[i].mean(), scan[ok], n_boot, seed)
                        rows.append({"threshold": t, "split": f"{p1}|{p2}", "orientation": k, "kind": kind,
                                     "estimator": est, "n": int(ok.sum()), "PAD_B_oracle": float(sq_o.mean()),
                                     "const_B": float(sq_c.mean()), "margin": float(diff.mean()),
                                     "margin_lo": lo, "margin_hi": hi})
    det = pd.DataFrame(rows)
    s = det.groupby(["threshold", "kind", "estimator"], as_index=False).agg(
        splits=("margin", "size"), n_mean=("n", "mean"), margin=("margin", "mean"),
        margin_min=("margin", "min"), margin_max=("margin", "max"))
    piv = s.pivot_table(index=["threshold", "estimator"], columns="kind", values="margin").reset_index()
    piv["retained_fraction_R"] = piv["disjoint"] / piv["same_pair_control"]
    piv["verdict_rule"] = np.where(piv["retained_fraction_R"] >= 0.5, "R>=0.5: not driven by shared raters",
                                   "R<0.5: inflated by shared raters")
    return det, s, piv


# ---------------- Task D analysis: extra open-weight models (predictions in the frozen plan) ----------------
EXTRA = [("gptoss120b", "openai/gpt-oss-120b"), ("qwen38_27b", "qwen/qwen3.8-27b"), ("gptoss20b", "openai/gpt-oss-20b")]


def _json_probs(path):
    import json
    import re
    from pathlib import Path
    out = []
    for m in re.finditer(r"\{[^{}]*\}", Path(path).read_text(encoding="utf-8")):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        v = o.get("probability")
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if 0 <= v <= 100:
            out.append((str(o.get("id", "")).strip(), v / 100.0))
    return out


def extra_models_analysis(n_boot: int = 2000, seed: int = 20261011):
    """Strict (id must match the corpus, pre-registered parse) and positional (exploratory rescue: a call whose
    object count equals its prompt's case count is aligned by position) analyses per model."""
    from pathlib import Path
    conf = config.RESULTS_DIR / "confirmatory"
    corpus = pd.read_parquet(config.CORPUS_PARQUET).reset_index(drop=True)
    corpus["order"] = np.arange(len(corpus))
    corpus["call"] = corpus["order"] // 100
    rat = rt.load_ratings().set_index("id")["ratings"]
    corpus["f"] = rt.vote_fraction(rat.loc[corpus["id"]], 3)
    p_star = brier.optimal_constant_p(corpus["f"].values)
    ids_all = set(corpus["id"])
    rows, calls = [], []
    for label, model in EXTRA:
        for mode in ("strict", "positional"):
            recs = []
            n_calls_ok = 0
            for k in range(1, 16):
                f = conf / "raw_extra" / f"{label}_{k:02d}.txt"
                prompt_ids = [l.split(" | ")[0] for l in
                              (conf / "prompts" / f"batch_{k:03d}.txt").read_text(encoding="utf-8")
                              .split("Cases:\n")[1].strip().split("\n")]
                objs = _json_probs(f)
                if mode == "strict":
                    seen = set()
                    for i, v in objs:
                        if i in ids_all and i not in seen:
                            seen.add(i)
                            recs.append((i, v))
                elif len(objs) == len(prompt_ids):
                    n_calls_ok += 1
                    recs.extend(zip(prompt_ids, [v for _, v in objs]))
            pr = pd.DataFrame(recs, columns=["id", "p_hat"]).drop_duplicates("id")
            d = corpus.merge(pr, on="id").merge(pd.read_parquet(config.CORPUS_PARQUET)[["id"] + SEEN], on="id",
                                                suffixes=("", "_x")).reset_index(drop=True)
            n = len(d)
            row = {"model": model, "mode": mode, "n": n, "coverage": n / len(corpus)}
            if n >= 150:
                sq, sqc = (d["p_hat"] - d["f"]).values ** 2, (p_star - d["f"]).values ** 2
                gen = lambda s: stats.nested_cluster_indices(d["call"].values, d["scan_idx"].values, n_boot, s)
                mean_p = d["p_hat"].values
                ci_m = np.percentile([mean_p[i].mean() - p_star for i in gen(seed)], [2.5, 97.5])
                ci_g = np.percentile([(sq - sqc)[i].mean() for i in gen(seed + 1)], [2.5, 97.5])
                row.update({"mean_p_hat": float(mean_p.mean()), "mean_f_covered": float(d["f"].mean()),
                            "mean_p_hat_minus_0.647": float(mean_p.mean() - p_star),
                            "mp_lo": float(ci_m[0]), "mp_hi": float(ci_m[1]),
                            "PAD_B": float(sq.mean()), "const_B": float(sqc.mean()), "gap": float((sq - sqc).mean()),
                            "gap_lo": float(ci_g[0]), "gap_hi": float(ci_g[1]),
                            "r_p_f": float(np.corrcoef(d["p_hat"], d["f"])[0, 1]) if d["p_hat"].nunique() > 1 else np.nan})
                if n >= 300:
                    X0, g_ = d[SEEN].values, d["scan_idx"].values
                    base = (oracle_b.cv_predict(X0, d["f"].values, g_, "gbt") - d["f"].values) ** 2
                    withp = (oracle_b.cv_predict(np.column_stack([X0, d["p_hat"].values]), d["f"].values, g_, "gbt")
                             - d["f"].values) ** 2
                    ab = base - withp
                    ci_a = np.percentile([ab[i].mean() for i in gen(seed + 2)], [2.5, 97.5])
                    row.update({"features_only": float(base.mean()), "features+p_hat": float(withp.mean()),
                                "ablation_gap": float(ab.mean()), "ab_lo": float(ci_a[0]), "ab_hi": float(ci_a[1]),
                                "P2_brackets_zero": bool(ci_a[0] <= 0 <= ci_a[1])})
                pc = d.groupby("call").apply(lambda x: pd.Series({
                    "n": len(x), "corr": np.corrcoef(x["p_hat"], x["f"])[0, 1] if x["p_hat"].nunique() > 1 else np.nan,
                    "mean_p": x["p_hat"].mean()}))
                pc = pc[pc["n"] >= 30]
                row.update({"calls_used": len(pc), "range_call_mean_p_hat": float(pc["mean_p"].max() - pc["mean_p"].min()),
                            "min_call_corr": float(pc["corr"].min()), "calls_neg_corr": int((pc["corr"] < 0).sum())})
            rows.append(row)
    return pd.DataFrame(rows)
