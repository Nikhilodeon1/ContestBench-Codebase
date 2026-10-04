"""Phases 5-9 of the TMLR revision, all under PAD-B (p folded from answer+confidence).

Everything runs at threshold t (3 = pre-specified primary, 4 = post-hoc sensitivity)
and uses scan-cluster bootstraps. MI permutation nulls shuffle f across nodules
(limitation: nodules within a scan are only approximately exchangeable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from contestbench import config
from contestbench.analysis import oracle_b, over_reliance as orl, panel_b
from contestbench.data import ratings as rt
from contestbench.data import vignette_audit
from contestbench.metrics import brier, mutual_info as MI, stats

THRESHOLDS = (3, 4)
BOOT = 1000


# ---------------- Phase 5: mutual information ----------------
def mi_table(threshold: int, n_perm: int = 500) -> pd.DataFrame:
    data = panel_b.load_panel_t(threshold)
    bonf = 0.05 / len(data)
    rows = []
    for label, d in data.items():
        f = d["f"].values
        r = MI.mi_with_null(d["p"].values, f, n_perm=n_perm)
        ks = {f"mi_k{k}": MI.mutual_information(d["p"].values, f, n_neighbors=k) for k in (3, 5, 10)}
        c = MI.mi_with_null(d["confidence"].values, f, n_perm=n_perm)
        rows.append({"threshold": threshold, "config": label, "target": "MI(p,f)", **r,
                     "bonferroni_sig": bool(r["p_value"] < bonf), **ks,
                     "mi_conf_only": c["mi"], "conf_only_excess": c["excess"],
                     "conf_only_p": c["p_value"]})
    c = oracle_b.corpus_with_f(threshold)
    pred = oracle_b.cv_predict(c[oracle_b.VIGNETTE].values, c["f"].values,
                               c["scan_idx"].values, "gbt")
    r = MI.mi_with_null(pred, c["f"].values, n_perm=n_perm)
    rows.append({"threshold": threshold, "config": "oracle:vignette", "target": "MI(p_hat,f)", **r,
                 "bonferroni_sig": bool(r["p_value"] < bonf)})
    return pd.DataFrame(rows)


# ---------------- Phase 6: mechanism ----------------
SEEN = ["subtlety", "spiculation", "margin"]


def _r2(X, y):
    X = np.column_stack([np.asarray(X, float), np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - X @ beta) ** 2).sum() / tot) if tot > 0 else float("nan")


def mechanism_table(threshold: int = 3):
    data = panel_b.load_panel_t(threshold)
    corpus = pd.read_parquet(config.CORPUS_PARQUET)[["id", "extent"] + SEEN]
    rows = []
    for label, d in data.items():
        d = d.merge(corpus, on="id")
        row = {"config": label}
        for f in SEEN + ["extent"]:
            row[f"r_p_{f}"] = pearsonr(d[f], d["p"])[0]
        row["R2_p_on_seen3"] = _r2(d[SEEN].values, d["p"].values)
        row["R2_c_on_seen3"] = _r2(d[SEEN].values, d["confidence"].values)
        row["R2_f_on_seen3"] = _r2(d[SEEN].values, d["f"].values)
        row["R2_p_lo"], row["R2_p_hi"] = panel_b._boot(
            d, lambda g: _r2(g[SEEN].values, g["p"].values), BOOT, 7)
        rows.append(row)
    mech = pd.DataFrame(rows)
    ps = {k: v.set_index("id")["p"] for k, v in data.items()}
    cs = {k: v.set_index("id")["confidence"] for k, v in data.items()}
    common = sorted(set.intersection(*[set(s.index) for s in ps.values()]))
    labs = list(ps)
    mp = pd.DataFrame({a: {b: pearsonr(ps[a].reindex(common), ps[b].reindex(common))[0] for b in labs}
                       for a in labs})
    mc = pd.DataFrame({a: {b: pearsonr(cs[a].reindex(common), cs[b].reindex(common))[0] for b in labs}
                       for a in labs})
    return mech, mp, mc, len(common)


# ---------------- Phase 7: interventions + reasoning ----------------
INTERV_DIR = config.RESULTS_DIR / "interv4"


def intervention_table(threshold: int, n_boot: int = BOOT) -> pd.DataFrame:
    from contestbench.eval import batch
    base_all = panel_b.load_panel_t(threshold)
    rat = rt.load_ratings().set_index("id")["ratings"]
    corpus = pd.read_parquet(config.CORPUS_PARQUET)[["id", "scan_idx"]]
    p_star = brier.optimal_constant_p(rt.vote_fraction(rat, threshold))
    ex = set()
    for k in "ABC":
        ex |= set(pd.read_csv(INTERV_DIR / f"exemplars_{k}.csv")["id"].astype(str))
    evalset = pd.read_parquet(INTERV_DIR / "evalset.parquet")["id"].astype(str)
    leak = bool(set(evalset) & ex)
    rows = []
    for m in ("haiku", "sonnet", "opus"):
        base = base_all[f"{m}:standard"].drop(columns=["scan_idx"])
        for design in "ABC":
            parts = [batch.parse_batch_file(INTERV_DIR / f"cl_{m}_{design}_{i}.txt") for i in (1, 2, 3)]
            inv = pd.concat(parts).drop_duplicates("id")
            inv["a"] = brier.answer_to_binary(inv["answer"])
            inv["p_after"] = brier.fold_forecast(inv["a"], inv["confidence"])
            d = base[["id", "f", "p"]].merge(inv[["id", "p_after"]], on="id").merge(corpus, on="id")
            sq_b, sq_a = (d["p"] - d["f"]) ** 2, (d["p_after"] - d["f"]) ** 2
            sq_c = (p_star - d["f"]) ** 2
            g = d["scan_idx"].values

            m_tests = 9
            q = [2.5, 97.5, 100 * 0.025 / m_tests, 100 - 100 * 0.025 / m_tests]

            def ci(arr):
                v = [arr[i].mean() for i in stats.scan_cluster_indices(g, n_boot, 11)]
                return [float(x) for x in np.percentile(v, q)]

            dl, dh, dlb, dhb = ci((sq_a - sq_b).values)
            gl, gh, glb, ghb = ci((sq_a - sq_c).values)
            rows.append({"threshold": threshold, "model": m, "design": design, "n": len(d),
                         "PAD_B_before": sq_b.mean(), "PAD_B_after": sq_a.mean(),
                         "delta": (sq_a - sq_b).mean(), "delta_lo": dl, "delta_hi": dh,
                         "const_B": sq_c.mean(), "gap_vs_const": (sq_a - sq_c).mean(),
                         "gap_lo": gl, "gap_hi": gh, "delta_lo_bonf": dlb, "delta_hi_bonf": dhb,
                         "gap_lo_bonf": glb, "gap_hi_bonf": ghb, "leak": leak})
    return pd.DataFrame(rows)


def reasoning_table(threshold: int, n_boot: int = BOOT) -> pd.DataFrame:
    data = panel_b.load_panel_t(threshold)
    rows = []
    for fam in ("haiku", "sonnet", "opus", "gemini", "deepseek"):
        s, t = data[f"{fam}:standard"], data[f"{fam}:thinking"]
        d = s[["id", "f", "p", "scan_idx"]].merge(t[["id", "p"]], on="id", suffixes=("_std", "_think"))
        e = ((d["p_think"] - d["f"]) ** 2 - (d["p_std"] - d["f"]) ** 2).values
        v = [e[i].mean() for i in stats.scan_cluster_indices(d["scan_idx"].values, n_boot, 13)]
        rows.append({"threshold": threshold, "family": fam, "n": len(d),
                     "delta_PAD_B": e.mean(), "lo": float(np.percentile(v, 2.5)),
                     "hi": float(np.percentile(v, 97.5)),
                     "lo_bonf": float(np.percentile(v, 100 * 0.025 / 4)),
                     "hi_bonf": float(np.percentile(v, 100 - 100 * 0.025 / 4)), "valid": fam != "gemini",
                     "note": "std=gemini-flash-latest vs thinking=gemini-2.5-flash: different models, invalid contrast"
                     if fam == "gemini" else ""})
    return pd.DataFrame(rows)


# ---------------- Phase 8: over-reliance, warranted reliance = s ----------------
THETAS, BETAS = [0.5, 0.6, 0.7, 0.8], [5, 10, 20]


def reliance_tables(threshold: int, n_boot: int = 300):
    data = panel_b.load_panel_t(threshold)
    corpus = pd.read_parquet(config.CORPUS_PARQUET)[["id"] + oracle_b.VIGNETTE]
    pools, per_model = [], []
    for label, d in data.items():
        d = d.merge(corpus, on="id")
        phat = oracle_b.cv_predict(d[oracle_b.VIGNETTE + ["p"]].values, d["f"].values,
                                   d["scan_idx"].values, "gbt")
        c_recal = np.where(d["a"].values == 1, phat, 1.0 - phat)
        x = pd.DataFrame({"config": label, "scan": d["scan_idx"].values, "s": d["s"].values,
                          "c_dec": d["confidence"].values, "c_rec": c_recal, "c_ideal": d["s"].values})
        pools.append(x)
        sim = orl.simulate({"decoupled": x.c_dec.values, "recalibrated": x.c_rec.values,
                            "ideal": x.c_ideal.values}, x.s.values, THETAS, BETAS)
        per_model.append({"threshold": threshold, "config": label, **sim})
    P = pd.concat(pools, ignore_index=True)
    regimes = {"decoupled": "c_dec", "recalibrated": "c_rec", "ideal": "c_ideal"}
    grid = []
    for t in THETAS:
        for b in BETAS:
            row = {"threshold": threshold, "theta": t, "beta": b}
            for n, col in regimes.items():
                row[n] = orl.over_reliance(P[col].values, P["s"].values, t, b)
            row["order_ok"] = bool(row["decoupled"] >= row["recalibrated"] >= row["ideal"])
            row["dec_gt_ideal"] = bool(row["decoupled"] > row["ideal"])
            grid.append(row)
    grid = pd.DataFrame(grid)
    pooled = {n: float(grid[n].mean()) for n in regimes}
    diffs = {"decoupled-recalibrated": ("decoupled", "recalibrated"),
             "recalibrated-ideal": ("recalibrated", "ideal"),
             "decoupled-ideal": ("decoupled", "ideal")}
    boots = {k: [] for k in diffs}
    for idx in stats.scan_cluster_indices(P["scan"].values, n_boot, 17):
        Q = P.iloc[idx]
        m = {n: np.mean([orl.over_reliance(Q[col].values, Q["s"].values, t, b)
                         for t in THETAS for b in BETAS]) for n, col in regimes.items()}
        for k, (u, v) in diffs.items():
            boots[k].append(m[u] - m[v])
    summ = [{"threshold": threshold, "contrast": k, "diff": pooled[u] - pooled[v],
             "lo": float(np.percentile(boots[k], 2.5)), "hi": float(np.percentile(boots[k], 97.5))}
            for k, (u, v) in diffs.items()]
    return pd.DataFrame(per_model), grid, pd.DataFrame(summ), pooled


# ---------------- Phase 9: robustness ----------------
def _gap(g, p_star):
    return brier.pad_b(g["p"], g["f"]) - brier.constant_pad_b(g["f"], p_star)


def robustness_tables(n_boot: int = BOOT):
    corpus = pd.read_parquet(config.CORPUS_PARQUET)
    rat = rt.load_ratings().set_index("id")["ratings"]
    f3 = rt.vote_fraction(rat.loc[corpus["id"]], 3)
    f4 = rt.vote_fraction(rat.loc[corpus["id"]], 4)
    stable_ids = set(corpus["id"][np.isclose(f3, f4)])
    cf = {}
    for t in (3, 4):
        c = oracle_b.corpus_with_f(t)
        mask = vignette_audit.non_colliding_mask(c.assign(pi_agree=c["f"]))  # one f per bucket
        cf[t] = set(c["id"][mask.values])
    c3 = oracle_b.corpus_with_f(3)
    cf_orig = set(c3["id"][vignette_audit.non_colliding_mask(c3).values])  # original: one pi per bucket
    sizes = {"collision_free_orig_pi": len(cf_orig), "collision_free_f3": len(cf[3]),
             "collision_free_f4": len(cf[4]), "threshold_stable": len(stable_ids),
             "threshold_sensitive": len(corpus) - len(stable_ids)}
    rows = []
    for t in (3, 4):
        data = panel_b.load_panel_t(t)
        subsets = {"all": None, "collision_free(orig pi)": cf_orig, f"collision_free(f{t})": cf[t],
                   "threshold_stable": stable_ids,
                   "threshold_sensitive": set(corpus["id"]) - stable_ids}
        for sname, ids in subsets.items():
            for label, d in data.items():
                dd = d if ids is None else d[d["id"].isin(ids)]
                p_star = brier.optimal_constant_p(dd["f"].values)
                lo, hi = panel_b._boot(dd, lambda g: _gap(g, p_star), n_boot // 2, 19)
                rows.append({"threshold": t, "subset": sname, "config": label, "n": len(dd),
                             "PAD_B": brier.pad_b(dd["p"], dd["f"]),
                             "const_B": brier.constant_pad_b(dd["f"], p_star),
                             "gap": _gap(dd, p_star), "gap_lo": lo, "gap_hi": hi,
                             "r_p_f": pearsonr(dd["p"], dd["f"])[0] if dd["f"].nunique() > 1 else np.nan})
    sub = pd.DataFrame(rows)

    data = panel_b.load_panel_t(3)
    rows = []
    for label, d in data.items():
        row = {"config": label, "n": len(d)}
        gaps = []
        for t in (2, 3, 4, 5):
            ft = d[f"f{t}"].values
            ps = brier.optimal_constant_p(ft)
            g_t = brier.pad_b(d["p"], ft) - brier.constant_pad_b(ft, ps)
            row[f"gap_t{t}"] = g_t
            gaps.append(g_t)
        row["gap_avg_t2to5"] = float(np.mean(gaps))
        # joint (single case-blind forecast for all thresholds) threshold-averaged Brier gap
        sq = np.mean([(d["p"].values - d[f"f{t}"].values) ** 2 for t in (2, 3, 4, 5)], axis=0)
        ps_avg = float(np.mean([d[f"f{t}"].mean() for t in (2, 3, 4, 5)]))
        sq_c = np.mean([(ps_avg - d[f"f{t}"].values) ** 2 for t in (2, 3, 4, 5)], axis=0)
        dd = d.assign(_e=sq - sq_c)
        row["gap_joint_t2to5"] = float(dd["_e"].mean())
        row["joint_lo"], row["joint_hi"] = panel_b._boot(dd, lambda g: g["_e"].mean(), n_boot // 2, 23)
        row["spearman_p_meanrating"] = spearmanr(d["p"], d["mean_rating"])[0]
        row["sp_lo"], row["sp_hi"] = panel_b._boot(
            d, lambda g: spearmanr(g["p"], g["mean_rating"])[0], n_boot // 2, 29)
        rows.append(row)
    return sub, pd.DataFrame(rows), sizes


# ---------------- Claim 8: accuracy-calibrated baseline under PAD-B ----------------
def acc_cal_table(threshold: int, n_boot: int = BOOT, k: int = 5, seed: int = 20260726) -> pd.DataFrame:
    """Platt-scale confidence to hit-rate vs the majority label (what ECE-minimising does),
    fold the result into p', and score PAD-B against f. Out-of-fold, folds grouped by scan.

    Variants: 'conf' = P(correct | c) (original design); 'conf+ans' = P(correct | c, a).
    Fit uses non-tied cases only (f != .5: majority label defined); applied to all cases.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    rat = rt.load_ratings().set_index("id")["ratings"]
    p_star = brier.optimal_constant_p(rt.vote_fraction(rat, threshold))
    rows = []
    for label, d in panel_b.load_panel_t(threshold).items():
        d = d.reset_index(drop=True)
        correct = ((d["a"] == 1) == (d["f"] > 0.5)).astype(int).values
        labelled = (d["f"] != 0.5).values
        for variant, cols in (("conf", ["confidence"]), ("conf+ans", ["confidence", "a"])):
            X = d[cols].to_numpy(float)
            c_acc = np.full(len(d), np.nan)
            for tr, te in GroupKFold(k, shuffle=True, random_state=seed).split(X, groups=d["scan_idx"].values):
                trl = tr[labelled[tr]]
                c_acc[te] = LogisticRegression().fit(X[trl], correct[trl]).predict_proba(X[te])[:, 1]
            p_new = brier.fold_forecast(d["a"].values, c_acc)
            sq_old, sq_new = (d["p"].values - d["f"].values) ** 2, (p_new - d["f"].values) ** 2
            sq_c = (p_star - d["f"].values) ** 2
            g = d["scan_idx"].values
            delta = sq_new - sq_old
            v = [delta[i].mean() for i in stats.scan_cluster_indices(g, n_boot, 31)]
            v2 = [(sq_new - sq_c)[i].mean() for i in stats.scan_cluster_indices(g, n_boot, 32)]
            rows.append({"threshold": threshold, "config": label, "variant": variant, "n": len(d),
                         "hit_rate": float(correct[labelled].mean()),
                         "PAD_B_raw": sq_old.mean(), "PAD_B_acc_cal": sq_new.mean(),
                         "delta": delta.mean(), "delta_lo": float(np.percentile(v, 2.5)),
                         "delta_hi": float(np.percentile(v, 97.5)),
                         "gap_vs_const": (sq_new - sq_c).mean(),
                         "gap_lo": float(np.percentile(v2, 2.5)), "gap_hi": float(np.percentile(v2, 97.5))})
    return pd.DataFrame(rows)


# ---------------- Checks requested 2026-10-02 ----------------
def excl3_table(n_boot: int = BOOT) -> pd.DataFrame:
    """Does the >=3 failure come from the indeterminate (rating 3) votes?

    'nodule_no3': keep only nodules with no rating of exactly 3 (then f3 == f4 identically).
    'rater_drop3': keep every nodule with >=1 non-3 rating; f' = (#ratings>=4)/(#ratings != 3),
                   i.e. clearly-benign (1-2) vs clearly-malignant (4-5) votes only.
    Constant baseline re-derived per variant (p* = mean f).
    """
    rat = rt.load_ratings().set_index("id")["ratings"]
    base = panel_b.load_panel_t(3)
    f_r = rat.map(lambda r: (np.sum(np.asarray(r) >= 4) / max(1, np.sum(np.asarray(r) != 3)))
                  if np.any(np.asarray(r) != 3) else np.nan)
    no3 = rat.map(lambda r: not np.any(np.asarray(r) == 3))
    rows = []
    for variant in ("all(>=3)", "nodule_no3", "rater_drop3"):
        for label, d in base.items():
            d = d.copy()
            if variant == "nodule_no3":
                d = d[d["id"].map(no3).values]
            elif variant == "rater_drop3":
                d["f"] = d["id"].map(f_r).values
                d = d.dropna(subset=["f"])
            ps = brier.optimal_constant_p(d["f"].values)
            lo, hi = panel_b._boot(d, lambda g: brier.pad_b(g["p"], g["f"]) - brier.constant_pad_b(g["f"], ps),
                                   n_boot, 41)
            nz = d[d["f"] != 0.5]
            rows.append({"variant": variant, "config": label, "n": len(d), "p_star": ps,
                         "const_B": brier.constant_pad_b(d["f"], ps), "PAD_B": brier.pad_b(d["p"], d["f"]),
                         "gap": brier.pad_b(d["p"], d["f"]) - brier.constant_pad_b(d["f"], ps),
                         "gap_lo": lo, "gap_hi": hi,
                         "r_p_f": pearsonr(d["p"], d["f"])[0],
                         "match_majority": float((nz["a"] == (nz["f"] > 0.5)).mean())})
    return pd.DataFrame(rows)


def base_rate_check(threshold: int, n_boot: int = BOOT, k: int = 5, seed: int = 20260726) -> pd.DataFrame:
    """Is accuracy-calibration just base-rate correction?

    All maps are fit out-of-fold (GroupKFold by scan) and scored by PAD-B against f:
      raw         : p
      shift       : p + delta, delta = mean(f_train) - mean(p_train)        (1 param, no per-case info)
      shift_scale : a + b*p (OLS)                                          (2 params)
      answer_only : mean(f_train | answer)                                 (2 params, uses the answer)
      platt_conf / platt_conf+ans : the existing accuracy-calibration baselines.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    rows = []
    for label, d in panel_b.load_panel_t(threshold).items():
        d = d.reset_index(drop=True)
        p, f, a, c = d["p"].values, d["f"].values, d["a"].values, d["confidence"].values
        correct = ((a == 1) == (f > 0.5)).astype(int)
        lab = f != 0.5
        out = {k_: np.full(len(d), np.nan) for k_ in ("shift", "shift_scale", "answer_only", "platt_conf", "platt_conf+ans")}
        for tr, te in GroupKFold(k, shuffle=True, random_state=seed).split(p, groups=d["scan_idx"].values):
            out["shift"][te] = np.clip(p[te] + (f[tr].mean() - p[tr].mean()), 0, 1)
            b, a0 = np.polyfit(p[tr], f[tr], 1)
            out["shift_scale"][te] = np.clip(a0 + b * p[te], 0, 1)
            m1, m0 = f[tr][a[tr] == 1].mean(), f[tr][a[tr] == 0].mean()
            out["answer_only"][te] = np.where(a[te] == 1, m1, m0)
            trl = tr[lab[tr]]
            for name, X in (("platt_conf", c.reshape(-1, 1)), ("platt_conf+ans", np.column_stack([c, a]))):
                pc = LogisticRegression().fit(X[trl], correct[trl]).predict_proba(X[te])[:, 1]
                out[name][te] = brier.fold_forecast(a[te], pc)
        sq_raw = (p - f) ** 2
        row = {"threshold": threshold, "config": label, "n": len(d), "raw": sq_raw.mean()}
        for name, pv in out.items():
            row[name] = ((pv - f) ** 2).mean()
        # share of Platt's improvement already achieved by the 1-parameter shift
        for plat in ("platt_conf", "platt_conf+ans"):
            gain = row["raw"] - row[plat]
            row[f"shift_share_of_{plat}"] = (row["raw"] - row["shift"]) / gain if abs(gain) > 1e-9 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------- Free noise estimate: call/chunk heterogeneity ----------------
CHUNK_SIZE = {"deepseek:standard": 471, "deepseek:thinking": 471}  # manual route: 3 calls; others 100/call


def chunk_heterogeneity(n_perm: int = 500, seed: int = 20261002) -> pd.DataFrame:
    """Do per-call (chunk) mean forecasts vary more than case sampling + case features predict?

    Chunks are consecutive slices of corpus order (NOT random), so composition differs between
    chunks. We (1) adjust p, confidence and the Malignant-answer indicator for the three vignette
    features (quadratic OLS), then (2) compare the variance of chunk-mean residuals with a null
    from randomly re-ordering whole scans into pseudo-chunks of the same sizes (keeps within-scan
    correlation). Excess SD = sqrt(max(0, V_obs - mean V_null)). Composition that is not captured
    by the features (e.g. site/scanner by patient id) still shows up as excess, so this is an
    UPPER bound on call-level drift.
    """
    corpus = pd.read_parquet(config.CORPUS_PARQUET).reset_index(drop=True)
    corpus["order"] = np.arange(len(corpus))
    rng = np.random.default_rng(seed)
    rows = []
    for label, d in panel_b.load_panel_t(3).items():
        d = d.drop(columns=["scan_idx"]).merge(corpus[["id", "order", "scan_idx"] + SEEN], on="id").sort_values("order")
        d = d.reset_index(drop=True)
        size = CHUNK_SIZE.get(label, 100)
        d["chunk"] = d["order"] // size
        X = np.column_stack([np.ones(len(d))] + [d[c] for c in SEEN] + [d[c] ** 2 for c in SEEN])
        for target, col in (("p", "p"), ("confidence", "confidence"), ("answer_malignant", "a")):
            y = d[col].values.astype(float)
            res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            def v_obs(r, chunk):
                m = pd.Series(r).groupby(chunk).mean()
                w = pd.Series(chunk).value_counts().sort_index()
                keep = w[w >= 30].index  # ignore the tiny trailing chunk
                return float(np.var(m.loc[keep].values, ddof=1))
            vo = v_obs(res, d["chunk"].values)
            scans = d["scan_idx"].unique()
            blocks = {s: np.flatnonzero(d["scan_idx"].values == s) for s in scans}
            null = []
            for _ in range(n_perm):
                perm = rng.permutation(scans)
                idx = np.concatenate([blocks[s] for s in perm])
                chunk = np.empty(len(d), int)
                chunk[idx] = np.arange(len(d)) // size
                null.append(v_obs(res, chunk))
            null = np.array(null)
            rows.append({"config": label, "target": target, "n_chunks": int(d["chunk"].nunique()),
                         "sd_chunk_means": float(np.sqrt(vo)), "sd_null_mean": float(np.sqrt(null.mean())),
                         "excess_sd": float(np.sqrt(max(0.0, vo - null.mean()))),
                         "p_perm": float((1 + (null >= vo).sum()) / (n_perm + 1))})
    return pd.DataFrame(rows)


def answer_only_repeat_stability(threshold: int = 3) -> pd.DataFrame:
    """Repeat-run PAD-B change for raw p vs the answer-only forecast (sonnet:standard chunks 1-3)."""
    import glob
    from contestbench.eval import batch
    new = pd.concat([batch.parse_batch_file(f) for f in
                     sorted(glob.glob(str(config.RESULTS_DIR / "recollect" / "noise" / "sonnet-standard-rerun_*.txt")))])
    new = new.drop_duplicates("id")
    new["a2"] = brier.answer_to_binary(new["answer"])
    new["p2"] = brier.fold_forecast(new["a2"], new["confidence"])
    d = panel_b.load_panel_t(threshold)["sonnet:standard"].merge(new[["id", "a2", "p2"]], on="id")
    f, g = d["f"].values, d["scan_idx"].values
    m1, m0 = f[d["a"] == 1].mean(), f[d["a"] == 0].mean()  # answer-only map from run 1 (2 params)
    out = {}
    for name, e1, e2 in (("raw p", (d["p"] - f) ** 2, (d["p2"] - f) ** 2),
                         ("answer-only", (np.where(d["a"] == 1, m1, m0) - f) ** 2,
                          (np.where(d["a2"] == 1, m1, m0) - f) ** 2)):
        diff = np.asarray(e2 - e1, float)
        v = [diff[i].mean() for i in stats.scan_cluster_indices(g, 3000, 3)]
        out[name] = {"threshold": threshold, "forecast": name, "n": len(d), "run1": float(np.mean(e1)), "run2": float(np.mean(e2)),
                     "diff": diff.mean(), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    return pd.DataFrame(out.values())


# ---------------- Noise floor (rule: results/noise_floor_rule.md) ----------------
REPEATS = {  # config -> (original-run glob, repeat-run glob)
    "sonnet:standard": ("results/recollect/sonnet-standard_*.txt", "results/recollect/noise/sonnet-standard-rerun_*.txt"),
    "gemini:standard": ("results/recollect/pinned/gemini-standard_*.txt",
                        "results/recollect/pinned_repeat/gemini-standard_*.txt"),
    "haiku:standard": ("results/recollect/haiku-standard_*.txt", "results/recollect/noise_haiku/haiku-standard-rerun_*.txt"),
    "gemini:thinking": ("results/recollect/pinned/gemini-thinking_*.txt",
                        "results/recollect/pinned_repeat_thinking/gemini-thinking_*.txt"),
}
STABLE = {"sonnet:standard", "gemini:standard"}  # pooled floor uses only these (see addendum to noise_floor_rule.md)


def noise_floor_tables():
    """Per-chunk repeat differences, per-config and pooled noise floor F_t = 1.96 * s / sqrt(14)."""
    import glob
    from contestbench.eval import batch
    rat = rt.load_ratings().set_index("id")["ratings"]
    chunks, summ = [], []
    for label, (g1, g2) in REPEATS.items():
        f1s, f2s = sorted(glob.glob(str(config.ROOT / g1))), sorted(glob.glob(str(config.ROOT / g2)))
        n = min(len(f1s), len(f2s))
        if n < 10:  # incomplete repeat (e.g. quota ran out): not used for any floor
            continue
        for k in range(n):
            r1, r2 = batch.parse_batch_file(f1s[k]), batch.parse_batch_file(f2s[k])
            m = r1.merge(r2, on="id", suffixes=("1", "2"))
            if len(m) < 30:
                continue
            for run in ("1", "2"):
                m[f"p{run}"] = brier.fold_forecast(brier.answer_to_binary(m[f"answer{run}"]), m[f"confidence{run}"])
            row = {"config": label, "chunk": k + 1, "n": len(m),
                   "answer_agree": float((m["answer1"] == m["answer2"]).mean()),
                   "mean_conf_run1": float(m["confidence1"].mean()), "mean_conf_run2": float(m["confidence2"].mean())}
            for t in (3, 4):
                f = rt.vote_fraction(rat.loc[m["id"]], t)
                row[f"d_t{t}"] = float(((m["p2"] - f) ** 2).mean() - ((m["p1"] - f) ** 2).mean())
            chunks.append(row)
    ch = pd.DataFrame(chunks)
    for t in (3, 4):
        col = f"d_t{t}"
        pooled_s = float(np.sqrt((ch[ch["config"].isin(STABLE)][col] ** 2).mean()))
        K = 14
        summ.append({"threshold": t, "scope": "pooled(stable configs)", "chunks": int(ch["config"].isin(STABLE).sum()), "s_chunk": pooled_s,
                     "noise_floor": 1.96 * pooled_s / np.sqrt(K),
                     "floor_vs_constant": 1.96 * pooled_s / np.sqrt(K) / np.sqrt(2)})
        for label, g in ch.groupby("config"):
            s = float(np.sqrt((g[col] ** 2).mean()))
            summ.append({"threshold": t, "scope": label, "chunks": len(g), "s_chunk": s,
                         "noise_floor": 1.96 * s / np.sqrt(K), "floor_vs_constant": 1.96 * s / np.sqrt(K) / np.sqrt(2),
                         "mean_d": float(g[col].mean())})
    return ch, pd.DataFrame(summ)


# ---------------- Protocol-stability rule (results/stability_rule.md, frozen) ----------------
REFERENCE = ["sonnet:standard", "sonnet:thinking", "opus:standard", "opus:thinking"]
STABILITY_MULT = 1.5


def stability_table() -> pd.DataFrame:
    corp = pd.read_parquet(config.CORPUS_PARQUET)[["id"]].reset_index().rename(columns={"index": "o"})
    rows = []
    for label, d in panel_b.load_panel_t(3).items():
        d = d.merge(corp, on="id")
        d["call"] = d["o"] // CHUNK_SIZE.get(label, 100)
        d = d[d.groupby("call")["id"].transform("size") >= 30]
        g = d.groupby("call").apply(lambda x: pd.Series({"mal": x["a"].mean(),
                                                           "cb": x.loc[x["a"] == 0, "confidence"].mean()}))
        rows.append({"config": label, "route": "manual" if label.startswith("deepseek") else "API",
                     "calls": len(g), "A_answer_rate_range": float(g["mal"].max() - g["mal"].min()),
                     "B_conf_benign_range": float(g["cb"].max() - g["cb"].min())})
    t = pd.DataFrame(rows)
    ref = t[t["config"].isin(REFERENCE)]
    a_max, b_max = STABILITY_MULT * ref["A_answer_rate_range"].max(), STABILITY_MULT * ref["B_conf_benign_range"].max()
    t["A_max"], t["B_max"] = a_max, b_max
    t["stable"] = ((t["A_answer_rate_range"] <= a_max) & (t["B_conf_benign_range"] <= b_max)
                   & (t["route"] == "API") & (t["calls"] >= 5))
    t["flag"] = np.where(t["stable"], "stable",
                         np.where((t["route"] == "manual") | (t["calls"] < 5), "manual / not assessable", "unstable"))
    return t


# ---------------- Confirmatory collection (results/confirmatory_preregistration.md) ----------------
CONF_DIR = config.RESULTS_DIR / "confirmatory"


def _parse_probability_file(path):
    import json
    import re
    rows, seen = [], set()
    for m in re.finditer(r"\{[^{}]*\}", Path(path).read_text(encoding="utf-8")):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        cid, v = str(o.get("id", "")).strip(), o.get("probability")
        if cid and v is not None and cid not in seen:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if 0 <= v <= 100:
                seen.add(cid)
                rows.append({"id": cid, "p_hat": v / 100.0})
    return pd.DataFrame(rows, columns=["id", "p_hat"])


def confirmatory_analysis(n_boot: int = 4000, seed: int = 20261004):
    """P1 (beats constant), P2 (adds nothing beyond features), P3 (Haiku flip gone), per prereg."""
    import glob
    corpus = pd.read_parquet(config.CORPUS_PARQUET).reset_index(drop=True)
    corpus["order"] = np.arange(len(corpus))
    corpus["call"] = corpus["order"] // 100
    rat = rt.load_ratings().set_index("id")["ratings"]
    corpus["f"] = rt.vote_fraction(rat.loc[corpus["id"]], 3)
    p_star = brier.optimal_constant_p(corpus["f"].values)
    p1, p2, p3 = [], [], []
    for label in ("sonnet", "opus", "haiku", "gemini"):
        files = sorted(glob.glob(str(CONF_DIR / "raw" / f"{label}_*.txt")))
        if not files:
            continue
        pr = pd.concat([_parse_probability_file(f) for f in files]).drop_duplicates("id")
        d = corpus.merge(pr, on="id").reset_index(drop=True)
        cov = len(d) / len(corpus)
        sq = (d["p_hat"] - d["f"]).values ** 2
        sqc = (p_star - d["f"]).values ** 2
        gap = sq - sqc
        v = [gap[i].mean() for i in stats.nested_cluster_indices(d["call"].values, d["scan_idx"].values, n_boot, seed)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        p1.append({"config": label, "n": len(d), "coverage": cov, "PAD_B": sq.mean(), "const_B": sqc.mean(),
                   "gap": gap.mean(), "gap_lo": float(lo), "gap_hi": float(hi),
                   "P1_pass": bool(hi < 0)})
        feats = d[oracle_b.VIGNETTE].values
        base = (oracle_b.cv_predict(feats, d["f"].values, d["scan_idx"].values, "gbt") - d["f"].values) ** 2
        withp = (oracle_b.cv_predict(np.column_stack([feats, d["p_hat"].values]), d["f"].values,
                                     d["scan_idx"].values, "gbt") - d["f"].values) ** 2
        ab = base - withp
        v2 = [ab[i].mean() for i in stats.nested_cluster_indices(d["call"].values, d["scan_idx"].values, n_boot, seed + 1)]
        lo2, hi2 = np.percentile(v2, [2.5, 97.5])
        p2.append({"config": label, "features_only": base.mean(), "features+p_hat": withp.mean(),
                   "ablation_gap": ab.mean(), "lo": float(lo2), "hi": float(hi2), "P2_pass": bool(lo2 <= 0 <= hi2)})
        pc = d.groupby("call").apply(lambda x: pd.Series({"n": len(x), "corr": np.corrcoef(x["p_hat"], x["f"])[0, 1]
                                                            if x["p_hat"].nunique() > 1 else np.nan,
                                                            "mean_p_hat": x["p_hat"].mean()}))
        pc = pc[pc["n"] >= 30]
        p3.append({"config": label, "calls": len(pc), "min_call_corr": float(pc["corr"].min()),
                   "calls_with_negative_corr": int((pc["corr"] < 0).sum()),
                   "range_call_mean_p_hat": float(pc["mean_p_hat"].max() - pc["mean_p_hat"].min()),
                   "P3_pass": bool((pc["corr"] >= 0).all())})
    return pd.DataFrame(p1), pd.DataFrame(p2), pd.DataFrame(p3)


def confirmatory_oof_shift(n_boot: int = 4000, seed: int = 20261005, k: int = 5) -> pd.DataFrame:
    """Out-of-fold, scan-grouped base-rate corrections of the confirmatory p_hat (exploratory, post-hoc).

    shift       : p_hat + (mean f_train - mean p_hat_train)             (1 parameter)
    shift_scale : a + b * p_hat fit by OLS on the training folds          (2 parameters)
    The constant baseline is also fit out-of-fold (mean f of the training folds), so every
    comparison uses only training-fold information.
    """
    import glob
    from sklearn.model_selection import GroupKFold
    corpus = pd.read_parquet(config.CORPUS_PARQUET).reset_index(drop=True)
    corpus["call"] = np.arange(len(corpus)) // 100
    rat = rt.load_ratings().set_index("id")["ratings"]
    corpus["f"] = rt.vote_fraction(rat.loc[corpus["id"]], 3)
    rows = []
    for label in ("sonnet", "opus", "haiku", "gemini"):
        files = sorted(glob.glob(str(CONF_DIR / "raw" / f"{label}_*.txt")))
        pr = pd.concat([_parse_probability_file(f) for f in files]).drop_duplicates("id")
        d = corpus.merge(pr, on="id").reset_index(drop=True)
        f, p = d["f"].values, d["p_hat"].values
        sh, ss, cn = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), np.nan)
        for tr, te in GroupKFold(k, shuffle=True, random_state=20260726).split(p, groups=d["scan_idx"].values):
            sh[te] = np.clip(p[te] + (f[tr].mean() - p[tr].mean()), 0, 1)
            b, a0 = np.polyfit(p[tr], f[tr], 1)
            ss[te] = np.clip(a0 + b * p[te], 0, 1)
            cn[te] = f[tr].mean()
        e_const = (cn - f) ** 2
        for name, pv in (("raw", p), ("shift", sh), ("shift_scale", ss)):
            e = (pv - f) ** 2
            gap = e - e_const
            v = [gap[i].mean() for i in stats.nested_cluster_indices(d["call"].values, d["scan_idx"].values, n_boot, seed)]
            rows.append({"config": label, "map": name, "PAD_B": e.mean(), "const_B": e_const.mean(),
                         "gap": gap.mean(), "gap_lo": float(np.percentile(v, 2.5)), "gap_hi": float(np.percentile(v, 97.5))})
    return pd.DataFrame(rows)
