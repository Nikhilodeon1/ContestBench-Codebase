"""ContestBench CLI.

Subcommands:
  build-corpus   parse LIDC XML -> corpus.parquet + results/exclusions.csv + summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from contestbench import config
from contestbench.data import corpus

# proportional to full-corpus tier weights (546/708/159), ~50 cases total
SWEEP_TARGETS = {"high": 19, "contested": 25, "ambiguous": 6}


def build_corpus_cmd() -> int:
    xml_paths = sorted(config.LIDC_XML_ROOT.rglob("*.xml"))
    print(f"Found {len(xml_paths)} XML files under {config.LIDC_XML_ROOT}")

    df, exclusions = corpus.build_corpus_from_files(xml_paths)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CORPUS_PARQUET, index=False)

    excl_df = (
        pd.DataFrame(sorted(exclusions.items()), columns=["reason", "count"])
        if exclusions else pd.DataFrame(columns=["reason", "count"])
    )
    excl_df.to_csv(config.EXCLUSIONS_CSV, index=False)

    _print_summary(df, excl_df)
    return 0


def _print_summary(df: pd.DataFrame, excl_df: pd.DataFrame) -> None:
    print("\n=== CORPUS SUMMARY ===")
    print(f"usable nodules (n_raters >= {config.MIN_READERS}): {len(df)}")
    print(f"distinct scans contributing: {df['scan_idx'].nunique()}")
    print("\ntier counts:")
    print(df["tier"].value_counts().to_string())
    print("\nn_raters distribution:")
    print(df["n_raters"].value_counts().sort_index().to_string())
    print("\npi distribution:")
    print(df["pi"].round(3).value_counts().sort_index().to_string())

    four = df[df["n_raters"] == 4]
    print(f"\n4-reader-only nodules: {len(four)}")
    print("4-reader-only tier counts:")
    print(four["tier"].value_counts().to_string())

    print("\n=== EXCLUSIONS ===")
    print(excl_df.to_string(index=False) if len(excl_df) else "(none)")
    print(f"\nwrote {config.CORPUS_PARQUET}")
    print(f"wrote {config.EXCLUSIONS_CSV}")


def run_sweep_cmd(efforts: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from contestbench.eval import registry, runner
    from contestbench.eval.cache import DiskCache
    from contestbench.eval.groq_adapter import groq_query

    load_dotenv(dotenv_path=config.ROOT / ".env")

    df = pd.read_parquet(config.CORPUS_PARQUET)
    sample = corpus.sample_per_tier(df, SWEEP_TARGETS, seed=config.RANDOM_SEED)
    print("sweep sample tier counts:")
    print(sample["tier"].value_counts().to_string())

    specs = registry.sweep_specs(efforts=efforts)
    print(f"\nrunning {len(specs)} specs x {len(sample)} cases = "
          f"{len(specs) * len(sample)} calls (cached)...")
    cache = DiskCache(config.DATA_DIR / "cache")
    responses = runner.run(sample, specs, groq_query, cache)

    suffix = "_" + "".join(e[0] for e in efforts) if efforts else ""
    out = config.DATA_DIR / f"responses_sweep{suffix}.parquet"
    responses.to_parquet(out, index=False)

    _report_sweep(responses)
    print(f"\nwrote {out}")
    return 0


def _agreement(responses: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    df = responses.copy()
    df["pi_agree"] = np.maximum(df["pi"], 1 - df["pi"])  # majority agreement in [0.5,1]
    return df


def _report_sweep(responses: pd.DataFrame) -> None:
    from scipy.stats import pearsonr

    df = _agreement(responses)
    print("\ndrop rate (missing confidence) by label:")
    drop = df.assign(miss=df["confidence"].isna()).groupby("label")["miss"].agg(["sum", "count"])
    print(drop.to_string())

    valid = df.dropna(subset=["confidence"]).copy()
    valid["pad"] = (valid["confidence"] - valid["pi_agree"]).abs()
    valid["pad_signed"] = valid["confidence"] - valid["pi_agree"]

    print(f"\n=== SWEEP RESULTS (agreement-pi) === "
          f"({len(valid)}/{len(df)} valid)")
    print(f"{'label':22}{'n':>4}{'r(c,pi)':>9}{'p':>7}{'conf_mean':>10}{'conf_sd':>8}"
          f"{'PADh':>7}{'PADc':>7}{'PADa':>7}")
    for label, g in valid.groupby("label"):
        if g["confidence"].nunique() > 1 and len(g) > 2:
            r, p = pearsonr(g["pi_agree"], g["confidence"])
        else:
            r, p = float("nan"), float("nan")
        tpad = lambda t: g[g["tier"] == t]["pad"].mean() if (g["tier"] == t).any() else float("nan")
        print(f"{label:22}{len(g):>4}{r:>9.3f}{p:>7.2g}{g['confidence'].mean():>10.3f}"
              f"{g['confidence'].std():>8.3f}{tpad('high'):>7.3f}{tpad('contested'):>7.3f}"
              f"{tpad('ambiguous'):>7.3f}")

    _scatter(valid)


def _scatter(valid: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(0)
    for label, g in valid.groupby("label"):
        jitter = rng.uniform(-0.012, 0.012, len(g))
        ax.scatter(g["pi_agree"] + jitter, g["confidence"], alpha=0.5, s=20, label=label)
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="ideal (c=pi)")
    ax.axhline(valid["confidence"].mean(), color="red", ls=":", lw=1,
               label=f"mean conf={valid['confidence'].mean():.2f}")
    ax.set_xlabel("physician majority agreement (pi in [0.5,1])")
    ax.set_ylabel("model confidence")
    ax.set_title("ContestBench sweep: confidence decoupled from agreement")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    figdir = config.RESULTS_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / "sweep_scatter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


def run_full_cmd(models: list[str], efforts: list[str], n: int | None = None) -> int:
    from dotenv import load_dotenv
    from contestbench.eval import runner
    from contestbench.eval.cache import DiskCache
    from contestbench.eval.groq_adapter import groq_query

    load_dotenv(dotenv_path=config.ROOT / ".env")
    corpus_df = pd.read_parquet(config.CORPUS_PARQUET)

    if n:
        # proportional stratified subset (preserves tier weights)
        frac = corpus_df["tier"].value_counts(normalize=True)
        targets = {t: max(1, round(frac[t] * n)) for t in frac.index}
        corpus_df = corpus.sample_per_tier(corpus_df, targets, seed=config.RANDOM_SEED)
        print(f"subset {len(corpus_df)} cases (proportional): "
              f"{dict(corpus_df['tier'].value_counts())}")

    specs = []
    for m in models:
        slug = m if "/" in m else f"openai/{m}"
        for e in efforts:
            specs.append({"label": f"{slug.split('/')[-1]}:{e}", "model": slug,
                          "params": {"reasoning_effort": e, "temperature": 0.0}})

    print(f"running {len(specs)} spec(s) x {len(corpus_df)} cases (full corpus, cached)...")
    cache = DiskCache(config.DATA_DIR / "cache")
    responses = runner.run(corpus_df, specs, groq_query, cache)

    tag = "_".join(s["label"].replace(":", "-") for s in specs)[:60]
    out = config.DATA_DIR / f"responses_full_{tag}.parquet"
    responses.to_parquet(out, index=False)
    _report_sweep(responses)
    print(f"\nwrote {out}")
    return 0


def score_cmd(responses_path: str) -> int:
    import numpy as np
    from contestbench.metrics import pad, stats, ece

    df = pd.read_parquet(responses_path).dropna(subset=["confidence"]).copy()
    df["pi_agree"] = stats.agreement_rate(df["pi"].values)

    lines = ["| model | n | r(c,pi) [95% CI] | PAD [95% CI] | signed PAD | "
             "signed PAD (high/cont/amb) | ECE\\* |",
             "|---|--:|---|---|--:|---|--:|"]
    for label, g in df.groupby("label"):
        c, pi = g["confidence"].values, g["pi_agree"].values
        r, rlo, rhi, p = stats.pearson_ci(pi, c, n_boot=2000)
        pv, plo, phi = stats.mean_ci(np.abs(c - pi), n_boot=2000)
        sgn = {t: pad.pad_signed(g[g.tier == t]["confidence"], g[g.tier == t]["pi_agree"])
               for t in ["high", "contested", "ambiguous"]}
        # ECE on non-ambiguous cases only (ambiguous has no majority -> degenerate)
        nz = g[g["pi"] != 0.5]
        correct = (nz["answer"] == "malignant").values == (nz["pi"] > 0.5).values
        ece_val = ece.ece(nz["confidence"].values, correct.astype(float), n_bins=10)
        lines.append(
            f"| {label} | {len(g)} | {r:.3f} [{rlo:.3f}, {rhi:.3f}] | "
            f"{pv:.3f} [{plo:.3f}, {phi:.3f}] | {pad.pad_signed(c, pi):+.3f} | "
            f"{sgn['high']:+.2f} / {sgn['contested']:+.2f} / {sgn['ambiguous']:+.2f} | "
            f"{ece_val:.3f} |")
    table = "\n".join(lines)
    n_amb = int((df["pi"] == 0.5).sum())
    note = (f"\n\\* ECE computed on non-ambiguous cases only; {n_amb} ambiguous "
            f"(pi=0.5) cases have no majority outcome and are undefined for ECE -- "
            f"the motivation for PAD.")
    print(table + note)

    tdir = config.RESULTS_DIR / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / (Path(responses_path).stem + "_metrics.md")
    out.write_text(table + note + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def run_claude_cmd(n: int | None, models: list[str] | None) -> int:
    from dotenv import load_dotenv
    from contestbench.eval import registry, runner
    from contestbench.eval.cache import DiskCache
    from contestbench.eval.anthropic_adapter import anthropic_query

    load_dotenv(dotenv_path=config.ROOT / ".env")
    corpus_df = pd.read_parquet(config.CORPUS_PARQUET)
    if n:
        frac = corpus_df["tier"].value_counts(normalize=True)
        targets = {t: max(1, round(frac[t] * n)) for t in frac.index}
        corpus_df = corpus.sample_per_tier(corpus_df, targets, seed=config.RANDOM_SEED)
        print(f"subset {len(corpus_df)}: {dict(corpus_df['tier'].value_counts())}")

    specs = registry.claude_panel()
    if models:
        specs = [s for s in specs if s["label"].split(":")[0] in models]
    print(f"running {len(specs)} Claude config(s) x {len(corpus_df)} cases (cached)...")
    cache = DiskCache(config.DATA_DIR / "cache")
    responses = runner.run(corpus_df, specs, anthropic_query, cache)

    out = config.DATA_DIR / "responses_claude_panel.parquet"
    responses.to_parquet(out, index=False)
    _report_sweep(responses)
    print(f"\nwrote {out}")
    return 0


def report_cmd() -> int:
    """Regenerate every results table and figure from the responses on disk."""
    import numpy as np

    from contestbench.analysis import figures, mechanism, oracle, recalibration, panel as P

    data = P.load_panel()
    if not data:
        print("no panel responses found in data/ — run/import models first")
        return 1
    tdir = config.RESULTS_DIR / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # 1. main panel table
    tbl = P.panel_table(data)
    tbl.to_csv(tdir / "panel_metrics.csv", index=False)
    print("=== PANEL (n, conf, r, PAD, gap vs case-blind constant) ===")
    for _, r in tbl.iterrows():
        print(f"{r['config']:20}{int(r['n']):>6}{r['conf_mean']:>6.2f}{r['r']:>8.3f}"
              f"{r['PAD']:>8.3f}   gap {r['gap']:+.3f} [{r['gap_lo']:+.3f},{r['gap_hi']:+.3f}]"
              f"   {r['signed_high']:+.2f}/{r['signed_contested']:+.2f}/{r['signed_ambiguous']:+.2f}")
    worse = int((tbl["gap_lo"] > 0).sum())
    print(f"\nconfigs significantly WORSE than the case-blind constant: {worse}/{len(tbl)}")

    # 2. reasoning effect (the null)
    eff = P.reasoning_effect(dict(zip(tbl["config"], tbl["PAD"])))
    print("\n=== reasoning effect (thinking - standard PAD) ===")
    for fam, delta in eff.items():
        print(f"  {fam:12}{delta:+.3f}")
    pd.DataFrame(sorted(eff.items()), columns=["family", "delta_PAD"]).to_csv(
        tdir / "reasoning_effect.csv", index=False)

    # 3. capability trend
    print("\n=== capability trend (signed PAD on ambiguous vs rank) ===")
    for setting in ["standard", "thinking"]:
        per_rank, ids = {}, None
        for fam, rank in P.CAPABILITY_RANK.items():
            d = data.get(f"{fam}:{setting}")
            if d is None:
                continue
            amb = d[d["pi"] == 0.5].set_index("id")["confidence"] - 0.5
            ids = sorted(set(amb.index) if ids is None else set(ids) & set(amb.index))
            per_rank[rank] = amb
        if len(per_rank) >= 2:
            aligned = {r: s.reindex(ids).values for r, s in per_rank.items()}
            s, lo, hi = P.capability_slope(aligned)
            print(f"  {setting:10} slope {s:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n_amb={len(ids)})")

    # 4. mechanism
    mech = mechanism.mechanism_table(data)
    mech.to_csv(tdir / "mechanism.csv", index=False)
    print("\n=== mechanism: R2 of confidence on locked feature set ===")
    for _, r in mech.iterrows():
        print(f"  {r['config']:20}R2={r['R2_all']:.3f}")
    xm = mechanism.cross_model_confidence_corr(data)
    xm.to_csv(tdir / "cross_model_corr.csv")
    off = xm.values[~np.eye(len(xm), dtype=bool)]
    print(f"\ncross-model confidence corr: mean {off.mean():+.2f}, "
          f"range [{off.min():+.2f}, {off.max():+.2f}]")

    # 4b. oracle / upper-bound baseline (predict agreement from cheap features)
    ocorp = oracle.load_corpus_with_target()
    orows = []
    for name, feats in [("oracle:all-features", oracle.ALL_FEATURES),
                        ("oracle:rating-independent", oracle.RATING_INDEPENDENT_FEATURES),
                        ("oracle:geometry-only", oracle.GEOMETRY_FEATURES)]:
        e = oracle.evaluate(ocorp, feats)
        orows.append({"baseline": name, "PAD": e["oracle_pad"],
                      "constant_PAD": e["constant_pad"]})
    odf = pd.DataFrame(orows)
    odf.to_csv(tdir / "oracle.csv", index=False)
    best_llm = tbl["PAD"].min()
    print("\n=== ORACLE / upper-bound baseline (5-fold CV, out-of-fold) ===")
    for _, r in odf.iterrows():
        print(f"  {r['baseline']:22}PAD={r['PAD']:.3f}   (constant {r['constant_PAD']:.3f}, "
              f"best LLM {best_llm:.3f})")

    # 4c. post-hoc recalibrator (Fix H): the proposed intervention
    orc_ri = odf.iloc[1]["PAD"]  # rating-independent oracle
    recal = recalibration.evaluate_panel(data, oracle_pad=orc_ri)
    recal.to_csv(tdir / "recalibration.csv", index=False)
    print("\n=== RECALIBRATION (per-config, out-of-fold): PAD before -> after ===")
    for _, r in recal.iterrows():
        print(f"  {r['config']:22}{r['pad_before']:.3f} -> {r['pad_after']:.3f}"
              f"   (gap to oracle closed {r['gap_closed_frac']*100:.0f}%)")
    print(f"  ABLATION: oracle (features only) {orc_ri:.3f} vs "
          f"recalibrator (features + confidence) {recal['pad_after'].mean():.3f} "
          f"-> confidence adds {orc_ri - recal['pad_after'].mean():+.3f} PAD")
    imp = recalibration.importance_panel(data)
    imp.to_csv(tdir / "confidence_importance.csv", index=False)
    print("  confidence permutation-importance (held-out): "
          f"mean {imp['permutation'].mean():.3f}, range "
          f"[{imp['permutation'].min():.3f}, {imp['permutation'].max():.3f}] "
          f"-> small but non-zero: confidence carries a little redundant signal")

    # 5. figures
    print()
    oracle_lines = [("oracle (all feats)", odf.iloc[0]["PAD"], "#228833"),
                    ("oracle (rating-indep)", odf.iloc[1]["PAD"], "#aa7733"),
                    ("oracle (geometry only)", odf.iloc[2]["PAD"], "#8833aa")]
    print("wrote", figures.baseline_floor(data, oracle_lines=oracle_lines))
    print("wrote", figures.capability_inversion(data, P.CAPABILITY_RANK))
    first = next(iter(P.PANEL.values()))
    print("wrote", figures.decoupling_scatter(config.DATA_DIR / f"{first}.parquet",
                                              out="decoupling_scatter.png"))
    st, path = figures.ece_degeneracy(config.DATA_DIR / f"{first}.parquet")
    print(f"wrote {path} (ECE band width {st['ece_ci_width']:.2f} vs PAD {st['pad']:.3f})")
    print(f"\ntables -> {tdir}")
    return 0


def figures_cmd() -> int:
    from contestbench.analysis import figures
    D = config.DATA_DIR
    models = {
        "gemini-3.5-flash": D / "responses_gemini.parquet",
        "gpt-oss-20b": D / "responses_full_gpt-oss-20b-low.parquet",
        "opus:standard": D / "responses_opus-standard.parquet",
        "opus:thinking": D / "responses_opus-thinking.parquet",
    }
    present = {k: str(v) for k, v in models.items() if v.exists()}

    if models["gemini-3.5-flash"].exists():
        print("wrote", figures.decoupling_scatter(str(models["gemini-3.5-flash"])))
        stats_, path = figures.ece_degeneracy(str(models["gemini-3.5-flash"]))
        print(f"wrote {path}  (ECE CI width {stats_['ece_ci_width']:.2f} vs "
              f"PAD {stats_['pad']:.3f} on {stats_['n_ambiguous']} ambiguous cases)")
    if len(present) >= 2:
        print("wrote", figures.multimodel_signed_pad(present))
    return 0


def export_batch_cmd(chunk_size: int, n: int | None = None) -> int:
    from contestbench.eval import batch
    df = pd.read_parquet(config.CORPUS_PARQUET)
    prefix = "batch"
    if n:
        frac = df["tier"].value_counts(normalize=True)
        targets = {t: max(1, round(frac[t] * n)) for t in frac.index}
        df = corpus.sample_per_tier(df, targets, seed=config.RANDOM_SEED)
        prefix = "claim3"
        print(f"subset {len(df)} cases (proportional): {dict(df['tier'].value_counts())}")
    out_dir = config.RESULTS_DIR / "batch_prompts"
    paths = batch.write_batch_files(df, out_dir, chunk_size=chunk_size, prefix=prefix)
    print(f"exported {len(df)} cases into {len(paths)} file(s) in {out_dir}:")
    for p in paths:
        print(f"  {p.name}")
    return 0


def import_batch_cmd(answer_paths: list[str], label: str) -> int:
    from contestbench.eval import batch
    corpus_df = pd.read_parquet(config.CORPUS_PARQUET)[["id", "pi", "tier"]]

    parts = [batch.parse_batch_file(p) for p in answer_paths]
    answers = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="first")

    merged = corpus_df.merge(answers, on="id", how="left")
    merged["label"] = label
    n_ans = merged["confidence"].notna().sum()
    print(f"parsed answers for {n_ans} cases (label={label})")

    safe = label.replace(":", "-").replace("/", "-").replace(".", "")
    out = config.DATA_DIR / f"responses_{safe}.parquet"
    scored = merged.dropna(subset=["confidence"])
    scored.to_parquet(out, index=False)
    _report_sweep(scored)
    print(f"\nwrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contestbench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-corpus", help="build corpus.parquet from LIDC XML")
    sweep = sub.add_parser("run-sweep", help="run the 50-case reasoning_effort sweep")
    sweep.add_argument("--efforts", nargs="+", choices=["low", "medium", "high"],
                       help="restrict to these reasoning efforts (default: all)")
    full = sub.add_parser("run-full", help="run model(s) x effort(s) over the FULL corpus")
    full.add_argument("--models", nargs="+", default=["gpt-oss-20b"])
    full.add_argument("--efforts", nargs="+", choices=["low", "medium", "high"],
                      default=["low"])
    full.add_argument("--n", type=int, default=None,
                      help="proportional stratified subset size (default: full corpus)")
    exp = sub.add_parser("export-batch", help="export prompt file(s) for file-in/out platform")
    exp.add_argument("--chunk-size", type=int, default=500,
                     help="cases per file (0 = single file). default 500")
    exp.add_argument("--n", type=int, default=None,
                     help="proportional stratified subset (e.g. 150 for claim-3)")
    imp = sub.add_parser("import-batch", help="parse returned answer file(s)")
    imp.add_argument("answers", nargs="+", help="returned answer file paths")
    imp.add_argument("--label", default="gemini-3.5-flash")
    sc = sub.add_parser("score", help="metrics table (PAD family, CIs, ECE) for a responses file")
    sc.add_argument("responses", help="path to a responses parquet")
    cl = sub.add_parser("run-claude", help="run the Claude capability x reasoning panel")
    cl.add_argument("--n", type=int, default=None, help="proportional subset size")
    cl.add_argument("--models", nargs="+", choices=["haiku", "sonnet", "opus"],
                    help="restrict to these Claude tiers (default: all)")
    sub.add_parser("figures", help="regenerate all paper figures from responses")
    sub.add_parser("report", help="regenerate ALL results tables + figures")

    args = parser.parse_args(argv)
    if args.cmd == "build-corpus":
        return build_corpus_cmd()
    if args.cmd == "run-sweep":
        return run_sweep_cmd(efforts=args.efforts)
    if args.cmd == "run-full":
        return run_full_cmd(args.models, args.efforts, args.n)
    if args.cmd == "export-batch":
        return export_batch_cmd(args.chunk_size, args.n)
    if args.cmd == "import-batch":
        return import_batch_cmd(args.answers, args.label)
    if args.cmd == "score":
        return score_cmd(args.responses)
    if args.cmd == "run-claude":
        return run_claude_cmd(args.n, args.models)
    if args.cmd == "figures":
        return figures_cmd()
    if args.cmd == "report":
        return report_cmd()
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
