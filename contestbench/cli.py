"""ContestBench CLI.

Subcommands:
  build-corpus   parse LIDC XML -> corpus.parquet + results/exclusions.csv + summary
"""

from __future__ import annotations

import argparse
import sys

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


def export_batch_cmd(chunk_size: int) -> int:
    from contestbench.eval import batch
    df = pd.read_parquet(config.CORPUS_PARQUET)
    out_dir = config.RESULTS_DIR / "batch_prompts"
    paths = batch.write_batch_files(df, out_dir, chunk_size=chunk_size)
    print(f"exported {len(df)} cases into {len(paths)} file(s) in {out_dir}:")
    for p in paths:
        print(f"  {p.name}")
    print("\nFeed each file to the platform; save each returned answer file, then run:")
    print("  python -m contestbench.cli import-batch results/batch_answers/*.txt")
    return 0


def import_batch_cmd(answer_paths: list[str], label: str) -> int:
    from contestbench.eval import batch
    corpus_df = pd.read_parquet(config.CORPUS_PARQUET)[["id", "pi", "tier"]]

    parts = [batch.parse_batch_file(p) for p in answer_paths]
    answers = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="first")

    merged = corpus_df.merge(answers, on="id", how="left")
    merged["label"] = label
    n_ans = merged["confidence"].notna().sum()
    print(f"parsed answers for {n_ans}/{len(corpus_df)} corpus cases "
          f"({len(corpus_df) - n_ans} missing)")

    out = config.DATA_DIR / "responses_gemini.parquet"
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
    exp = sub.add_parser("export-batch", help="export prompt file(s) for file-in/out platform")
    exp.add_argument("--chunk-size", type=int, default=500,
                     help="cases per file (0 = single file). default 500")
    imp = sub.add_parser("import-batch", help="parse returned answer file(s)")
    imp.add_argument("answers", nargs="+", help="returned answer file paths")
    imp.add_argument("--label", default="gemini-3.5-flash")

    args = parser.parse_args(argv)
    if args.cmd == "build-corpus":
        return build_corpus_cmd()
    if args.cmd == "run-sweep":
        return run_sweep_cmd(efforts=args.efforts)
    if args.cmd == "export-batch":
        return export_batch_cmd(args.chunk_size)
    if args.cmd == "import-batch":
        return import_batch_cmd(args.answers, args.label)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
