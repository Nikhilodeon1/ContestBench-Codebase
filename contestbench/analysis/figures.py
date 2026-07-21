"""Reproducible paper figures + the ECE-degeneracy demonstration.

Reads responses parquets and writes PNGs under results/figures/. Called by
`cli figures` and reproduce.sh so every figure regenerates from data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contestbench import config
from contestbench.metrics import ece, pad, stats

FIGDIR = config.RESULTS_DIR / "figures"
TIERS = ["high", "contested", "ambiguous"]


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path).dropna(subset=["confidence"]).copy()
    df["pi_agree"] = stats.agreement_rate(df["pi"].values)
    return df


def decoupling_scatter(responses_path, out: str = "decoupling_scatter.png") -> Path:
    df = _load(responses_path)
    r, lo, hi, _ = stats.pearson_ci(df["pi_agree"].values, df["confidence"].values, n_boot=3000)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rng = np.random.default_rng(0)
    ax.scatter(df["pi_agree"] + rng.uniform(-0.02, 0.02, len(df)),
               df["confidence"], s=8, alpha=0.25, color="#4477aa")
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1.2, label="ideal (confidence = agreement)")
    tm = df.groupby("pi_agree")["confidence"].mean()
    ax.scatter(tm.index, tm.values, color="red", s=70, marker="D", zorder=5,
               label="mean confidence per agreement level")
    ax.set_xlabel("physician majority agreement  pi = max(f, 1-f)")
    ax.set_ylabel("model confidence")
    ax.set_title(f"Confidence decoupled from agreement\nr={r:.3f} [{lo:.2f}, {hi:.2f}], n={len(df)}")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    return _save(fig, out)


def multimodel_signed_pad(paths: dict[str, str], out: str = "multimodel_signed_pad.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(TIERS))
    w = 0.8 / max(len(paths), 1)
    for i, (label, path) in enumerate(paths.items()):
        df = _load(path)
        vals = [pad.pad_signed(df[df.tier == t]["confidence"], df[df.tier == t]["pi_agree"])
                for t in TIERS]
        ax.bar(x + (i - (len(paths) - 1) / 2) * w, vals, w, label=label)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["high\n(pi=1.0)", "contested\n(pi=0.75)", "ambiguous\n(pi=0.5)"])
    ax.set_ylabel("signed PAD (confidence - agreement)")
    ax.set_title("Under-confident on easy, over-confident on hard (grows with capability)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def ece_degeneracy(responses_path, n_draws: int = 2000, seed: int = 20260720,
                   out: str = "ece_degeneracy.png"):
    """Show ECE is degenerate on ambiguous cases while PAD is stable.

    Ambiguous cases (raw f=0.5) have no majority outcome, so the binary label ECE
    needs is undefined. Drawing it at random (physicians are 50/50) makes ECE swing
    across a wide band; PAD needs no label and is a single stable value.
    """
    df = _load(responses_path)
    amb = df[df["pi"] == 0.5]
    conf = amb["confidence"].values
    rng = np.random.default_rng(seed)
    eces = np.array([ece.ece(conf, rng.integers(0, 2, len(conf)).astype(float), n_bins=10)
                     for _ in range(n_draws)])
    pad_val = pad.pad(conf, amb["pi_agree"].values)
    lo, hi = np.percentile(eces, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(eces, bins=40, color="#ee6677", alpha=0.7,
            label=f"ECE under random label\n(95% band [{lo:.2f}, {hi:.2f}], width {hi-lo:.2f})")
    ax.axvline(pad_val, color="#4477aa", lw=2.5, label=f"PAD = {pad_val:.3f} (stable, label-free)")
    ax.set_xlabel("calibration error on ambiguous tier")
    ax.set_ylabel("bootstrap draws")
    ax.set_title(f"ECE is undefined on contested cases; PAD is not (n={len(conf)} ambiguous)")
    ax.legend(fontsize=8)
    path = _save(fig, out)
    return {"ece_ci_width": float(hi - lo), "pad": pad_val, "n_ambiguous": int(len(conf))}, path


def _save(fig, name) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    p = FIGDIR / name
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
