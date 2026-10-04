"""Tests for the metrics subsystem: PAD family, stats (agreement, CIs), ECE."""

import math

import numpy as np

from contestbench.metrics import pad, stats, ece


# --- PAD family ---

def test_pad_unsigned():
    assert pad.pad_l1([0.6, 0.6], [1.0, 0.5]) == 0.25          # mean(0.4, 0.1)


def test_pad_signed_shows_underconfidence_negative():
    assert abs(pad.pad_signed([0.6, 0.6], [1.0, 0.5]) - (-0.15)) < 1e-9  # mean(-0.4, +0.1)


# --- stats: agreement rate ---

def test_agreement_rate_maps_frac_to_majority():
    assert stats.agreement_rate(0.0) == 1.0
    assert stats.agreement_rate(1.0) == 1.0
    assert stats.agreement_rate(0.25) == 0.75
    assert stats.agreement_rate(0.5) == 0.5


def test_agreement_rate_vectorized():
    out = stats.agreement_rate(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert list(out) == [1.0, 0.75, 0.5, 0.75, 1.0]


# --- stats: bootstrap CIs ---

def test_pearson_ci_strong_positive():
    x = np.linspace(0, 1, 50)
    y = x + np.random.default_rng(0).normal(0, 0.01, 50)
    r, lo, hi, p = stats.pearson_ci(x, y, n_boot=500, seed=1)
    assert r > 0.9
    assert lo > 0.5 and hi <= 1.0
    assert p < 0.05


def test_pearson_ci_null_brackets_zero():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 200)
    y = rng.normal(0, 1, 200)
    r, lo, hi, p = stats.pearson_ci(x, y, n_boot=500, seed=1)
    assert lo < 0 < hi          # CI includes zero for independent data


def test_mean_ci_brackets_mean():
    vals = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    m, lo, hi = stats.mean_ci(vals, n_boot=500, seed=1)
    assert m == 0.4
    assert lo < 0.4 < hi


def test_slope_ci_positive_trend():
    # signed-PAD-ambiguous rising with capability rank
    x = np.array([1, 1, 2, 2, 3, 3])
    y = np.array([0.10, 0.12, 0.21, 0.19, 0.33, 0.34])
    slope, lo, hi = stats.slope_ci(x, y, n_boot=1000, seed=1)
    assert slope > 0
    assert lo > 0            # CI excludes zero -> real upward trend


def test_slope_ci_flat_brackets_zero():
    rng = np.random.default_rng(3)
    x = np.array([1, 1, 2, 2, 3, 3])
    y = rng.normal(0.2, 0.05, 6)
    slope, lo, hi = stats.slope_ci(x, y, n_boot=1000, seed=1)
    assert lo < 0 < hi


# --- ECE ---

def test_ece_perfect_calibration_is_zero():
    # confidence exactly matches outcome frequency in each bin
    conf = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([0, 0, 1, 1])
    assert ece.ece(conf, correct, n_bins=2) == 0.0


def test_ece_detects_overconfidence():
    conf = np.array([0.9, 0.9, 0.9, 0.9])   # very confident
    correct = np.array([1, 0, 0, 0])        # but usually wrong
    val = ece.ece(conf, correct, n_bins=1)
    assert val > 0.5                         # |0.9 - 0.25|
