"""Tests for panel-level analysis (gap vs constant, reasoning effect, trend)."""

import numpy as np
import pandas as pd

from contestbench.analysis import panel


def test_gap_vs_constant_positive_when_model_worse():
    pi = np.array([0.5, 0.75, 1.0] * 40)
    conf = np.full(len(pi), 0.5)          # badly under-confident on high-agreement
    gap, lo, hi = panel.gap_vs_constant(conf, pi, c_star=0.75, n_boot=300, seed=1)
    assert gap > 0
    assert lo > 0                          # significantly worse than the constant


def test_gap_vs_constant_zero_for_the_constant_itself():
    pi = np.array([0.5, 0.75, 1.0] * 40)
    conf = np.full(len(pi), 0.75)          # exactly the constant strategy
    gap, lo, hi = panel.gap_vs_constant(conf, pi, c_star=0.75, n_boot=300, seed=1)
    assert abs(gap) < 1e-9


def test_reasoning_effect_pairs_by_family():
    pads = {"opus:standard": 0.20, "opus:thinking": 0.18,
            "gemini:standard": 0.17, "gemini:thinking": 0.19}
    eff = panel.reasoning_effect(pads)
    assert abs(eff["opus"] - (-0.02)) < 1e-9
    assert abs(eff["gemini"] - (0.02)) < 1e-9


def test_reasoning_effect_ignores_unpaired():
    eff = panel.reasoning_effect({"solo:standard": 0.2})
    assert eff == {}


def test_capability_slope_positive():
    # ambiguous-tier signed PAD rising with rank
    per_rank = {1: np.array([0.10, 0.12]), 2: np.array([0.20, 0.22]), 3: np.array([0.30, 0.34])}
    slope, lo, hi = panel.capability_slope(per_rank, n_boot=300, seed=1)
    assert slope > 0
    assert lo > 0
