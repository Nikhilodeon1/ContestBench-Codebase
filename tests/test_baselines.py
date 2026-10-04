"""Tests for baseline strategies (constant-confidence floor)."""

import numpy as np

from contestbench.metrics import baselines


def test_optimal_constant_is_median():
    assert baselines.optimal_constant([0.5, 0.75, 1.0]) == 0.75


def test_constant_pad_value():
    # c=0.5 vs pi=[0.5, 1.0] -> mean(0, 0.5) = 0.25
    assert baselines.constant_pad([0.5, 1.0], 0.5) == 0.25


def test_optimal_constant_minimizes_pad():
    pi = np.array([0.5, 0.5, 0.75, 1.0, 1.0])
    c_star = baselines.optimal_constant(pi)
    pad_star = baselines.constant_pad(pi, c_star)
    for c in [0.4, 0.6, 0.7, 0.9, 1.0]:
        assert pad_star <= baselines.constant_pad(pi, c) + 1e-9
