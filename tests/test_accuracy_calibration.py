"""Tests for the accuracy-calibrated baseline (weekend fix #2)."""

import numpy as np

from contestbench.analysis import accuracy_calibration as ac


def test_platt_predictions_out_of_fold_are_probabilities():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 200)
    correct = (rng.uniform(0, 1, 200) < conf).astype(int)
    p = ac.platt_cv(conf, correct, k=5, seed=1)
    assert len(p) == 200
    assert p.min() >= 0 and p.max() <= 1


def test_calibrated_confidence_tracks_accuracy_not_input_scale():
    # model is 70% accurate regardless of its (inflated) confidence -> calibrated ~0.7
    rng = np.random.default_rng(1)
    conf = np.full(400, 0.95)
    correct = (rng.uniform(0, 1, 400) < 0.70).astype(int)
    p = ac.platt_cv(conf, correct, k=5, seed=1)
    assert abs(p.mean() - 0.70) < 0.08          # calibrated toward true accuracy
