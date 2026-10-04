"""Validation of PAD-B fold logic (Phase 1.5 sanity check)."""

import numpy as np

from contestbench.metrics import brier, pad


def test_fold_forecast():
    a = np.array([1.0, 0.0])
    c = np.array([0.8, 0.8])
    assert np.allclose(brier.fold_forecast(a, c), [0.8, 0.2])


def test_support():
    assert np.allclose(brier.support([1.0, 0.0], [0.75, 0.75]), [0.75, 0.25])


def test_confidently_wrong_unanimous_case():
    # all readers say malignant (f=1, pi=1); model says Benign at c=0.8 -> p=0.2
    f, pi = np.array([1.0]), np.array([1.0])
    p = brier.fold_forecast([0.0], [0.8])
    assert abs(brier.pad_b(p, f) - 0.64) < 1e-12
    # same case seen by PAD-L1 is only mildly penalised; at c=1.0 it scores 0
    assert abs(pad.pad_l1([0.8], pi) - 0.2) < 1e-12
    assert pad.pad_l1([1.0], pi) == 0.0
    assert brier.pad_b(brier.fold_forecast([0.0], [1.0]), f) == 1.0


def test_confidently_right_scores_near_zero():
    p = brier.fold_forecast([1.0], [0.95])
    assert abs(brier.pad_b(p, [1.0]) - 0.0025) < 1e-12


def test_proper_scoring_truthful_forecast_minimises():
    f = np.array([0.75] * 4)
    scores = {p: brier.pad_b([p] * 4, f) for p in (0.5, 0.75, 0.95)}
    assert scores[0.75] == 0.0 and scores[0.75] < scores[0.5] < scores[0.95] + 1


def test_optimal_constant_is_mean_f():
    f = np.array([0.0, 0.0, 1.0, 0.5])
    assert abs(brier.optimal_constant_p(f) - 0.375) < 1e-12
    assert abs(brier.constant_pad_b(f) - np.var(f)) < 1e-12


def test_answer_parsing_rejects_junk():
    import pytest
    assert list(brier.answer_to_binary(["Malignant", "benign"])) == [1.0, 0.0]
    with pytest.raises(ValueError):
        brier.answer_to_binary(["maybe"])
