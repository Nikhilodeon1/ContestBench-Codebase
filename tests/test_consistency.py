"""Tests for sample-consistency confidence (Fix weekend-#1)."""

import numpy as np

from contestbench.analysis import consistency


def test_identical_samples_give_confidence_one():
    emb = np.tile([1.0, 0.0, 0.0], (5, 1))          # all identical
    assert abs(consistency.pairwise_cosine_confidence(emb) - 1.0) < 1e-9


def test_orthogonal_samples_give_low_confidence():
    emb = np.eye(4)                                  # mutually orthogonal
    assert abs(consistency.pairwise_cosine_confidence(emb)) < 1e-9


def test_confidence_normalizes_unnormalized_vectors():
    emb = np.array([[2.0, 0.0], [5.0, 0.0]])         # same direction, diff magnitude
    assert abs(consistency.pairwise_cosine_confidence(emb) - 1.0) < 1e-9


def test_single_sample_is_nan():
    assert np.isnan(consistency.pairwise_cosine_confidence(np.array([[1.0, 0.0]])))
