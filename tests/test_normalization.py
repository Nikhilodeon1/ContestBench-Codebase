"""Tests for agreement-rate normalization across heterogeneous source formats."""

import math

from contestbench.data import normalization as norm


def test_percent_divides_by_100():
    assert norm.normalize_percent(72) == 0.72
    assert norm.normalize_percent(100) == 1.0


def test_cohens_kappa_to_implied_agreement():
    # agreement = kappa*(1 - p_e) + p_e
    assert abs(norm.normalize_kappa(0.6, p_expected=0.5) - 0.8) < 1e-9


def test_kappa_zero_returns_chance_agreement():
    assert norm.normalize_kappa(0.0, p_expected=0.4) == 0.4


def test_icc_is_clipped_to_unit_interval():
    assert norm.normalize_icc(0.28) == 0.28
    assert norm.normalize_icc(1.2) == 1.0
    assert norm.normalize_icc(-0.1) == 0.0


def test_sensitivity_identity_on_percent_source():
    # three normalization choices on a percentage source agree (no format ambiguity)
    variants = norm.sensitivity_variants(value=72, source="percent")
    assert all(abs(v - 0.72) < 1e-9 for v in variants.values())
