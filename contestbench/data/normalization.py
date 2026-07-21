"""Normalize heterogeneous inter-rater agreement formats to a common rate [0,1].

LIDC needs no normalization (it gives raw agreement counts). This module exists
so a second source (BI-RADS percentages, DSM-5 ICCs, kappa-reported studies) can
be folded onto the same scale, with a sensitivity check across the three choices
(spec 4.5, PDF novelty claim 4).
"""

from __future__ import annotations


def normalize_percent(pct: float) -> float:
    """Raw percentage agreement -> [0,1]."""
    return pct / 100.0


def normalize_kappa(kappa: float, p_expected: float) -> float:
    """Cohen's kappa -> implied observed agreement = kappa*(1-p_e) + p_e."""
    return kappa * (1.0 - p_expected) + p_expected


def normalize_icc(icc: float) -> float:
    """ICC as a direct agreement proxy, clipped to [0,1] (an approximation)."""
    return min(1.0, max(0.0, icc))


def sensitivity_variants(value: float, source: str, p_expected: float = 0.5) -> dict:
    """Return the agreement rate under each defensible normalization choice.

    For an unambiguous source (percent) all three agree; for kappa/ICC sources the
    spread across choices is the sensitivity signal reported in the appendix.
    """
    return {
        "as_percent": normalize_percent(value) if source == "percent" else value,
        "as_kappa": normalize_kappa(value, p_expected) if source == "kappa"
        else (normalize_percent(value) if source == "percent" else value),
        "as_icc": normalize_icc(value) if source == "icc"
        else (normalize_percent(value) if source == "percent" else value),
    }
