"""PAD-B: Brier-based Physician Agreement Deviation (primary metric).

A model gives an answer a (Malignant=1) and a confidence c in that answer. Fold
them into a single probability that the nodule is malignant:
    p = c if a == 1 else 1 - c
Score against f, the raw fraction of readers rating malignancy >= 3:
    PAD-B = mean (p - f)^2
This is the Brier score with the physician vote fraction as the (soft) target, a
proper scoring rule: expected score is minimized by reporting the true malignant
vote fraction, and a confidently wrong answer is penalised.

s = support for the model's chosen answer among physicians:
    s = f if a == 1 else 1 - f
(the warranted-reliance target used by the over-reliance simulation).
"""

from __future__ import annotations

import numpy as np


def answer_to_binary(answers) -> np.ndarray:
    """1.0 for 'malignant', 0.0 for 'benign' (case-insensitive)."""
    a = np.asarray([str(x).strip().lower() for x in answers])
    bad = ~np.isin(a, ["malignant", "benign"])
    if bad.any():
        raise ValueError(f"unrecognised answer values: {sorted(set(a[bad]))[:5]}")
    return (a == "malignant").astype(float)


def fold_forecast(a, c) -> np.ndarray:
    """p = c if a == 1 else 1 - c."""
    a, c = np.asarray(a, float), np.asarray(c, float)
    return np.where(a == 1.0, c, 1.0 - c)


def support(a, f) -> np.ndarray:
    """s = f if a == 1 else 1 - f."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    return np.where(a == 1.0, f, 1.0 - f)


def pad_b(p, f) -> float:
    """PAD-B = mean (p - f)^2. Proper (Brier) score against the vote fraction."""
    p, f = np.asarray(p, float), np.asarray(f, float)
    return float(np.mean((p - f) ** 2))


def optimal_constant_p(f) -> float:
    """PAD-B-minimising case-blind forecast p* = mean(f) (not 0.75)."""
    return float(np.mean(np.asarray(f, float)))


def constant_pad_b(f, p: float | None = None) -> float:
    """PAD-B of a case-blind constant forecast (default: the optimal one, = var(f))."""
    f = np.asarray(f, float)
    p = optimal_constant_p(f) if p is None else p
    return float(np.mean((p - f) ** 2))
