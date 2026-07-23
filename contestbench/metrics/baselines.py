"""Baseline strategies for contextualizing PAD (Fix 4).

- Constant-confidence: a model that ignores the case and emits one number. The
  optimal constant (the PAD-minimizing value) is the best a case-blind strategy
  can do; if a real model doesn't beat it, the model isn't using case difficulty.
- ECE-vs-PAD divergence is computed in analysis from the existing metrics: a model
  can be well accuracy-calibrated (low ECE) yet poorly agreement-calibrated (high
  PAD) — the thing ECE cannot see.
"""

from __future__ import annotations

import numpy as np


def optimal_constant(pi) -> float:
    """PAD-minimizing constant confidence = median of the agreement rates."""
    return float(np.median(np.asarray(pi, float)))


def constant_pad(pi, c: float) -> float:
    """PAD of a constant-confidence-c strategy = mean|c - pi|."""
    pi = np.asarray(pi, float)
    return float(np.mean(np.abs(c - pi)))
