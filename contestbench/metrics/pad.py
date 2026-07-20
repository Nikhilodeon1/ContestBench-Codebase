"""PAD family — Physician Agreement Deviation.

All take model confidence ``c`` and physician agreement ``pi`` (majority
agreement in [0.5,1]; see stats.agreement_rate). PAD is the L1 analogue of the
Brier proper scoring rule with the physician agreement rate as target.
"""

from __future__ import annotations

import numpy as np


def pad(c, pi) -> float:
    """Unsigned PAD = mean|c - pi| (headline miscalibration)."""
    c, pi = np.asarray(c, float), np.asarray(pi, float)
    return float(np.mean(np.abs(c - pi)))


def pad_signed(c, pi) -> float:
    """Signed PAD = mean(c - pi). Negative = under-confident vs agreement."""
    c, pi = np.asarray(c, float), np.asarray(pi, float)
    return float(np.mean(c - pi))


def pad_squared(c, pi) -> float:
    """PAD^2 = mean((c - pi)^2), the Brier-score variant."""
    c, pi = np.asarray(c, float), np.asarray(pi, float)
    return float(np.mean((c - pi) ** 2))
