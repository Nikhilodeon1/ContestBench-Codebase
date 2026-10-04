"""PAD-L1 family -- descriptive distance statistics (NOT proper scoring rules).

All take model confidence ``c`` and physician agreement ``pi`` (majority
agreement in [0.5,1]; see stats.agreement_rate). PAD-L1 = mean|c - pi| is a
descriptive distance between stated confidence and agreement. It is NOT a proper
scoring rule: it ignores which answer the model chose, so a confidently-wrong
answer on a unanimous case can score 0. The primary metric is PAD-B
(contestbench.metrics.brier), a proper Brier score against the vote fraction f.
"""

from __future__ import annotations

import numpy as np


def pad_l1(c, pi) -> float:
    """PAD-L1 = mean|c - pi|. Descriptive distance statistic, not a proper scoring rule."""
    c, pi = np.asarray(c, float), np.asarray(pi, float)
    return float(np.mean(np.abs(c - pi)))


def pad_signed(c, pi) -> float:
    """Signed PAD-L1 = mean(c - pi). Negative = under-confident vs agreement. Descriptive only."""
    c, pi = np.asarray(c, float), np.asarray(pi, float)
    return float(np.mean(c - pi))
