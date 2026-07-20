"""Expected Calibration Error, and its degeneracy on contested cases.

ECE needs a binary correct/incorrect outcome per case. On ambiguous-tier cases
(physicians split, no majority) that outcome is undefined, so ECE is degenerate
there — the motivation for PAD (spec 4.3, PDF 4.4).
"""

from __future__ import annotations

import numpy as np


def ece(confidence, correct, n_bins: int = 10) -> float:
    """Standard ECE: sum over confidence bins of |avg_conf - accuracy| * weight."""
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    n = len(confidence)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # last bin is closed on the right so conf==1.0 lands somewhere
        in_bin = (confidence >= lo) & (confidence < hi)
        if hi == 1.0:
            in_bin |= confidence == 1.0
        if not in_bin.any():
            continue
        avg_conf = confidence[in_bin].mean()
        acc = correct[in_bin].mean()
        total += (in_bin.sum() / n) * abs(avg_conf - acc)
    return float(total)
