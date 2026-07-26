"""Over-reliance simulation (Fix K) -- illustrative, not a clinical claim.

Anchored in Lee & See (2004): appropriate reliance tracks capability;
over-reliance is reliance in excess of what is warranted. We operationalize
warranted reliance on a case as the physician agreement rate pi (how trustworthy
the majority call is) and model a clinician who relies on the AI with probability
logistic in its expressed confidence:  P(rely) = sigmoid(beta * (c - theta)).
Over-reliance on a case = max(0, P(rely) - pi), averaged over cases.

Three confidence regimes on the SAME cases:
  - decoupled    : observed model confidence, POOLED across all 10 configs
                   (avoids cherry-picking a single model)
  - recalibrated : the Fix H post-hoc recalibrator's output (deployable mitigation)
  - ideal        : c = pi (perfectly agreement-calibrated ceiling)

Assumptions are explicit and the result is swept over theta/beta to show the
ordering is robust to the clinician-behavior parameters. This illustrates the
*mechanism* by which decoupled confidence induces over-reliance; it is not an
empirical harm estimate.
"""

from __future__ import annotations

import numpy as np


def reliance_prob(c, theta: float, beta: float):
    c = np.asarray(c, float)
    return 1.0 / (1.0 + np.exp(-beta * (c - theta)))


def over_reliance(c, pi, theta: float, beta: float) -> float:
    """Mean over-reliance = mean max(0, P(rely|c) - pi)."""
    pi = np.asarray(pi, float)
    excess = reliance_prob(c, theta, beta) - pi
    return float(np.mean(np.clip(excess, 0.0, None)))


def simulate(regimes: dict[str, np.ndarray], pi, thetas, betas) -> dict[str, float]:
    """Mean over-reliance per regime, averaged over the theta x beta grid."""
    out = {}
    for name, c in regimes.items():
        vals = [over_reliance(c, pi, t, b) for t in thetas for b in betas]
        out[name] = float(np.mean(vals))
    return out


def simulate_grid(regimes: dict[str, np.ndarray], pi, thetas, betas):
    """Full per-(theta,beta) table for each regime (for the sweep figure)."""
    grid = {}
    for name, c in regimes.items():
        grid[name] = {(t, b): over_reliance(c, pi, t, b) for t in thetas for b in betas}
    return grid
