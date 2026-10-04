"""Tests for the over-reliance simulation (Fix K)."""

import numpy as np

from contestbench.analysis import over_reliance as orl


def test_reliance_prob_monotonic_in_confidence():
    assert orl.reliance_prob(0.9, theta=0.6, beta=10) > orl.reliance_prob(0.5, theta=0.6, beta=10)


def test_over_reliance_nonnegative():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.5, 1, 200); pi = rng.uniform(0.5, 1, 200)
    assert orl.over_reliance(c, pi, theta=0.6, beta=10) >= 0


def test_calibrated_beats_overconfident_constant():
    # ideal c=pi should incur less over-reliance than always-max-confidence
    rng = np.random.default_rng(1)
    pi = rng.uniform(0.5, 1, 500)
    ideal = orl.over_reliance(pi, pi, theta=0.6, beta=10)
    overconf = orl.over_reliance(np.ones_like(pi), pi, theta=0.6, beta=10)
    assert ideal < overconf


def test_simulate_returns_regime_means():
    rng = np.random.default_rng(2)
    pi = rng.uniform(0.5, 1, 300)
    regimes = {"decoupled": rng.uniform(0.6, 0.9, 300), "ideal": pi}
    res = orl.simulate(regimes, pi, thetas=[0.5, 0.7], betas=[5, 10])
    assert set(res) == {"decoupled", "ideal"}
    assert res["ideal"] < res["decoupled"]      # calibrated confidence over-relies less
