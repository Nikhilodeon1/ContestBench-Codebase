"""Tests for the vignette-template audit (weekend fix #5)."""

import pandas as pd

from contestbench.data import vignette_audit as va


def _df(rows):
    return pd.DataFrame(rows, columns=["id", "subtlety", "spiculation", "margin", "pi_agree"])


def test_identical_vignette_different_pi_is_a_collision():
    # two cases the model sees identically, but different agreement -> forced decoupling
    df = _df([("a", 3.0, 2.0, 4.0, 1.0), ("b", 3.0, 2.0, 4.0, 0.5),
              ("c", 5.0, 1.0, 2.0, 0.75)])
    r = va.collision_report(df)
    assert r["n_unique_vignettes"] == 2
    assert r["cases_in_pi_varying_collisions"] == 2      # a and b
    assert r["frac_forced"] == 2 / 3


def test_all_distinct_vignettes_no_collision():
    df = _df([("a", 3.0, 2.0, 4.0, 1.0), ("b", 4.0, 1.0, 5.0, 0.5)])
    r = va.collision_report(df)
    assert r["cases_in_pi_varying_collisions"] == 0
    assert r["frac_forced"] == 0.0


def test_bucket_mean_baseline_pad():
    # bucket {a,b}: pi 1.0 & 0.5 -> mean 0.75 -> errors 0.25,0.25; bucket {c}: 0
    df = _df([("a", 3.0, 2.0, 4.0, 1.0), ("b", 3.0, 2.0, 4.0, 0.5),
              ("c", 5.0, 1.0, 2.0, 0.75)])
    assert abs(va.bucket_mean_pad(df) - (0.25 + 0.25 + 0.0) / 3) < 1e-9


def test_non_colliding_mask_flags_single_pi_buckets():
    df = _df([("a", 3.0, 2.0, 4.0, 1.0), ("b", 3.0, 2.0, 4.0, 0.5),
              ("c", 5.0, 1.0, 2.0, 0.75)])
    mask = va.non_colliding_mask(df)
    assert list(mask) == [False, False, True]     # only c is collision-free


def test_rounding_matches_presented_precision():
    # differ only in the 3rd decimal -> presented identically at 2dp -> collide
    df = _df([("a", 3.001, 2.0, 4.0, 1.0), ("b", 3.002, 2.0, 4.0, 0.5)])
    r = va.collision_report(df, ndigits=2)
    assert r["cases_in_pi_varying_collisions"] == 2
