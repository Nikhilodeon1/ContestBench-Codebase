"""Tests for corpus construction: pi, tiering, record building, sampling."""

import pandas as pd

from contestbench.data.lidc_loader import ReaderNodule, ScanParse
from contestbench.data import corpus


def rn(malig, x=100.0, y=100.0, z=0.0, sub=None, spic=None, marg=None):
    return ReaderNodule(nodule_id="N", x=x, y=y, z=z, malignancy=malig,
                        subtlety=sub, spiculation=spic, margin=marg)


# --- pi ---

def test_compute_pi_is_fraction_rated_malignant():
    assert corpus.compute_pi([1, 2, 4, 5]) == 0.5
    assert corpus.compute_pi([3, 3, 3, 3]) == 1.0
    assert corpus.compute_pi([1, 2]) == 0.0


# --- tiers ---

def test_tier_high_for_unanimous():
    assert corpus.assign_tier(0.0) == "high"
    assert corpus.assign_tier(1.0) == "high"


def test_tier_ambiguous_for_even_split():
    assert corpus.assign_tier(0.5) == "ambiguous"


def test_tier_contested_otherwise():
    assert corpus.assign_tier(0.25) == "contested"
    assert corpus.assign_tier(0.75) == "contested"
    assert corpus.assign_tier(2 / 3) == "contested"


# --- record building ---

def test_build_records_computes_pi_tier_and_nraters():
    scan = ScanParse(readers=[[rn(2)], [rn(2)], [rn(4)], [rn(4)]])
    records, _ = corpus.build_records([scan])
    assert len(records) == 1
    r = records[0]
    assert r["n_raters"] == 4
    assert r["pi"] == 0.5
    assert r["tier"] == "ambiguous"


def test_build_records_excludes_and_counts_under_min_readers():
    scan = ScanParse(readers=[[rn(3)], [rn(4)]])  # only 2 readers, min is 3
    records, exclusions = corpus.build_records([scan], min_readers=3)
    assert records == []
    assert exclusions.get("under_min_readers") == 1


def test_build_records_averages_vignette_features():
    scan = ScanParse(readers=[
        [rn(4, sub=4, spic=2, marg=3)],
        [rn(4, sub=2, spic=2, marg=5)],
        [rn(5, sub=3, spic=2, marg=4)],
    ])
    records, _ = corpus.build_records([scan])
    r = records[0]
    assert r["subtlety"] == 3.0        # mean(4,2,3)
    assert r["margin"] == 4.0          # mean(3,5,4)


def test_build_records_emits_mechanism_features():
    # malignancies 1 and 5 -> mean 3.0, extremity mean|m-3| = 2.0
    scan = ScanParse(readers=[[rn(1)], [rn(5)], [rn(1)]])
    records, _ = corpus.build_records([scan])
    r = records[0]
    assert r["mean_malignancy"] == (1 + 5 + 1) / 3
    assert abs(r["malignancy_extremity"] - (2 + 2 + 2) / 3) < 1e-9
    assert "extent" in r


def test_build_records_propagates_loader_drops():
    scan = ScanParse(readers=[[rn(3)], [rn(3)], [rn(3)]],
                     dropped={"no_malignancy": 2})
    _, exclusions = corpus.build_records([scan])
    assert exclusions.get("no_malignancy") == 2


# --- sampling ---

def test_sample_per_tier_caps_at_target_and_is_deterministic():
    df = pd.DataFrame({
        "id": range(10),
        "tier": ["high"] * 6 + ["ambiguous"] * 4,
        "pi": [1.0] * 6 + [0.5] * 4,
    })
    out1 = corpus.sample_per_tier(df, {"high": 3, "ambiguous": 10}, seed=1)
    out2 = corpus.sample_per_tier(df, {"high": 3, "ambiguous": 10}, seed=1)
    assert (out1["tier"] == "high").sum() == 3          # capped
    assert (out1["tier"] == "ambiguous").sum() == 4     # fewer than target -> all
    assert list(out1["id"]) == list(out2["id"])         # deterministic
