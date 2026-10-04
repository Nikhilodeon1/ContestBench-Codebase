import numpy as np

from contestbench.analysis import final_round as F


def test_case_floor_values():
    assert abs(F.case_floor(0.5, 4) - 0.25 / 3) < 1e-12
    assert abs(F.case_floor(0.0, 3)) < 1e-12 and abs(F.case_floor(1.0, 4)) < 1e-12


def test_case_floor_is_unbiased_for_vote_noise():
    rng = np.random.default_rng(0)
    rho = rng.beta(0.6, 0.33, 400_000)
    n = np.where(rng.random(rho.size) < 0.36, 3, 4)
    fh = rng.binomial(n, rho) / n
    assert abs(F.case_floor(fh, n).mean() - (rho * (1 - rho) / n).mean()) < 5e-4


def test_pair_splits_are_the_three_partitions():
    seen = {frozenset([frozenset(a), frozenset(b)]) for a, b in F.PAIR_SPLITS}
    assert len(seen) == 3
    for a, b in F.PAIR_SPLITS:
        assert sorted(a + b) == [0, 1, 2, 3]


def test_empty_probability_file_has_float_dtype(tmp_path):
    import pandas as pd

    from contestbench.analysis import revision_b as R
    p = tmp_path / "refused.txt"
    p.write_text("I'm sorry, but I can't fulfill that request.", encoding="utf-8")
    empty = R._parse_probability_file(p)
    ok = tmp_path / "ok.txt"
    ok.write_text('{"id": "1:0", "probability": 40}\n{"id": "1:2", "probability": 70}', encoding="utf-8")
    merged = pd.concat([empty, R._parse_probability_file(ok)])
    assert merged["p_hat"].dtype == float and len(merged) == 2
