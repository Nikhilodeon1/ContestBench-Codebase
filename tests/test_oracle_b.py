import numpy as np
from sklearn.model_selection import GroupKFold

from contestbench.analysis import oracle_b


def test_feature_sets_exclude_extent_from_vignette():
    assert "extent" not in oracle_b.VIGNETTE
    assert oracle_b.EXTENDED == oracle_b.VIGNETTE + ["extent"]
    assert "malignancy_extremity" not in oracle_b.EXTENDED and "n_raters" not in oracle_b.EXTENDED


def test_cv_predict_covers_all_rows_and_is_clipped():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)); y = np.clip(X[:, 0] * 0.3 + 0.5, 0, 1)
    g = np.repeat(np.arange(50), 4)
    p = oracle_b.cv_predict(X, y, g, "linear")
    assert not np.isnan(p).any() and p.min() >= 0 and p.max() <= 1


def test_group_folds_never_split_a_scan():
    g = np.repeat(np.arange(40), 5)
    for tr, te in GroupKFold(5, shuffle=True, random_state=1).split(np.zeros(len(g)), groups=g):
        assert not set(g[tr]) & set(g[te])


def test_l1_from_forecast_uses_implied_confidence():
    # forecast .2 -> implied answer Benign at c=.8; pi=.8 -> L1 0
    assert abs(oracle_b._l1_from_forecast([0.2], [0.8])) < 1e-12
