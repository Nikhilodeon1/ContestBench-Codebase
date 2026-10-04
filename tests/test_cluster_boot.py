import numpy as np

from contestbench.metrics import stats


def test_scan_cluster_indices_keep_scans_whole():
    groups = np.array([0, 0, 0, 1, 1, 2])
    for idx in stats.scan_cluster_indices(groups, 20, 1):
        for g in np.unique(groups[idx]):
            # each resampled scan appears a whole multiple of its size
            assert (groups[idx] == g).sum() % (groups == g).sum() == 0


def test_cluster_ci_wider_than_iid_for_clustered_data():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(100), 5)
    y = np.repeat(rng.normal(size=100), 5) + 0.1 * rng.normal(size=500)
    cl = [y[i].mean() for i in stats.scan_cluster_indices(g, 400, 2)]
    iid = [y[rng.integers(0, 500, 500)].mean() for _ in range(400)]
    assert np.std(cl) > 1.5 * np.std(iid)
