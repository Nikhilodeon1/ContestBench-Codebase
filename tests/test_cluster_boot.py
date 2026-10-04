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


def test_nested_bootstrap_keeps_scans_whole_within_calls():
    calls = np.repeat([0, 1, 2], 20)
    scans = np.repeat(np.arange(15), 4)
    for idx in stats.nested_cluster_indices(calls, scans, 20, 3):
        for s in np.unique(scans[idx]):
            assert (scans[idx] == s).sum() % (scans == s).sum() == 0


def test_nested_ci_wider_than_scan_ci_when_calls_drift():
    rng = np.random.default_rng(1)
    calls = np.repeat(np.arange(10), 30)
    scans = np.repeat(np.arange(100), 3)
    y = np.repeat(rng.normal(size=10), 30) + 0.1 * rng.normal(size=300)
    a = [y[i].mean() for i in stats.scan_cluster_indices(scans, 400, 1)]
    b = [y[i].mean() for i in stats.nested_cluster_indices(calls, scans, 400, 1)]
    assert np.std(b) > 1.5 * np.std(a)
