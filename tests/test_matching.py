"""Tests for XML-native centroid matching across reading sessions."""

from contestbench.data.lidc_loader import ReaderNodule
from contestbench.data import matching


def rn(malig, x, y, z=0.0, nid="N"):
    return ReaderNodule(nodule_id=nid, x=x, y=y, z=z, malignancy=malig)


def test_colocated_nodules_across_two_readers_form_one_group():
    readers = [[rn(2, 100, 100)], [rn(4, 105, 102)]]  # within 30px, 3mm
    groups = matching.match_nodules(readers, xy_thresh=30, z_thresh=3)
    assert len(groups) == 1
    assert groups[0].n_readers == 2
    assert sorted(groups[0].malignancies) == [2, 4]


def test_far_apart_nodules_do_not_match():
    readers = [[rn(2, 100, 100)], [rn(4, 300, 300)]]  # >30px apart
    groups = matching.match_nodules(readers, xy_thresh=30, z_thresh=3)
    assert len(groups) == 2
    assert all(g.n_readers == 1 for g in groups)


def test_z_distance_beyond_threshold_prevents_match():
    readers = [[rn(2, 100, 100, z=0.0)], [rn(4, 100, 100, z=10.0)]]  # xy same, z far
    groups = matching.match_nodules(readers, xy_thresh=30, z_thresh=3)
    assert len(groups) == 2
    assert all(g.n_readers == 1 for g in groups)


def test_anchor_is_reader_with_most_nodules():
    # reader 1 has 2 nodules, both co-located with reader 0's single nodule region
    readers = [
        [rn(3, 100, 100)],
        [rn(4, 102, 101), rn(5, 500, 500)],
    ]
    groups = matching.match_nodules(readers, xy_thresh=30, z_thresh=3)
    # anchor = reader 1 (most nodules): 2 anchor nodules -> 2 groups
    assert len(groups) == 2
    matched = [g for g in groups if g.n_readers == 2]
    assert len(matched) == 1
    assert sorted(matched[0].malignancies) == [3, 4]


def test_each_reader_nodule_claimed_at_most_once():
    # two anchor nodules both near reader 0's single nodule; only closest claims it
    readers = [
        [rn(1, 100, 100)],
        [rn(2, 101, 100), rn(3, 120, 100)],  # both within 30px of (100,100)
    ]
    groups = matching.match_nodules(readers, xy_thresh=30, z_thresh=3)
    # reader-0 nodule can be claimed by only one anchor nodule
    total_reader0 = sum(g.n_readers - 1 for g in groups if g.n_readers > 1)
    assert total_reader0 == 1
