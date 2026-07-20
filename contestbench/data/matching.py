"""Match nodules across reading sessions by centroid proximity (XML-native).

Greedy, anchored on the reader with the most nodules. Each other reader's
nodule is claimed by at most one anchor nodule (nearest wins). Returns every
group including singletons; tier/`MIN_READERS` filtering happens downstream so
this stays a pure, testable geometric step.
"""

from __future__ import annotations

from dataclasses import dataclass

from contestbench.config import MATCH_XY_THRESH, MATCH_Z_THRESH
from contestbench.data.lidc_loader import ReaderNodule


@dataclass
class MatchedGroup:
    members: list[ReaderNodule]

    @property
    def n_readers(self) -> int:
        return len(self.members)

    @property
    def malignancies(self) -> list[int]:
        return [m.malignancy for m in self.members]

    @property
    def centroid(self) -> tuple[float, float, float]:
        anchor = self.members[0]
        return (anchor.x, anchor.y, anchor.z)


def _xy_dist(a: ReaderNodule, b: ReaderNodule) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def match_nodules(
    readers: list[list[ReaderNodule]],
    xy_thresh: float = MATCH_XY_THRESH,
    z_thresh: float = MATCH_Z_THRESH,
) -> list[MatchedGroup]:
    if not readers:
        return []

    anchor_idx = max(range(len(readers)), key=lambda i: len(readers[i]))
    used: list[set[int]] = [set() for _ in readers]

    groups: list[MatchedGroup] = []
    for anchor in readers[anchor_idx]:
        members = [anchor]
        for ri, reader in enumerate(readers):
            if ri == anchor_idx:
                continue
            best_k, best_d = None, float("inf")
            for k, cand in enumerate(reader):
                if k in used[ri]:
                    continue
                if abs(cand.z - anchor.z) > z_thresh:
                    continue
                d = _xy_dist(anchor, cand)
                if d <= xy_thresh and d < best_d:
                    best_k, best_d = k, d
            if best_k is not None:
                used[ri].add(best_k)
                members.append(reader[best_k])
        groups.append(MatchedGroup(members=members))

    # unmatched nodules from non-anchor readers become their own singletons
    for ri, reader in enumerate(readers):
        if ri == anchor_idx:
            continue
        for k, cand in enumerate(reader):
            if k not in used[ri]:
                groups.append(MatchedGroup(members=[cand]))

    return groups
