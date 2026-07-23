"""Build the ContestBench corpus from LIDC scans.

Pipeline: parsed scans -> match nodules across readers -> keep groups with
>= MIN_READERS -> physician agreement pi -> tier -> per-tier sample. Every
dropped raw annotation is accounted for in the exclusions log (spec 4.1).
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from contestbench import config
from contestbench.data import lidc_loader, matching
from contestbench.data.lidc_loader import ScanParse
from contestbench.data.matching import MatchedGroup


def compute_pi(malignancies: list[int]) -> float:
    """Fraction of readers rating malignancy >= MALIGNANCY_POSITIVE."""
    if not malignancies:
        return float("nan")
    votes = sum(1 for m in malignancies if m >= config.MALIGNANCY_POSITIVE)
    return votes / len(malignancies)


def assign_tier(pi: float) -> str:
    """Tier from discrete pi (spec 4.1): high={0,1}, ambiguous={.5}, else contested."""
    if math.isclose(pi, 0.0) or math.isclose(pi, 1.0):
        return config.TIER_HIGH
    if math.isclose(pi, 0.5):
        return config.TIER_AMBIGUOUS
    return config.TIER_CONTESTED


def _mean_feature(group: MatchedGroup, attr: str) -> float:
    vals = [getattr(m, attr) for m in group.members if getattr(m, attr) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def build_records(
    scans: Iterable[ScanParse],
    min_readers: int = config.MIN_READERS,
) -> tuple[list[dict], dict]:
    """Return (records, exclusions).

    exclusions aggregates loader drop reasons plus 'under_min_readers'
    (matched groups with too few readers to score).
    """
    records: list[dict] = []
    exclusions: Counter = Counter()

    for scan_idx, scan in enumerate(scans):
        for reason, count in scan.dropped.items():
            exclusions[reason] += count

        groups = matching.match_nodules(scan.readers)
        for grp_idx, group in enumerate(groups):
            if group.n_readers < min_readers:
                exclusions["under_min_readers"] += 1
                continue
            pi = compute_pi(group.malignancies)
            x, y, z = group.centroid
            maligs = group.malignancies
            records.append({
                "id": f"{scan_idx}:{grp_idx}",
                "scan_idx": scan_idx,
                "n_raters": group.n_readers,
                "pi": pi,
                "tier": assign_tier(pi),
                "subtlety": _mean_feature(group, "subtlety"),
                "spiculation": _mean_feature(group, "spiculation"),
                "margin": _mean_feature(group, "margin"),
                "extent": _mean_feature(group, "extent"),
                "mean_malignancy": sum(maligs) / len(maligs),
                # how far from the 'uncertain' midpoint raters sat, on average
                "malignancy_extremity": sum(abs(m - 3) for m in maligs) / len(maligs),
                "centroid_x": x,
                "centroid_y": y,
                "centroid_z": z,
            })

    return records, dict(exclusions)


def sample_per_tier(df: pd.DataFrame, targets: dict[str, int], seed: int) -> pd.DataFrame:
    """Sample up to targets[tier] rows per tier (all rows if fewer). Deterministic."""
    parts = []
    for tier, group in df.groupby("tier", sort=True):
        target = targets.get(tier)
        if target is not None and len(group) > target:
            group = group.sample(n=target, random_state=seed)
        parts.append(group)
    return pd.concat(parts).sort_index()


def build_corpus_from_files(
    xml_paths: Iterable[Path],
    min_readers: int = config.MIN_READERS,
) -> tuple[pd.DataFrame, dict]:
    """Parse XML files -> records DataFrame + exclusions."""
    scans = []
    for p in xml_paths:
        try:
            scans.append(lidc_loader.parse_scan_file(p))
        except Exception:
            # unparseable XML is itself an exclusion category
            scans.append(ScanParse(readers=[], dropped={"unparseable_xml": 1}))
    records, exclusions = build_records(scans, min_readers=min_readers)
    return pd.DataFrame(records), exclusions
