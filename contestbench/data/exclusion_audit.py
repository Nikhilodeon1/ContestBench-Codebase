"""Characterize what LIDC marks are excluded, by size proxy (supplementary).

Reviewer concern (benchmarks track): does dropping uncharacterized / <3mm marks
skew the corpus toward a particular malignancy profile? Dropped marks have no
malignancy rating by definition, so we characterize them by ROI extent instead:
a single-point ROI is a <3mm locus mark (uncharacterizable by LIDC design), a
multi-point ROI has a drawn contour. ``nonNodule`` marks are counted separately.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter

from contestbench.config import NIH_NS


def _roi_point_count(node) -> int:
    return sum(len(roi.findall("n:edgeMap", NIH_NS))
               for roi in node.findall("n:roi", NIH_NS))


def audit_scan_root(root) -> Counter:
    counts: Counter = Counter()
    for session in root.findall(".//n:readingSession", NIH_NS):
        for nod in session.findall("n:unblindedReadNodule", NIH_NS):
            char = nod.find("n:characteristics", NIH_NS)
            has_malig = char is not None and char.find("n:malignancy", NIH_NS) is not None
            if has_malig:
                counts["kept_characterized"] += 1
            elif _roi_point_count(nod) <= 1:
                counts["no_malig_locus"] += 1
            else:
                counts["no_malig_contour"] += 1
        counts["non_nodule"] += len(session.findall("n:nonNodule", NIH_NS))
    return counts


def audit_scan_string(xml: str) -> Counter:
    return audit_scan_root(ET.fromstring(xml))


def audit_scan_file(path) -> Counter:
    return audit_scan_root(ET.parse(path).getroot())
