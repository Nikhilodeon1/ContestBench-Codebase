"""Load LIDC-IDRI annotation XML into per-reader nodule centroids + malignancy.

No DICOM is required or available: centroids come from the ROI ``edgeMap``
pixel coordinates and ``imageZposition`` carried in the XML itself.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from contestbench.config import NIH_NS


@dataclass
class ReaderNodule:
    nodule_id: str
    x: float
    y: float
    z: float
    malignancy: int
    subtlety: int | None = None
    spiculation: int | None = None
    margin: int | None = None
    extent: float = 0.0   # in-plane bounding-box diagonal (nodule size proxy)


@dataclass
class ScanParse:
    readers: list[list[ReaderNodule]]
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def n_readers(self) -> int:
        return len(self.readers)


def _text(node, tag):
    child = node.find(f"n:{tag}", NIH_NS)
    return child.text if child is not None else None


def _parse_nodule(nod) -> tuple[ReaderNodule | None, str | None]:
    """Return (ReaderNodule, None) or (None, drop_reason)."""
    char = nod.find("n:characteristics", NIH_NS)
    malig_txt = _text(char, "malignancy") if char is not None else None
    if malig_txt is None:
        return None, "no_malignancy"

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for roi in nod.findall("n:roi", NIH_NS):
        z = _text(roi, "imageZposition")
        if z is not None:
            zs.append(float(z))
        for em in roi.findall("n:edgeMap", NIH_NS):
            xc = _text(em, "xCoord")
            yc = _text(em, "yCoord")
            if xc is not None and yc is not None:
                xs.append(float(xc))
                ys.append(float(yc))
    if not xs:
        return None, "no_coords"

    def _feat(tag):
        v = _text(char, tag)
        return int(v) if v is not None else None

    nid = _text(nod, "noduleID") or ""
    extent = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    centroid = ReaderNodule(
        nodule_id=nid,
        x=sum(xs) / len(xs),
        y=sum(ys) / len(ys),
        z=(sum(zs) / len(zs)) if zs else 0.0,
        malignancy=int(malig_txt),
        subtlety=_feat("subtlety"),
        spiculation=_feat("spiculation"),
        margin=_feat("margin"),
        extent=extent,
    )
    return centroid, None


def parse_scan_root(root) -> ScanParse:
    readers: list[list[ReaderNodule]] = []
    dropped: dict[str, int] = {}
    for session in root.findall(".//n:readingSession", NIH_NS):
        nodules: list[ReaderNodule] = []
        for nod in session.findall("n:unblindedReadNodule", NIH_NS):
            parsed, reason = _parse_nodule(nod)
            if parsed is not None:
                nodules.append(parsed)
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
        readers.append(nodules)
    return ScanParse(readers=readers, dropped=dropped)


def parse_scan_string(xml: str) -> ScanParse:
    return parse_scan_root(ET.fromstring(xml))


def parse_scan_file(path) -> ScanParse:
    return parse_scan_root(ET.parse(path).getroot())
