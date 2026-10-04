"""Tests for LIDC XML loading: per-reader nodule centroids + malignancy."""

import textwrap

from contestbench.data import lidc_loader


def _scan_xml(sessions_xml: str) -> str:
    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <LidcReadMessage xmlns="http://www.nih.gov">
    {sessions_xml}
    </LidcReadMessage>
    """)


def _session(nodules_xml: str) -> str:
    return f"<readingSession>{nodules_xml}</readingSession>"


def _nodule(nid, malig, rois, subtlety=None, spiculation=None, margin=None):
    """rois: list of (z, [(x,y), ...])."""
    if malig is None:
        char = ""
    else:
        extra = ""
        if subtlety is not None:
            extra += f"<subtlety>{subtlety}</subtlety>"
        if spiculation is not None:
            extra += f"<spiculation>{spiculation}</spiculation>"
        if margin is not None:
            extra += f"<margin>{margin}</margin>"
        char = f"<characteristics><malignancy>{malig}</malignancy>{extra}</characteristics>"
    roi_xml = ""
    for z, pts in rois:
        edges = "".join(f"<edgeMap><xCoord>{x}</xCoord><yCoord>{y}</yCoord></edgeMap>"
                        for x, y in pts)
        roi_xml += f"<roi><imageZposition>{z}</imageZposition>{edges}</roi>"
    return f"<unblindedReadNodule><noduleID>{nid}</noduleID>{char}{roi_xml}</unblindedReadNodule>"


def test_centroid_is_mean_of_edgemaps_and_z():
    xml = _scan_xml(_session(_nodule("N1", 4, [(10.0, [(0, 0), (4, 0), (4, 4), (0, 4)])])))
    scan = lidc_loader.parse_scan_string(xml)
    assert scan.n_readers == 1
    nod = scan.readers[0][0]
    assert nod.malignancy == 4
    assert nod.x == 2.0        # mean of 0,4,4,0
    assert nod.y == 2.0        # mean of 0,0,4,4
    assert nod.z == 10.0


def test_centroid_averages_z_across_multiple_rois():
    xml = _scan_xml(_session(_nodule(
        "N1", 3, [(10.0, [(0, 0)]), (12.0, [(2, 2)])])))
    scan = lidc_loader.parse_scan_string(xml)
    nod = scan.readers[0][0]
    assert nod.z == 11.0       # mean of 10 and 12
    assert nod.x == 1.0        # mean of 0 and 2


def test_nodule_without_malignancy_is_dropped_and_counted():
    xml = _scan_xml(_session(_nodule("N1", None, [(10.0, [(0, 0)])])))
    scan = lidc_loader.parse_scan_string(xml)
    assert scan.readers[0] == []
    assert scan.dropped.get("no_malignancy") == 1


def test_nodule_without_coords_is_dropped_and_counted():
    xml = _scan_xml(_session(_nodule("N1", 5, [])))
    scan = lidc_loader.parse_scan_string(xml)
    assert scan.readers[0] == []
    assert scan.dropped.get("no_coords") == 1


def test_vignette_features_are_parsed():
    xml = _scan_xml(_session(_nodule(
        "N1", 4, [(10.0, [(0, 0)])], subtlety=5, spiculation=2, margin=3)))
    nod = lidc_loader.parse_scan_string(xml).readers[0][0]
    assert nod.subtlety == 5
    assert nod.spiculation == 2
    assert nod.margin == 3


def test_extent_is_bounding_box_diagonal():
    # points (0,0),(3,0),(3,4),(0,4) -> bbox 3 x 4 -> diagonal 5
    xml = _scan_xml(_session(_nodule(
        "N1", 4, [(10.0, [(0, 0), (3, 0), (3, 4), (0, 4)])])))
    nod = lidc_loader.parse_scan_string(xml).readers[0][0]
    assert nod.extent == 5.0


def test_extent_zero_for_single_point():
    xml = _scan_xml(_session(_nodule("N1", 4, [(10.0, [(7, 7)])])))
    nod = lidc_loader.parse_scan_string(xml).readers[0][0]
    assert nod.extent == 0.0


def test_missing_vignette_features_are_none():
    xml = _scan_xml(_session(_nodule("N1", 4, [(10.0, [(0, 0)])])))
    nod = lidc_loader.parse_scan_string(xml).readers[0][0]
    assert nod.subtlety is None


def test_separate_reading_sessions_become_separate_readers():
    xml = _scan_xml(
        _session(_nodule("N1", 2, [(10.0, [(0, 0)])]))
        + _session(_nodule("N1", 4, [(10.0, [(0, 0)])])))
    scan = lidc_loader.parse_scan_string(xml)
    assert scan.n_readers == 2
    assert scan.readers[0][0].malignancy == 2
    assert scan.readers[1][0].malignancy == 4
