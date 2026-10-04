"""Tests for the exclusion size-breakdown audit (supplementary appendix table)."""

import textwrap

from contestbench.data import exclusion_audit


def _scan(inner: str) -> str:
    return textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <LidcReadMessage xmlns="http://www.nih.gov">
    <readingSession>{inner}</readingSession>
    </LidcReadMessage>
    """)


def _unblinded(malig, n_points):
    char = f"<characteristics><malignancy>{malig}</malignancy></characteristics>" if malig else ""
    edges = "".join(f"<edgeMap><xCoord>{i}</xCoord><yCoord>{i}</yCoord></edgeMap>"
                    for i in range(n_points))
    return f"<unblindedReadNodule>{char}<roi><imageZposition>1</imageZposition>{edges}</roi></unblindedReadNodule>"


def _non_nodule():
    return "<nonNodule><roi><imageZposition>1</imageZposition><edgeMap><xCoord>5</xCoord><yCoord>5</yCoord></edgeMap></roi></nonNodule>"


def test_characterized_nodule_is_not_an_exclusion():
    counts = exclusion_audit.audit_scan_string(_scan(_unblinded(malig=4, n_points=6)))
    assert counts.get("kept_characterized") == 1
    assert counts.get("no_malig_locus", 0) == 0


def test_uncharacterized_single_point_is_locus():
    counts = exclusion_audit.audit_scan_string(_scan(_unblinded(malig=None, n_points=1)))
    assert counts.get("no_malig_locus") == 1


def test_uncharacterized_multipoint_is_contour():
    counts = exclusion_audit.audit_scan_string(_scan(_unblinded(malig=None, n_points=8)))
    assert counts.get("no_malig_contour") == 1


def test_non_nodule_markings_counted_separately():
    counts = exclusion_audit.audit_scan_string(_scan(_non_nodule()))
    assert counts.get("non_nodule") == 1
