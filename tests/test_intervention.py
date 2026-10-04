"""Tests for the prompting-intervention analysis (Fix I)."""

import pandas as pd

from contestbench.analysis import intervention


def test_leak_check_flags_exemplar_in_eval():
    eval_ids = {"1:0", "2:0", "180:3"}          # 180:3 is a held-out exemplar
    exemplars = {"180:3", "341:3"}
    leaked = intervention.leaked_exemplars(eval_ids, exemplars)
    assert leaked == {"180:3"}


def test_compare_aligns_to_shared_ids():
    base = pd.DataFrame({"id": ["a", "b", "c"], "pi_agree": [0.5, 0.75, 1.0],
                         "confidence": [0.9, 0.9, 0.9], "tier": ["ambiguous", "contested", "high"]})
    inv = pd.DataFrame({"id": ["a", "b"], "pi_agree": [0.5, 0.75],
                        "confidence": [0.6, 0.7], "tier": ["ambiguous", "contested"]})
    res = intervention.compare(base, inv)
    assert res["n"] == 2                          # only shared ids
    assert res["pad_after"] < res["pad_before"]  # lower confidence closer to pi here
