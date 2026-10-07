import numpy as np
import pandas as pd

from contestbench.analysis import informed as A
from contestbench.eval import informed as E


def _corpus(n=250):
    return pd.DataFrame({"id": [f"{i // 3}:{i % 3}" for i in range(n)], "subtlety": 3.0, "spiculation": 2.5, "margin": 4.0})


def test_prompts_chunked_in_order_and_distinct(tmp_path):
    c = _corpus()
    pa = E.write_prompts(c, tmp_path, "scales")
    pb = E.write_prompts(c, tmp_path, "scales_rate")
    assert len(pa) == len(pb) == 3
    ta, tb = pa[0].read_text(), pb[0].read_text()
    assert "LESS suspicious" in ta and "65%" not in ta
    assert "65%" in tb and "3 or higher" in tb
    assert ta.count(" | subtlety ") == 100 and pa[2].read_text().count(" | subtlety ") == 50
    assert "0:0 | subtlety 3.00" in ta


def test_prediction_scoring_rules():
    t = pd.DataFrame([
        {"threshold": 3, "config": "sonnet:standard", "cond": "scales", "gap_new": 0.05, "gap_new_lo": 0.02, "gap_new_hi": 0.08, "gap_orig": 0.052},
        {"threshold": 3, "config": "sonnet:standard", "cond": "scales_rate", "gap_new": 0.02, "gap_new_lo": -0.01, "gap_new_hi": 0.04, "gap_orig": 0.052},
    ])
    s = A.score_predictions(t).set_index("prediction")["held"]
    assert s["P1_scales_still_worse"] and s["P2_rate_halves_gap"] and not s["P3_prompt_dependent"]
