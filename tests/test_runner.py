"""Tests for the eval runner: sweep expansion, parsing, caching, provenance."""

import pandas as pd

from contestbench.eval.cache import DiskCache
from contestbench.eval import runner


def _corpus():
    return pd.DataFrame({
        "id": ["a", "b"],
        "pi": [0.5, 1.0],
        "tier": ["ambiguous", "high"],
        "subtlety": [3.0, 4.0],
        "spiculation": [2.0, 1.0],
        "margin": [4.0, 5.0],
    })


def test_run_parses_answer_and_confidence(tmp_path):
    specs = [{"label": "oss20-low", "model": "m20",
              "params": {"reasoning_effort": "low", "temperature": 0.0}}]

    def query_fn(model, params, prompt):
        return "Answer: Malignant\nConfidence: 70%"

    df = runner.run(_corpus(), specs, query_fn, DiskCache(tmp_path))
    assert len(df) == 2
    assert set(df["confidence"]) == {0.70}
    assert set(df["answer"]) == {"malignant"}
    assert set(df["label"]) == {"oss20-low"}
    assert set(df["reasoning_effort"]) == {"low"}


def test_sweep_expands_specs_across_rows(tmp_path):
    specs = [
        {"label": "lo", "model": "m", "params": {"reasoning_effort": "low"}},
        {"label": "hi", "model": "m", "params": {"reasoning_effort": "high"}},
    ]
    df = runner.run(_corpus(), specs, lambda *_: "Answer: Benign\nConfidence: 40%",
                    DiskCache(tmp_path))
    assert len(df) == 4  # 2 rows x 2 specs
    assert set(df["reasoning_effort"]) == {"low", "high"}


def test_cached_calls_are_not_repeated(tmp_path):
    specs = [{"label": "x", "model": "m", "params": {"reasoning_effort": "low"}}]
    calls = {"n": 0}

    def query_fn(model, params, prompt):
        calls["n"] += 1
        return "Answer: Benign\nConfidence: 30%"

    cache = DiskCache(tmp_path)
    runner.run(_corpus(), specs, query_fn, cache)
    assert calls["n"] == 2
    runner.run(_corpus(), specs, query_fn, cache)  # second run: all cached
    assert calls["n"] == 2  # no new calls


def test_provenance_columns_present(tmp_path):
    specs = [{"label": "x", "model": "m20",
              "params": {"reasoning_effort": "low", "temperature": 0.0}}]
    df = runner.run(_corpus(), specs, lambda *_: "Answer: Benign\nConfidence: 30%",
                    DiskCache(tmp_path))
    for col in ["id", "pi", "tier", "model", "reasoning_effort", "prompt_hash"]:
        assert col in df.columns
