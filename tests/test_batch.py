"""Tests for batch file export/import (file-in/file-out Gemini workflow)."""

import math

import pandas as pd

from contestbench.eval import batch


def _corpus():
    return pd.DataFrame({
        "id": ["3:12", "3:15", "7:0"],
        "pi": [0.5, 1.0, 0.25],
        "tier": ["ambiguous", "high", "contested"],
        "subtlety": [3.5, 4.0, 2.0],
        "spiculation": [2.0, 1.0, 3.0],
        "margin": [4.0, 5.0, 3.5],
    })


# --- export ---

def test_export_single_file_contains_header_and_all_cases(tmp_path):
    paths = batch.write_batch_files(_corpus(), tmp_path, chunk_size=0)
    assert len(paths) == 1
    text = paths[0].read_text(encoding="utf-8")
    assert "Malignant" in text                      # instruction header
    assert '"answer"' in text and '"confidence"' in text  # standardized JSON format
    for cid in ["3:12", "3:15", "7:0"]:
        assert cid in text
    assert "3.50" in text and "2.00" in text  # feature values


def test_parse_json_lines_output():
    text = ('{"id": "3:12", "answer": "Benign", "confidence": 55}\n'
            '{"id": "3:15", "answer": "Malignant", "confidence": 80}\n')
    df = batch.parse_batch_text(text)
    row = df.set_index("id").loc["3:12"]
    assert row["answer"] == "benign"
    assert row["confidence"] == 0.55
    assert df.set_index("id").loc["3:15"]["confidence"] == 0.80


def test_parse_json_lines_tolerates_prose():
    text = ('Here are my assessments:\n'
            '{"id": "3:12", "answer": "Benign", "confidence": 55}\n'
            'and the second one:\n'
            '{"id": "3:15", "answer": "Malignant", "confidence": 80}\n')
    df = batch.parse_batch_text(text)
    assert set(df["id"]) == {"3:12", "3:15"}


def test_export_chunks_split_cases(tmp_path):
    paths = batch.write_batch_files(_corpus(), tmp_path, chunk_size=2)
    assert len(paths) == 2  # 3 cases -> 2 + 1
    combined = "".join(p.read_text(encoding="utf-8") for p in paths)
    for cid in ["3:12", "3:15", "7:0"]:
        assert cid in combined


# --- import / parse ---

def test_parse_wellformed_lines():
    text = "3:12 | Benign | 55\n3:15 | Malignant | 80\n"
    df = batch.parse_batch_text(text)
    row = df.set_index("id").loc["3:12"]
    assert row["answer"] == "benign"
    assert row["confidence"] == 0.55
    assert df.set_index("id").loc["3:15"]["confidence"] == 0.80


def test_parse_tolerates_prose_and_junk_lines():
    text = ("Here are my assessments:\n"
            "3:12 | Benign | 55\n"
            "(note: 3:15 was difficult)\n"
            "3:15 , Malignant , 80\n"      # comma delimiter
            "Thank you.\n")
    df = batch.parse_batch_text(text)
    assert set(df["id"]) == {"3:12", "3:15"}


def test_parse_out_of_range_confidence_is_nan():
    df = batch.parse_batch_text("7:0 | Benign | 250\n")
    assert math.isnan(df.iloc[0]["confidence"])


def test_parse_empty_returns_empty_frame():
    df = batch.parse_batch_text("no parseable content here\n")
    assert len(df) == 0
