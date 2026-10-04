"""Tests for parsing model output into answer + confidence."""

import math

from contestbench.eval import parse


def test_confidence_percent_to_unit_interval():
    assert parse.extract_confidence("Answer: Benign\nConfidence: 85%") == 0.85


def test_confidence_none_input_is_nan():
    assert math.isnan(parse.extract_confidence(None))


def test_confidence_missing_is_nan():
    assert math.isnan(parse.extract_confidence("Answer: Benign"))


def test_confidence_out_of_range_is_nan():
    assert math.isnan(parse.extract_confidence("Confidence: 250%"))


def test_extract_answer_malignant_and_benign():
    assert parse.extract_answer("Answer: Malignant\nConfidence: 60%") == "malignant"
    assert parse.extract_answer("Answer: Benign") == "benign"


def test_extract_answer_missing_is_none():
    assert parse.extract_answer("Confidence: 60%") is None
