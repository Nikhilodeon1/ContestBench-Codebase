"""Tests for parse_response: JSON-first extraction with regex fallback."""

import math

from contestbench.eval import parse


def test_valid_json_object():
    ans, conf = parse.parse_response('{"answer": "Malignant", "confidence": 80}')
    assert ans == "malignant"
    assert conf == 0.80


def test_json_embedded_in_text():
    ans, conf = parse.parse_response('Here is my answer:\n{"answer":"Benign","confidence":45}')
    assert ans == "benign"
    assert conf == 0.45


def test_malformed_json_falls_back_to_regex():
    # not valid JSON, but has the legacy plaintext format
    ans, conf = parse.parse_response("Answer: Malignant\nConfidence: 65%")
    assert ans == "malignant"
    assert conf == 0.65


def test_none_input():
    ans, conf = parse.parse_response(None)
    assert ans is None
    assert math.isnan(conf)


def test_no_extractable_confidence_is_nan():
    ans, conf = parse.parse_response("I cannot determine malignancy from these features.")
    assert math.isnan(conf)


def test_json_confidence_out_of_range_is_nan():
    _, conf = parse.parse_response('{"answer":"Benign","confidence":250}')
    assert math.isnan(conf)
