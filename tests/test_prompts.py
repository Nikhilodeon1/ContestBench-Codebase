"""Tests for the radiologist vignette prompt."""

from contestbench.eval import prompts


def test_prompt_includes_feature_values():
    p = prompts.create_prompt(subtlety=4.2, spiculation=1.4, margin=4.6)
    assert "4.20" in p
    assert "1.40" in p
    assert "4.60" in p


def test_prompt_requests_json_answer_and_confidence_fields():
    p = prompts.create_prompt(subtlety=3.0, spiculation=3.0, margin=3.0)
    assert '"answer"' in p
    assert '"confidence"' in p
    assert "JSON" in p
