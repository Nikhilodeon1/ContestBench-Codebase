"""Tests for the disk cache keyed by (model, params, prompt)."""

from contestbench.eval.cache import DiskCache


def test_get_returns_none_on_miss(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("gpt-oss-20b", {"reasoning_effort": "low"}, "prompt A") is None


def test_put_then_get_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("gpt-oss-20b", {"reasoning_effort": "low"}, "prompt A", "response A")
    assert cache.get("gpt-oss-20b", {"reasoning_effort": "low"}, "prompt A") == "response A"


def test_key_is_sensitive_to_params_and_prompt(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("gpt-oss-20b", {"reasoning_effort": "low"}, "prompt A", "resp low")
    # different reasoning_effort -> different key -> miss
    assert cache.get("gpt-oss-20b", {"reasoning_effort": "high"}, "prompt A") is None
    # different prompt -> miss
    assert cache.get("gpt-oss-20b", {"reasoning_effort": "low"}, "prompt B") is None


def test_persists_across_instances(tmp_path):
    DiskCache(tmp_path).put("m", {}, "p", "r")
    assert DiskCache(tmp_path).get("m", {}, "p") == "r"
