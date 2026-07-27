"""Anthropic API query function for the Claude capability x reasoning panel.

Uses the SAME standardized JSON ``create_prompt`` as every other model (Fix 1),
and runs through the real API with full provenance (Fix 2) — no manual/black-box
batch route. Extended thinking is toggled per spec; standard = thinking off.
Requires ANTHROPIC_API_KEY in the environment (.env).
"""

from __future__ import annotations

import os
import random
import time

import anthropic

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ClaudeKey")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY / ClaudeKey not set (add it to .env)")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def anthropic_query(model: str, params: dict, prompt: str, max_retries: int = 4) -> str | None:
    client = _get_client()
    if params.get("thinking"):
        budget = params.get("thinking_budget", 2000)
        kwargs = {
            "model": model,
            "max_tokens": budget + 512,          # must exceed the thinking budget
            "temperature": 1.0,                  # required when thinking is enabled
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        kwargs = {
            "model": model,
            "max_tokens": 512,
            "temperature": params.get("temperature", 0.0),
            "messages": [{"role": "user", "content": prompt}],
        }

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**kwargs)
            for block in resp.content:          # skip thinking blocks, take the text
                if block.type == "text":
                    return block.text
            return None
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"  anthropic_query failed ({model}): {e}")
                return None
            time.sleep(2 ** attempt + random.uniform(0, 1.0))
    return None
