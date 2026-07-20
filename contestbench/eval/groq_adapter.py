"""Groq query function for gpt-oss models, with retry/backoff.

gpt-oss exposes a ``reasoning_effort`` knob (low/medium/high) that Groq passes
through -- this is the within-model lever for novelty claim 3. Requires
GROQ_API_KEY in the environment (.env).
"""

from __future__ import annotations

import os
import random
import time

from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set (add it to .env)")
        _client = Groq(api_key=key)
    return _client


def groq_query(model: str, params: dict, prompt: str, max_retries: int = 4) -> str | None:
    client = _get_client()
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params.get("temperature", 0.0),
        # force a JSON object so the confidence field can't be omitted
        "response_format": {"type": "json_object"},
    }
    if params.get("reasoning_effort"):
        kwargs["reasoning_effort"] = params["reasoning_effort"]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - surface after retries
            if attempt == max_retries - 1:
                print(f"  groq_query failed ({model}): {e}")
                return None
            time.sleep(2 ** attempt + random.uniform(0, 1.0))
    return None
