"""Run a corpus x model-spec sweep, with caching, parsing, and provenance.

``query_fn(model, params, prompt) -> text`` is injected so the loop is testable
without any live API (the Groq adapter provides the real one).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from contestbench.eval import parse, prompts
from contestbench.eval.cache import DiskCache

QueryFn = Callable[[str, dict, str], str]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def run(
    corpus: pd.DataFrame,
    specs: list[dict],
    query_fn: QueryFn,
    cache: DiskCache,
) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in specs:
        label = spec["label"]
        model = spec["model"]
        params = spec.get("params", {})
        for _, r in corpus.iterrows():
            prompt = prompts.create_prompt(r["subtlety"], r["spiculation"], r["margin"])
            text = cache.get(model, params, prompt)
            if text is None:
                text = query_fn(model, params, prompt)
                if text is not None:
                    cache.put(model, params, prompt, text)
            answer, confidence = parse.parse_response(text)
            rows.append({
                "id": r["id"],
                "pi": r["pi"],
                "tier": r["tier"],
                "label": label,
                "model": model,
                "reasoning_effort": params.get("reasoning_effort"),
                "temperature": params.get("temperature"),
                "answer": answer,
                "confidence": confidence,
                "prompt_hash": _prompt_hash(prompt),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    return pd.DataFrame(rows)
