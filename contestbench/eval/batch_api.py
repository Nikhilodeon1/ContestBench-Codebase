"""Send the SAME batch prompt files through provider APIs (secondary panel).

The Claude panel was run manually on batched JSON prompts. To make cross-vendor
PAD comparisons valid, Gemini and gpt-oss must see the *identical* batched prompt
rather than single-call prompts — but via real APIs, so temperature and model
string are logged (Fix 2). One API call per batch file.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path


def _groq_batch(model: str, prompt: str, temperature: float, max_tokens: int) -> str | None:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _gemini_batch(model: str, prompt: str, temperature: float, max_tokens: int,
                  api_key: str | None = None) -> str | None:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens),
    )
    return resp.text


def gemini_keys() -> list[str]:
    """All GEMINI_API_KEY[_N] values present in the environment, in order."""
    keys = []
    for name in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
                 "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"]:
        v = os.getenv(name)
        if v:
            keys.append(v)
    return keys


def gemini_rotating(model: str, prompt: str, keys: list[str], state: dict,
                    temperature: float, max_tokens: int) -> str | None:
    """Call Gemini, advancing through keys on quota/rate errors.

    ``state['i']`` is the current key index, carried across calls so a run keeps
    using the last working key. Returns None only when ALL keys are exhausted.
    """
    n = len(keys)
    for _ in range(n):
        i = state["i"]
        try:
            return _gemini_batch(model, prompt, temperature, max_tokens, api_key=keys[i])
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")) or "quota" in msg.lower():
                state["i"] = (i + 1) % n
                print(f"  key #{i+1} unavailable ({msg[:40]}) -> key #{state['i']+1}")
                continue
            raise
    return None  # all keys exhausted


def batch_query(provider: str, model: str, prompt: str, temperature: float = 0.0,
                max_tokens: int = 16384, max_retries: int = 3) -> str | None:
    fn = {"groq": _groq_batch, "gemini": _gemini_batch}[provider]
    for attempt in range(max_retries):
        try:
            return fn(model, prompt, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"  batch_query failed ({provider}/{model}): {e}")
                return None
            time.sleep(2 ** attempt + random.uniform(0, 1.0))
    return None


def run_batch_files(provider: str, model: str, label: str, batch_dir,
                    temperature: float = 0.0, delay_s: float = 0.0,
                    max_tokens: int = 16384, out_dir=None) -> list[Path]:
    """Send each batch_*.txt through the API; write replies as <label>_N.txt.

    ``delay_s`` paces requests for providers with tokens-per-minute caps.
    """
    batch_dir = Path(batch_dir)
    out_dir = Path(out_dir) if out_dir else batch_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    safe = label.replace(":", "-")
    files = sorted(batch_dir.glob("batch_*.txt"))
    for i, p in enumerate(files):
        idx = p.stem.split("_")[-1]
        out = out_dir / f"{safe}_{idx}.txt"
        if out.exists() and out.stat().st_size > 0:
            out_paths.append(out)
            continue
        text = batch_query(provider, model, p.read_text(encoding="utf-8"),
                           temperature, max_tokens)
        if text:
            out.write_text(text, encoding="utf-8")
            out_paths.append(out)
            print(f"  {p.name} -> {out.name} ({text.count(chr(34) + 'id' + chr(34))} answers)")
        if delay_s and i < len(files) - 1:
            time.sleep(delay_s)
    return out_paths
