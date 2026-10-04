"""Re-collect gemini standard/thinking on ONE pinned model ID with full per-call logging.

Fixes the confound in the original panel (standard = rolling alias gemini-flash-latest,
thinking = gemini-2.5-flash). Both conditions use MODEL; thinking is explicit
(budget 0 = off, 2048 = on) and the resolved model_version and thinking-token count are
logged for every call. Same prompts/chunks (100 cases) as the Claude panel.
Usage: python results/recollect/gemini_pinned_run.py [--probe]
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
from google import genai
from google.genai import types

from contestbench.eval.batch_api import gemini_keys

MODEL = "gemini-2.5-flash"
CONDS = [("standard", 0), ("thinking", 2048)]
D = Path("results/recollect")
OUT = D / "pinned"
OUT.mkdir(exist_ok=True)
PROV = D / "provenance_pinned.jsonl"
keys = gemini_keys()
state = {"i": 0}


def call(prompt, budget):
    for _ in range(len(keys) * 2):
        i = state["i"]
        try:
            c = genai.Client(api_key=keys[i])
            cfg = types.GenerateContentConfig(
                temperature=0, max_output_tokens=8000,
                thinking_config=types.ThinkingConfig(thinking_budget=budget))
            return c.models.generate_content(model=MODEL, contents=prompt, config=cfg)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                state["i"] = (i + 1) % len(keys)
                time.sleep(2)
                continue
            raise
    return None


probe = "--probe" in sys.argv
for cond, budget in CONDS:
    for ci in range(1, 16):
        out = OUT / f"gemini-{cond}_{ci:02d}.txt"
        if out.exists() and out.stat().st_size > 50:
            continue
        prompt = (D / "prompts" / f"batch_{ci:03d}.txt").read_text(encoding="utf-8")
        r = call(prompt, budget)
        if r is None or not r.text:
            print(f"{cond} chunk {ci}: no output", flush=True)
            continue
        um = r.usage_metadata
        rec = {"label": f"gemini:{cond}", "requested_model": MODEL,
               "resolved_model_version": getattr(r, "model_version", None),
               "thinking_budget": budget, "temperature": 0, "chunk": ci, "chunk_size": 100,
               "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
               "n_answers": r.text.count('"confidence"'),
               "thoughts_tokens": getattr(um, "thoughts_token_count", None),
               "prompt_tokens": getattr(um, "prompt_token_count", None),
               "output_tokens": getattr(um, "candidates_token_count", None),
               "timestamp": datetime.now(timezone.utc).isoformat()}
        out.write_text(r.text, encoding="utf-8")
        with open(PROV, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(cond, ci, rec["resolved_model_version"], "thoughts", rec["thoughts_tokens"],
              "answers", rec["n_answers"], flush=True)
        if probe:
            sys.exit(0)
        time.sleep(1)
print("PINNED DONE", flush=True)
