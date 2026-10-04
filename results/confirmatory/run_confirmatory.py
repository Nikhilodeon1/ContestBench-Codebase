"""Confirmatory collection (design frozen in results/confirmatory_preregistration.md).

One run per config, default settings, thinking disabled, 100-case calls in corpus order.
Usage: python results/confirmatory/run_confirmatory.py anthropic|gemini
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

D = Path("results/confirmatory")
PROV = D / "provenance.jsonl"
MODE = sys.argv[1]


def log(rec):
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(PROV, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def prompt_of(ci):
    return (D / "prompts" / f"batch_{ci:03d}.txt").read_text(encoding="utf-8")


if MODE == "anthropic":
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ClaudeKey"))
    CONFIGS = [("sonnet", "claude-sonnet-5"), ("opus", "claude-opus-4-8"),
               ("haiku", "claude-haiku-4-5-20251001")]
    for label, model in CONFIGS:
        for ci in range(1, 16):
            out = D / "raw" / f"{label}_{ci:02d}.txt"
            out.parent.mkdir(exist_ok=True)
            if out.exists() and out.stat().st_size > 50:
                continue
            prompt = prompt_of(ci)
            r = client.messages.create(model=model, max_tokens=8000, thinking={"type": "disabled"},
                                       messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in r.content if b.type == "text")
            out.write_text(text, encoding="utf-8")
            log({"label": label, "requested_model": model, "resolved_model": r.model, "thinking": "disabled",
                 "temperature": "default", "chunk": ci, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                 "n_answers": text.count('"probability"'), "input_tokens": r.usage.input_tokens,
                 "output_tokens": r.usage.output_tokens})
            print(label, ci, r.model, text.count('"probability"'), flush=True)
else:
    from google import genai
    from google.genai import types

    from contestbench.eval.batch_api import gemini_keys

    keys = gemini_keys()
    MODEL = "gemini-2.5-flash"
    served = False
    for ci in range(1, 16):
        out = D / "raw" / f"gemini_{ci:02d}.txt"
        out.parent.mkdir(exist_ok=True)
        if out.exists() and out.stat().st_size > 50:
            continue
        prompt = prompt_of(ci)
        resp = None
        for k in keys:  # one pass over the keys per call; no waiting loops
            try:
                c = genai.Client(api_key=k)
                resp = c.models.generate_content(
                    model=MODEL, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0, max_output_tokens=8000,
                                                       thinking_config=types.ThinkingConfig(thinking_budget=0)))
                break
            except Exception as e:  # noqa: BLE001
                last = str(e)[:150]
        if resp is None or not resp.text:
            log({"label": "gemini", "requested_model": MODEL, "chunk": ci, "status": "unavailable", "error": last})
            print("gemini unavailable at chunk", ci, last, flush=True)
            break
        out.write_text(resp.text, encoding="utf-8")
        um = resp.usage_metadata
        log({"label": "gemini", "requested_model": MODEL, "resolved_model": getattr(resp, "model_version", None),
             "thinking_budget": 0, "temperature": 0, "chunk": ci,
             "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
             "n_answers": resp.text.count('"probability"'),
             "thoughts_tokens": getattr(um, "thoughts_token_count", None)})
        print("gemini", ci, resp.text.count('"probability"'), flush=True)
        time.sleep(1)
print("DONE", MODE, flush=True)
