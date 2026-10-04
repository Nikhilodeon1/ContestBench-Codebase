"""Repeat-run noise floor: re-run sonnet:standard chunks 1-3 with the IDENTICAL prompts/params.

Original run: results/recollect/run.py -> _anthropic_batch(model, prompt, 0, 8000, thinking=False)
(claude-sonnet-5, thinking disabled, default temperature, max_tokens 8000, 100 cases per call).
This script repeats exactly that for all 15 chunks (chunks 1-3 were run first) so any difference is pure sampling noise.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
import anthropic

MODEL = "claude-haiku-4-5-20251001"
D = Path("results/recollect")
OUT = D / "noise_haiku"
OUT.mkdir(exist_ok=True)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ClaudeKey"))
for ci in range(1, 16):
    out = OUT / f"haiku-standard-rerun_{ci:02d}.txt"
    if out.exists() and out.stat().st_size > 50:
        continue
    prompt = (D / "prompts" / f"batch_{ci:03d}.txt").read_text(encoding="utf-8")
    r = client.messages.create(model=MODEL, max_tokens=8000, thinking={"type": "disabled"},
                               messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in r.content if b.type == "text")
    out.write_text(text, encoding="utf-8")
    rec = {"label": "haiku:standard-rerun", "requested_model": MODEL, "resolved_model": r.model,
           "thinking": "disabled", "temperature": "default(deprecated)", "chunk": ci,
           "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
           "n_answers": text.count('"confidence"'), "input_tokens": r.usage.input_tokens,
           "output_tokens": r.usage.output_tokens, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(D / "provenance_noise_haiku.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(ci, r.model, rec["n_answers"], flush=True)
print("NOISE DONE")
