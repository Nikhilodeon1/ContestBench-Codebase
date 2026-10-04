"""Task D (plan: results/analysis_plan_final.md): confirmatory P(>=3) prompt on three hosted open-weight models.

Verifies the frozen prompt hashes first. One run per model, default sampling (temperature unset), 100-case calls in
corpus order, reasoning minimised where the API allows, resolved model id and token usage logged per call.
Usage: python results/confirmatory/run_extra_models.py
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
from groq import Groq

D = Path("results/confirmatory")
RAW = D / "raw_extra"
RAW.mkdir(exist_ok=True)
PROV = D / "provenance_extra.jsonl"

# 1) verify frozen prompt hashes before any call
frozen = {}
for line in Path("results/frozen_hashes.txt").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^([0-9a-f]{64}) \*(results/confirmatory/prompts/batch_\d+\.txt)$", line.strip())
    if m:
        frozen[m.group(2)] = m.group(1)
assert len(frozen) == 15, f"expected 15 frozen prompt hashes, found {len(frozen)}"
for rel, h in frozen.items():
    got = hashlib.sha256((Path(rel)).read_bytes()).hexdigest()
    assert got == h, f"prompt hash mismatch for {rel}"
print("frozen prompt hashes verified (15/15)", flush=True)

MODELS = [("gptoss120b", "openai/gpt-oss-120b", {"reasoning_effort": "low"}),
          ("qwen38_27b", "qwen/qwen3.8-27b", {"reasoning_effort": "none"}),
          ("gptoss20b", "openai/gpt-oss-20b", {"reasoning_effort": "low"})]
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def log(rec):
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(PROV, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def call(model, prompt, extra):
    kw = dict(extra)
    for attempt in range(12):
        try:
            return client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                  max_tokens=8000, **kw), kw
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "reasoning_effort" in msg and "reasoning_effort" in kw and ("400" in msg or "invalid" in msg.lower()):
                kw.pop("reasoning_effort")  # parameter not supported for this model: fall back, logged below
                continue
            if "429" in msg or "rate" in msg.lower():
                wait = 30
                m = re.search(r"try again in ([0-9.]+)s", msg)
                if m:
                    wait = min(120, float(m.group(1)) + 2)
                print(f"  rate limited, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise
    return None, kw


for label, model, extra in MODELS:
    for ci in range(1, 16):
        out = RAW / f"{label}_{ci:02d}.txt"
        if out.exists() and out.stat().st_size > 50:
            continue
        prompt = (D / "prompts" / f"batch_{ci:03d}.txt").read_text(encoding="utf-8")
        resp, used_kw = call(model, prompt, extra)
        if resp is None:
            log({"label": label, "requested_model": model, "chunk": ci, "status": "failed"})
            print(label, ci, "FAILED", flush=True)
            continue
        text = resp.choices[0].message.content or ""
        out.write_text(text, encoding="utf-8")
        u = resp.usage
        rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
        log({"label": label, "requested_model": model, "resolved_model": resp.model, "chunk": ci,
             "settings": {"temperature": "default", "max_tokens": 8000, **used_kw},
             "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
             "n_answers": text.count('"probability"'), "prompt_tokens": u.prompt_tokens,
             "completion_tokens": u.completion_tokens, "reasoning_tokens": rt})
        print(label, ci, resp.model, text.count('"probability"'), flush=True)
        time.sleep(2)
print("DONE extra models", flush=True)
