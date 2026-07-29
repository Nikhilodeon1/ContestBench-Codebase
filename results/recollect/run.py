"""Re-collect the primary panel via provenanced APIs (weekend provenance-close)."""
import os, sys, time, json, hashlib
sys.path.insert(0, ".")
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
from contestbench.eval.batch_api import _anthropic_batch, gemini_keys
from google import genai
from google.genai import types

CONFIGS = [
    ("haiku:standard", "anthropic", "claude-haiku-4-5-20251001", False),
    ("haiku:thinking", "anthropic", "claude-haiku-4-5-20251001", True),
    ("sonnet:standard", "anthropic", "claude-sonnet-5", False),
    ("sonnet:thinking", "anthropic", "claude-sonnet-5", True),
    ("opus:standard", "anthropic", "claude-opus-4-8", False),
    ("opus:thinking", "anthropic", "claude-opus-4-8", True),
    ("gemini:standard", "google", "gemini-2.5-flash", False),
    ("gemini:thinking", "google", "gemini-2.5-flash", True),
]
keys = gemini_keys(); gstate = {"i": 0}
d = Path("results/recollect")
prov = open(d / "provenance.jsonl", "a", encoding="utf-8")

# Log DeepSeek manual exception explicit check
ds_log = {"label": "deepseek:standard/thinking", "provider": "deepseek", "model": "deepseek-r1/v3",
          "route": "manual_exception", "api_available": False,
          "note": "No direct DeepSeek API key present; documented manual route exception per Task 2 standard",
          "timestamp": datetime.now(timezone.utc).isoformat()}
prov.write(json.dumps(ds_log) + "\n"); prov.flush()
phash = lambda p: hashlib.sha256(p.encode()).hexdigest()[:16]

def gemini_call(model, prompt, thinking):
    for _ in range(len(keys) * 2):
        i = gstate["i"]
        try:
            c = genai.Client(api_key=keys[i])
            cfg = dict(temperature=0, max_output_tokens=8000)
            if thinking:  # standard omits thinking_config (budget=0 is rejected)
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=2048)
            r = c.models.generate_content(model=model, contents=prompt,
                config=types.GenerateContentConfig(**cfg))
            return r.text
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "404")):
                gstate["i"] = (i + 1) % len(keys); continue
            print(f"  gemini error: {msg[:80]}", flush=True)
            return None
    return None

for label, ptype, model, thinking in CONFIGS:
    safe = label.replace(":", "-")
    for ci in range(1, 16):
        out = d / f"{safe}_{ci:02d}.txt"
        if out.exists() and out.stat().st_size > 50:
            continue
        prompt = (d / "prompts" / f"batch_{ci:03d}.txt").read_text(encoding="utf-8")
        t = None
        try:  # per-chunk guard: one failure never kills the run
            if ptype == "anthropic":
                mt = 16000 if thinking else 8000
                for attempt in range(3):
                    try:
                        t = _anthropic_batch(model, prompt, 0, mt, thinking=thinking); break
                    except Exception as e:
                        print(f"  {out.name} retry: {str(e)[:60]}", flush=True)
                        time.sleep(2 ** attempt + 1)
            else:
                t = gemini_call(model, prompt, thinking)
        except Exception as e:
            print(f"  {out.name} chunk error: {str(e)[:80]}", flush=True)
        if t:
            out.write_text(t, encoding="utf-8")
            prov.write(json.dumps({"label": label, "provider": ptype, "model": model,
                "thinking": thinking, "temperature": 0 if ptype == "gemini" else "default(deprecated)",
                "prompt_chunk": ci, "prompt_hash": phash(prompt),
                "n_answers": t.count('"confidence"'),
                "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n"); prov.flush()
            print(f"{out.name}: {t.count(chr(34) + 'confidence')}", flush=True)
        else:
            print(f"{out.name}: EMPTY", flush=True)
        time.sleep(1)
print("RECOLLECT DONE", flush=True)
