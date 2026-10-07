"""Collect the informed-prompt conditions (rule: results/informed/prespec.md).

python results/informed/run_informed.py --dry                       # write prompts + hashes, no API
python results/informed/run_informed.py --go --conds scales scales_rate --configs sonnet:standard opus:standard
Raw replies -> results/informed/raw/<cfg>__<cond>_<chunk>.txt ; provenance -> results/informed/provenance.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.getcwd())
import pandas as pd
from dotenv import load_dotenv

from contestbench import config
from contestbench.eval import informed
from contestbench.eval.batch import parse_batch_text

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
D = Path("results/informed")
MODELS = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-4-8"}

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
ap.add_argument("--go", action="store_true")
ap.add_argument("--conds", nargs="+", default=list(informed.CONDITIONS))
ap.add_argument("--configs", nargs="+", default=["sonnet:standard", "opus:standard"])
a = ap.parse_args()
assert a.dry or a.go, "pass --dry or --go"

corpus = pd.read_parquet(config.CORPUS_PARQUET).reset_index(drop=True)
hashes = {}
for cond in a.conds:
    for p in informed.write_prompts(corpus, D / "prompts", cond):
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(D / "prompt_hashes.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8", newline="\n")
print(f"{len(hashes)} prompt files written; hashes -> {D / 'prompt_hashes.json'}")
if a.dry:
    sys.exit(0)

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ClaudeKey"))
(D / "raw").mkdir(exist_ok=True)
for cfg in a.configs:
    fam, mode = cfg.split(":")
    for cond in a.conds:
        for ci in range(1, len(corpus) // 100 + 2):
            pf = D / "prompts" / f"{cond}_{ci:03d}.txt"
            if not pf.exists():
                break
            out = D / "raw" / f"{fam}-{mode}__{cond}_{ci:03d}.txt"
            want = min(100, len(corpus) - (ci - 1) * 100)
            if out.exists() and len(parse_batch_text(out.read_text(encoding="utf-8"))) >= want:
                continue  # complete; a partial or empty reply is re-requested (prespec: fill failed calls)
            prompt = pf.read_text(encoding="utf-8")
            kw = {"model": MODELS[fam], "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 48000 if mode == "thinking" else 8000}
            if mode == "thinking":  # identical to the main panel's thinking condition
                kw["thinking"] = {"type": "adaptive"}
                kw["output_config"] = {"effort": "high"}
            else:
                kw["thinking"] = {"type": "disabled"}
            with client.messages.stream(**kw) as s:
                r = s.get_final_message()
            text = "".join(b.text for b in r.content if b.type == "text")
            out.write_text(text, encoding="utf-8")
            rec = {"config": cfg, "cond": cond, "chunk": ci, "requested_model": MODELS[fam],
                   "resolved_model": r.model, "prompt_hash": hashes[pf.name][:16],
                   "n_answers": text.count('"id"'), "input_tokens": r.usage.input_tokens,
                   "output_tokens": r.usage.output_tokens, "timestamp": datetime.now(timezone.utc).isoformat()}
            with open(D / "provenance.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            got = len(parse_batch_text(text))
            print(cfg, cond, ci, f"parsed {got}/{want}", rec["output_tokens"], flush=True)
            if got < want:
                print("  WARNING: incomplete reply; rerun the same command to re-request this chunk", flush=True)
print("DONE")
