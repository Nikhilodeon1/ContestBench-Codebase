"""Import all recollected text files into data/responses_*.parquet."""
import sys, glob
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
from contestbench.eval import batch

MAPPINGS = [
    ("haiku-standard", "haiku:standard", "responses_haiku-standard.parquet"),
    ("haiku-thinking", "haiku:thinking", "responses_haiku-thinking.parquet"),
    ("sonnet-standard", "sonnet:standard", "responses_sonnet-standard.parquet"),
    ("sonnet-thinking", "sonnet:thinking", "responses_sonnet-thinking.parquet"),
    ("opus-standard", "opus:standard", "responses_opus-standard.parquet"),
    ("opus-thinking", "opus:thinking", "responses_opus-thinking.parquet"),
    ("gemini-standard", "gemini:standard", "responses_gemini-m-standard.parquet"),
    ("gemini-thinking", "gemini:thinking", "responses_gemini-m-thinking.parquet"),
]

corpus_df = pd.read_parquet("data/corpus.parquet")[["id", "pi", "tier"]]

for prefix, label, filename in MAPPINGS:
    files = sorted(glob.glob(f"results/recollect/{prefix}_*.txt"))
    if not files:
        print(f"Skipping {prefix}: no files found")
        continue
    parts = []
    for f in files:
        p = batch.parse_batch_file(f)
        if not p.empty:
            parts.append(p)
    if not parts:
        print(f"Skipping {prefix}: parsed empty")
        continue
    answers = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="first")
    merged = corpus_df.merge(answers, on="id", how="left")
    merged["label"] = label
    scored = merged.dropna(subset=["confidence"])
    out = Path("data") / filename
    scored.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(scored)} rows, parsed from {len(files)} files)")

print("Import script completed.")
