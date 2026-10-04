"""Import the pinned gemini-2.5-flash re-collection into data/responses_gemini-pinned-*.parquet."""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from contestbench.eval import batch

corpus_df = pd.read_parquet("data/corpus.parquet")[["id", "pi", "tier"]]
for cond in ("standard", "thinking"):
    files = sorted(glob.glob(f"results/recollect/pinned/gemini-{cond}_*.txt"))
    parts = [batch.parse_batch_file(f) for f in files]
    answers = pd.concat([p for p in parts if not p.empty]).drop_duplicates("id", keep="first")
    merged = corpus_df.merge(answers, on="id", how="left")
    merged["label"] = f"gemini:{cond}"
    scored = merged.dropna(subset=["confidence"])
    out = Path("data") / f"responses_gemini-pinned-{cond}.parquet"
    scored.to_parquet(out, index=False)
    print(f"{out}: {len(scored)} rows from {len(files)} files")
