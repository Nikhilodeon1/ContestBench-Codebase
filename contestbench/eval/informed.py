"""Informed-prompt conditions: same batched task as the main panel, prompt header made explicit.

A  ("scales"):      feature definitions, 1-5 ranges and the direction of each feature.
B  ("scales_rate"): A + what "Malignant" means (most radiologists rate >= 3) + the corpus base rate.
Case lines and chunking (100 cases, corpus order) are identical to the main panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from contestbench.eval.batch import format_case

BASE_RATE = 0.65  # mean vote fraction at t=3 (0.647), the same number the constant baseline gets

_INTRO = """\
You are a radiologist evaluating lung nodules. For EACH case below, decide whether
the nodule is Malignant or Benign and state your confidence (0-100) in that answer.

Each feature is the mean over several radiologists, so values are decimals between 1 and 5:
- subtlety: 1 = extremely subtle, 5 = obvious (how easy the nodule is to see)
- spiculation: 1 = none, 5 = marked (higher = more spiculated = more suspicious)
- margin: 1 = poorly defined, 5 = sharp (higher = sharper edge = LESS suspicious; poorly defined or irregular margins are MORE suspicious)
"""

_LABEL = """\
"Malignant" means you expect most radiologists would rate the nodule suspicious (3 or higher on the 1-5 malignancy scale).
For reference, about {pct:.0f}% of the nodules in this set are rated suspicious by the average radiologist.
"""

_OUTRO = """\
Your confidence MUST depend on the features; do NOT default to a constant value.

Output EXACTLY one JSON object per line, one per case, and nothing else:
{"id": "<CASE_ID>", "answer": "Malignant" or "Benign", "confidence": <integer 0-100>}

Cases:
"""

CONDITIONS = ("scales", "scales_rate")


def header(cond: str) -> str:
    if cond == "scales":
        return _INTRO + "\n" + _OUTRO
    if cond == "scales_rate":
        return _INTRO + "\n" + _LABEL.format(pct=100 * BASE_RATE) + "\n" + _OUTRO
    raise ValueError(cond)


def write_prompts(corpus: pd.DataFrame, out_dir, cond: str, chunk_size: int = 100) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ci, start in enumerate(range(0, len(corpus), chunk_size), 1):
        body = "\n".join(format_case(r) for _, r in corpus.iloc[start:start + chunk_size].iterrows())
        p = out_dir / f"{cond}_{ci:03d}.txt"
        p.write_text(header(cond) + body + "\n", encoding="utf-8", newline="\n")  # LF on every OS: stable hashes
        paths.append(p)
    return paths
