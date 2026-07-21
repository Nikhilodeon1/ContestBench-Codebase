"""Batch file export/import for a file-in / file-out model platform.

Workflow: ``write_batch_files`` emits self-contained prompt file(s) (instruction
header + all cases) that the user feeds to the platform; the platform returns a
file of answers that ``parse_batch_text`` reads back. Used for the free
unlimited Gemini route on the full 1,413-case corpus.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

HEADER = """\
You are a radiologist evaluating lung nodules. For EACH case below, decide whether
the nodule is Malignant or Benign and state your confidence (0-100) in that answer.
Higher spiculation and irregular margins are associated with malignancy. Your
confidence MUST depend on the features; do NOT default to a constant value.

Output EXACTLY one line per case, and nothing else, in this format:
<CASE_ID> | <Malignant or Benign> | <Confidence 0-100>

Cases:
"""

# a line like "3:12 | Benign | 55" or "3:15 , Malignant , 80"
_LINE_RE = re.compile(
    r"([0-9]+:[0-9]+)\s*[|,\t]\s*(malignant|benign)\s*[|,\t]\s*(\d{1,3})",
    re.IGNORECASE,
)


def format_case(row) -> str:
    return (f"{row['id']} | subtlety {row['subtlety']:.2f} | "
            f"spiculation {row['spiculation']:.2f} | margin {row['margin']:.2f}")


def write_batch_files(corpus: pd.DataFrame, out_dir, chunk_size: int = 300,
                      prefix: str = "batch") -> list[Path]:
    """Write prompt file(s). chunk_size=0 -> a single file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(corpus)
    size = n if chunk_size in (0, None) else chunk_size
    paths: list[Path] = []
    for ci, start in enumerate(range(0, n, size), 1):
        chunk = corpus.iloc[start:start + size]
        body = "\n".join(format_case(r) for _, r in chunk.iterrows())
        path = out_dir / f"{prefix}_{ci:03d}.txt"
        path.write_text(HEADER + body + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def parse_batch_text(text: str) -> pd.DataFrame:
    """Parse returned answers; tolerant of prose, junk lines, and , | tab delimiters."""
    records = []
    for m in _LINE_RE.finditer(text):
        cid, ans, conf = m.group(1), m.group(2).lower(), float(m.group(3))
        records.append({
            "id": cid,
            "answer": ans,
            "confidence": conf / 100 if 0 <= conf <= 100 else math.nan,
        })
    return pd.DataFrame(records, columns=["id", "answer", "confidence"])


def parse_batch_file(path) -> pd.DataFrame:
    return parse_batch_text(Path(path).read_text(encoding="utf-8"))
