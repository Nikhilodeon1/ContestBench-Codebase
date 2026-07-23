"""Batch file export/import for a file-in / file-out model platform.

Workflow: ``write_batch_files`` emits self-contained prompt file(s) (instruction
header + all cases) that the user feeds to the platform; the platform returns a
file of answers that ``parse_batch_text`` reads back. Used for the free
unlimited Gemini route on the full 1,413-case corpus.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

HEADER = """\
You are a radiologist evaluating lung nodules. For EACH case below, decide whether
the nodule is Malignant or Benign and state your confidence (0-100) in that answer.
Higher spiculation and irregular margins are associated with malignancy. Your
confidence MUST depend on the features; do NOT default to a constant value.

Output EXACTLY one JSON object per line, one per case, and nothing else:
{"id": "<CASE_ID>", "answer": "Malignant" or "Benign", "confidence": <integer 0-100>}

Cases:
"""

# fallback for legacy pipe format: "3:12 | Benign | 55" or "3:15 , Malignant , 80"
_LINE_RE = re.compile(
    r"([0-9]+:[0-9]+)\s*[|,\t]\s*(malignant|benign)\s*[|,\t]\s*(\d{1,3})",
    re.IGNORECASE,
)
_OBJ_RE = re.compile(r"\{[^{}]*\}")


def _conf(val) -> float:
    v = float(val)
    return v / 100 if 0 <= v <= 100 else math.nan


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
    """Parse returned answers: JSON object per line (primary), pipe format (fallback).

    Tolerant of surrounding prose and junk lines.
    """
    records = []
    seen = set()
    # primary: JSON objects (one per case)
    for m in _OBJ_RE.finditer(text):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        cid = str(o.get("id", "")).strip()
        ans = str(o.get("answer", "")).lower()
        if cid and ans in ("malignant", "benign") and o.get("confidence") is not None:
            if cid not in seen:
                seen.add(cid)
                records.append({"id": cid, "answer": ans, "confidence": _conf(o["confidence"])})
    # fallback: legacy pipe format for any ids not already captured
    for m in _LINE_RE.finditer(text):
        cid = m.group(1)
        if cid not in seen:
            seen.add(cid)
            records.append({"id": cid, "answer": m.group(2).lower(), "confidence": _conf(m.group(3))})
    return pd.DataFrame(records, columns=["id", "answer", "confidence"])


def parse_batch_file(path) -> pd.DataFrame:
    return parse_batch_text(Path(path).read_text(encoding="utf-8"))
