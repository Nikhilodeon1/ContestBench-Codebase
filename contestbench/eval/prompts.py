"""Radiologist vignette prompt (feature-based, no images available)."""

from __future__ import annotations


def create_prompt(subtlety: float, spiculation: float, margin: float) -> str:
    # Wording/feature-format deliberately mirror the Gemini batch header
    # (contestbench/eval/batch.py) so cross-provider rows are comparable; only the
    # output channel differs (JSON here for single-call parse reliability, pipe in
    # the batch route). See spec 4.2.
    return f"""\
You are a radiologist evaluating a lung nodule. Decide whether the nodule is \
Malignant or Benign and state your confidence (0-100) in that answer. Higher \
spiculation and irregular margins are associated with malignancy. Your confidence \
MUST depend on the features; do NOT default to a constant value.

Nodule features: subtlety {subtlety:.2f} | spiculation {spiculation:.2f} | margin {margin:.2f}

Respond with ONLY a JSON object, no other text:
{{"answer": "Malignant" or "Benign", "confidence": <integer 0-100>}}"""
