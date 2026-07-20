"""Radiologist vignette prompt (feature-based, no images available)."""

from __future__ import annotations


def create_prompt(subtlety: float, spiculation: float, margin: float) -> str:
    return f"""\
You are a radiologist evaluating a lung nodule.

Radiological features:
- Subtlety (1-5): {subtlety:.2f}
- Spiculation (1-5): {spiculation:.2f}
- Margin (1-5): {margin:.2f}

Higher spiculation and irregular margins are associated with malignancy.

Question:
Is this nodule malignant?

Your confidence MUST depend on the features above. Do NOT default to a constant value.

Respond with ONLY a JSON object, no other text:
{{"answer": "Malignant" or "Benign", "confidence": <integer 0-100>}}"""
