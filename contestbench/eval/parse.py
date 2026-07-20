"""Parse model output into a malignancy answer + a unit-interval confidence."""

from __future__ import annotations

import json
import re

import numpy as np

_CONF_RE = re.compile(r"(\d{1,3})\s*%")
_ANSWER_RE = re.compile(r"answer\s*:?\s*[\"']?(malignant|benign)", re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _conf_to_unit(val: float) -> float:
    return val / 100 if 0 <= val <= 100 else np.nan


def extract_confidence(text: str | None) -> float:
    """First ``NN%`` in text, scaled to [0,1]; NaN if absent or out of range."""
    if text is None:
        return np.nan
    m = _CONF_RE.search(text)
    if not m:
        return np.nan
    val = float(m.group(1))
    return val / 100 if 0 <= val <= 100 else np.nan


def extract_answer(text: str | None) -> str | None:
    """'malignant' / 'benign' / None."""
    if text is None:
        return None
    m = _ANSWER_RE.search(text)
    return m.group(1).lower() if m else None


def parse_response(text: str | None) -> tuple[str | None, float]:
    """(answer, confidence) from a model reply: JSON first, regex fallback.

    Returns confidence in [0,1] or NaN. Robust to JSON embedded in prose and to
    the legacy ``Answer:/Confidence:`` plaintext format.
    """
    if text is None:
        return None, np.nan

    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = obj.get("answer")
            ans = ans.lower() if isinstance(ans, str) and ans.lower() in ("malignant", "benign") else None
            conf = obj.get("confidence")
            conf = _conf_to_unit(float(conf)) if conf is not None else np.nan
            if ans is not None or not np.isnan(conf):
                return ans, conf
        except (ValueError, TypeError, AttributeError):
            pass

    # regex fallback (malformed JSON or legacy plaintext)
    return extract_answer(text), extract_confidence(text)
