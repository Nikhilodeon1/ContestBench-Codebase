"""Sample-consistency confidence (weekend fix #1).

The PDF's original elicitation: sample the model K times per case at temperature
0.8, embed each response (e5-small-v2), and take confidence = mean pairwise cosine
similarity of the K embeddings. High self-consistency (the model says the same
thing every time) = high confidence. This is a genuinely different UQ signal from
verbalized confidence -- a robustness check that the decoupling isn't an artifact
of the (least-reliable) verbalized method.
"""

from __future__ import annotations

import numpy as np


def pairwise_cosine_confidence(embeddings) -> float:
    """Mean pairwise cosine similarity across K sample embeddings (K>=2)."""
    e = np.asarray(embeddings, float)
    if e.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    u = e / norms
    sim = u @ u.T
    iu = np.triu_indices(len(u), k=1)   # upper triangle, excludes self-pairs
    return float(sim[iu].mean())
