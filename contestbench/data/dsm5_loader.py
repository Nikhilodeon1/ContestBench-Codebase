"""Psychiatry domain loader (DSM-5 field trials) — STUB / interface only.

Phase 2. The ICC->agreement-rate conversion is the pipeline's weakest link and is
deferred until an MD reviews the vignette construction for face validity (spec
4.4). This stub fixes the interface so the harness/metrics stay domain-agnostic;
it deliberately ships no data.
"""

from __future__ import annotations

import pandas as pd

from contestbench.data import normalization

# Published DSM-5 field-trial ICCs (Regier et al. 2013) — reference values for
# phase 2, NOT yet turned into scored vignettes.
FIELD_TRIAL_ICC = {
    "major_depressive_disorder": 0.28,
    "generalized_anxiety_disorder": 0.20,
    "borderline_personality_disorder": 0.54,
}

CORPUS_COLUMNS = ["id", "pi", "n_raters", "subtlety", "spiculation", "margin", "tier"]


def icc_to_agreement(icc: float, p_chance: float) -> float:
    """pi = ICC*(1 - p_chance) + p_chance (PDF 5.2). Provided for phase 2."""
    return normalization.normalize_kappa(icc, p_chance)


def load_corpus() -> pd.DataFrame:
    """Not implemented — psychiatry is phase 2 (stub only)."""
    raise NotImplementedError(
        "DSM-5 psychiatry domain is a phase-2 stub; not populated. "
        "See spec 4.4."
    )
