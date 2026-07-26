"""Frozen configuration constants for ContestBench.

Matcher thresholds are FROZEN at the reported default (see spec 3.3, verified
2026-07-18: total N and the ambiguous tier are stable across 20/2, 30/3, 40/4).
Any change here is a NEW sensitivity run, never a silent update to base numbers.
"""

from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LIDC_XML_ROOT = DATA_DIR / "LIDC-IDRI"  # moved under data/ 2026-07-26
RESULTS_DIR = ROOT / "results"
CORPUS_PARQUET = DATA_DIR / "corpus.parquet"
EXCLUSIONS_CSV = RESULTS_DIR / "exclusions.csv"

# --- LIDC XML namespace ---
NIH_NS = {"n": "http://www.nih.gov"}

# --- nodule matching (FROZEN) ---
MATCH_XY_THRESH = 30.0  # pixels, in-plane centroid distance
MATCH_Z_THRESH = 3.0    # mm, slice-position distance

# --- physician agreement / tiering ---
MALIGNANCY_POSITIVE = 3   # malignancy rating >= 3 counts as "malignant" vote
MIN_READERS = 3           # a matched nodule needs >= this many reader ratings

# Tier definition adapted to discrete 4-rater support (spec 4.1):
#   high = pi in {0, 1}, contested = pi in {.25, .75}, ambiguous = pi == .5
# 3-reader nodules also produce {1/3, 2/3}; those fall in "contested" (not
# high, not exact .5). Encoded as a function-free rule table for testability
# in corpus.py -- constants only here.
TIER_HIGH = "high"
TIER_CONTESTED = "contested"
TIER_AMBIGUOUS = "ambiguous"

# --- sampling ---
RANDOM_SEED = 20260718
