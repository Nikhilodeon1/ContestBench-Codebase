"""Per-nodule raw malignancy ratings, rebuilt from the LIDC XML.

corpus.parquet stores only the thresholded vote fraction. Threshold-sensitivity
analyses need the raw 1-5 ratings, so this reparses the XML with the frozen
matcher, checks the result reproduces corpus.parquet (ids and pi at threshold 3),
and writes data/corpus_ratings.parquet (id, ratings). corpus.parquet is untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from contestbench import config
from contestbench.data import lidc_loader, matching
from contestbench.data.corpus import ScanParse

RATINGS_PARQUET = config.DATA_DIR / "corpus_ratings.parquet"


def xml_root() -> Path:
    env = os.getenv("LIDC_XML_ROOT")
    return Path(env) if env else config.LIDC_XML_ROOT


def build_ratings(xml_paths) -> pd.DataFrame:
    rows = []
    for scan_idx, p in enumerate(xml_paths):
        try:
            scan = lidc_loader.parse_scan_file(p)
        except Exception:
            scan = ScanParse(readers=[], dropped={"unparseable_xml": 1})
        for grp_idx, g in enumerate(matching.match_nodules(scan.readers)):
            if g.n_readers < config.MIN_READERS:
                continue
            rows.append({"id": f"{scan_idx}:{grp_idx}", "ratings": [int(m) for m in g.malignancies]})
    return pd.DataFrame(rows)


def verify_against_corpus(ratings: pd.DataFrame, corpus: pd.DataFrame) -> None:
    m = corpus[["id", "pi"]].merge(ratings, on="id", how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), m["_merge"].value_counts().to_dict()
    pi3 = m["ratings"].map(lambda r: np.mean(np.asarray(r) >= 3))
    assert np.allclose(pi3, m["pi"]), "threshold-3 fraction does not reproduce corpus pi"


def vote_fraction(ratings: pd.Series, threshold: int) -> np.ndarray:
    return ratings.map(lambda r: float(np.mean(np.asarray(r) >= threshold))).to_numpy()


def load_ratings() -> pd.DataFrame:
    return pd.read_parquet(RATINGS_PARQUET)


if __name__ == "__main__":
    paths = sorted(xml_root().rglob("*.xml"))
    print(f"{len(paths)} XML files under {xml_root()}")
    r = build_ratings(paths)
    verify_against_corpus(r, pd.read_parquet(config.CORPUS_PARQUET))
    r.to_parquet(RATINGS_PARQUET, index=False)
    print(f"verified vs corpus.parquet; wrote {RATINGS_PARQUET} ({len(r)} rows)")
