"""Per-reader ratings and vignette features for each corpus nodule (needed by the disjoint-reader check).

corpus.parquet stores mean features across readers, corpus_ratings.parquet the malignancy list. The
disjoint-reader oracle needs each reader's own subtlety/spiculation/margin, so this reparses the XML with
the frozen matcher, verifies the result reproduces corpus.parquet (ids, mean features, malignancy lists),
and writes data/corpus_reader_features.parquet (id, member, malignancy, subtlety, spiculation, margin).
Usage: LIDC_XML_ROOT=<dir with the XML files> python -m contestbench.data.reader_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contestbench import config
from contestbench.data import lidc_loader, matching, ratings
from contestbench.data.corpus import ScanParse

OUT = config.DATA_DIR / "corpus_reader_features.parquet"


def build(xml_paths) -> pd.DataFrame:
    rows = []
    for scan_idx, p in enumerate(xml_paths):
        try:
            scan = lidc_loader.parse_scan_file(p)
        except Exception:
            scan = ScanParse(readers=[], dropped={"unparseable_xml": 1})
        for grp_idx, g in enumerate(matching.match_nodules(scan.readers)):
            if g.n_readers < config.MIN_READERS:
                continue
            for k, m in enumerate(g.members):
                rows.append({"id": f"{scan_idx}:{grp_idx}", "member": k, "malignancy": int(m.malignancy),
                             "subtlety": m.subtlety, "spiculation": m.spiculation, "margin": m.margin})
    df = pd.DataFrame(rows)
    for c in ("subtlety", "spiculation", "margin"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def verify(df: pd.DataFrame, corpus: pd.DataFrame, rat: pd.DataFrame) -> None:
    assert set(df["id"]) == set(corpus["id"]), "ids differ from corpus.parquet"
    mean = df.groupby("id")[["subtlety", "spiculation", "margin"]].mean()
    c = corpus.set_index("id").loc[mean.index]
    for col in ("subtlety", "spiculation", "margin"):
        assert np.allclose(mean[col].values, c[col].values, equal_nan=True, atol=1e-9), f"{col} mean mismatch"
    lists = df.sort_values(["id", "member"]).groupby("id")["malignancy"].apply(list)
    r = rat.set_index("id")["ratings"].map(list)
    assert (lists.loc[r.index] == r).all(), "malignancy lists differ from corpus_ratings.parquet"


if __name__ == "__main__":
    paths = sorted(ratings.xml_root().rglob("*.xml"))
    print(f"{len(paths)} XML files under {ratings.xml_root()}")
    d = build(paths)
    verify(d, pd.read_parquet(config.CORPUS_PARQUET), pd.read_parquet(ratings.RATINGS_PARQUET))
    d.to_parquet(OUT, index=False)
    print(f"verified vs corpus.parquet and corpus_ratings.parquet; wrote {OUT} ({len(d)} rows, "
          f"{d['id'].nunique()} nodules)")
