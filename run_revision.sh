#!/usr/bin/env bash
# Regenerate every revision table from the frozen inputs, then write the data manifest.
# Needs: data/ (corpus.parquet, corpus_ratings.parquet, responses_*.parquet incl. gemini-pinned-*),
#        results/interv4/, results/recollect/{noise,noise_haiku,pinned,pinned_repeat,prompts}/,
#        results/confirmatory/{raw,prompts}/.  Seeds are fixed in the code.
set -euo pipefail
PY=${PY:-python}
$PY -m pytest tests -q
$PY -m contestbench.cli panel-b
$PY -m contestbench.cli oracle-b
$PY -m contestbench.cli revision-b
$PY scripts_freeze_manifest.py
