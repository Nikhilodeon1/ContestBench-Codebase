#!/usr/bin/env bash
# One entry point for a fresh pod (after `git clone`). Usage: bash pod_run.sh <stage>
#   setup     venv + pinned deps + input checks
#   repro     tests + full frozen-table rebuild (parallel); then diff against the committed tables
#   informed  build prompts, run the API collection (needs ANTHROPIC_API_KEY in the shell), analyse
#   analyse   score already-collected informed replies only (no API)
set -euo pipefail
cd "$(dirname "$0")"
stage=${1:-}
PYBIN=${PYBIN:-$(command -v python3.12 || command -v python3.11 || command -v python3)}

activate() { . venv/bin/activate; }

case "$stage" in
setup)
  $PYBIN -m venv venv && activate
  pip install -q --upgrade pip && pip install -q -r requirements.txt
  python - <<'EOF'
import os, pathlib, sys
need = ["data/corpus.parquet", "data/corpus_ratings.parquet", "data/corpus_reader_features.parquet",
        "data/responses_sonnet-standard.parquet", "results/confirmatory/raw_extra", "results/recollect/noise",
        "results/informed/run_informed.py", "contestbench/analysis/informed.py"]
bad = [p for p in need if not pathlib.Path(p).exists()]
print("cores:", os.cpu_count(), "| python", sys.version.split()[0], "| missing:", bad or "none")
sys.exit(1 if bad else 0)
EOF
  free -g | head -2 || true
  ;;
repro)
  activate
  # the pipeline overwrites results/tables in place; with fixed seeds a clean `git diff` means it reproduced
  bash run_revision_parallel.sh
  echo "--- tables changed vs committed (expect none; data_manifest.json timestamp is fine) ---"
  git diff --stat -- results/tables results/data_manifest.json | tail -15
  ;;
informed)
  activate
  : "${ANTHROPIC_API_KEY:?export ANTHROPIC_API_KEY in this shell first (do not write it to .env on the pod)}"
  python results/informed/run_informed.py --dry
  python results/informed/run_informed.py --go --conds scales scales_rate --configs sonnet:standard opus:standard
  python -m pytest tests/test_informed.py -q
  python -m contestbench.analysis.informed
  ;;
analyse)
  activate
  python -m contestbench.analysis.informed
  ;;
*) echo "usage: bash pod_run.sh setup|repro|informed|analyse"; exit 2 ;;
esac
