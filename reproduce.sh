#!/usr/bin/env bash
# ContestBench end-to-end reproduction.
# Corpus, scoring, and figures are fully automated. Model responses are partly
# manual: Gemini/Opus run through an external file-in/file-out platform, gpt-oss
# through Groq (needs GROQ_API_KEY in .env). Provenance for the manual runs lives
# in results/provenance_gemini.json.
set -euo pipefail

PY="python -m contestbench.cli"

echo "== tests =="
python -m pytest tests/ -q

echo "== build corpus (LIDC XML -> data/corpus.parquet + exclusion logs) =="
$PY build-corpus

echo "== model responses =="
# gpt-oss (automated, needs GROQ_API_KEY):
#   $PY run-full --models gpt-oss-20b --efforts low
# Gemini / Opus (manual file-in/file-out platform):
#   $PY export-batch --chunk-size 500              # -> results/batch_prompts/batch_*.txt
#   (run each file through the platform, save replies)
#   $PY import-batch results/batch_prompts/response_*.txt --label gemini-3.5-flash
# Claim-3 (Opus thinking vs standard, manual):
#   $PY export-batch --n 150 --chunk-size 0        # -> claim3_001.txt
#   $PY import-batch <reply> --label opus:thinking
echo "(model runs are manual/API — see comments above; responses cached in data/)"

echo "== score each responses file =="
for f in data/responses_*.parquet; do
  [ -e "$f" ] && echo "--- $f ---" && $PY score "$f"
done

echo "== figures =="
$PY figures

echo "== done: results/figures/, results/tables/ =="
