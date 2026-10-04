#!/usr/bin/env bash
# Parallel version of run_revision.sh for a many-core machine. Every job writes its own CSVs
# (no shared outputs), so they can run side by side. One thread per job avoids BLAS oversubscription.
set -euo pipefail
PY=${PY:-python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs results/tables
jobs=(
  "-m pytest tests -q"
  "-m contestbench.cli panel-b"
  "-m contestbench.cli oracle-b"
  "-m contestbench.cli revision-b mi"
  "-m contestbench.cli revision-b mech"
  "-m contestbench.cli revision-b interv"
  "-m contestbench.cli revision-b reliance"
  "-m contestbench.cli revision-b robust"
  "-m contestbench.cli revision-b acccal"
  "-m contestbench.cli revision-b excl3"
  "-m contestbench.cli revision-b baserate"
  "-m contestbench.cli revision-b noise"
  "-m contestbench.cli revision-b stability"
  "-m contestbench.cli revision-b confirm"
  "-m contestbench.cli revision-b ece"
  "-m contestbench.cli revision-b tiers"
  "-m contestbench.cli revision-b consistency"
  "-m contestbench.cli revision-b resid"
  "-m contestbench.cli revision-b floor"
  "-m contestbench.cli revision-b disjoint"
  "-m contestbench.cli revision-b extra"
)
pids=(); names=()
for j in "${jobs[@]}"; do
  name=$(echo "$j" | tr ' ' '_' | tr -d '-' | cut -c1-60)
  $PY $j > "logs/$name.log" 2>&1 &
  pids+=($!); names+=("$name")
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "ok    ${names[$i]}"; else echo "FAIL  ${names[$i]} (see logs/${names[$i]}.log)"; fail=1; fi
done
[ $fail -eq 0 ] || { echo "some jobs failed"; exit 1; }
$PY scripts_freeze_manifest.py
echo "ALL DONE"
