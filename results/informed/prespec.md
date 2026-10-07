# Informed-prompt check: pre-specification (written before any informed call)

Motivation: the main prompt gives three bare numbers, no scales, no margin direction (high margin = sharp = benign,
r = -0.36 with the vote fraction) and no base rate. Does that underspecification explain the models' deficit vs the constant?

Design: same batched JSON-lines task, 100 cases per call in corpus order, same configs and settings as the main panel
(`claude-sonnet-5`, `claude-opus-4-8`; thinking disabled/adaptive-high). One run per config x condition, no retries beyond
filling failed calls. Conditions: `scales` (definitions + directions) and `scales_rate` (+ meaning of Malignant + 65% base rate).
Prompts: `contestbench/eval/informed.py`; hashes in `prompt_hashes.json`.

Metric: gap = PAD-B(model) - PAD-B(case-blind constant), t=3 primary, t=4 secondary; nested call->scan bootstrap, paired
against the original run on the same cases. Standard configs first; thinking configs only if standard ones are informative.

Predictions (standard configs, t=3):
- P1: under `scales`, the gap stays above zero (CI lower bound > 0) for both.
- P2: under `scales_rate`, the gap falls to at most half its original value for both.
- P3: under `scales_rate`, a gap CI upper bound < 0 for either (then the headline "worse than the constant" is prompt-dependent
  and the paper must say so).

Scored by `python -m contestbench.analysis.informed`; failed predictions are reported as failures, whatever the outcome.
