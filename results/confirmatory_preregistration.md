# Confirmatory collection: pre-registered design and predictions (frozen before any call)

Claim under test: the failure of answer-folded confidence at the pre-specified >= 3 target is largely a
question/target mismatch (the prompt asked Malignant vs Benign; the target counts rating >= 3, which includes
"indeterminate"). This collection asks the question the target actually measures. It does NOT confirm the
post-hoc exclude-3 result; it confirms only "when the question matches the target".

## Design
- Prompt: `results/confirmatory/prompts/batch_001..015.txt` (hashes recorded in `results/frozen_hashes.txt`).
  One number per case: percent probability that a radiologist rates the nodule >= 3. Same three vignette
  features, same 100-case chunks in corpus order as the original panel. No answer field, so no answer/confidence
  folding: the forecast is p_hat = probability / 100.
- Configs (one run each, default sampling settings, thinking disabled, logged model string/date per call):
  claude-sonnet-5, claude-opus-4-8, claude-haiku-4-5-20251001 (protocol test), and pinned gemini-2.5-flash
  (thinking_budget 0, temperature 0) if an API key still serves it; if not, Gemini is dropped and the
  unavailability is reported.
- Target: f = fraction of readers rating malignancy >= 3 (pre-specified threshold). Score:
  PAD-B = mean (p_hat - f)^2. Constant baseline: p* = full-corpus mean f.
- CIs: two-stage cluster bootstrap (resample calls, then scans within calls), 4000 draws, fixed seed.
- Cost cap: 15 USD (estimated total about 4-5 USD).

## Predictions (committed before any call)
- P1: For Sonnet and Opus (and Gemini if collected), PAD-B(p_hat, f) is below the constant's, with the 95%
  nested-bootstrap CI of the gap (model minus constant) entirely below zero. If a config's CI includes zero or
  lies above zero, P1 FAILS for that config and is reported as a failure.
- P2: For each config, adding p_hat to the three vignette features does not lower PAD-B: the 95% CI of
  PAD-B(features only) - PAD-B(features + p_hat) (gbt, 5-fold GroupKFold by scan, same folds) includes zero.
- P3: Haiku's per-call semantic flip disappears: in each of its calls, the within-call Pearson correlation
  between p_hat and f is positive (no call with a negative correlation), reported next to the range across calls
  of the call-mean p_hat. No prediction is made about Haiku's rank or its PAD-B level.
- No prediction is made about ordering among configs. Differences smaller than the noise floor in
  `results/noise_floor_b.csv` (F_3 = 0.0073 for stable configs) are not interpreted.

## Rules
- One run per config. No re-running after seeing results. If the prompt is changed after any result is seen,
  every result obtained with the changed prompt is labelled exploratory.
- All analyses outside P1-P3 (for example the >= 4 threshold, exclude-3 subsets, rank correlations) are
  labelled exploratory.
- If P1 fails for a config, it is reported as a failure of the prediction.
