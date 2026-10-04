# ContestBench: full writeup of the revision since the AAAI draft

Prepared 2026-10-04 from the frozen pod-run-2 tables (`results/tables/*_b.csv`, `results/data_manifest.json`,
hashes in `results/frozen_hashes.txt`). Claim-by-claim table: `results/reconciliation.md`. Phase-by-phase
status: `results/methodology_checklist.md`. No paper text has been rewritten yet; this document is the record
the rewrite will be built from.

## 1. Executive summary

The AAAI draft claimed that no model beats a case-blind constant at tracking physician agreement, that
confidence is uncorrelated with agreement, and that reasoning does nothing. A reviewer pointed out that the
metric (PAD) ignored which answer the model chose. Fixing that, and auditing everything around it, changed
the picture:

1. **What survived intact (and is now the lead result).** Stated confidence adds nothing beyond the answer
   and three simple vignette features. This holds at both malignancy thresholds, for all ten configs
   (ablation CIs bracket zero 10/10 at each threshold), and a second, differently worded elicitation (direct
   P(rating >= 3)) reproduced it for all four configs tested (pre-registered P2).
2. **What became threshold-dependent.** At the pre-specified labeling (malignant = rating >= 3) no headline
   model beats the constant under either elicitation. At the post-hoc >= 4 labeling, Sonnet and Opus do.
   The deficit at >= 3 is mainly a base-rate offset: models state P(rating >= 3) far below the true 0.647,
   and one out-of-fold constant shift flips Sonnet and Opus to beating the constant.
3. **What reversed.** Accuracy-calibration (Platt) improves PAD-B for every config at >= 3 (the AAAI draft said it
   worsens PAD for every config). ECE and PAD-B are no longer "uncorrelated" at >= 4. Thinking slightly
   hurts at >= 3 (small, threshold-specific). Prompting design C helps the larger Claude models at >= 3 only.
4. **What was wrong in the draft and is now fixed.** The "pipe-batch" elicitation sentence, "exactly what the
   LLMs see" for a 4-feature oracle (`extent` was never in any prompt), a rolling Gemini alias that made
   gemini:standard look worst, a Gemini standard/thinking contrast that compared two different models, iid
   bootstrap CIs on scan-clustered data, and a sample-consistency appendix that claimed 10 samples (median is 5).
5. **New findings about the instrument itself.** Per-call protocol instability (Haiku flips the meaning of its
   confidence field between calls; gemini:thinking's Malignant rate swings 0.11 to 0.83 across calls), a
   measured repeat-run noise floor, and the observation that the answer-only forecast is several times
   less noisy than the stated-confidence forecast.
6. **Scope narrowed honestly.** Headline panel is the four Anthropic configs (one vendor), chosen by a rule
   committed before it was applied. DeepSeek stays manual and out of headline claims.

## 2. Starting point: the AAAI draft

- Metric: PAD = mean |c - pi|, with pi = max(f, 1-f) (an L1 distance between stated confidence and physician
  agreement), described as the L1 analogue of the Brier score.
- Ten configs (Haiku, Sonnet, Opus, Gemini, DeepSeek x standard/thinking), 1,413 nodules, 709 scans.
- Headline claims: all ten significantly worse than a case-blind constant (always 0.75, PAD 0.143); confidence
  uncorrelated with agreement; a 4-feature "rating-independent" oracle reaches PAD about 0.127 (described as
  exactly what the LLMs see); a recalibrator closes the gap; confidence statistically redundant but a small
  detectable signal survives; reasoning a mixed-sign null; accuracy-calibration worsens PAD for every config;
  prompting helps only the smallest model; ECE uncorrelated with PAD; over-reliance simulation shows
  recalibration worsens over-reliance for already-cautious models.
- Format: AAAI-2027, hard 7 pages of content (appendices included) plus references on pages 8-9.

## 3. Why it changed

- A reviewer (PC7rPC) objected that PAD compares confidence with agreement and never uses the answer, so a
  confidently wrong model on a unanimous case can score zero. A second reviewer (eUFh) asked why DeepSeek was
  collected manually.
- The venue changed to TMLR (no page limit); the TMLR style files are in `indivCB/tmlr/`.
- A ten-phase rebuild spec was run (Phases 0-10), followed by rounds of strategy-agent review that added
  checks (threshold sensitivity, rating-3 exclusion, base-rate check, noise floor, stability rule, a
  pre-registered confirmatory collection).

## 4. Pre-flight audit findings (Phase 0)

- **Elicitation format.** All ten configs used the same batched prompt asking for one JSON object per case. The
  "pipe format" belonged only to a legacy manual Gemini run that is not in the panel. Chunk sizes differ: 100 cases
  per call for API configs, about 471 for DeepSeek (manual, three calls).
- **`extent` was never shown.** Prompts contain only subtlety, spiculation and margin. The 4-feature oracle was
  therefore not a same-information comparison. Adding `extent` changes oracle PAD-B by only 0.004 to 0.006.
- **Grouping.** The original CV and bootstraps ignored that 1,413 nodules come from 709 scans (338 scans have more
  than one nodule). CV leakage turned out negligible (GroupKFold changes the oracle by <= 0.001), but iid
  bootstrap CIs were too narrow.
- **Chunks are not random.** Each model call covered a consecutive slice of corpus order, so call composition
  differs (ANOVA of f across chunks p = 0.02; tier x chunk p < 0.0001). This matters for any claim about
  call-level behaviour and is stated as a limitation.

## 5. The new metric and statistical methods

- **PAD-B** = mean (p - f)^2 where p = c if the answer is Malignant else 1 - c, and f is the raw fraction of
  raters rating malignancy >= t. This is the Brier score with the physician vote fraction as target; unlike
  the old metric it penalises a confidently wrong answer (a Benign answer at confidence 0.8 on a unanimous
  malignant case scores 0.64; the old metric scored 0.2, and 0.0 at confidence 1.0).
- **PAD-L1** (the old metric) is kept as a descriptive distance statistic, explicitly not a proper scoring rule.
- **Case-blind constant** is re-derived as p* = mean f = 0.647 (PAD-B 0.0986 at t = 3; 0.0970 at t = 4), not 0.75.
- **Warranted-reliance target** s = support for the chosen answer (f if Malignant, 1 - f if Benign) replaces pi in
  the over-reliance simulation.
- **Thresholds.** t = 3 is primary (in the initial design notes and first commit, not externally registered); t = 4 is
  a labelled post-hoc sensitivity analysis. Tiers (high / contested / ambiguous) stay defined on pi (case
  difficulty); scoring uses f.
- **CIs.** Scan-cluster bootstrap everywhere; a two-stage call-then-scan bootstrap wherever an LLM call exists.
  CV is 5-fold GroupKFold by scan. Hyperparameters unchanged: GBT 200 trees, depth 3, learning rate 0.05;
  ridge alpha 1; fixed seeds in code.
- **Honest oracle.** Vignette-optimal oracle uses only the three features in the prompt; an "extended" oracle
  adds `extent` and is labelled an upper bound. `n_raters` and `malignancy_extremity` are dropped (derived
  from the same ratings as f).
- **Per-rater ratings** were rebuilt from the LIDC XML (`data/corpus_ratings.parquet`) and verified to reproduce all
  1,413 ids and the >= 3 vote fractions exactly.

## 6. Data collection and provenance work since the draft

| Item | What was done | Outcome |
|---|---|---|
| Gemini pinning | Re-collected standard and thinking on one explicit `gemini-2.5-flash`, thinking budget 0 vs 2048, resolved model version and thinking-token count logged per call (30 calls, 1,413 rows each) | The old gemini:standard used the rolling alias `gemini-flash-latest`; PAD-B moved from 0.218 to 0.135. The old standard/thinking contrast compared two different models. |
| Model availability | Four of six API keys now return "gemini-2.5-flash is no longer available to new users" | Replicability caveat to state in the paper; only the legacy key still serves it. |
| Repeat runs (noise floor) | Identical-prompt repeats, 14 calls each, of sonnet:standard, gemini:standard (pinned), haiku:standard; gemini:thinking only 3/15 (key quota exhausted, not used) | See section 8.9. Cost about $0.8 for Anthropic plus free-tier Gemini. |
| Confirmatory collection | New prompt asking directly for P(a radiologist rates the nodule >= 3), one number; Sonnet, Opus, Haiku, pinned Gemini; 60 calls | Design, predictions and prompt hashes frozen before any call (`results/confirmatory_preregistration.md`, `results/frozen_hashes.txt`). Estimated cost about $3.25. |
| DeepSeek | No API key exists; both configs remain manual (3 calls each, unknown settings) | Kept in the full table, excluded from headline claims. The abstract cannot claim three vendors or the R1-lineage point. |
| Pre-specification package | `indivCB/supplement_prespec/` (design spec, first-commit `config.py`, commit timeline without authors, README) | Scoped honestly: the >= 3 threshold and pi were in the design notes and first commit; PAD-B itself was not. Commit timestamps are self-reported. |

Stability rule (frozen before it was applied, `results/stability_rule.md`): a config is stable if its range of
Malignant-answer rate across calls is <= 0.375, the range of mean confidence on Benign answers is <= 0.2385
(1.5 x the maxima of the four Anthropic reference configs), and its route is API. Result: the four Anthropic
configs are stable; Haiku x2, gemini:standard and gemini:thinking are unstable; both DeepSeek configs are manual
and not assessable. Disclosures: the cutoffs came from the reference set, so those four pass by construction;
gemini:standard fails on answer-rate range (0.51) although its repeat-run answer agreement is 98%
(the variation is mostly chunk composition).

## 7. Reproducibility and pipeline state

- 17 CLI jobs (`run_revision_parallel.sh`) regenerate every table from stored data; 133 tests pass. Zero API calls are
  needed for the analysis. A pod run (32 cores) took on the order of minutes; two pod runs produced identical
  point estimates (only CI columns changed when nested clustering was added).
- Inputs are hashed (`results/data_manifest.json`, 229 files). Parquet inputs are byte-identical between the
  laptop and the pod; text inputs differ only by line endings.
- Laptop and pod results agreed to about 1e-15 on every table both computed after the Gemini swap.
- Known leftovers: `answer_only_repeat_full_b.csv` was computed on the laptop (deterministic); intervention and
  over-reliance CIs are scan-clustered only; the manifest records data and tables, not Python/BLAS versions.

## 8. Results

All numbers are 95% CIs from the frozen tables. "Headline four" = sonnet:standard, sonnet:thinking,
opus:standard, opus:thinking.

### 8.1 Lead result: confidence adds nothing beyond the answer and three simple features
- 3-feature oracle (grouped CV): PAD-B 0.066 (GBT) / 0.076 (ridge, linear) at t = 3; margin over the constant
  -0.033 [-0.039, -0.027] / -0.023 [-0.028, -0.019]. At t = 4: 0.046 / 0.050 against a constant of 0.097.
  Extended oracle (adds `extent`, not in the prompt): 0.062 / 0.071 at t = 3.
- Recalibrated model forecast + features reaches the oracle (gap closed 0.976 to 0.995 for the headline four).
- Ablation (features only minus features + p): bracketing zero for all 10 configs at both thresholds; headline gaps
  -0.0004 to -0.0020 at t = 3.
- Replicated with a different elicitation: P2 passed for all four confirmatory configs (gaps -0.0011 to -0.0019).
- Caveat to state: noisy p-hat from an unstable config would pass this test by itself, so the claim leans on
  Sonnet and Opus.

### 8.2 Threshold dependence (second finding)
- t = 3: gap to the constant (model minus constant, PAD-B): sonnet:std +0.052 [0.034, 0.069], sonnet:think +0.070
  [0.046, 0.094], opus:std +0.059 [0.040, 0.077], opus:think +0.077 [0.053, 0.097]. Full ten: all worse except
  deepseek:std (+0.018 [-0.001, 0.039], manual).
- t = 4: sonnet/opus beat the constant by 0.021 to 0.025 (e.g. sonnet:std [-0.036, -0.007]); gemini:std tied
  (+0.004 [-0.019, 0.027]); haiku +0.12.
- Threshold-averaged over t = 2..5 with one joint constant: only sonnet:std clears zero (-0.007 [-0.014, -0.001]).
  Spearman(p, mean rating) is about 0.47 to 0.52 for the headline four, 0.40 for deepseek:std, near 0 for Haiku.
- Decomposition: on nodules whose f is identical at t = 3 and t = 4 (n = 347), the headline four beat the constant
  by 0.077 to 0.085 (r(p,f) about 0.74); on the 1,066 threshold-sensitive nodules they are +0.116 to +0.152 worse
  at t = 3 and tied at t = 4. Collision-free subset (n = 638) reproduces the full-corpus pattern.
- Exploratory exclude-3 (post-hoc): headline four beat the constant on nodules with no rating 3 (n = 347, 24.6% of
  the corpus, skewed to clear cases: 63% high tier) by 0.077 to 0.085 and, with individual 3-votes dropped
  (n = 1,356), by 0.035 to 0.043. Haiku x2, gemini:thinking and deepseek:thinking still fail.

### 8.3 Base-rate account of the >= 3 deficit
- Out-of-fold, scan-grouped, one-parameter shift of the confirmatory P(>=3) forecasts: Sonnet -0.012 [-0.020,
  -0.005] and Opus -0.015 [-0.023, -0.008] now beat a constant that is also fit out-of-fold; Haiku +0.012 [-0.001,
  0.027] and Gemini +0.032 [0.014, 0.050] do not. Two-parameter versions: -0.014, -0.016, -0.003 (ns), 0.000 (ns).
- Models state P(>=3) far below the true mean 0.647: Sonnet 0.31, Opus 0.41, Haiku 0.49, Gemini 0.63.
  Discrimination is real for Sonnet and Opus (r(p,f) 0.37 and 0.40) and absent for Gemini (0.04).
- Base-rate check on the answer-folded forecasts (mean PAD-B over ten configs, t = 3): raw 0.159; one-parameter
  shift 0.110; two-parameter linear map 0.088; answer-only map 0.090; Platt on confidence 0.117; Platt on
  confidence + answer 0.095.

### 8.4 Accuracy-calibration reverses
- Platt scaling of confidence to hit rate (vs the majority label) improved PAD-B for all ten configs at t = 3
  (confidence + answer: e.g. sonnet:std 0.150 -> 0.089, delta -0.061 [-0.082, -0.037]); the headline four then beat
  the constant by 0.009 to 0.013. The AAAI claim that accuracy-calibration worsens PAD for every config is withdrawn.
  Interpretation: it corrects the base rate and scale of confidence, not case-level calibration.

### 8.5 Confidence vs agreement, mutual information, mechanism
- r(c,f) for the headline four is -0.26 to -0.33; r(p,f) is +0.43 to +0.45 (Haiku +0.08 to +0.11). So the answer
  direction carries information and the confidence number does not.
- MI(p, f): 11/11 pass Bonferroni at both thresholds (headline MI 0.14 to 0.20 at t = 3; oracle 0.285).
  MI cannot separate the answer from the confidence.
- Mechanism: R2 of p on the three vignette features is 0.80 to 0.92 for the headline four (Haiku 0.03 to 0.05);
  headline cross-model r(p,p) is 0.84 to 0.94, so stable models share a feature-driven signal.

### 8.6 Reasoning and prompting
- Reasoning (thinking minus standard, PAD-B, nested CIs): at t = 3 sonnet +0.019 [0.004, 0.035], opus +0.018
  [0.007, 0.028] (about 2.5x the noise floor); Bonferroni across four valid families leaves Opus only. At t = 4
  both are null. Haiku null; DeepSeek manual; Gemini (pinned) +0.005 at t = 3, +0.072 at t = 4 but unstable.
- Prompting (n = 300 held-out, scan-clustered only, 3 calls per design): design C at t = 3 helps Sonnet -0.039
  [-0.059, -0.022] and Opus -0.061 [-0.081, -0.042] (Bonferroni), both tying the constant; Haiku not significant.
  At t = 4 the same prompts hurt. These prompts taught "confidence = physician agreement" (targets pi, not the
  quantity PAD-B scores); a properly targeted study is future work.

### 8.7 ECE
- (a) Empirical correlation across configs: t = 3 Pearson +0.43 [-0.04, 0.84], Spearman 0.41 (perm p = 0.24,
  not significant); t = 4 Pearson +0.84 [0.47, 0.96], Spearman 0.87 (p = 0.002). The "uncorrelated" claim is retired.
- (b) Structural: ECE needs a binary label that does not exist on 159 split cases (11.3% of the corpus); assigning
  it at random swings ECE across a 95% band 0.143 to 0.161 wide per config. This argument is intact.
- Supplementary gpt-oss-20b (single-call JSON, 93% Malignant answers): ECE 0.050 but PAD-B 0.115, gap +0.017
  [0.013, 0.022] (scan-only CI).

### 8.8 Over-reliance simulation (warranted reliance = s)
- t = 3: recalibration lowers headline over-reliance from 0.195 to 0.246 down to 0.042 to 0.045; pooled
  decoupled minus recalibrated +0.155 [0.147, 0.163]; ordering decoupled > recalibrated > ideal in 12/12 cells.
- t = 4: recalibration raises over-reliance for sonnet:std (0.055 -> 0.070) and opus:std (0.062 -> 0.069) and lowers it
  slightly for the thinking configs; pooled +0.026 [0.019, 0.032]; ordering 11/12. The old "recalibration worsens
  over-reliance for cautious models" point survives only at t = 4 and only for the standard configs.

### 8.9 Noise and stability
- Repeat-run noise floor (rule written before running; F = 1.96 x RMS per-call repeat difference / sqrt(14)):
  F_3 = 0.0073, F_4 = 0.0040 for stable configs; Haiku F_3 = 0.014, F_4 = 0.075. An earlier 3-call Sonnet repeat
  (+0.023) was unrepresentative and is not to be quoted.
- Answer-only forecast repeat noise (RMS per call): sonnet 0.0036 vs 0.0153 for raw p at t = 3 (4.2x), gemini
  0.0029 vs 0.0124 (4.3x), Haiku 0.0012 vs 0.0267 (21.9x). At t = 4 equal for the stable configs.
- Call-level instability: Haiku reports confidence-in-answer (about 0.8) in some calls and P(malignant) (about
  0.2 on Benign answers) in others (mean confidence on Benign answers per call ranges 0.17 to 0.87).
  gemini:thinking's Malignant rate per call ranges 0.11 to 0.83; deepseek:thinking 0.14 to 0.67 across its three
  manual calls. Sonnet and Opus are stable (SD of confidence on Benign answers about 0.035 to 0.046).
- Because the stable configs ran the same unrandomized chunks and do not swing, composition cannot explain the
  swings in the unstable ones (stated for the paper in one sentence).

### 8.10 Confirmatory collection (P(>=3) elicitation, pre-registered)
- P1 (model beats the constant at >= 3, nested CI excludes zero): FAILED for all four. Gaps: Sonnet +0.103 [0.077,
  0.129] (PAD-B 0.201), Opus +0.040 [0.023, 0.057], Haiku +0.037 [0.015, 0.062], Gemini +0.032 [0.015, 0.051].
- P2 (probability adds nothing beyond three features): PASSED 4/4.
- P3 (Haiku's per-call flip disappears; within-call correlation of p-hat with f positive in every call): Sonnet and
  Opus pass; Haiku fails (4 of 14 calls negative, min -0.13, SE about 0.1); Gemini fails (6 of 14, min -0.21),
  which with r = 0.04 means no discrimination rather than instability.
- This does not confirm the "question/target mismatch explains the >= 3 failure" hypothesis; the surviving
  explanation is a base-rate offset (8.3). The exclude-3 result stays exploratory.

### 8.11 Descriptive extras
- PAD-B by pi-tier (t = 3): headline models lean Benign in every tier (signed p - f -0.17 to -0.33); e.g. sonnet:std
  high 0.234 (constant 0.203), contested 0.110 (0.036), ambiguous 0.039 (0.022).
- Sample-consistency spot check (Gemini, T = 0.8): n = 73 cases, median 5 samples per case (the old appendix said
  10), mean agreement 0.962, r(agreement, pi) = -0.111, PAD-B 0.333 vs constant 0.092 (saturated confidence).
- Capability trend: not supported under PAD-B (ambiguous-tier PAD-B: Haiku 0.095, Sonnet 0.039, Opus 0.047) and dropped.

### 8.12 Final methodology round (exploratory; plan frozen in `results/analysis_plan_final.md` before running)
- **A, residualized test.** After removing what the three vignette features predict (OOF, GBT and linear), the
  partial correlation of the model's answer, confidence or folded forecast with f is statistically
  indistinguishable from zero for the headline four at both thresholds (no CI excludes zero in 16 cells; R2 of p on
  the features 0.82 to 0.93). The pre-written prediction (|partial r(c,f)| < 0.1) held in 14 of 16 cells; the two
  misses were linear-residualization cells at t = 3 (-0.104, -0.112), likely unremoved non-linearity. Consequence:
  the lead claim can be stated more strongly and more precisely: the models' outputs carry no detectable
  information about the vote fraction beyond the three features; the answer's marginal correlation with f is
  routed through the features.
- **B, irreducible noise.** Per-case floor f(1-f)/(n_raters-1): 0.0513 at t = 3, 0.0386 at t = 4 (simulation:
  unbiased, forecaster differences unaffected). The 3-feature oracle's PAD-B of 0.066 is 0.015 above the floor at
  t = 3 (extended 0.011); at t = 4 the extended oracle is at the floor (0.0394 vs 0.0386). Headline raw p is 0.099
  to 0.124 above the floor, shift-corrected 0.028 to 0.029, recalibrated 0.015 to 0.017. Caveats: fixed panel, noisy plug-in.
- **C, disjoint readers (n = 907).** Features from one reader pair, target from the other: the oracle's margin over
  the constant is -0.015 (GBT) / -0.014 (ridge) at t = 3 versus -0.033 / -0.027 for same-pair features (retained
  fraction 0.46 / 0.52), and at t = 4 -0.029 / -0.031 versus -0.046 / -0.047 (0.62 / 0.67). Every disjoint split
  has a CI excluding zero. Shared-rater dependence therefore inflates the oracle margin (about half at t = 3) but
  does not create it.
- **D, other model families (Groq; resolved ids logged; prompt hashes verified).** Qwen3.8-27B: complete (n = 1,413),
  mean P(>=3) 0.513, gap to the constant +0.046 [0.027, 0.065], r(p,f) 0.07, ablation brackets zero. gpt-oss-120b and
  gpt-oss-20b mangled case ids and refused whole calls (strict recovery 35% and 41%; positional rescue 58% and 43%),
  so their results are partial and descriptive. Prediction 1 (mean P(>=3) below 0.647) held for 1 of 3 models;
  prediction 2 (ablation brackets zero) held for 3 of 3, with weak discrimination (r 0.03 to 0.09) so the test has
  little power. The stability rule cannot be applied unchanged (no answer or confidence field).

## 9. Old vs new headline claims

| AAAI draft | Now |
|---|---|
| All ten significantly worse than the constant | At the pre-specified >= 3: all headline configs worse under two elicitations; at >= 4, Sonnet/Opus better. Mainly a base-rate offset |
| Confidence uncorrelated with agreement | Answer direction correlates (r(p,f) +0.43 to +0.45); confidence magnitude adds nothing beyond features |
| 4-feature oracle = what LLMs see, PAD 0.127 | 3-feature oracle PAD-B 0.066; `extent` never shown |
| Small detectable confidence signal survives | None: ablation brackets zero 10/10 at both thresholds |
| Reasoning: mixed-sign null | Small harm at >= 3 for Sonnet/Opus (about 2.5x floor), null at >= 4 |
| Accuracy-calibration worsens PAD | Improves PAD-B for all ten at >= 3 (base-rate correction) |
| Prompting helps only the smallest model | Design C helps Sonnet/Opus at >= 3 only; hurts at >= 4 |
| ECE uncorrelated with PAD | Correlated at >= 4, not significant at >= 3; undefined-on-split argument kept |
| Ten configs, three vendors | Four-config Anthropic headline; ten in the full table; DeepSeek manual |

## 10. Status against the original masterprompt and the AAAI feedback

**Masterprompt (Phases 0-10): complete.** Every phase has a frozen table (see `methodology_checklist.md`).
One deviation: the spec's "worked example" was missing from the pasted text; an own example is in
`tests/test_brier.py`.

**AAAI-era feedback.**
- Done and carried forward: MI k-sensitivity (k = 3, 5, 10 point estimates); malignancy-threshold diagnostics (extended
  to the threshold curve, stable/sensitive split, exclude-3); bib file renamed to `references.bib`; 300 DPI
  figure setting; deletion of two unverifiable placeholder citations; the corrected facts from the consistency
  pass (gpt-oss was a differently elicited supplementary model; the colliding-pair example is pair 581:4 / 581:5;
  gap bound 0.033; "only half of configurations").
- Superseded by the venue change: AAAI page budget, section order, appendix placement, AAAI reproducibility checklist.
- Not yet done because they live in the text: rewritten methods/elicitation/provenance paragraph, hyperparameter
  and seed disclosure in prose, updated data appendix, updated limitations, bibliography check against
  `tmlr.bst`, TMLR broader-impact statement, rebuilt anonymised code supplement, `LIDC_XML_ROOT` repoint.

## 11. Honest limitations to carry into the paper
- One vendor in the headline; Gemini availability is closing; DeepSeek manual.
- Chunks are consecutive slices of corpus order, not random; no per-config repeat for most configs; the noise
  floor rests on three configs.
- Pre-specification covers the threshold and pi, not PAD-B; the exclude-3 analysis and the out-of-fold shift
  are post-hoc.
- P1 failed; P3 partly failed; the stability rule's cutoffs came from the configs that pass it.
- Intervention and over-reliance CIs are scan-clustered only; CV leakage and the oracle assume the vignette
  features carry the available signal.
- Text vignette, not images; LIDC-IDRI is the only source; construct validity of rating-based "difficulty" untested.

## 12. Remaining plan
1. Figures from the frozen tables (300 DPI).
2. TMLR rewrite: methods and results, then intro and abstract, then title last. Lead with the "adds nothing beyond
   three features, under two elicitations" thesis; threshold dependence second; report P1 as a failed prediction.
3. Cleanup: supplement rebuild and anonymity scrub (include pre-registration hashes), `LIDC_XML_ROOT`, bibliography,
   broader-impact statement.
4. Two small decisions: keep or drop the consistency appendix and the gpt-oss one-liner.
5. Your side: TMLR submission quota and a complete OpenReview profile.
