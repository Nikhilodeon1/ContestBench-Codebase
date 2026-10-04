# ContestBench — Full Build Design

**Date:** 2026-07-18
**Status:** Approved design, pre-implementation (updated after strategy-mentor review)
**Scope:** Full-paper build of the `contestbench_overview` PDF, single-domain (radiology) first, as a Python package.

---

## 1. Overview

ContestBench measures whether a medical LLM's expressed confidence tracks the
*empirical distribution of physician opinion* on a case, rather than a single gold
label. On genuinely contested clinical cases (equally-qualified physicians
disagree), a well-behaved model should hedge; existing accuracy/calibration
benchmarks cannot detect a model that stays uniformly confident.

**Framing — decoupling, not overconfidence (updated 2026-07-19 after pilot sweep).**
The PDF hypothesized systematic *overconfidence* on contested cases (PAD⁺ > 0). The
50-case pilot sweep found something different and stronger: **model confidence is
flat (~0.60) and essentially independent of physician agreement in either
direction** — near-zero correlation, and *under*-confidence on high-agreement cases
rather than overconfidence on contested ones. We therefore frame the finding as
**confidence decoupled from clinical difficulty**, not overconfidence. This is more
novel (undocumented in the field), matches the data, and is harder for a reviewer to
dismiss than "sometimes overconfident."

**Central hypothesis (reframed):** model confidence is statistically independent of
physician agreement — the correlation is near zero, and confidence barely moves while
the physician agreement rate ranges from 0.5 (split) to 1.0 (unanimous).

**Physician agreement rate `π` (definition — critical).** `π = fraction of readers
agreeing with the majority judgment = max(f, 1−f)`, where `f = frac(malignancy ≥ 3)`.
So `π ∈ [0.5, 1]`: 0.5 = even split, 1.0 = unanimous. This matches model `confidence`
(certainty in the model's *own* answer, also ~[0.5, 1]). NOTE: `f` itself (frac
malignant, ∈ [0,1]) is only an intermediate — comparing confidence to `f` directly is
a category error (a correctly-confident model on a unanimous-benign case, f=0, would
score a huge spurious PAD). Tiers are computed from `π` and are unaffected (symmetric).

**Core metric — PAD (Physician Agreement Deviation):** `PAD = |c − π|`, the L1
analogue of the Brier proper scoring rule with the physician agreement rate as the
target. Unsigned PAD is the headline miscalibration number.

**Primary statistic:** **Pearson correlation between model confidence and π** plus
the **confidence-vs-π scatter as the lead figure** (it shows the flat band directly).
Signed PAD by tier is reported as *evidence of decoupling* (negative on high-agreement,
near-zero on ambiguous — confidence stays put while ideal rises 0.5→1.0), NOT as an
overconfidence claim. Tier breakdown is secondary/illustrative.

**Target venue:** near-term NeurIPS 2026 / ML4H workshop (deadlines ~Aug–Sept 2026);
full NeurIPS D&B paper targets the 2027 cycle. This spec covers the radiology-only
build that serves both.

## 2. Current state (pilot notebook)

`main.ipynb` is a working prototype worth porting: LIDC XML parse (CELL 2),
π aggregation (CELL 3), radiologist vignette prompt (CELL 4), Gemini client (CELL 5),
confidence regex (CELL 6), retry loop (CELL 7), correlation + PAD + scatter
(CELL 9–11). Pilot n=10: Pearson −0.21 (p=0.56), PAD-by-tier low 0.585 → high 0.10.
Direction right, sample far too small.

**Defects fixed in the full build:**

1. **Nodule matching.** Pilot `groupby('id')` uses the noduleID string, which is not
   globally unique across scans → merges unrelated nodules. **We do NOT use pylidc**
   (it requires DICOM volumes; we have none — see §3.1). Instead we do **XML-native
   centroid matching**: each reader-nodule's ROI carries `edgeMap` (x,y) +
   `imageZposition` (z); match nodules across the 4 reading sessions within a scan by
   centroid proximity.
2. **Corpus size hack.** Pilot stops after 50 files, keeps `head(5)+tail(5)`. Replace
   with full walk + per-tier sampling.
3. **π discreteness.** With 4 raters, `π ∈ {0,.25,.5,.75,1}`; 3-reader nodules add
   {.333,.667}. Handled by the redefined tiers in §4.1 and by using Pearson as the
   primary statistic.

## 2a. Primary result (full corpus, 2026-07-19)

Gemini-3.5-Flash over the full corpus via the free batch route. **Coverage 1,400/1,413
(99.1%)**; 13 missing (12 contested, 1 high, **0 ambiguous** — load-bearing tier fully
covered).

- **Headline:** Pearson(confidence, agreement π) = **−0.029, 95% CI [−0.081, 0.024],
  p=0.28** (bootstrap 5k). A tight, well-powered null (N ≈ 3× the power target) — model
  confidence is statistically independent of physician agreement.
- **Decoupling by tier:** mean confidence is flat ~0.56–0.60 while ideal rises 0.5→1.0.
  Signed PAD: high −0.44 (badly under-confident on unanimous cases), ambiguous +0.10.
  Mean PAD 0.311.
- **Anti-calibration twist (abstract-worthy):** confidence is *highest* on ambiguous
  cases (0.596) and *lowest* on unanimous ones (0.559) — mildly the wrong direction.
- **Variance twist:** unlike gpt-oss (nearly constant ~0.60, sd 0.07), Gemini confidence
  *varies* (sd 0.25) but on something other than clinical difficulty — a stronger claim
  than flatness. (Discussion should speculate what drives the variance: case length?
  disease category? nothing? — a reviewer will ask.)
- Figure: `results/figures/gemini_full_decoupling.png` (lead figure).
- Two models now agree (Gemini n=1,400; gpt-oss pilot n=50), across very different
  confidence distributions.

## 2c. FINAL cross-vendor panel (2026-07-20, supersedes 2b)

10 configs, 3 vendors, **full N=1,413 each**, all on identical standardized JSON
batch elicitation. Regenerate with `python -m contestbench.cli report`.

| config | conf | r(c,π) | PAD | gap vs constant [95% CI] | signed h/c/a |
|---|--:|--:|--:|---|---|
| haiku:std / think | .74/.75 | +.03/−.13 | .163/.157 | +.020[+.016,+.024] / +.014[+.011,+.016] | −.26/+.02/+.25 · −.26/+.04/+.26 |
| sonnet:std / think | .73/.65 | +.18/+.24 | .167/.189 | +.024[+.018,+.030] / +.047[+.040,+.053] | −.24/−.01/+.21 · −.31/−.08/+.13 |
| opus:std / think | .85/.82 | −.22/−.03 | .197/.182 | +.054[+.047,+.061] / +.039[+.032,+.046] | −.18/+.15/+.39 · −.18/+.10/+.35 |
| gemini:std / think | .75/.78 | −.02/−.13 | .177/.191 | +.034[+.029,+.040] / +.048[+.042,+.054] | −.25/+.04/+.27 · −.24/+.07/+.32 |
| deepseek:std / think | .64/.70 | +.26/−.07 | .198/.179 | +.055[+.049,+.061] / +.036[+.032,+.040] | −.32/−.10/+.12 · −.31/−.00/+.20 |

**Headline (lead the paper with this):** the **case-blind constant** (always answer
0.75, never look at the case) achieves **PAD 0.144**, and **10/10 configs are
significantly worse than it** — every gap CI strictly positive. No LLM tested
extracts usable case-difficulty signal. Figure `baseline_floor_all.png`.

**Oracle / upper-bound baseline (directive A, 2026-07-23).** Gradient-boosted tree
predicting agreement π from case features, 5-fold CV, out-of-fold (no leakage), at
three levels of independence from the malignancy ratings that define π:

| oracle variant | features | PAD |
|---|---|--:|
| all-features | + n_raters, malignancy_extremity (share source with π) | **0.095** |
| rating-independent | subtlety, spiculation, margin, extent | **0.127** |
| geometry-only (fully rater-independent) | extent (pure ROI polygon geometry) | **0.141** |
| — case-blind constant | — | 0.143 |
| — best LLM (haiku:thinking) | — | 0.157 |

**Two readings, both real:** (1) **geometry alone barely helps** (0.141 ≈ constant) —
recoverable agreement signal is *not* in pure nodule size. (2) The signal lives in the
**subjective Likert imaging assessments** (subtlety/spiculation/margin), and the
rating-independent oracle on them (**0.127**) still beats *every LLM* — even though the
LLM prompt **contains those very features**. So LLMs are handed the exact inputs a
gradient-boosted stump uses to beat them, and still fail to use them. Lead the paper
with the rating-independent 0.127 (independent of the malignancy rating); geometry-only
0.141 is the fully rater-independent bound; all-feature 0.095 is the upper bound with
the shared-source caveat. Not a novel prediction method (cite LIDC panel-opinion 2009,
multi-rater disagreement 2019).

**Post-hoc recalibrator (Fix H, 2026-07-23) — the proposed intervention.** A
per-config GBT mapping `[model confidence + rating-independent features] → π`, same
5-fold out-of-fold protocol. Every config's PAD collapses to ≈ the oracle: before
0.157–0.198 → **after 0.126–0.127**, closing ~100% of the gap to the rating-
independent oracle, and **all 10 configs now beat the case-blind constant.** Two
honest readings: (1) the decoupling is **fully fixable post-hoc** with a trivial
feature-based layer; (2) presented as an **ablation** — oracle (features only) 0.127
vs recalibrator (features + confidence) 0.126 — **model confidence adds only ~0.001
PAD.** Verified two ways after a reviewer-style check (2026-07-23): the confidence
column is real and varying for every config (0 nulls, std 0.05–0.13, 15–50 unique
values — the 0.126 convergence is a finding, not a dropped column); and confidence's
held-out **permutation importance is small but non-zero (0.02–0.04, above noise for
most configs)** — so confidence is not literally discarded, it carries a *small,
largely redundant* signal. Correct framing (NOT "contributes nothing"): the model's
confidence is **nearly redundant with four cheap features**; recalibration succeeds
overwhelmingly by using the features, and the model's confidence adds a marginal
sliver on top. The mitigation works by routing *around* the model's confidence, not
by extracting from it.

**Over-reliance simulation (Fix K, 2026-07-25).** Lee & See (2004) reliance model:
warranted reliance = π; clinician relies with `P(rely)=σ(β(c−θ))`; over-reliance =
mean `max(0, P(rely)−π)`, swept over θ×β, on confidences **pooled across all 10
configs**. Decoupled 0.064 → ideal (c=π) 0.028 (calibration ~halves over-reliance).
The recalibrator (H) lands at 0.070 — it **levels** all models: helps the
over-confident (opus 0.114→0.070) but *hurts* the naturally-cautious (deepseek
0.025→0.070), because it inherits the oracle's residual over-confidence on ambiguous
cases. Verified per-tier and per-model, robust across the grid; reproducible via
`cli reliance`.

**WRITEUP FRAMING (mentor directive, use later):** present K's recalibrator result
NOT as "H has a flaw" but as a **second-order contribution — PAD-minimization is not
the same target as over-reliance-minimization.** Optimizing an average calibration
metric (PAD, ECE, any average-based metric) can move a safety-relevant downstream
quantity the wrong way for a subset of models — a distinction average metrics cannot
see on their own. This motivates pairing deployment-relevant simulations with
calibration metrics, a general point beyond this paper's fix.

**Vignette-template audit + oracle relabel (weekend #5, 2026-07-26) — CORE FRAMING.**
- **Disclose in the corpus/method section (not just limitations):** 1,413 cases →
  only **690 unique presented vignettes**; **54.8% of cases share an identical
  vignette (subtlety/spiculation/margin at 2dp) with a different-π case** (mean
  π-spread 0.30). For those, the model gets an identical prompt and *cannot* track π.
- **The oracle is relabeled "vignette-optimal" everywhere** — it is the best
  achievable from the coarse prompt, NOT an absolute ceiling. Bucket-mean baselines
  (no ML): in-sample 0.067 (leaky floor), **leave-one-out 0.140 ≈ the case-blind
  constant 0.143**. The GBT/linear oracle (0.127/0.133) beats the LOO bucket-mean
  only by *generalizing across similar* vignettes; the vignette affords ~0.015–0.02
  PAD of usable signal over ignoring the case.
- **Claim, precisely worded:** no LLM reaches the vignette-optimal ceiling (0.127) —
  in fact all are worse than a case-blind constant — even though that ceiling is
  provably achievable from the identical prompt the models see.
- **Elicitation was NOT temperature-0:** the 10-config panel was collected via
  manual chat apps at app-default (stochastic, uncontrolled) temperature, so
  colliding cases are NOT identical-by-construction. And the **collision-free 45%
  subset independently shows the decoupling**: all 10 configs PAD 0.16–0.22 > the
  subset constant 0.150; pooled r(conf,π) = +0.074 (negligible). So the finding is
  not a collision artifact.

**Reasoning null (replicated across 3 vendors / 5 families):** thinking − standard
PAD = haiku −0.006, sonnet +0.023, opus −0.015, gemini +0.014, deepseek −0.019.
Mixed signs, all |Δ| ≤ 0.023. Reasoning neither fixes nor worsens the decoupling.

**Capability trend (suggestive only):** signed-PAD-ambiguous vs rank —
standard slope +0.072 [+0.067,+0.077], thinking +0.048 [+0.039,+0.057]. Caveat:
3 rungs, non-monotonic (Sonnet dips below Haiku), Opus-driven; the CI is a cluster
bootstrap over the 159 shared ambiguous cases and does **not** capture model-count
uncertainty. Report as a hypothesis for future work, not a law.

**Mechanism (Fix 5, locked feature set):** confidence is *not* noise for most models
(R² 0.19–0.76 on subtlety/spiculation/margin/extent/n_raters/malignancy-extremity),
but the mappings **contradict each other across models** (spiculation +0.60 in
sonnet:thinking vs −0.61 in opus:standard). Cross-model confidence correlation:
mean +0.23, range **[−0.50, +0.91]** — no shared latent difficulty signal. Models
apply idiosyncratic surface-feature heuristics unrelated to where physicians disagree.

**ECE vs PAD:** ECE and PAD are near-uncorrelated across models (gpt-oss: ECE 0.050
but PAD 0.170). ECE cannot see agreement-decoupling.

**Excluded:** gpt-oss batched re-run (Groq free tier, 8k tokens/min, could not
complete a matched run — best 324/1413). Its earlier *single-call* data (n=1,317,
PAD 0.170) is available as a footnote but is not elicitation-matched.

## 2b. Earlier cross-model results (superseded by 2c; kept for history)

Four model/conditions. Gemini + gpt-oss at full N; Opus 4.8 at n=150 (thinking vs
standard, same cases — the controlled reasoning contrast). π = agreement in [0.5,1].

| model | n | conf | r(c,π) [95% CI] | PAD | signed PAD high/cont/amb |
|---|--:|--:|---|--:|---|
| gemini-3.5-flash | 1400 | 0.57 | −0.029 [−0.08, +0.02] | 0.311 | −0.44 / −0.14 / +0.10 |
| gpt-oss-20b | 1317 | 0.71 | −0.027 [−0.08, +0.02] | 0.170 | −0.29 / −0.00 / +0.21 |
| opus:standard | 150 | 0.81 | +0.131 [−0.02, +0.28] | 0.155 | −0.16 / +0.06 / +0.33 |
| opus:thinking | 150 | 0.79 | −0.072 [−0.23, +0.09] | 0.179 | −0.21 / +0.06 / +0.34 |

**Findings:**
1. **Decoupling is universal** — every r's 95% CI brackets 0, including a flagship
   reasoning model. Not a small/cheap-model artifact.
2. **Capability raises confidence level (0.57→0.81) but not its correlation with
   agreement (~0).**
3. **Capability→overconfidence inversion (abstract-worthy):** signed PAD on ambiguous
   cases *grows* with capability (+0.10 → +0.21 → +0.33) while shrinking on easy cases
   (−0.44 → −0.16). More capable models are more overconfident exactly where physicians
   disagree most. Figure `results/figures/multimodel_signed_pad.png`.
4. **Claim 3 (reasoning effect, Opus, controlled):** extended thinking mildly *worsens*
   calibration — PAD 0.155→0.179, ECE 0.344→0.356, correlation +0.131→−0.072. Direction
   matches the PDF's claim 3 (reasoning ≠ safer), but n=150 CIs overlap — report as
   **suggestive, not significant.**

**Lead figures:** `gemini_full_decoupling.png` (flat confidence vs rising ideal),
`multimodel_signed_pad.png` (the inversion). **Caveats:** Opus n=150; gpt-oss used JSON
elicitation while Gemini/Opus used the pipe batch prompt (documented, §4.2).

## 3. Data availability & corpus size (verified 2026-07-18)

### 3.1 What we actually have
- **1,319 LIDC annotation XML files. Zero DICOM images.** (`find … -iname '*.dcm'` = 0.)
- Therefore pylidc is not an option; matching is XML-native (§2 defect 1).
- XML provides per reading session (reader): `unblindedReadNodule` → `roi`
  (`imageZposition`, `imageSOP_UID`, `edgeMap` x/y) and `characteristics` →
  `malignancy` (1–5). Nodules <3mm lack full characteristics and are excluded.

### 3.2 Usable-N estimate (rough matcher: greedy, xy<30px, z<3mm)
- **~1,413 matched nodules** with malignancy from ≥3 readers, across 1,036 scans.
- Tier counts (see §4.1 definition): **high 546, contested 708, ambiguous 159.**
- **Clears the PDF power target (N=480 for 80% power on ΔPAD=0.1) ~3×** on LIDC alone.
  BI-RADS is not needed for power (may be added later only for finer-resolution π).
- **Caveats:** the ambiguous tier (exact π=0.5) is thinnest (~159) and structurally
  needs 4-reader even splits (it is 100% 4-reader nodules at every threshold tested).
  The exact **4-reader-only** N (cleanest π support) is reported once the production
  matcher is built.

### 3.3 Matcher threshold-sensitivity (verified 2026-07-18)
Because matching thresholds feed tier assignment (the primary DV via π), we checked
tier stability across three settings. This is the corpus-level analogue of the
normalization sensitivity analysis (§4.5).

| Matcher (xy/z) | Total N | high | contested | ambiguous | ambiguous stability |
|----------------|--------:|-----:|----------:|----------:|---------------------|
| tight 20px/2mm | 1375 | 535 | 688 | 152 | Jaccard 0.944 vs base |
| base 30px/3mm  | 1413 | 546 | 708 | 159 | — (reference) |
| loose 40px/4mm | 1418 | 551 | 706 | 161 | Jaccard 0.963 vs base |

**Finding:** total N swings only ~±3%; the ambiguous tier moves 152↔161 (±5%) and is
essentially the *same nodule set* (Jaccard ≥0.94). Tier assignment is **not** a hidden
researcher degree-of-freedom. **Ships in Appendix A of the paper, next to the
normalization sensitivity check** — same species of robustness argument.
(Numbers from the rough estimator; production matcher will refresh them.)

## 4. Architecture

Python package, no notebooks, one concern per file. Top-level layout follows the
PDF's reviewer-facing structure (`data/ eval/ metrics/ analysis/ reproduce.sh`).

```
contestbench/
  config.py              # paths, model registry, tier thresholds, seeds
  data/
    lidc_loader.py       # parse XML → per-reader nodule ROIs + malignancy
    matching.py          # XML-native centroid matching across reading sessions
    corpus.py            # π, tier assignment, sampling → corpus.parquet
    dsm5_loader.py       # STUB/interface only (psychiatry, phase 2 — not populated)
    normalization.py     # %/κ/ICC → common scale + 3-choice sensitivity (placeholder)
  eval/
    base.py              # ModelAdapter interface: query(prompt) -> text
    groq_adapter.py      # gpt-oss-20b/120b (+reasoning_effort), deepseek-r1-distill, llama-3.3-70b
    registry.py          # model name → (adapter, params incl. reasoning_effort, temperature)
    prompts.py           # vignette templates
    parse.py             # extract answer + confidence
    runner.py            # corpus × models loop, retry/backoff, cache, provenance log
    cache.py             # disk cache keyed (model, params, prompt-hash)
  metrics/
    pad.py               # PAD, PAD⁺, PAD² (Brier)
    ece.py               # ECE + degeneracy demo on ambiguous tier
    stats.py             # Pearson/Spearman (primary), bootstrap CIs
  analysis/
    figures.py           # scatter+calibration, PAD-by-tier, reasoning_effort sweep
    tables.py            # markdown/LaTeX results tables
  cli.py                 # build-corpus | run | score | figures
reproduce.sh             # end-to-end: build-corpus → run → score → figures
tests/                   # unit tests for pure functions
data/                    # generated: corpus.parquet, responses.parquet, cache/
results/                 # figures/, tables/, provenance logs
```

**Data flow:**
`XML → lidc_loader → matching → corpus (π+tiers) → corpus.parquet → eval.runner (× models) → responses.parquet → metrics → results → analysis → figures/tables`

### 4.1 Corpus builder (`data/`)
- `lidc_loader`: per reader-nodule → centroid (mean edgeMap x,y; mean z) + malignancy.
- `matching`: cluster reader-nodules within a scan by centroid proximity.
  **Matcher config is FROZEN at the reported default `xy<30px, z<3mm`** (§3.3 proved
  it stable). It lives in `config.py`; any later change is treated as a *new
  sensitivity run*, never a silent update to the base N — so N cannot drift across
  paper sections.
- **Exclusion log (required):** the builder emits `results/exclusions.csv` accounting
  for every raw annotation dropped and why — `<3 readers`, no malignancy
  characteristic (<3mm nodule), unmatched by the matcher, etc. This is the
  corpus-construction-bias rebuttal for benchmark-track reviewers: exact drop counts
  by reason, not a scramble later.
- `corpus`: `π = frac(malignancy ≥ 3)`; **tiers redefined for discrete π**:
  **high = π∈{0,1}, contested = π∈{.25,.75}, ambiguous = π=.5** — stated in the paper
  as a deliberate adaptation to 4-rater support, not an oversight. Sample per tier;
  write `corpus.parquet[id, scan, pi, n_raters, subtlety, spiculation, margin, tier]`.
  Provide a **4-reader-only** view for the clean-π tier analysis.

### 4.2 Model harness (`eval/`) — Groq-only for now
- `ModelAdapter.query(prompt)->text`. `groq_adapter` implements it.
- **Primary analysis:** `gpt-oss-20b` and `gpt-oss-120b` × `reasoning_effort` ∈
  {low, medium, high} — a within-model dose-response test of novelty claim 3
  (reasoning worsens calibration on contested cases). Cleaner causal design than the
  PDF's cross-model comparison.
- **Robustness panel (replication, not headline):** `deepseek-r1-distill-llama-70b`
  (reasoning — note: a *distill* of R1, not full R1; flagged as a limitation) +
  `llama-3.3-70b` (non-reasoning). Report same direction or null either way.
- No paid-provider keys currently (no full R1 / GPT-4o / Claude). Revisit if a key
  becomes available.
- `runner`: corpus × models with retry/backoff + **disk cache** + **provenance log**
  (exact model id, temperature, reasoning_effort, prompt-hash per call).
- **Batch route (`eval/batch.py`) — free Gemini, file-in/file-out.** `export-batch`
  writes self-contained prompt file(s) (instruction header + all cases, pipe-format);
  the user runs them through a free unlimited Gemini-3.5-Flash platform manually and
  saves the returned answer files; `import-batch` parses (tolerant regex), matches by
  ID to π/tier, and scores. This produced the **primary full-corpus result** (§2a).
  **Two limitations, both logged in `results/provenance_gemini.json`:**
  (1) the batch pipe-format prompt is **NOT identical** to the gpt-oss JSON single-call
  prompt — same clinical framing/feature semantics, different format — so direct
  cross-provider row comparison is confounded until a second model is re-run through the
  *same* batch prompt (planned for the robustness model). Each within-model result is
  internally valid. (2) The platform is a **black box on temperature and exact model
  version** (not exposed); the preserved prompt+response files are the reproducibility
  artifact, and temperature/version are flagged UNKNOWN.
- **Elicitation & parsing (updated 2026-07-19).** The pilot sweep showed
  `reasoning_effort=high` dropped the confidence line 32–42% of the time (the model
  reasons at length and omits format) — missing-not-at-random on the exact condition
  we test. Fix: **JSON structured output as primary** (`response_format=json_object`,
  schema forces `answer` + `confidence` fields so they can't be omitted) **plus a
  regex fallback** for malformed JSON. If, after this, high-effort drops do NOT fall to
  ~0, that is a **real finding** about verbosity vs. extractable confidence at high
  reasoning effort — report it, do not engineer it away.

### 4.3 Metrics (`metrics/`)
- `π = max(f, 1−f) ∈ [0.5,1]` (majority agreement) is the target everywhere — never
  raw `f`. See Overview.
- `pad`: `PAD = mean|cᵢ−πᵢ|` (headline, unsigned). `PAD⁺ = mean(cᵢ−πᵢ)` is reported
  **as evidence of decoupling** (expected negative on high-agreement, ~0 on ambiguous —
  confidence stays flat while π rises), NOT relabeled as overconfidence. `PAD² =
  mean(cᵢ−πᵢ)²` (Brier) in the appendix.
- **Primary:** Pearson(c, π) + p-value + bootstrap 95% CI, and the **confidence-vs-π
  scatter as the lead figure**. Spearman secondary.
- PAD family per model × per tier × overall, each with bootstrap CI (illustrative).
- Also report **mean/std of confidence per model** — the flatness (small std,
  agreement-independent mean) is the direct quantitative form of the decoupling claim.
- `ece`: binarize by majority vote, compute ECE, demonstrate it is degenerate/wide-CI
  on the ambiguous tier (PDF §4.4 — itself a result). **Demonstrated (2026-07-20):** on
  the 159 ambiguous cases the binary outcome is undefined, so ECE swings across a
  **0.15-wide** 95% band under random label assignment, while PAD is a single stable
  label-free value (0.275). Figure `results/figures/ece_degeneracy.png`.

### 4.4 Psychiatry (`data/dsm5_loader.py`) — STUB ONLY
- Ship the interface matching the corpus schema; **do not populate.** ICC→agreement
  conversion is the pipeline's weakest link; deferred to phase 2 with MD face-validity
  review. Keeps the architecture domain-agnostic without shipping shaky data.

### 4.4b BI-RADS second-source search — CHECKED-AND-REJECTED CANDIDATES
A continuous-agreement radiology source (BI-RADS inter-reader) would improve π
resolution. Candidates checked and rejected as **consensus-only** (one label per
case, no independent per-reader scores — unusable for π):
- **VinDr-Mammo** — consensus-only.
- **CMMD** — consensus-only.
- **CBIS-DDSM (checked 2026-07-26):** 4 case-description CSVs (mass/calc × train/test),
  3,568 rows / 2,050 lesions. **No reader/annotator/session column.** Every
  (lesion + view) is exactly one row; the only multiplicity is CC vs MLO *views* of
  the same lesion (same radiologist, two projections), disagreeing on assessment for
  just 55/2,050 lesions. No legacy DDSM `.OVERLAY`/`.ics` multi-radiologist outline
  files present (metadata-only download). Same failure mode as VinDr/CMMD.

Per mentor directive (2026-07-26): **stop searching for BI-RADS sources** after five
checked candidates; LIDC-IDRI remains the sole π source. BI-RADS integration is off
the table unless a genuinely per-reader source surfaces later.

### 4.5 Normalization (`data/normalization.py`) — placeholder
- Identity on LIDC (counts need no normalization). Scaffold the **three-choice
  sensitivity analysis** now (raw % ÷100; κ→implied agreement; ICC→proxy) so it's
  wired before any second source (BI-RADS) is added.

### 4.6 Analysis (`analysis/`)
- Scatter c-vs-π + calibration line; PAD-by-tier bars per model; **reasoning_effort
  sweep** (the claim-3 figure). Tables in markdown + LaTeX.

## 5. Reproducibility (from day one, not the end)
- Top-level `data/ eval/ metrics/ analysis/ reproduce.sh` layout.
- Every model call disk-cached; never re-pay on rerun or added model.
- Per-call provenance: model id, temperature, reasoning_effort, prompt-hash, timestamp.
- `reproduce.sh` runs the full pipeline end-to-end from raw XML to figures.

## 6. Build order
1. Corpus builder (XML loader + matching + tiering + sampling).
2. Model harness (Groq adapter + cache + provenance + elicitation).
3. Metrics (Pearson primary, PAD family, ECE, bootstrap).
4. **Pilot re-run deliverable:** corrected tiers + reasoning_effort sweep on the
   10-nodule pilot; report scatter + PAD-by-tier to confirm the signal survives the
   new tiering **before** committing to the full ~1,400 build.
5. Full corpus build + full run.
6. Normalization sensitivity placeholder + analysis/figures/tables.
7. (Phase 2, later) psychiatry population.

## 7. Testing
Pure functions (XML parsing, centroid/matching, π, tier assignment, PAD/ECE math,
normalization formulas, confidence parsing) get unit tests. Adapters/runner are
exercised via the pilot re-run, not unit-tested.

## 8. Known limitations (carry into paper)
- π is coarse (mostly 4 raters → 5 levels); finer π needs a continuous-agreement
  source (e.g., BI-RADS) — future work.
- Radiology vignettes use extracted features, not raw CT images (no DICOM available;
  stated as an explicit modeling choice).
- Robustness-panel reasoning model is an R1 *distill*, not full R1.
- Tier bands are adapted to discrete 4-rater support, diverging from the PDF's
  continuous percentage bands (deliberate).

## 9. Non-goals (YAGNI)
- No raw-image / vision pipeline. No fine-tuning. No web UI / hosted leaderboard.
- No psychiatry data in this build (stub only).

## 10. Safest-default decisions pending a real venue (2026-07-29)

No target venue is locked (see paper draft's venue-context note); several
AAAI-2027-template formatting/structural choices were made against the
*safest universal default* — the option valid under the widest range of
possible real venue rules — rather than a venue-specific instruction, since
no such instruction exists yet to check against. Mentor directive
(2026-07-29): do not adopt a placeholder venue to resolve these now (false
precision, re-verification needed anyway); keep applying the safest default
and log it here so a real venue choice is a five-minute audit against this
list, not a paper re-read. Add an entry any time this pattern recurs.

1. **Appendix placement (2026-07-29).** AAAI distinguishes "Content
   Appendices" (unconditionally part of the main paper, before the Ethical
   Statement, lettered sections via `\appendix`) from "Supplementary/
   Technical Appendices" (after References, allowed only if the specific
   venue permits). Placed all appendix material (matcher robustness,
   malignancy-threshold sensitivity, capability-trend detail,
   non-verbalized-confidence detail) in the first, unconditionally-safe
   category. **Re-check if a real venue is chosen:** confirm its content-
   appendix page-limit policy (content appendices count toward the page
   limit) and whether it would prefer some of this material moved to a
   post-references technical appendix instead.
