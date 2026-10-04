# Final methodology round: analysis plan and predictions (frozen before any of it is run)

Status of everything below: EXPLORATORY, post-hoc, review-driven. No paper text depends on it being confirmatory.
Common rules: scan-cluster bootstrap (two-stage call -> scan wherever a call id exists), GroupKFold by scan (5 folds,
fixed seeds), thresholds t = 3 (primary) and t = 4, all ten configs reported, headline four flagged
(sonnet:standard, sonnet:thinking, opus:standard, opus:thinking). Results are reported as they come out; a result
that undercuts a locked claim is reported as such.

## Task A. Residualized test
For each config and threshold: out-of-fold predictions (OLS and GBT, features subtlety/spiculation/margin) of
p (answer-folded forecast), c (stated confidence), a (binary answer) and f; residuals; partial correlation of
(v - v_hat) with (f - f_hat) for v in {p, c, a}, with CIs; OOF R2 of each v on the features.
Prediction (written first): for the headline four, the partial correlation of c with f is about zero, |r| < 0.1
(both estimators, t = 3 and t = 4). A positive partial correlation for the answer a would be informative
and is NOT a failure of the prediction. Whatever occurs is reported. Deliverable: whether the lead claim
("confidence adds nothing beyond features") stays as stated or is narrowed to "the answer may carry something,
the confidence magnitude does not".
Note: residuals are out-of-fold and fixed; CIs resample (call -> scan) clusters of the fixed residual pairs
and do not refit the feature models inside the bootstrap.

## Task B. Irreducible-noise floor for PAD-B
Per case, floor = f(1 - f) / (n_raters - 1) with n_raters in {3, 4}: the unbiased estimator of
Var(f_hat) = rho(1 - rho)/n for iid votes. Averaged over the corpus (and per config's cases) at each threshold,
overall and by pi-tier, with scan-cluster CIs. Validation by simulation: rho from a plausible Beta distribution
with mean 0.647, n = 3 and n = 4 in the corpus proportion (506 vs 907), confirm the estimator is unbiased and that
the PAD-B difference between two fixed forecasters is unaffected by the noise term.
Report excess over floor (PAD-B minus floor) for: the constant, the 3-feature oracle (gbt, ridge), the extended
oracle, and the headline four (raw, one-parameter-shift-corrected, recalibrated features + p).
Caveats to be stated: the raters are a fixed panel, not an iid sample; the plug-in floor is itself noisy.

## Task C. Disjoint-reader oracle check
Restrict to nodules with four readers (n = 907). For each of the 3 ways to split the four readers into two pairs
and both orientations (6 in total): features = mean subtlety/spiculation/margin of pair 1, target f = fraction of
pair 2 rating >= t (values in {0, 0.5, 1}). Same OOF grouped oracle (GBT, ridge). Report the margin over the
constant (oracle PAD-B minus the constant's PAD-B on the same cases) for (i) disjoint pairs and (ii) same-pair
features and target (control), averaged over splits.
Interpretation rule (written first): let R = disjoint margin / same-pair margin (averaged over splits). If R >= 0.5
the disjoint margin "retains most" of the same-pair margin and shared-rater dependence is judged not to be
driving the oracle; if R < 0.5 the oracle comparison is judged to be substantially inflated by shared raters and
this is reported with what it does to the oracle comparison. Margins are noisier because pair-based f takes only
{0, 0.5, 1}. Cases with a missing pair feature (nodules without characteristics) are dropped per split and the
n per split is reported.

## Task D (optional, gated). Other model families
Run only if ALL of: (a) an API key for a hosted open-weight provider already exists in the environment (no accounts
created); (b) projected total cost for all models <= 5 USD; (c) at most three models, each with an explicit
resolved model id logged per call. Otherwise write "not run".
Prompt: the confirmatory P(rating >= 3) prompt, unchanged (hashes in results/frozen_hashes.txt are verified before
any call), same 100-case chunks, default settings, thinking off where applicable, one run per model.
Predictions committed before any call: (1) the stated mean P(>=3) is below the true 0.647 for each model;
(2) the ablation (3 features vs 3 features + p, gbt, grouped CV) has a CI that brackets zero for each model.
No prediction is made about whether any model beats the constant. The committed stability rule is applied unchanged.
