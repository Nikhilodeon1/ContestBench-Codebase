# Methodology checklist against the original 10-phase spec and later rulings

Status: DONE = computed and in frozen pod tables (results/tables/*_b.csv, results/data_manifest.json);
NEW-RERUN = code written and smoke-tested on the laptop, needs the next pod run to be frozen.

## Original spec (phases 0-10)
| Phase | Requirement | Status | Table / file |
|---|---|---|---|
| 0.1 | Elicitation format truth (JSON batch, chunk sizes, model strings) | DONE | results/reconciliation.md row 17; provenance*.jsonl |
| 0.2 | `extent` not shown to models | DONE | oracle_b.csv (vignette vs extended) |
| 0.3 | Scan-grouped CV and bootstrap | DONE | GroupKFold in oracle_b.py; scan + nested (call->scan) bootstrap |
| 1 | p, PAD-B, PAD-L1 relabel, s_i, validation test | DONE | brier.py, tests/test_brier.py (the spec's worked example was missing from the pasted spec; own example used) |
| 2 | Ten-config panel, constant p*=mean f, r(c,f), r(p,f), majority-match share | DONE | panel_b.csv |
| 3 | Honest oracle (3-feature) + extended (upper bound), 3 estimators, PAD-B and PAD-L1 | DONE | oracle_b.csv |
| 4 | Recalibrator + features-only vs features+p ablation, grouped CV | DONE | recalibration_b.csv |
| 5 | MI(p,f), permutation null, Bonferroni, k-sensitivity | DONE | mi_b.csv (k=3,5,10 point estimates) |
| 6 | Mechanism (p on features), cross-model correlation on p | DONE | mechanism_b.csv, cross_model_p_corr_b.csv |
| 7 | Prompting designs A/B/C and reasoning, held-out | DONE | intervention_b.csv, reasoning_b.csv (Bonferroni columns) |
| 8 | Over-reliance with s_i, per-cell (theta,beta) check | DONE | over_reliance_*_b.csv |
| 9 | Collision-free, threshold decomposition, threshold-free view | DONE | robust_subsets_b.csv, robust_threshold_curve_b.csv |
| 10 | Reconciliation table | NEEDS REFRESH after next pod run | results/reconciliation.md is pre-pinned-Gemini; regenerate from frozen tables |

## Later rulings
| Item | Status | File |
|---|---|---|
| Threshold sensitivity (>=3 primary pre-specified; >=4 post-hoc) | DONE | panel/threshold tables |
| Exclude-3 check (exploratory) | DONE | excl3_b.csv |
| Constant shift vs Platt (base-rate check) | DONE | base_rate_check_b.csv |
| Accuracy-calibration baseline under PAD-B | DONE | acc_cal_b.csv |
| Pinned Gemini re-collection, resolved model logged | DONE | provenance_pinned*.jsonl |
| Noise floor (rule written first), per-config floors | DONE | noise_floor_rule.md, noise_floor_b.csv |
| Chunk heterogeneity (free check), unrandomized chunks | DONE | chunk_heterogeneity_b.csv |
| Stability rule (frozen), applied | DONE | stability_rule.md, stability_b.csv |
| Confirmatory collection, P1-P3, out-of-fold shift | DONE | results/confirmatory/, confirmatory_*_b.csv |
| Pre-specification package | DONE | indivCB/supplement_prespec/ |
| Call-level (nested) clustering in CIs | NEW-RERUN for reasoning, acc_cal, recalibration/ablation, robustness, excl3, tiers (panel gap and confirmatory already nested) | panel_b._boot, oracle_b._cluster_ci, revision_b |
| ECE vs PAD-B correlation with CI + permutation p; ambiguous-tier random-label band; gpt-oss PAD-B | NEW-RERUN | ece_b.csv, ece_corr_b.csv, ece_ambiguous_band_b.csv, gptoss_supplementary_b.csv |
| PAD-B by pi-tier | NEW-RERUN | tiers_b.csv |
| Sample-consistency spot check under PAD-B | NEW-RERUN | consistency_b.csv |

## Known scope limits (to state in the paper)
- Intervention CIs are scan-clustered only (the 300-case held-out set is covered by 3 calls per design).
- Over-reliance CIs pool configs and are scan-clustered only.
- Oracle margins and the ablation use scan clustering for the oracle part (not an LLM call).
- DeepSeek has 3 manual calls: its nested CI is nearly degenerate; kept out of headline claims.
- gemini-2.5-flash is closing to new users; the pinned collection used the only key that still served it.
- Noise floor rests on 3 configs (Sonnet, Gemini standard, Haiku); gemini:thinking repeat incomplete.

## Open decisions
1. Sample-consistency appendix: keep as a descriptive appendix (n=73 cases, median 5 samples, agreement near 0.96) or drop.
2. gpt-oss-20b: keep as a one-line supplementary example of low ECE with worse-than-constant PAD-B, or drop.
