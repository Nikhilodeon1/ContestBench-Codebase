# Pre-specification record

What this folder documents: the malignancy binarization used for the primary analysis
(a rater "votes malignant" when malignancy >= 3 on the 1-5 LIDC scale; f = fraction of
readers voting malignant) was fixed in the project design before any model results were
analysed. The alternative >= 4 threshold is reported only as a post-hoc sensitivity analysis.

Files:
- `design_spec.md`: the project design document, first written 2026-07-18. It is a living
  document that was edited after that date (later sections record results and decisions), so
  its file timestamp alone is not proof; line 32 (definition of f with the >= 3 threshold) is
  the pre-specified statement.
- `config_first_commit_d648a2e.py`: `config.py` as of the repository's first commit
  (2026-07-20 11:17 -0700). It already contains `MALIGNANCY_POSITIVE = 3`.
- `commit_timeline.txt`: commit hashes, timestamps and subject lines for the whole history
  (author fields removed). The metrics-scoring CLI first appears in commit `70d7de8`
  (2026-07-20 17:37 -0700), after the first commit.

What was and was not pre-specified: the >= 3 threshold and the agreement rate
pi = max(f, 1-f) were pre-specified. The scoring rule used for the revised paper, PAD-B
(Brier score of the answer-folded forecast against f), was NOT: it replaces the original
L1 PAD in response to review. Commit timestamps are produced by local git and are
self-reported.
