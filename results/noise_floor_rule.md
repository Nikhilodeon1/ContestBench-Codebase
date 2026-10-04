# Noise-floor rule (written BEFORE the repeat runs of 2026-10-02)

Purpose: decide which differences in single-run PAD-B are distinguishable from repeat-run
(call-level) variation. Bootstrap CIs over scans capture case sampling only.

Design: repeat the identical protocol (same 100-case prompt chunks, same model string, same
thinking setting, same sampling defaults) for two configs from different families:
sonnet:standard (claude-sonnet-5, thinking disabled; chunks 1-3 already repeated, chunks 4-15 added)
and gemini:standard (gemini-2.5-flash, thinking_budget 0, temperature 0; all 15 chunks).

Estimator (fixed now):
1. For each config c, threshold t in {3,4}, and chunk k, let d_{c,k} = PAD-B(run 2, chunk k) - PAD-B(run 1, chunk k),
   PAD-B computed on that chunk's cases only. Chunks with fewer than 30 cases are dropped.
2. Pool d over both configs and all kept chunks (K_c chunks each). Let s = sqrt(mean of d^2) (uncentred, so
   a systematic run-level offset counts as noise).
3. For a comparison of two single-run corpus-level PAD-B values (each an average over about K = 14 calls),
   the repeat-to-repeat difference has SD s / sqrt(K). The NOISE FLOOR at threshold t is
   F_t = 1.96 * s / sqrt(14).
4. Reporting rule: any difference between single-run PAD-B values with |difference| < F_t is reported as
   "not distinguishable from repeat-run variation". Configs whose PAD-B values differ by less than F_t are
   NOT ranked against each other. Gaps to the constant are compared with F_t as well (the constant has no
   run-to-run noise, so for that comparison the floor is F_t / sqrt(2)).
5. Evidence base is two configs from two families; this is stated in the paper. A per-config floor is
   also reported. Configs with large per-call instability (see chunk-heterogeneity table) are flagged.

Secondary checks (fixed now): (a) answer-only forecast repeat-run PAD-B change should be near zero
(answers repeat 98-100%); (b) the standard-run chunk-mean confidence offsets between runs are reported
as a table.

## Addendum (written before the haiku/gemini-thinking repeats, after the first two repeats)
Result of the first two full repeats (sonnet:standard, gemini:standard; 14 chunks each): pooled
noise floor F_3 = 0.0073 and F_4 = 0.0040. The earlier 3-chunk Sonnet repeat (PAD-B +0.023 at t=3)
turned out to be an unrepresentative subset: over 14 chunks the mean repeat difference is +0.003.
The free chunk-heterogeneity check shows haiku, gemini:thinking and deepseek:thinking have
call-level swings far larger than these two stable configs, so the pooled floor probably
understates their noise. Before any claim about those configs we repeat haiku:standard and
gemini:thinking (all 15 chunks; cost ~ $0.2 and free tier). Rule unchanged: per-config floors
use the same estimator (1.96 * s / sqrt(14)); the pooled floor applies only to configs whose
per-call behaviour is stable; configs with a larger per-config floor use their own.
