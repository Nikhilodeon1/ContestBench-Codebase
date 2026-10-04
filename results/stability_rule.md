# Protocol-stability rule (committed before it is applied to the ten configs)

Purpose: decide which configs enter the headline panel using only outcome-independent quantities
(no PAD-B, no correlations with physician votes, no comparison with the constant).

Quantities, computed per config over its API/manual calls (one call = one prompt chunk; chunks with fewer
than 30 cases dropped; 100 cases per call, deepseek manual 471):
  A = range (max - min) across calls of the Malignant-answer rate.
  B = range across calls of the mean stated confidence on Benign answers.
  R = route (API with logged model string, or manual).
Reference set used ONLY to set cutoffs: the four Anthropic API configs sonnet:standard, sonnet:thinking,
opus:standard, opus:thinking (same vendor, same API route, API-logged settings; chosen because they form the
pre-planned capability ladder of the original design, not because of their results).
Cutoffs: A_max = 1.5 x max over the reference set of A; B_max = 1.5 x max over the reference set of B.
The multiplier 1.5 is a convention fixed here, not tuned. Observed reference maxima (descriptive, computed
immediately before writing this rule): A = 0.25, B = 0.159  ->  A_max = 0.375, B_max = 0.2385.
Rule: a config is STABLE iff A <= A_max AND B <= B_max AND R = API. Configs with fewer than 5 calls or a manual
route are labelled "manual / not assessable" and are excluded from headline claims regardless of A and B.
All ten configs are reported in the full table with A, B, R and the flag; nothing is dropped from the data.

Disclosure: the ranges of all ten configs were visible to the author when this rule was written. The rule
and the multiplier were nevertheless fixed before any headline table was regenerated under it.
