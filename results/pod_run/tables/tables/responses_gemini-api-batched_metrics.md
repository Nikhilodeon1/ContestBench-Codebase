| model | n | r(c,pi) [95% CI] | PAD [95% CI] | signed PAD | signed PAD (high/cont/amb) | ECE\* |
|---|--:|---|---|--:|---|--:|
| gemini-api-batched | 1413 | 0.051 [0.001, 0.104] | 0.252 [0.241, 0.262] | -0.164 | -0.35 / -0.09 / +0.14 | 0.222 |
\* ECE computed on non-ambiguous cases only; 159 ambiguous (pi=0.5) cases have no majority outcome and are undefined for ECE -- the motivation for PAD.
