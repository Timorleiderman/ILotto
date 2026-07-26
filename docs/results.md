# Results

!!! info "Generated output"

    Rebuilt 2026-07-26 05:52 UTC from 1,629 draws through 2026-07-25. Do not edit by hand — `scripts/build_report.py` overwrites this page.

## Strategy scoreboard

Walk-forward over the last 300 draws. A random ticket averages **0.9730** matches; a uniform guess scores **3.6109** log loss. Log loss is shown only where the strategy emits genuine inclusion probabilities.

| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Lag-1 persistence | 1.0667 | +9.6% | +1.94 | 0.053 | 3.6126 | <span class="pill ok">no signal</span> |
| Hot numbers (last 100 draws) | 1.0567 | +8.6% | +1.73 | 0.084 | 3.6353 | <span class="pill ok">no signal</span> |
| EWMA frequency (half-life 50) | 1.0267 | +5.5% | +1.11 | 0.267 | 3.6297 | <span class="pill ok">no signal</span> |
| Rank ensemble | 1.0133 | +4.1% | +0.83 | 0.404 | — | <span class="pill ok">no signal</span> |
| Overdue (longest absence) | 1.0000 | +2.8% | +0.56 | 0.576 | — | <span class="pill ok">no signal</span> |
| Hot numbers (all-time) | 0.9967 | +2.4% | +0.49 | 0.624 | 3.6123 | <span class="pill ok">no signal</span> |
| Unpopular numbers (EV-aware) | 0.9967 | +2.4% | +0.49 | 0.624 | — | <span class="pill ok">no signal</span> |
| Neural (gru, set prediction) | 0.9867 | +1.4% | +0.28 | 0.777 | 3.6200 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (Markov) | 0.9733 | +0.0% | +0.01 | 0.994 | — | <span class="pill ok">no signal</span> |
| Cold numbers (all-time) | 0.9667 | -0.6% | -0.13 | 0.896 | — | <span class="pill ok">no signal</span> |
| Random ticket | 0.9333 | -4.1% | -0.82 | 0.413 | — | <span class="pill ok">no signal</span> |
| Date-conditioned generative (CB-6/37) | 0.9333 | -4.1% | -0.82 | 0.413 | 3.6349 | <span class="pill ok">no signal</span> |
| Sum-targeted combination | 0.9300 | -4.4% | -0.89 | 0.374 | — | <span class="pill ok">no signal</span> |
| Neural (transformer, set prediction) | 0.8533 | -12.3% | -2.47 | 0.013 | 3.6194 | <span class="pill ok">no signal</span> |

Verdicts apply Holm–Bonferroni across all 14 strategies. For scale, the luckiest of 2,000 purely random strategies scored **1.147** over the same draws — against the best strategy here at **1.067**.

## Randomness tests

| Test | Statistic | p | After correction |
|---|---:|---:|---|
| Ball frequency uniformity (chi-square) | 35.25 | 0.5038 | <span class="pill ok">no signal</span> |
| Strong-number uniformity (chi-square) | 2.70 | 0.8449 | <span class="pill ok">no signal</span> |
| Lag-1 serial independence (hot/due folklore) | 1.01 | 0.3151 | <span class="pill ok">no signal</span> |
| Appearance-gap distribution vs geometric | 11.14 | 0.4317 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (666 pairs, Sidak-corrected) | 3.73 | 0.1217 | <span class="pill ok">no signal</span> |
| Draw-sum distribution (two-sample KS vs simulated fair draws) | 0.02 | 0.7096 | <span class="pill ok">no signal</span> |
| First half vs second half frequency drift | 26.92 | 0.8634 | <span class="pill ok">no signal</span> |

## Memorisation vs prediction

The date-conditioned generative model takes a date rather than a history, and there is exactly one draw per calendar date — so the date is a **unique key**. A model with enough capacity can store a lookup table and reproduce past draws perfectly. **DKRR** (Date-Keyed Reconstruction Rate) is that seductive number: matched numbers on dates the model was *trained on*. It measures storage, not skill, and is never shown without the walk-forward column beside it.

| Model | DKRR (in-sample) | In-sample log loss | Walk-forward matches | p |
|---|---:|---:|---:|---:|
| Date-conditioned generative (CB-6/37) | 1.263 / 6 | 3.5930 | 0.9333 | 0.413 |
| Date lookup table (memorisation control) | 6.000 / 6 | 1.8001 | 0.9300 | 0.374 |

Chance is **0.9730** matched numbers and **3.6109** log loss. The lookup control reconstructs every training draw perfectly and is worth nothing on a date it has not seen. That gap is the entire point of the exercise.

## Next-draw picks

Published because they were asked for. These have exactly the same chance as any other six numbers.

| Strategy | Numbers | Strong |
|---|---|---:|
| Random ticket | 1 · 4 · 5 · 9 · 24 · 26 | 3 |
| Lag-1 persistence | 11 · 20 · 25 · 26 · 27 · 29 | 1 |
| Neural (gru, set prediction) | 5 · 8 · 9 · 26 · 28 · 29 | 3 |
| Unpopular numbers (EV-aware) | 32 · 33 · 34 · 35 · 36 · 37 | 1 |

---

Full charts and methodology: [the benchmark report](benchmark/index.html).
