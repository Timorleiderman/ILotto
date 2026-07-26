# Results

!!! info "Generated output"

    Rebuilt 2026-07-26 08:00 UTC from 1,629 draws through 2026-07-25. Do not edit by hand — `scripts/build_report.py` overwrites this page.

## Strategy scoreboard

Walk-forward over the last 200 draws. A random ticket averages **0.9730** matches; a uniform guess scores **3.6109** log loss. Log loss is shown only where the strategy emits genuine inclusion probabilities.

| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Hot numbers (last 100 draws) | 1.0550 | +8.4% | +1.38 | 0.166 | 3.6365 | <span class="pill ok">no signal</span> |
| Sum-targeted combination | 1.0550 | +8.4% | +1.38 | 0.166 | — | <span class="pill ok">no signal</span> |
| Lag-1 persistence | 1.0400 | +6.9% | +1.13 | 0.258 | 3.6134 | <span class="pill ok">no signal</span> |
| Hot numbers (all-time) | 1.0100 | +3.8% | +0.62 | 0.532 | 3.6134 | <span class="pill ok">no signal</span> |
| Cold numbers (all-time) | 0.9950 | +2.3% | +0.37 | 0.710 | — | <span class="pill ok">no signal</span> |
| EWMA frequency (half-life 50) | 0.9950 | +2.3% | +0.37 | 0.710 | 3.6311 | <span class="pill ok">no signal</span> |
| Date-conditioned generative (CB-6/37) | 0.9900 | +1.7% | +0.29 | 0.774 | 3.6326 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (Markov) | 0.9800 | +0.7% | +0.12 | 0.906 | — | <span class="pill ok">no signal</span> |
| Overdue (longest absence) | 0.9750 | +0.2% | +0.03 | 0.973 | — | <span class="pill ok">no signal</span> |
| Rank ensemble | 0.9750 | +0.2% | +0.03 | 0.973 | — | <span class="pill ok">no signal</span> |
| Neural (gru, set prediction) | 0.9700 | -0.3% | -0.05 | 0.960 | 3.6224 | <span class="pill ok">no signal</span> |
| Unpopular numbers (EV-aware) | 0.9650 | -0.8% | -0.13 | 0.893 | — | <span class="pill ok">no signal</span> |
| Random ticket | 0.8750 | -10.1% | -1.65 | 0.098 | — | <span class="pill ok">no signal</span> |
| Neural (transformer, set prediction) | 0.8200 | -15.7% | -2.58 | 0.010 | 3.6216 | <span class="pill ok">no signal</span> |

Verdicts apply Holm–Bonferroni across all 14 strategies. For scale, the luckiest of 1,000 purely random strategies scored **1.185** over the same draws — against the best strategy here at **1.055**.

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
| Date-conditioned generative (CB-6/37) | 1.263 / 6 | 3.5930 | 0.9900 | 0.774 |
| Date lookup table (memorisation control) | 6.000 / 6 | 1.8001 | 1.0050 | 0.589 |

Chance is **0.9730** matched numbers and **3.6109** log loss. The lookup control reconstructs every training draw perfectly and is worth nothing on a date it has not seen. That gap is the entire point of the exercise.

## Next-draw picks

Published because they were asked for. These have exactly the same chance as any other six numbers.

| Strategy | For draw of | Numbers | Strong |
|---|---|---|---:|
| Random ticket | next draw | 1 · 3 · 5 · 7 · 9 · 25 | 3 |
| Hot numbers (last 100 draws) | next draw | 5 · 7 · 8 · 18 · 21 · 37 | 2 |
| Neural (gru, set prediction) | next draw | 5 · 8 · 9 · 26 · 28 · 29 | 3 |
| Unpopular numbers (EV-aware) | next draw | 32 · 33 · 34 · 35 · 36 · 37 | 1 |
| Date-conditioned generative (CB-6/37) | 2026-07-28 | 11 · 13 · 27 · 28 · 34 · 35 | 5 |

---

Full charts and methodology: [the benchmark report](benchmark/index.html).
