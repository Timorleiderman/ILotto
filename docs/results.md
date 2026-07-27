# Results

!!! info "Generated output"

    Rebuilt 2026-07-27 20:50 UTC from 1,629 draws through 2026-07-25. Do not edit by hand — `scripts/build_report.py` overwrites this page after every draw, so the numbers below shuffle from build to build. That shuffling is not a bug in the strategies; it is what a table of noise looks like when you keep re-rolling it.

## Strategy scoreboard

Every strategy was asked to pick 6 numbers for each of the last **300 draws**, seeing only the draws before each one. The columns:

| Column | Meaning | What counts as unremarkable |
|---|---|---|
| **Mean matches** | Of the 6 numbers picked, how many were actually drawn, averaged over the 300 test draws | A random ticket averages 6×6/37 = **0.9730**. For this window, anything in **0.876 – 1.070** is within ±2 standard errors of chance |
| **Lift** | Mean matches relative to chance, as a percentage | ±10% is the ±2-standard-error band — single-digit lifts are noise at this sample size |
| **z** | How many standard errors the mean sits from chance | \|z\| < 2 is unremarkable, and with 18 strategies the *largest* \|z\| is expected to exceed 2 by chance alone |
| **p** | Probability that a skill-less strategy would score at least this far from chance | **Not** the probability the strategy works. One row near p ≈ 0.06 is expected among 18 rows |
| **Log loss** | Grades the full probability distribution the strategy assigned, not just its top 6. A uniform guess scores exactly ln 37 = **3.6109**. Genuine information would show up as a value *consistently* below that across rebuilds — a hair below it on one window is sampling noise, and no significance test is applied to this column | “—” means the strategy outputs ranks, not probabilities, so this score would be meaningless for it |
| **Verdict** | Significance after Holm–Bonferroni correction across all 18 rows | “no signal” is the expected outcome; a genuine “signal” would survive the correction *and* recur in the next rebuild |

| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Lag-1 persistence | 1.0667 | +9.6% | +1.94 | 0.053 | 3.6126 | <span class="pill ok">no signal</span> |
| Hot numbers (last 100 draws) | 1.0567 | +8.6% | +1.73 | 0.084 | 3.6353 | <span class="pill ok">no signal</span> |
| Perfect-fit function (DFT) | 1.0467 | +7.6% | +1.52 | 0.128 | — | <span class="pill ok">no signal</span> |
| EWMA frequency (half-life 50) | 1.0267 | +5.5% | +1.11 | 0.267 | 3.6297 | <span class="pill ok">no signal</span> |
| Rank ensemble | 1.0133 | +4.1% | +0.83 | 0.404 | — | <span class="pill ok">no signal</span> |
| Overdue (longest absence) | 1.0000 | +2.8% | +0.56 | 0.576 | — | <span class="pill ok">no signal</span> |
| Hot numbers (all-time) | 0.9967 | +2.4% | +0.49 | 0.624 | 3.6123 | <span class="pill ok">no signal</span> |
| Unpopular numbers (EV-aware) | 0.9967 | +2.4% | +0.49 | 0.624 | — | <span class="pill ok">no signal</span> |
| Perfect-fit function (spline) | 0.9967 | +2.4% | +0.49 | 0.624 | — | <span class="pill ok">no signal</span> |
| Perfect-fit function (polynomial) | 0.9900 | +1.7% | +0.35 | 0.725 | — | <span class="pill ok">no signal</span> |
| Neural (gru, set prediction) | 0.9867 | +1.4% | +0.28 | 0.777 | 3.6200 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (Markov) | 0.9733 | +0.0% | +0.01 | 0.994 | — | <span class="pill ok">no signal</span> |
| Cold numbers (all-time) | 0.9667 | -0.6% | -0.13 | 0.896 | — | <span class="pill ok">no signal</span> |
| Random ticket | 0.9333 | -4.1% | -0.82 | 0.413 | — | <span class="pill ok">no signal</span> |
| Date-conditioned generative (CB-6/37) | 0.9333 | -4.1% | -0.82 | 0.413 | 3.6349 | <span class="pill ok">no signal</span> |
| Sum-targeted combination | 0.9300 | -4.4% | -0.89 | 0.374 | — | <span class="pill ok">no signal</span> |
| Method of analogues (chaos) | 0.9233 | -5.1% | -1.03 | 0.305 | 3.7195 | <span class="pill ok">no signal</span> |
| Neural (transformer, set prediction) | 0.8533 | -12.3% | -2.47 | 0.013 | 3.6194 | <span class="pill ok">no signal</span> |

Two rulers for the table above. First, the luckiest of 5,000 **purely random** strategies scored **1.147** over these same draws — against this build's top row at **1.067** — so topping this table is well within what luck alone produces. Second, note both tails: strategies land *below* chance as often as above it, and a low row is exactly as (un)meaningful as a high one.

## Randomness tests

Each row asks a different way the machine could be unfair: biased ball frequencies, memory between draws, pairs that travel together, drift over the years, a compressible generating pattern. **Statistic** is each test's own measure (a chi-square, a KS distance, bits/draw, …) — the values are not comparable *between* rows; the p-value is the comparable column. Small p would mean structure a model could exploit. Because many tests run at once, the verdict column applies Holm–Bonferroni across the family.

| Test | Statistic | p | After correction |
|---|---:|---:|---|
| Ball frequency uniformity (chi-square) | 35.25 | 0.5038 | <span class="pill ok">no signal</span> |
| Strong-number uniformity (chi-square) | 2.70 | 0.8449 | <span class="pill ok">no signal</span> |
| Lag-1 serial independence (hot/due folklore) | 1.01 | 0.3151 | <span class="pill ok">no signal</span> |
| Appearance-gap distribution vs geometric | 11.14 | 0.4317 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (666 pairs, Sidak-corrected) | 3.73 | 0.1217 | <span class="pill ok">no signal</span> |
| Draw-sum distribution (two-sample KS vs simulated fair draws) | 0.02 | 0.7096 | <span class="pill ok">no signal</span> |
| First half vs second half frequency drift | 26.92 | 0.8634 | <span class="pill ok">no signal</span> |
| Incompressibility (no generating function within compressor reach) | 24.05 | 1.0000 | <span class="pill ok">no signal</span> |
| Surrogate-data determinism (method of analogues vs shuffles) | 0.88 | 0.8537 | <span class="pill ok">no signal</span> |

## Memorisation vs prediction

The date-conditioned generative model takes a date rather than a history, and there is exactly one draw per calendar date — so the date is a **unique key**. A model with enough capacity can store a lookup table and reproduce past draws perfectly. **DKRR** (Date-Keyed Reconstruction Rate) is that seductive number: matched numbers on dates the model was *trained on*. It measures storage, not skill, and is never shown without the walk-forward column beside it.

Column guide: **DKRR** — matched numbers when asked about dates it was trained on (6.000 = perfect recall); **in-sample log loss** — same idea for the full distribution (lower = more memorised; a uniform guess scores 3.6109); **walk-forward matches** — the honest column, on draws it had never seen (chance 0.9730). Read each row left to right as "how impressive it looks" versus "what it is worth".

| Model | DKRR (in-sample) | In-sample log loss | Walk-forward matches | p |
|---|---:|---:|---:|---:|
| Date-conditioned generative (CB-6/37) | 1.263 / 6 | 3.5930 | 0.9333 | 0.413 |
| Date lookup table (memorisation control) | 6.000 / 6 | 1.8001 | 0.9300 | 0.374 |

Chance is **0.9730** matched numbers and **3.6109** log loss. The lookup control reconstructs every training draw perfectly and is worth nothing on a date it has not seen. That gap is the entire point of the exercise.

## Next-draw picks

Published because they were asked for. These have exactly the same chance as any other six numbers. **For draw of** — most strategies condition on the history, so their pick simply applies to whichever draw comes next; the date-conditioned model needs an actual date, so it names the next scheduled one.

| Strategy | For draw of | Numbers | Strong |
|---|---|---|---:|
| Random ticket | next draw | 1 · 4 · 5 · 9 · 24 · 26 | 3 |
| Lag-1 persistence | next draw | 11 · 20 · 25 · 26 · 27 · 29 | 1 |
| Neural (gru, set prediction) | next draw | 5 · 8 · 9 · 26 · 28 · 29 | 3 |
| Unpopular numbers (EV-aware) | next draw | 32 · 33 · 34 · 35 · 36 · 37 | 1 |
| Date-conditioned generative (CB-6/37) | 2026-07-28 | 11 · 13 · 27 · 28 · 34 · 35 | 5 |

---

Full charts and methodology: [the benchmark report](benchmark/index.html).
