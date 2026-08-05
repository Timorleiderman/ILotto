# Results

!!! info "Generated output"

    Rebuilt 2026-08-05 20:38 UTC from 1,632 draws through 2026-08-04. Do not edit by hand — `scripts/build_report.py` overwrites this page after every draw, so the numbers below shuffle from build to build. That shuffling is not a bug in the strategies; it is what a table of noise looks like when you keep re-rolling it.

## Strategy scoreboard

Every strategy was asked to pick 6 numbers for each of the last **300 draws**, seeing only the draws before each one. The columns:

| Column | Meaning | What counts as unremarkable |
|---|---|---|
| **Mean matches** | Of the 6 numbers picked, how many were actually drawn, averaged over the 300 test draws | A random ticket averages 6×6/37 = **0.9730**. For this window, anything in **0.876 – 1.070** is within ±2 standard errors of chance |
| **Lift** | Mean matches relative to chance, as a percentage | ±10% is the ±2-standard-error band — single-digit lifts are noise at this sample size |
| **z** | How many standard errors the mean sits from chance | \|z\| < 2 is unremarkable, and with 20 strategies the *largest* \|z\| is expected to exceed 2 by chance alone |
| **p** | Probability that a skill-less strategy would score at least this far from chance | **Not** the probability the strategy works. One row near p ≈ 0.05 is expected among 20 rows |
| **Log loss** | Grades the full probability distribution the strategy assigned, not just its top 6. A uniform guess scores exactly ln 37 = **3.6109**. Genuine information would show up as a value *consistently* below that across rebuilds — a hair below it on one window is sampling noise, and no significance test is applied to this column | “—” means the strategy outputs ranks, not probabilities, so this score would be meaningless for it |
| **Verdict** | Significance after Holm–Bonferroni correction across all 20 rows | “no signal” is the expected outcome; a genuine “signal” would survive the correction *and* recur in the next rebuild |

| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Hot numbers (last 100 draws) | 1.0567 | +8.6% | +1.73 | 0.084 | 3.6360 | <span class="pill ok">no signal</span> |
| Lag-1 persistence | 1.0533 | +8.3% | +1.66 | 0.097 | 3.6129 | <span class="pill ok">no signal</span> |
| Perfect-fit function (DFT) | 1.0433 | +7.2% | +1.45 | 0.146 | — | <span class="pill ok">no signal</span> |
| Golden ratio (Kronecker schedule) | 1.0333 | +6.2% | +1.25 | 0.212 | — | <span class="pill ok">no signal</span> |
| EWMA frequency (half-life 50) | 1.0267 | +5.5% | +1.11 | 0.267 | 3.6302 | <span class="pill ok">no signal</span> |
| Neural (transformer, set prediction) | 1.0200 | +4.8% | +0.97 | 0.331 | 3.6152 | <span class="pill ok">no signal</span> |
| Random ticket | 1.0167 | +4.5% | +0.90 | 0.366 | — | <span class="pill ok">no signal</span> |
| Unpopular numbers (EV-aware) | 1.0067 | +3.5% | +0.70 | 0.486 | — | <span class="pill ok">no signal</span> |
| Rank ensemble | 1.0033 | +3.1% | +0.63 | 0.530 | — | <span class="pill ok">no signal</span> |
| Perfect-fit function (spline) | 1.0000 | +2.8% | +0.56 | 0.576 | — | <span class="pill ok">no signal</span> |
| Perfect-fit function (polynomial) | 0.9933 | +2.1% | +0.42 | 0.674 | — | <span class="pill ok">no signal</span> |
| Hot numbers (all-time) | 0.9900 | +1.7% | +0.35 | 0.725 | 3.6125 | <span class="pill ok">no signal</span> |
| Overdue (longest absence) | 0.9867 | +1.4% | +0.28 | 0.777 | — | <span class="pill ok">no signal</span> |
| Cold numbers (all-time) | 0.9700 | -0.3% | -0.06 | 0.951 | — | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (Markov) | 0.9667 | -0.6% | -0.13 | 0.896 | — | <span class="pill ok">no signal</span> |
| Fibonacci ticket (control) | 0.9600 | -1.3% | -0.27 | 0.789 | — | <span class="pill ok">no signal</span> |
| Date-conditioned generative (CB-6/37) | 0.9200 | -5.4% | -1.10 | 0.273 | 3.6347 | <span class="pill ok">no signal</span> |
| Method of analogues (chaos) | 0.9167 | -5.8% | -1.16 | 0.244 | 3.7187 | <span class="pill ok">no signal</span> |
| Neural (gru, set prediction) | 0.9067 | -6.8% | -1.37 | 0.170 | 3.6207 | <span class="pill ok">no signal</span> |
| Sum-targeted combination | 0.8900 | -8.5% | -1.72 | 0.086 | — | <span class="pill ok">no signal</span> |

Two rulers for the table above. First, the luckiest of 5,000 **purely random** strategies scored **1.187** over these same draws — against this build's top row at **1.057** — so topping this table is well within what luck alone produces. Second, note both tails: strategies land *below* chance as often as above it, and a low row is exactly as (un)meaningful as a high one.

## Randomness tests

Each row asks a different way the machine could be unfair: biased ball frequencies, memory between draws, pairs that travel together, drift over the years, a compressible generating pattern. **Statistic** is each test's own measure (a chi-square, a KS distance, bits/draw, …) — the values are not comparable *between* rows; the p-value is the comparable column. Small p would mean structure a model could exploit. Because many tests run at once, the verdict column applies Holm–Bonferroni across the family.

| Test | Statistic | p | After correction |
|---|---:|---:|---|
| Ball frequency uniformity (chi-square) | 35.11 | 0.5106 | <span class="pill ok">no signal</span> |
| Strong-number uniformity (chi-square) | 2.48 | 0.8704 | <span class="pill ok">no signal</span> |
| Lag-1 serial independence (hot/due folklore) | 0.90 | 0.3440 | <span class="pill ok">no signal</span> |
| Appearance-gap distribution vs geometric | 11.22 | 0.4247 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (666 pairs, Sidak-corrected) | 3.71 | 0.1285 | <span class="pill ok">no signal</span> |
| Draw-sum distribution (two-sample KS vs simulated fair draws) | 0.02 | 0.7328 | <span class="pill ok">no signal</span> |
| First half vs second half frequency drift | 27.48 | 0.8452 | <span class="pill ok">no signal</span> |
| Incompressibility (no generating function within compressor reach) | 24.05 | 1.0000 | <span class="pill ok">no signal</span> |
| Surrogate-data determinism (method of analogues vs shuffles) | 0.88 | 0.9512 | <span class="pill ok">no signal</span> |

## Memorisation vs prediction

The date-conditioned generative model takes a date rather than a history, and there is exactly one draw per calendar date — so the date is a **unique key**. A model with enough capacity can store a lookup table and reproduce past draws perfectly. **DKRR** (Date-Keyed Reconstruction Rate) is that seductive number: matched numbers on dates the model was *trained on*. It measures storage, not skill, and is never shown without the walk-forward column beside it.

Column guide: **DKRR** — matched numbers when asked about dates it was trained on (6.000 = perfect recall); **in-sample log loss** — same idea for the full distribution (lower = more memorised; a uniform guess scores 3.6109); **walk-forward matches** — the honest column, on draws it had never seen (chance 0.9730). Read each row left to right as "how impressive it looks" versus "what it is worth".

| Model | DKRR (in-sample) | In-sample log loss | Walk-forward matches | p |
|---|---:|---:|---:|---:|
| Date-conditioned generative (CB-6/37) | 1.259 / 6 | 3.5930 | 0.9200 | 0.273 |
| Date lookup table (memorisation control) | 6.000 / 6 | 1.8001 | 0.9133 | 0.218 |

Chance is **0.9730** matched numbers and **3.6109** log loss. The lookup control reconstructs every training draw perfectly and is worth nothing on a date it has not seen. That gap is the entire point of the exercise.

## Next-draw picks

Published because they were asked for. These have exactly the same chance as any other six numbers. **For draw of** — most strategies condition on the history, so their pick simply applies to whichever draw comes next; the date-conditioned model needs an actual date, so it names the next scheduled one.

| Strategy | For draw of | Numbers | Strong |
|---|---|---|---:|
| Random ticket | next draw | 1 · 4 · 5 · 9 · 24 · 26 | 3 |
| Hot numbers (last 100 draws) | next draw | 5 · 7 · 8 · 18 · 19 · 21 | 2 |
| Neural (gru, set prediction) | next draw | 3 · 5 · 8 · 9 · 11 · 29 | 3 |
| Unpopular numbers (EV-aware) | next draw | 32 · 33 · 34 · 35 · 36 · 37 | 1 |
| Date-conditioned generative (CB-6/37) | 2026-08-06 | 26 · 27 · 28 · 30 · 34 · 35 | 4 |

---

Full charts and methodology: [the benchmark report](benchmark/index.html).
