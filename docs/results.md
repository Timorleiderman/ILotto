# Results

!!! info "Generated output"

    Rebuilt 2026-11-14 18:13 UTC from 1,663 draws through 2026-11-10. Do not edit by hand — `lotto/experiments/benchmark_report.py` overwrites this page after every draw, so the numbers below shuffle from build to build. That shuffling is not a bug in the strategies; it is what a table of noise looks like when you keep re-rolling it.

## Strategy scoreboard

Every strategy was asked to pick 6 numbers for each of the last **300 draws**, seeing only the draws before each one. The columns:

| Column | Meaning | What counts as unremarkable |
|---|---|---|
| **Mean matches** | Of the 6 numbers picked, how many were actually drawn, averaged over the 300 test draws | A random ticket averages 6×6/37 = **0.9730**. For this window, anything in **0.876 – 1.070** is within ±2 standard errors of chance |
| **Lift** | Mean matches relative to chance, as a percentage | ±10% is the ±2-standard-error band — single-digit lifts are noise at this sample size |
| **z** | How many standard errors the mean sits from chance | \|z\| < 2 is unremarkable, and with 19 strategies the *largest* \|z\| is expected to exceed 2 by chance alone |
| **p** | Doubled smaller inclusive tail of the convolved hypergeometric null for total matches | **Not** the probability the strategy works. One row near p ≈ 0.05 is expected among 19 rows |
| **Log loss** | Grades normalized marginal ball probabilities, not just its top 6. A uniform guess scores exactly ln 37 = **3.6109**. Genuine information would show up as a value below that in a predeclared evaluation with appropriate uncertainty. No significance test is applied to this column | “—” means the strategy outputs ranks, not probabilities, so this score would be meaningless for it |
| **Verdict** | Significance after Holm–Bonferroni correction across all 19 rows | A flagged result needs independent validation; repeated overlapping rebuilds are not independent replications |

| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Hot numbers (last 100 draws) | 1.0833 | +11.3% | +2.28 | 0.026 | 3.6354 | <span class="pill ok">no signal</span> |
| Lag-1 persistence | 1.0700 | +10.0% | +2.01 | 0.050 | 3.6126 | <span class="pill ok">no signal</span> |
| EWMA frequency (half-life 50) | 1.0367 | +6.5% | +1.32 | 0.201 | 3.6293 | <span class="pill ok">no signal</span> |
| Perfect-fit function (DFT) | 1.0367 | +6.5% | +1.32 | 0.201 | — | <span class="pill ok">no signal</span> |
| Hot numbers (all-time) | 1.0333 | +6.2% | +1.25 | 0.226 | 3.6125 | <span class="pill ok">no signal</span> |
| Golden ratio (Kronecker schedule) | 1.0200 | +4.8% | +0.97 | 0.348 | — | <span class="pill ok">no signal</span> |
| Perfect-fit function (spline) | 1.0133 | +4.1% | +0.83 | 0.423 | — | <span class="pill ok">no signal</span> |
| Perfect-fit function (polynomial) | 1.0067 | +3.5% | +0.70 | 0.506 | — | <span class="pill ok">no signal</span> |
| Method of analogues (chaos) | 1.0033 | +3.1% | +0.63 | 0.551 | 3.7024 | <span class="pill ok">no signal</span> |
| Overdue (longest absence) | 0.9933 | +2.1% | +0.42 | 0.696 | — | <span class="pill ok">no signal</span> |
| Cold numbers (all-time) | 0.9833 | +1.1% | +0.21 | 0.853 | — | <span class="pill ok">no signal</span> |
| Rank ensemble | 0.9833 | +1.1% | +0.21 | 0.853 | — | <span class="pill ok">no signal</span> |
| Random ticket | 0.9800 | +0.7% | +0.15 | 0.907 | — | <span class="pill ok">no signal</span> |
| Fibonacci ticket (control) | 0.9800 | +0.7% | +0.15 | 0.907 | — | <span class="pill ok">no signal</span> |
| Sum-targeted combination | 0.9733 | +0.0% | +0.01 | 1.000 | — | <span class="pill ok">no signal</span> |
| Date-conditioned generative (CB-6/37) | 0.9600 | -1.3% | -0.27 | 0.819 | 3.6354 | <span class="pill ok">no signal</span> |
| Neural (transformer, set prediction) | 0.9500 | -2.4% | -0.47 | 0.663 | 3.6185 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (Markov) | 0.9467 | -2.7% | -0.54 | 0.613 | — | <span class="pill ok">no signal</span> |
| Neural (gru, set prediction) | 0.9200 | -5.4% | -1.10 | 0.288 | 3.6225 | <span class="pill ok">no signal</span> |

Two rulers for the table above. First, the luckiest of 5,000 **purely random** strategies scored **1.140** over these same draws — against this build's top row at **1.083**. This illustrates selection effects; the simulated maximum is not a significance threshold. Second, the two-sided tests can detect performance below as well as above chance. Family correction in a single build does not control repeated decisions across rebuilds.

## Randomness tests

Each row asks a different way the machine could be unfair: biased ball frequencies, memory between draws, pairs that travel together, drift over the years, a compressible generating pattern. **Statistic** is each test's own measure (a chi-square, a KS distance, bits/draw, …) — the values are not comparable *between* rows; the p-value is the comparable column. Small p warrants investigation, not a claim of exploitable skill. Because many tests run at once, the verdict column applies Holm–Bonferroni across the family.

| Test | Statistic | p | After correction |
|---|---:|---:|---|
| Ball frequency uniformity (without-replacement chi-square) | 42.49 | 0.2115 | <span class="pill ok">no signal</span> |
| Strong-number uniformity (chi-square) | 3.58 | 0.7337 | <span class="pill ok">no signal</span> |
| Lag-1 serial independence (hot/due folklore) | 1.51 | 0.2189 | <span class="pill ok">no signal</span> |
| Appearance-gap distribution vs geometric | 12.57 | 0.3221 | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (666 pairs, exact-binomial Bonferroni) | 3.73 | 0.4759 | <span class="pill ok">no signal</span> |
| Draw-sum distribution (chi-square vs exact null) | 20.02 | 0.3936 | <span class="pill ok">no signal</span> |
| First half vs second half frequency drift (without-replacement Wald) | 31.67 | 0.6745 | <span class="pill ok">no signal</span> |
| Incompressibility (no generating function within compressor reach) | 24.05 | 1.0000 | <span class="pill ok">no signal</span> |
| Surrogate-data determinism (method of analogues vs shuffles) | 1.03 | 0.2725 | <span class="pill ok">no signal</span> |

## Memorisation vs prediction

The date-conditioned generative model takes a date rather than a history, and there is exactly one draw per calendar date — so the date is a **unique key**. A model with enough capacity can store a lookup table and reproduce past draws perfectly. **DKRR** (Date-Keyed Reconstruction Rate) is that seductive number: matched numbers on dates the model was *trained on*. It measures storage, not skill, and is never shown without the walk-forward column beside it.

Column guide: **DKRR** — matched numbers when asked about dates it was trained on (6.000 = perfect recall); **in-sample log loss** — same idea for the full distribution (lower = more memorised; a uniform guess scores 3.6109); **walk-forward matches** — the honest column, on draws it had never seen (chance 0.9730). Read each row left to right as "how impressive it looks" versus "what it is worth".

| Model | DKRR (in-sample) | In-sample log loss | Walk-forward matches | p |
|---|---:|---:|---:|---:|
| Date-conditioned generative (CB-6/37) | 1.271 / 6 | 3.5934 | 0.9600 | 0.819 |
| Date lookup table (memorisation control) | 6.000 / 6 | 1.8001 | 1.0233 | 0.314 |

Chance is **0.9730** matched numbers and **3.6109** log loss. The lookup control reconstructs every training draw perfectly and is worth nothing on a date it has not seen. That gap is the entire point of the exercise.

## Next-draw picks

Under independent fair draws, every full ticket has the same jackpot probability. **For draw of** — most strategies condition on the history, so their pick simply applies to whichever draw comes next; the date-conditioned model needs an actual date, so it names the next scheduled one.

| Strategy | For draw of | Numbers | Strong |
|---|---|---|---:|
| Random ticket | next draw | 1 · 4 · 5 · 9 · 24 · 26 | 3 |
| Hot numbers (last 100 draws) | next draw | 1 · 2 · 7 · 18 · 21 · 35 | 2 |
| Neural (gru, set prediction) | next draw | 2 · 15 · 16 · 20 · 25 · 36 | 4 |
| Date-conditioned generative (CB-6/37) | 2026-11-17 | 7 · 10 · 23 · 25 · 27 · 29 | 5 |

---

Full charts and methodology: [the benchmark report](benchmark/index.html).
