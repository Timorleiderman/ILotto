# Results

!!! info "Generated output"

    Rebuilt 2026-07-26 04:33 UTC from 1,629 draws through 2026-07-25. Do not edit by hand — `scripts/build_report.py` overwrites this page.

## Strategy scoreboard

Walk-forward over the last 150 draws. A random ticket averages **0.9730** matches; a uniform guess scores **3.6109** log loss. Log loss is shown only where the strategy emits genuine inclusion probabilities.

| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Lag-1 persistence | 1.0533 | +8.3% | +1.17 | 0.240 | 3.6144 | <span class="pill ok">no signal</span> |
| Hot numbers (last 100 draws) | 1.0333 | +6.2% | +0.88 | 0.378 | 3.6379 | <span class="pill ok">no signal</span> |
| Hot numbers (all-time) | 1.0267 | +5.5% | +0.78 | 0.433 | 3.6137 | <span class="pill ok">no signal</span> |
| Cold numbers (all-time) | 0.9933 | +2.1% | +0.30 | 0.766 | — | <span class="pill ok">no signal</span> |
| Sum-targeted combination | 0.9933 | +2.1% | +0.30 | 0.766 | — | <span class="pill ok">no signal</span> |
| Unpopular numbers (EV-aware) | 0.9800 | +0.7% | +0.10 | 0.918 | — | <span class="pill ok">no signal</span> |
| EWMA frequency (half-life 50) | 0.9533 | -2.0% | -0.29 | 0.774 | 3.6318 | <span class="pill ok">no signal</span> |
| Overdue (longest absence) | 0.9467 | -2.7% | -0.38 | 0.701 | — | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (Markov) | 0.9267 | -4.8% | -0.68 | 0.498 | — | <span class="pill ok">no signal</span> |
| Rank ensemble | 0.9267 | -4.8% | -0.68 | 0.498 | — | <span class="pill ok">no signal</span> |
| Random ticket | 0.9000 | -7.5% | -1.07 | 0.286 | — | <span class="pill ok">no signal</span> |

Verdicts apply Holm–Bonferroni across all 11 strategies. For scale, the luckiest of 800 purely random strategies scored **1.207** over the same draws — against the best strategy here at **1.053**.

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

## Next-draw picks

Published because they were asked for. These have exactly the same chance as any other six numbers.

| Strategy | Numbers | Strong |
|---|---|---:|
| Random ticket | 19 · 20 · 23 · 32 · 36 · 37 | 6 |
| Lag-1 persistence | 11 · 20 · 25 · 26 · 27 · 29 | 1 |
| Unpopular numbers (EV-aware) | 32 · 33 · 34 · 35 · 36 · 37 | 1 |

---

Full charts and methodology: [the benchmark report](benchmark/index.html).
