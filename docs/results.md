# Results

!!! info "Generated output"

    Rebuilt 2026-07-26 05:08 UTC from 1,629 draws through 2026-07-25. Do not edit by hand — `scripts/build_report.py` overwrites this page.

## Strategy scoreboard

Walk-forward over the last 100 draws. A random ticket averages **0.9730** matches; a uniform guess scores **3.6109** log loss. Log loss is shown only where the strategy emits genuine inclusion probabilities.

| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Random ticket | 1.0600 | +8.9% | +1.04 | 0.299 | — | <span class="pill ok">no signal</span> |
| Unpopular numbers (EV-aware) | 1.0400 | +6.9% | +0.80 | 0.424 | — | <span class="pill ok">no signal</span> |
| Lag-1 persistence | 1.0300 | +5.9% | +0.68 | 0.496 | 3.6140 | <span class="pill ok">no signal</span> |
| Hot numbers (last 100 draws) | 1.0000 | +2.8% | +0.32 | 0.747 | 3.6432 | <span class="pill ok">no signal</span> |
| Hot numbers (all-time) | 0.9700 | -0.3% | -0.04 | 0.972 | 3.6147 | <span class="pill ok">no signal</span> |
| Overdue (longest absence) | 0.9700 | -0.3% | -0.04 | 0.972 | — | <span class="pill ok">no signal</span> |
| Pairwise co-occurrence (Markov) | 0.9500 | -2.4% | -0.27 | 0.784 | — | <span class="pill ok">no signal</span> |
| Cold numbers (all-time) | 0.9400 | -3.4% | -0.39 | 0.694 | — | <span class="pill ok">no signal</span> |
| EWMA frequency (half-life 50) | 0.9100 | -6.5% | -0.75 | 0.452 | 3.6383 | <span class="pill ok">no signal</span> |
| Rank ensemble | 0.9000 | -7.5% | -0.87 | 0.384 | — | <span class="pill ok">no signal</span> |
| Sum-targeted combination | 0.8200 | -15.7% | -1.83 | 0.068 | — | <span class="pill ok">no signal</span> |

Verdicts apply Holm–Bonferroni across all 11 strategies. For scale, the luckiest of 300 purely random strategies scored **1.200** over the same draws — against the best strategy here at **1.060**.

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
| Random ticket | 3 · 5 · 8 · 11 · 18 · 24 | 6 |
| Unpopular numbers (EV-aware) | 32 · 33 · 34 · 35 · 36 · 37 | 1 |

---

Full charts and methodology: [the benchmark report](benchmark/index.html).
