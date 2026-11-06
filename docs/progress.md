# Progress and evidence report — 2026-11-06

Snapshot of completed work through commit [`6ca353a`](https://github.com/Timorleiderman/ILotto/commit/6ca353a0aa09a76f880d72dc66e79741525733f3). The benchmark was generated at 2026-11-06T18:00:16+00:00, using 1,661 modern draws through 2026-11-03. This is a fixed report; the regular results page can change on later rebuilds.

**Measured gains are better ticket coverage and faster computation. No predictive advantage, improved equal-budget jackpot odds, or profit advantage has been established.**

[Download the complete research data bundle](data/progress-data-2026-11-06.zip). It contains the two benchmark snapshots, every calibration result and temporal fold, every tested portfolio and exact swap, statistical power results, the draw archives, seed-search summary, original reports, and a SHA-256 manifest. Tables below are rounded; JSON preserves full precision.

## What has been completed

| Workstream | Delivered | Measured result / status |
| --- | --- | --- |
| Project foundation | Four legacy neural architectures; statistical, set-prediction, date-conditioned, interpolation, chaos and golden-ratio approaches; automated archive and documentation workflows | 20 strategies in the current walk-forward benchmark |
| Prediction evaluation | Past-only backtesting, exact match-count null, multiple-testing correction, calibration grid and temporal stability checks | No strategy or calibration configuration passes its accuracy family correction |
| Ticket coverage | Greedy ten-row generator, exhaustive probability evaluator, two exact refinement passes and per-swap audit | Mean at-least-one-prize probability: 38.3536% → 38.5779% on 20 paired portfolios |
| Statistical review | Correct prize tiers, without-replacement uniformity and drift, exact pair tails, stronger validation | Nine current diagnostic tests; none has p < 0.05 |
| Performance | Direct hypergeometric simulation; compiled exact coverage; native CPU/CUDA seed engines | 14.1× null sampling; 93.3× warm exact evaluation; CPU A/B 182M → 237M seeds/s |
| Search infrastructure | Checkpointing, resume, disjoint allocation, remote SSH workers, checkpoint validation and campaign deadlines | 952,605,319,763 unique real-history candidate seeds tested; zero two-draw or full-prefix survivors |
| Documentation and delivery | Research reports, method explanations, report-label corrections, reproduction commands, committed source and CI | Latest implementation pushed as 6ca353a; CI passed |

## Evidence scope

| Item | Value |
| --- | --- |
| Compatible modern draw archive | 1,661 draws, 2012-01-03 through 2026-11-03 |
| Primary model evaluation | Last 300 draws, 2024-04-13 through 2026-11-03 |
| Temporal calibration checks | Four disjoint blocks of 300 draws; all retrospective |
| Theoretical mean main matches | 36/37 = 0.972973 per six-number ticket |
| Uniform marginal ball log loss | ln(37) = 3.610917913; lower is better |
| Uniform strong-number accuracy / log loss | 1/7 = 14.2857%; ln(7) = 1.945910149 |
| Main / complete outcome counts | C(37,6) = 2,324,784 / 7 × C(37,6) = 16,273,488 |
| Raw archive SHA-256 | `0034509bdfa510c3978abcff14baee705f6bebf85701c7e6978ac07ecc171560` |

The archive has already been explored. Walk-forward model inputs exclude later draws, but these results are not a fresh prospective holdout. Family correction within one run does not correct repeated researcher selection across runs. Coverage probabilities below assume independent uniform fair draws and fixed tickets.

## All 20 prediction methods

Each method was scored on 300 draws. The p-value is the nominal two-sided accuracy test; the final column reports the 20-method Holm correction. “3+” is the observed fraction of historical test draws with at least three main matches. A dash in log loss means the method emits rankings rather than calibrated probabilities.

| Method | Mean matches | 3+ rate | Strong accuracy | Ball log loss | Nominal p | Holm significant |
| --- | --- | --- | --- | --- | --- | --- |
| Hot numbers (last 100 draws) | 1.07333 | 6.33% | 13.33% | 3.634882 | 0.04300 | No |
| Lag-1 persistence | 1.07000 | 4.67% | 11.33% | 3.612381 | 0.05043 | No |
| EWMA frequency (half-life 50) | 1.03667 | 4.67% | 14.00% | 3.628970 | 0.20093 | No |
| Perfect-fit function (DFT) | 1.03667 | 5.00% | 11.33% | — | 0.20093 | No |
| Hot numbers (all-time) | 1.02667 | 4.33% | 12.67% | 3.612283 | 0.28247 | No |
| Golden ratio (Kronecker schedule) | 1.02667 | 3.00% | 14.67% | — | 0.28247 | No |
| Perfect-fit function (polynomial) | 1.01667 | 4.33% | 11.33% | — | 0.38418 | No |
| Perfect-fit function (spline) | 1.01333 | 3.67% | 11.33% | — | 0.42259 | No |
| Random ticket | 0.99333 | 3.67% | 18.00% | — | 0.69568 | No |
| Rank ensemble | 0.98333 | 5.00% | 16.00% | — | 0.85317 | No |
| Overdue (longest absence) | 0.98000 | 4.67% | 13.33% | — | 0.90743 | No |
| Fibonacci ticket (control) | 0.97667 | 5.00% | 11.33% | — | 0.96216 | No |
| Unpopular numbers (EV-aware) | 0.97000 | 3.33% | 11.33% | — | 0.98288 | No |
| Neural (gru, set prediction) | 0.97000 | 3.67% | 16.33% | 3.624247 | 0.98288 | No |
| Date-conditioned generative (CB-6/37) | 0.96667 | 5.33% | 17.00% | 3.634527 | 0.92796 | No |
| Cold numbers (all-time) | 0.95667 | 3.67% | 18.33% | — | 0.76601 | No |
| Pairwise co-occurrence (Markov) | 0.95667 | 4.67% | 11.33% | — | 0.76601 | No |
| Sum-targeted combination | 0.94333 | 3.67% | 11.33% | — | 0.56557 | No |
| Method of analogues (chaos) | 0.94333 | 4.67% | 16.67% | 3.712342 | 0.56557 | No |
| Neural (transformer, set prediction) | 0.86000 | 2.00% | 11.33% | 3.617951 | 0.01999 | No |

Hot-100 leads this particular window at 1.07333 mean matches, but its nominal p = 0.0430 does not survive correction. Every probability-emitting model in this benchmark has worse marginal log loss than uniform. Neural results reflect a single training run per method, not a multi-initialization study. [Current full benchmark JSON](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/docs/benchmark/results.json).

## All 18 calibration configurations

The grid combines four history windows with four uniform-prior strengths, plus uniform and adaptive baselines. Ball Brier and both log losses are lower-is-better probability scores. Every accuracy Holm result is **No**. A uniform predictor’s realized top-six match score can fluctuate; its 1.05 in this run does not replace the theoretical 36/37 baseline.

| Configuration | Mean matches | Ball log loss | Ball Brier | Strong log loss | 95% interval: ball loss − uniform |
| --- | --- | --- | --- | --- | --- |
| frequency window=50 prior_draws=0 | 0.94000 | 3.673253 | 0.138740 | 2.014730 | [0.049242, 0.075473] |
| frequency window=50 prior_draws=100 | 0.94000 | 3.618314 | 0.136250 | 1.956316 | [0.003582, 0.011368] |
| frequency window=50 prior_draws=500 | 0.94000 | 3.611831 | 0.135913 | 1.947722 | [-0.000131, 0.002005] |
| frequency window=50 prior_draws=2000 | 0.94000 | 3.611082 | 0.135874 | 1.946323 | [-0.000117, 0.000459] |
| frequency window=100 prior_draws=0 | 1.07333 | 3.634882 | 0.137053 | 1.969454 | [0.015634, 0.032438] |
| frequency window=100 prior_draws=100 | 1.07333 | 3.615854 | 0.136120 | 1.952034 | [0.000746, 0.009043] |
| frequency window=100 prior_draws=500 | 1.07333 | 3.611103 | 0.135875 | 1.946784 | [-0.001245, 0.001558] |
| frequency window=100 prior_draws=2000 | 1.07333 | 3.610823 | 0.135861 | 1.946043 | [-0.000514, 0.000299] |
| frequency window=300 prior_draws=0 | 1.04667 | 3.620193 | 0.136344 | 1.946356 | [0.002645, 0.015570] |
| frequency window=300 prior_draws=100 | 1.04667 | 3.615776 | 0.136117 | 1.945119 | [-0.000053, 0.009549] |
| frequency window=300 prior_draws=500 | 1.04667 | 3.611788 | 0.135910 | 1.944670 | [-0.001564, 0.003172] |
| frequency window=300 prior_draws=2000 | 1.04667 | 3.610866 | 0.135863 | 1.945285 | [-0.000899, 0.000740] |
| frequency window=all prior_draws=0 | 1.02667 | 3.612283 | 0.135935 | 1.946489 | [-0.001301, 0.003997] |
| frequency window=all prior_draws=100 | 1.02667 | 3.612078 | 0.135924 | 1.946391 | [-0.001336, 0.003632] |
| frequency window=all prior_draws=500 | 1.02667 | 3.611560 | 0.135898 | 1.946145 | [-0.001356, 0.002641] |
| frequency window=all prior_draws=2000 | 1.02667 | 3.611005 | 0.135870 | 1.945896 | [-0.001057, 0.001238] |
| uniform | 1.05000 | 3.610918 | 0.135866 | 1.945910 | [0.000000, 0.000000] |
| adaptive past-loss selection | 1.05000 | 3.611359 | 0.135888 | 1.945681 | [-0.000723, 0.001572] |

Intervals use 4,000 circular moving-block bootstrap replicates, block length 20; they are exploratory, not family-adjusted. The adaptive selector chooses using past losses every 50 draws after a warmup. Shrinkage improves overconfident raw frequency probabilities without changing their number ranking; the small apparent improvement over uniform remains uncertain.

Hot-100 temporal stability:

| Test block | Draws | Mean matches | Ball log loss |
| --- | --- | --- | --- |
| 2016-04-16 → 2018-12-04 | 300 | 0.91000 | 3.637080 |
| 2018-12-08 → 2021-08-21 | 300 | 0.97333 | 3.642192 |
| 2021-08-24 → 2024-04-11 | 300 | 0.98667 | 3.635430 |
| 2024-04-13 → 2026-11-03 | 300 | 1.07333 | 3.634882 |

All 72 configuration-by-block records, p-values, uncertainty estimates, and adaptive selection counts are in [the research JSON](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/prediction_probability_2026_11_06.json). Prediction defaults were not changed based on this exploratory calibration grid.

## Ticket coverage: first comparison

Each row below is an exact evaluation of a ten-ticket portfolio, across all 2,324,784 main draws. The initial comparison used five seeds per method. “Any prize” means at least three main matches on at least one ticket; it does not mean recovering the portfolio cost.

| Method | Portfolios | Mean any prize | Minimum | Maximum | Mean 4+ main matches |
| --- | --- | --- | --- | --- | --- |
| random | 5 | 34.5556% | 33.8711% | 35.0725% | 3.0333% |
| legacy_smart | 5 | 25.9790% | 24.6072% | 27.7345% | 2.7024% |
| legacy_coverage | 5 | 32.2078% | 31.8895% | 32.3703% | 3.0091% |
| greedy_3plus | 5 | 38.4027% | 38.2411% | 38.5419% | 3.0801% |

| Method | Seed | Any prize | 4+ main matches | 5+ main matches |
| --- | --- | --- | --- | --- |
| random | 0 | 33.8711% | 2.9993% | 0.0803% |
| legacy_smart | 0 | 24.6072% | 2.6013% | 0.0788% |
| legacy_coverage | 0 | 32.1491% | 3.0070% | 0.0804% |
| greedy_3plus | 0 | 38.4020% | 3.0792% | 0.0804% |
| random | 1 | 35.0116% | 3.0556% | 0.0804% |
| legacy_smart | 1 | 27.7345% | 2.7793% | 0.0798% |
| legacy_coverage | 1 | 31.8895% | 2.9970% | 0.0804% |
| greedy_3plus | 1 | 38.2411% | 3.0807% | 0.0804% |
| random | 2 | 33.9877% | 3.0272% | 0.0804% |
| legacy_smart | 2 | 26.7614% | 2.7772% | 0.0801% |
| legacy_coverage | 2 | 32.3559% | 3.0195% | 0.0804% |
| greedy_3plus | 2 | 38.2892% | 3.0792% | 0.0804% |
| random | 3 | 35.0725% | 3.0428% | 0.0804% |
| legacy_smart | 3 | 25.6171% | 2.7140% | 0.0799% |
| legacy_coverage | 3 | 32.3703% | 3.0149% | 0.0804% |
| greedy_3plus | 3 | 38.5391% | 3.0807% | 0.0804% |
| random | 4 | 34.8354% | 3.0416% | 0.0804% |
| legacy_smart | 4 | 25.1746% | 2.6403% | 0.0783% |
| legacy_coverage | 4 | 32.2743% | 3.0073% | 0.0804% |
| greedy_3plus | 4 | 38.5419% | 3.0807% | 0.0804% |

## Exact refinement: all 20 paired portfolios

Each initial portfolio used 512 candidates and 30,000 simulated fair draws. Two refinement passes then considered exact one-number exchanges and accepted only positive integer coverage gains. Both columns below use exhaustive evaluation, not Monte Carlo estimates.

| Seed | Initial any prize | Refined any prize | Gain (percentage points) | Accepted swaps | Reference verified |
| --- | --- | --- | --- | --- | --- |
| 0 | 38.4020% | 38.5770% | 0.1750 | 9 | Yes |
| 1 | 38.2411% | 38.5770% | 0.3359 | 9 | Yes |
| 2 | 38.2892% | 38.5849% | 0.2956 | 10 | Yes |
| 3 | 38.5391% | 38.5849% | 0.0458 | 10 | Yes |
| 4 | 38.5419% | 38.5849% | 0.0430 | 9 | Yes |
| 5 | 38.5364% | 38.5849% | 0.0485 | 9 | — |
| 6 | 38.4371% | 38.5849% | 0.1478 | 11 | — |
| 7 | 38.2945% | 38.5825% | 0.2880 | 10 | — |
| 8 | 38.3350% | 38.5825% | 0.2475 | 9 | — |
| 9 | 38.3072% | 38.5770% | 0.2698 | 9 | — |
| 10 | 38.3847% | 38.5715% | 0.1868 | 8 | — |
| 11 | 38.2507% | 38.5662% | 0.3155 | 8 | — |
| 12 | 38.3432% | 38.5769% | 0.2337 | 9 | — |
| 13 | 38.2854% | 38.5825% | 0.2971 | 13 | — |
| 14 | 38.2918% | 38.5689% | 0.2771 | 9 | — |
| 15 | 38.4480% | 38.5770% | 0.1290 | 7 | — |
| 16 | 38.1963% | 38.5849% | 0.3886 | 11 | — |
| 17 | 38.4261% | 38.5635% | 0.1374 | 8 | — |
| 18 | 38.2855% | 38.5770% | 0.2915 | 11 | — |
| 19 | 38.2358% | 38.5689% | 0.3331 | 10 | — |

| Aggregate | Initial | Refined |
| --- | --- | --- |
| Mean | 38.3536% | 38.5779% |
| Minimum | 38.1963% | 38.5635% |
| Maximum | 38.5419% | 38.5849% |

All 20 portfolios improved; mean paired gain: **0.2243 percentage points**. The earlier 38.4027% greedy average uses five portfolios; the 38.3536% baseline here uses twenty. They are different cohorts. Seeds 0–4 were independently checked against the reference evaluator, including full before/after seven-bin histograms. The other seeds used the compiled exhaustive evaluator.

For seed 0, the number of covered main outcomes rose from 892,764 to 896,832. All forty initial/refined ten-row portfolios, full histograms, generation/refinement/evaluation timings, and every accepted swap are in [the refinement JSON](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/coverage_refinement_2026_11_06.json). The bounded search is not a proven global optimum; optimizing any-prize coverage can reduce other prize-event probabilities.

Every tested ten-row portfolio has ten distinct full combinations, so its jackpot probability stays **10 / 16,273,488**, approximately **1 in 1,627,349**. Coverage optimization uses no historical draw data and supplies no evidence of improved next-draw forecasting.

## Exact single-ticket prize probabilities

These are the eight-tier definitions used in the saved study. Rows without “strong” mean the strong number does not match; “no prize” includes every result with fewer than three main matches. Probabilities are over all 16,273,488 full outcomes.

| Outcome / tier | Full outcomes | Probability |
| --- | --- | --- |
| no prize | 15,594,054 | 95.82490244% |
| 7th (3+strong) | 89,900 | 0.55243228% |
| 8th (3) | 539,400 | 3.31459365% |
| 5th (4+strong) | 6,975 | 0.04286112% |
| 6th (4) | 41,850 | 0.25716675% |
| 3rd (5+strong) | 186 | 0.00114296% |
| 4th (5) | 1,116 | 0.00685778% |
| 1st (6+strong) | 1 | 0.00000614% |
| 2nd (6) | 6 | 0.00003687% |

Total probability of any prize for one ticket: **4.17509756%**. Cash return and probability of net profit were not evaluated. The original study links its official rules source in [the prize-definition discussion](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/prediction_probability_research.md).

## Current randomness diagnostics

| Test | Statistic | df | p-value |
| --- | --- | --- | --- |
| Ball frequency uniformity (without-replacement chi-square) | 42.521217 | 36 | 0.210687 |
| Strong-number uniformity (chi-square) | 3.596629 | 6 | 0.731072 |
| Lag-1 serial independence (hot/due folklore) | 1.437071 | 1 | 0.230614 |
| Appearance-gap distribution vs geometric | 12.468739 | 11 | 0.329460 |
| Pairwise co-occurrence (666 pairs, exact-binomial Bonferroni) | 3.735694 | — | 0.462012 |
| Draw-sum distribution (two-sample KS vs simulated fair draws) | 0.015981 | — | 0.791892 |
| First half vs second half frequency drift (without-replacement Wald) | 30.013078 | 36 | 0.748304 |
| Incompressibility (no generating function within compressor reach) | 24.052980 | — | 1.000000 |
| Surrogate-data determinism (method of analogues vs shuffles) | 0.950000 | — | 0.682927 |

None has p < 0.05, so none passes a Holm family correction either. The pair result uses exact doubled-inclusive binomial tails and a Bonferroni bound over 666 dependent pairs. Drift uses the fixed-size draw covariance. These replace the earlier snapshot’s pair/drift calculations. Non-significance does not prove randomness; gap, pooled serial, sum, and drift reference approximations still have limits.

Uniformity sensitivity check: 512 simulated 1,661-draw histories per setting, with only ball 1’s inclusion probability multiplied. “Corrected” below refers to the without-replacement statistic, before correction across a larger test family.

| Ball 1 multiplier | Old rejection rate | Corrected rejection rate | Corrected 95% Wilson interval |
| --- | --- | --- | --- |
| 1.0 | 1.17% | 5.08% | 3.49%–7.34% |
| 1.2 | 16.60% | 39.06% | 34.93%–43.36% |
| 1.5 | 100.00% | 100.00% | 99.26%–100.00% |

## Performance measurements

| Component / scope | Before | After / observed | Interpretation |
| --- | --- | --- | --- |
| Random null: 5,000 trials × 300 draws | 1.163049 s | 0.082746 s | 14.06× faster |
| Ten-ticket exact evaluator, warm three-run medians | 757.012 ms | 8.111 ms | 93.33× faster |
| First exact evaluation in saved process | — | 182.7 ms | Includes lazy import / existing JIT cache load; first compilation can take longer |
| Rust CPU, 14-thread A/B benchmark | 182 million seeds/s | 237 million seeds/s | About 30% faster; capacity report rounds rates |
| Rust CPU, earlier sustained campaign | — | 177.46 million seeds/s | Earlier build/run; 45.17 minutes wall time |
| RTX 3090, old binary, local benchmark | — | 6.818 billion seeds/s | Local timed workload; excludes SSH end-to-end overhead |
| RTX 3090, old binary, three remote runs | — | About 6.66 billion seeds/s | End-to-end SSH rate reported in capacity study |
| RTX 3090, optimized current source | — | Not measured | Requires worker rebuild, validation, and benchmark |

These are component timings, not whole-application speedups. The exact evaluator caches about 31 MiB of immutable outcome arrays. The saved timing environment and individual repetitions are in the refinement JSON. CPU machine: Apple M3 Max, 14 cores, 36 GiB; GPU: RTX 3090, 24 GiB. The capacity report’s 6.9 billion seeds/s combined figure is a sum of rates, not a measured concurrent run. Source: [capacity report](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/seed_search_capacity_report.md).

## Artificial seed search: completed work and final state

`calendar-sha256-v1` is an artificial generator with a 64-bit seed, not a model of the physical lottery. The search must match the same seed across 1,657 dated draws, with four historical draws withheld for survivors. It rejects candidates at their first mismatch; throughput therefore does not mean complete histories generated per second.

| Measurement | Result |
| --- | --- |
| Toy control | Seed 45,123 recovered from 65,536 candidates; all four withheld artificial draws reproduced |
| Initial real-data pilot | 1,000,000 seeds; no first-draw survivor; older 1,632-draw snapshot |
| Archive update | 29 added modern draws; now 1,661; compatible old rejections inherited without recounting |
| Unique CPU seeds tested | 490,110,813,982 |
| Unique GPU seeds tested | 462,494,505,781 |
| Total unique real-history seeds tested | 952,605,319,763 |
| Fraction of 64-bit seed space | 0.000005164% |
| Matched at least first draw | 58,548 |
| Matched at least first two draws | 0 |
| Matched complete prefix | 0 |
| Planned one-hour CPU run | Finished successfully after user stop; not a completed one-hour run |
| Actual CPU job start / finish (UTC) | 2026-11-06T15:06:06.780996+00:00 / 2026-11-06T15:51:16.933914+00:00 |
| CPU job wall time / newly tested seeds | 45.17 minutes / 480,887,541,155 |

The total includes the initial GPU snapshot (372,494,505,781 seeds) plus three later disjoint 30-billion-seed remote allocations. Launch notes and the initial GPU summary predate final completion; this report uses the final job and committed checkpoint records. Full local run artifacts are under `experiments/seed_search_runs/` and are ignored by Git. The bundle includes a derived seed summary with source hashes, completed GPU range bounds, cumulative match histograms, and final CPU status; operational logs and binaries remain local.

## Review fixes and validation

| Finding | Completed change |
| --- | --- |
| Missing prize outcome | Added the eighth tier for three main matches without the strong number |
| Incorrect or approximate inference | Finite convolved-hypergeometric match null, corrected uniformity, exact pair tails and without-replacement drift covariance |
| Slow coverage scoring / incomplete optimization | Compiled bit-mask evaluator, bounded exact exchanges, count-consistency checks and swap audit |
| Reporting could overstate findings | Dynamic conclusions; ranking scores distinguished from probabilities; actual chart-band outliers counted; unsupported payout-edge wording removed |
| Inconsistent tied predictions | Report generation now reuses the backtest top-six tie rule |
| Malformed or excessive inputs | Reject noninteger/out-of-range counts, nonfinite scores, duplicate calibration settings, and allocations above a conservative memory budget |
| Native search correctness and operation | Rust/CUDA outcome agreement checks, resumable range checkpoints, disjoint remote reservations, old-worker capability fallback and lock-aware deadlines |

| Validation record | Outcome | Scope / caveat |
| --- | --- | --- |
| Latest local Python suite | 157 passed, 2 skipped | Live CUDA not configured; host C++ probe could not link on this local toolchain |
| Rust native tests | 7 passed | Recorded in the preceding native-search iteration; latest coverage commit did not change native code |
| Historical GPU validation | 1,003 vectors and 36 batch comparisons passed | Old deployed worker; current optimized GPU build still needs measurement |
| Latest local lint / whitespace / docs | Ruff, git diff --check, strict MkDocs passed | Recorded in second review report |
| Full benchmark generation | 20-model report build passed | 1,661-draw archive; 300-draw evaluation |
| Default coverage CLI | Saved seed-0 tickets, odds and full swap audit reproduced | Two refinement passes |
| GitHub CI for 6ca353a | Completed successfully | [CI run 42272345897](https://github.com/Timorleiderman/ILotto/actions/runs/42272345897) |

These are saved validation outcomes for the implementation; assembling this documentation did not rerun neural training, search campaigns, or the full experiment suite. [Detailed second-review validation](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/coverage_review_iteration2.md).

## Delivery history

| Commit | Date | Change |
| --- | --- | --- |
| [6ca353a](https://github.com/Timorleiderman/ILotto/commit/6ca353a) | 2026-11-06 | Optimize exact ticket coverage and correct statistical reporting |
| [b206628](https://github.com/Timorleiderman/ILotto/commit/b206628) | 2026-11-06 | Optimize native seed search and honor campaign deadlines |
| [672c4c6](https://github.com/Timorleiderman/ILotto/commit/672c4c6) | 2026-11-06 | Coordinate disjoint remote GPU ranges and resumable SSH searches |
| [7e08fcd](https://github.com/Timorleiderman/ILotto/commit/7e08fcd) | 2026-11-06 | Add resumable Rust and CUDA seed search with validated range checkpoints |
| [ab73fe3](https://github.com/Timorleiderman/ILotto/commit/ab73fe3) | 2026-10-14 | chore: refresh draw archive [skip ci] |
| [7cf7300](https://github.com/Timorleiderman/ILotto/commit/7cf7300) | 2026-08-12 | chore: refresh draw archive [skip ci] |
| [ac060bd](https://github.com/Timorleiderman/ILotto/commit/ac060bd) | 2026-08-09 | chore: refresh draw archive [skip ci] |
| [11aa3a3](https://github.com/Timorleiderman/ILotto/commit/11aa3a3) | 2026-08-07 | Add a one-minute abstract to every approach page (#14) |
| [e091f9e](https://github.com/Timorleiderman/ILotto/commit/e091f9e) | 2026-08-05 | The golden ratio: measured, inverted, and given its one honest job (#13) |
| [d926057](https://github.com/Timorleiderman/ILotto/commit/d926057) | 2026-07-28 | Add the chaos theory approach: Takens, analogues, surrogates — measured (#12) |
| [39cc869](https://github.com/Timorleiderman/ILotto/commit/39cc869) | 2026-07-27 | Make the results page self-explanatory (#11) |
| [47aae12](https://github.com/Timorleiderman/ILotto/commit/47aae12) | 2026-07-27 | Add the perfect-fit function approach: spacetime interpolation, measured (#10) |
| [ea67f18](https://github.com/Timorleiderman/ILotto/commit/ea67f18) | 2026-07-26 | docs: explain why each approach fails, with measurements (#9) |
| [bb8ed98](https://github.com/Timorleiderman/ILotto/commit/bb8ed98) | 2026-07-26 | Polish: remove drifted duplicate dependency list, fix stale counts, complete exports (#8) |
| [cb8f0c8](https://github.com/Timorleiderman/ILotto/commit/cb8f0c8) | 2026-07-26 | Make CUDA an opt-in extra so the default install works everywhere (#7) |
| [7a4a38e](https://github.com/Timorleiderman/ILotto/commit/7a4a38e) | 2026-07-26 | Add a date-conditioned generative model (#6) |
| [3f31030](https://github.com/Timorleiderman/ILotto/commit/3f31030) | 2026-07-26 | Security hardening: fix workflow script injection, pin actions, guard the fetch (#5) |
| [551a617](https://github.com/Timorleiderman/ILotto/commit/551a617) | 2026-07-26 | fix(docs): render Material icon shortcodes (#4) |
| [39ba731](https://github.com/Timorleiderman/ILotto/commit/39ba731) | 2026-07-26 | Add a MkDocs site with diagrammed explanations of every approach (#3) |
| [c1fea4f](https://github.com/Timorleiderman/ILotto/commit/c1fea4f) | 2026-07-26 | fix: correct lint path in publish workflow after package rename (#2) |

The bundle contains the complete reachable Git commit history through the snapshot, including earlier foundation work. This consolidated report is a new documentation addition and is not part of commit 6ca353a.

## Complete data and report inventory

| Artifact | Contents |
| --- | --- |
| [Current benchmark](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/docs/benchmark/results.json) | All 20 strategy metrics, nine diagnostics, null summary and five published prediction records |
| [Current visual benchmark](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/docs/benchmark/index.html) | Standalone HTML report with charts |
| [Earlier benchmark snapshot](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/prediction_benchmark_2026_11_06.json) | First research iteration; pair/drift inference is superseded by the current benchmark |
| [Prediction / probability data](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/prediction_probability_2026_11_06.json) | 18 calibration configurations, 72 temporal folds, 20 initial portfolios, power study, prize odds and null timings |
| [Exact refinement data](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/coverage_refinement_2026_11_06.json) | 20 before/after portfolios, all tickets, swaps, seven-bin histograms, probabilities and timing repetitions |
| [Original draw archive](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/input/Orig_IL_lotto.csv) | Full downloaded historical archive, including older game formats |
| [Modern draw archive](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/input/lotto_IL_modern.csv) | 1,661 compatible modern draws used by the benchmark |
| [Research narrative](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/prediction_probability_research.md) | Methods, interpretation, reproduction commands and research sources |
| [Second review narrative](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/coverage_review_iteration2.md) | Fixes, exact refinement, validation and limitations |
| [Search capacity](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/seed_search_capacity_report.md) | CPU/GPU measurements and explicitly modeled projections |
| [Initial seed pilot](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/seed_search_pilot.md) | Toy control and initial real-data pilot |
| [Search campaign](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/seed_search_campaign.md) | Range ledger, archive refresh and inheritance |
| [Native search progression](https://github.com/Timorleiderman/ILotto/blob/6ca353a0aa09a76f880d72dc66e79741525733f3/experiments/parallel_search_plan.md) | Engine scaling, GPU work and historical launch notes |

Additional bundle entries: `seed_search_summary.json`, `git_history.tsv`, `MANIFEST.json`, and a copy of this report. Source files are stored under their repository paths. Each manifest entry records byte length and SHA-256. Experiment JSON contains aggregate and per-configuration records; it is not a log of every neural training step or every individual rejected seed. Full local seed run records are not committed or all included in the bundle.

## What remains open

| Next step | Reason / current limitation |
| --- | --- |
| Freeze a prospective forecasting protocol | Commit predictions before future draws; specify models, scoring, sample size and correction before observing outcomes |
| Repeat neural evaluation across initialization seeds | Separate training variation from an apparent model advantage |
| Extend probability scoring and null calibration | Full joint scoring and better calibration checks for the remaining approximate diagnostics |
| Rebuild and benchmark optimized CUDA worker | Optimizations are committed; performance of the rebuilt GPU worker has not been measured |
| Continue coverage research with fixed budgets | Bounded refinement improves these portfolios but does not prove optimal coverage |
| Evaluate payouts only with relevant data | Prize amounts, ticket costs and sharing assumptions are needed to estimate monetary outcomes |

The work completed so far supports a more accurate evaluation system and broader ticket coverage. Establishing better future-draw prediction remains an open research objective.
