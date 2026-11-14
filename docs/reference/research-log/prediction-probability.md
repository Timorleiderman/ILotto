# Prediction accuracy and win probability: research and experiments

Research date: 2026-11-06. Archive: 1,661 compatible modern draws through
2026-11-03. Raw archive SHA-256:
`0034509bdfa510c3978abcff14baee705f6bebf85701c7e6978ac07ecc171560`.

This is the first-iteration snapshot. The [second review iteration](coverage-review.md)
adds exact refinement, corrects pair/drift inference, and updates report wording.
The dated JSON files here preserve the original measurements; the generated
documentation scoreboard reflects the latest code.

## Findings and decisions

**The measurable improvement is portfolio coverage. A forecasting advantage has
not been established.** For ten tickets, the new coverage optimizer averaged a
38.40% chance of at least one prize, compared with 34.56% for random portfolios
and 25.98% for the existing smart generator. These probabilities enumerate every
possible main-number draw. They assume a fair draw and do not imply a higher
jackpot probability or a positive financial return.

The current 20-model benchmark found no significant advantage after correction
for testing multiple strategies. Shrinking frequency forecasts toward uniform
substantially reduces their overconfidence, but its small apparent advantage over
uniform probabilities is uncertain. The production prediction defaults therefore
remain unchanged; the calibration grid is an explicit experiment.

Implemented during this research:

- Exact portfolio evaluation and a greedy coverage generator with a runnable CLI.
- The missing eighth prize tier: three main matches without the strong number.
- A without-replacement correction to the main-ball uniformity test.
- Finite-sample match-count p-values from the convolved hypergeometric null.
- Direct hypergeometric sampling, about 14.1 times faster for the random baseline.
- Reproducible calibration, temporal-stability, test-power, and coverage experiments.

Machine-readable evidence: [full model benchmark](https://github.com/Timorleiderman/ILotto/blob/main/experiments/prediction_benchmark_2026_11_06.json)
and [calibration, coverage, power, and timing results](https://github.com/Timorleiderman/ILotto/blob/main/experiments/prediction_probability_2026_11_06.json).
The latter includes every portfolio, integer coverage histogram, experiment seed,
and archive hash.

## What “win probability” means

The current game chooses six main numbers from 37 and one strong number from seven.
The official prize table includes eight tiers; three main matches alone win the
eighth tier. See [Mifal HaPais game instructions](https://www.pais.co.il/info/lotto-how-to-play.aspx)
and [the official rules, section 21, PDF page 15](https://www.pais.co.il/download/files/Lotto-8306.pdf).
The rules PDF is the 2019 document; this study uses its tier definitions, not historic
cash amounts.

Under independent uniform draws:

| Quantity | Exact expression | Meaning |
|---|---|---|
| Main outcomes | C(37,6) = 2,324,784 | All unordered sets of six main numbers |
| Full outcomes | 7 × C(37,6) = 16,273,488 | Main set plus strong number |
| One-row jackpot probability | 1 / 16,273,488 | Every full combination has the same probability |
| Ten distinct rows: jackpot | 10 / 16,273,488 | About 1 in 1,627,349, regardless of main-number overlap |
| At least one prize in a portfolio | P(maximum main matches ≥ 3) | Depends on overlap between tickets |

Duplicate full rows cannot increase the probability of at least one jackpot.
Different strong numbers on the same main set cover different jackpot outcomes,
but they add no coverage for the event “at least three main matches.”

At least one prize is not the same event as recovering the cost of the whole
portfolio. No profit probability or expected monetary return is estimated here.
For fixed per-tier payouts, linearity of expectation makes the expected total
payout of equally sized portfolios the same under a fair draw; coverage changes
the distribution of wins. Shared prize pools and other players' choices require
additional information.

## Prediction evidence

### Current model benchmark

All 20 strategies were evaluated on the latest 300 draws with the existing
walk-forward harness, including GRU, transformer, and date-conditioned CB models.
Models receive only preceding draws, plus the target date where applicable.

| Strategy | Mean main matches | Marginal log loss, lower is better |
|---|---:|---:|
| Uniform theoretical baseline | 0.97297 expected | 3.610918 |
| Hot numbers, last 100 | 1.07333 | 3.634882 |
| Lag-1 persistence | 1.07000 | 3.612381 |
| EWMA, half-life 50 | 1.03667 | 3.628970 |
| Hot numbers, full history | 1.02667 | 3.612283 |
| Random ticket, realized run | 0.99333 | — |
| GRU set prediction | 0.97000 | 3.624247 |
| Date-conditioned CB | 0.96667 | 3.634527 |
| Transformer set prediction | 0.86000 | 3.617951 |

No strategy passed Holm correction across the 20-model family. Hot-100's nominal
two-sided exact p-value was 0.0430, which does not survive that correction.
Every probability-emitting model in this run had worse marginal log loss than
uniform. A single stochastic neural training run is not a complete comparison
over initialization seeds; it supplies no evidence for promoting either model.

The random baseline's expected main matches are 36/37. The actual score of one
random ticket sequence, or one deterministically tie-broken uniform predictor,
will fluctuate around that expectation. In the calibration experiment the tied
uniform predictor happened to score 1.05; that is not a replacement for the
theoretical baseline.

### Calibration and temporal stability

Tested 16 frequency configurations: windows of 50, 100, 300, and all preceding
draws, each with 0, 100, 500, or 2,000 equivalent uniform-prior draws. For number j:

`p_j = (past_count_j + prior_draws × 6/37) / (history_length + prior_draws)`.

Strong-number probabilities use the corresponding 1/7 prior. Prefix sums exclude
the target draw. A seventeenth configuration is uniform; an eighteenth adaptively
selects a configuration every 50 draws using its preceding 300 losses after a
200-prediction warmup. Ball and strong forecasts are selected separately.

| Configuration | Latest-300 main matches | Marginal log loss | Loss difference from uniform: exploratory 95% interval |
|---|---:|---:|---:|
| Raw Hot-100 | 1.07333 | 3.634882 | +0.015634 to +0.032438 |
| Hot-100, 2,000 prior draws | 1.07333 | 3.610823 | −0.000514 to +0.000299 |
| Adaptive past-loss selection | 1.05000 | 3.611359 | −0.000723 to +0.001572 |
| Uniform | 0.97297 expected | 3.610918 | Reference |

Intervals use 4,000 paired circular moving-block bootstrap replicates, with block
length 20. They are exploratory, unadjusted intervals, not evidence that the
best retrospectively selected configuration generalizes. All 18 configurations
also failed the one-sided accuracy test after Holm correction within this family.

Uniform shrinkage preserves the ranking for a fixed frequency window. Here it
improved probability scoring without improving top-six accuracy. The adaptive
strategy's strong log loss was 1.945681 versus uniform's ln(7) = 1.945910; that
small retrospective difference is not established predictive skill either.

Four disjoint temporal blocks, each containing 300 targets, expose instability:

| Target dates | Raw Hot-100 mean matches |
|---|---:|
| 2016-04-16 to 2018-12-04 | 0.91000 |
| 2018-12-08 to 2021-08-21 | 0.97333 |
| 2021-08-24 to 2024-04-11 | 0.98667 |
| 2024-04-13 to 2026-11-03 | 1.07333 |

The archive has already influenced earlier project experiments. Strictly past-only
forecast code prevents direct lookahead, but it cannot undo researcher selection
on previously examined data. Holm correction within the two reported families
does not correct the entire project's experiment history. This is the selection
problem described by [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).
New model promotion should use a frozen prospective evaluation or a genuinely
untouched outer test period.

The existing log-loss metric grades normalized marginal ball probabilities, not
the joint probability of a complete six-ball set. The new experiment also records
37-dimensional inclusion Brier scores and strong-number log loss. A future joint
model comparison should score its normalized set likelihood separately.

## Exact portfolio coverage

Compared four generators at the same ten-row size over seeds 0–4. The new method
draws 512 candidate tickets and 30,000 synthetic fair outcomes, then greedily adds
the ticket covering the most previously uncovered outcomes with three or more
main matches. Strong numbers are distributed across 1–7.

Each returned portfolio is then evaluated against **all 2,324,784 possible main
draws**, independently of the optimization sample. The resulting probabilities
for these fixed portfolios have no Monte Carlo estimation error.

| Generator | Mean P(at least one prize) | Range across five portfolios | Mean P(at least one 4+ main hit) |
|---|---:|---:|---:|
| Random tickets | 34.56% | 33.87–35.07% | 3.033% |
| Existing smart generator | 25.98% | 24.61–27.73% | 2.702% |
| Existing coverage generator | 32.21% | 31.89–32.37% | 3.009% |
| New greedy coverage | **38.40%** | **38.24–38.54%** | **3.080%** |

The average increase is 3.85 percentage points versus random, or 11.1% relative,
and 12.42 percentage points versus the existing smart generator. This is evidence
about the tested portfolios and configurations. Five seeds do not prove universal
superiority of the generator, and greedy search does not prove global optimality.
The random comparison is the mean of five actual portfolios, not an exact
expectation over every possible random portfolio.

All 20 portfolios contain ten distinct full combinations and therefore have
identical jackpot probability. The exact evaluation resolves overlap between
tickets instead of incorrectly treating their prize events as independent.

Generate a reproducible portfolio and its exact odds:

```bash
uv run python scripts/optimize_ticket_coverage.py --count 10 --seed 0 --refine-passes 0 --out /tmp/lotto-coverage.json
```

The CLI supports even row counts from 2 to 14 for an ordinary form. It emits the
complete configuration, tickets, exact match histogram, any-prize probability,
and jackpot probability. It performs no purchase. The underlying library also
supports research objectives at other match thresholds.

The explicit zero refinement passes above reproduce this experiment's original
portfolio. The current CLI defaults to two exact improvement passes, evaluated
in the second-iteration report.

## Statistical corrections and detection power

### Without-replacement uniformity

The old main-ball test applied the ordinary Pearson chi-square reference to
dependent within-draw indicators. If p = k/N, each indicator has variance p(1−p)
and distinct indicators have covariance −p(1−p)/(N−1). On the sum-zero subspace,
the correct asymptotic statistic is:

`Q_corrected = Q_Pearson × (N−1)/(N−k) = Q_Pearson × 36/31`, with 36 degrees of freedom.

This derivation is independently checked against the full covariance-matrix Wald
statistic in the tests. The need for a different lottery goodness-of-fit reference
is documented by [Genest, Lockhart, and Stephens (2002)](https://doi.org/10.1111/1467-9884.00315).
This remains an asymptotic test, unlike the exact finite-sample match-count test.

For the actual archive, the corrected statistic is 42.5212 and p = 0.2107; the
uncorrected p-value was about 0.4401. The conclusion remains “no detected marginal
nonuniformity,” with a better-calibrated test.

Simulated 512 independent histories at each bias level, each with 1,661 draws.
Ball 1 is included with probability multiplier × 6/37; conditional on inclusion,
the remaining five or six balls are chosen uniformly from the other 36.

| Ball-1 inclusion multiplier | Old test rejection rate | Corrected rejection rate | Corrected Wilson 95% interval |
|---|---:|---:|---:|
| 1.0: fair draw | 1.17% | 5.08% | 3.49–7.34% |
| 1.2: 20% relative increase | 16.60% | 39.06% | 34.93–43.36% |
| 1.5: 50% relative increase | 100% | 100% | 99.26–100% |

These are individual-test rates at alpha 0.05, before family correction. The test
still misses many histories under the moderate-bias alternative. Passing a
randomness test does not prove that every exploitable bias is absent.

### Finite-sample accuracy inference and runtime

For a fixed or strictly past-adaptive six-number ticket, the conditional match
distribution under independent fair draws is Hypergeometric(37,6,6). Convolving
that PMF gives the null distribution for total matches across the evaluation
window. The benchmark now reports a one-sided upper tail and a conservative
two-sided doubled-inclusive-tail p-value; the z-score remains descriptive.

The separate Monte Carlo baseline now samples that same hypergeometric law
directly in bounded chunks. For 5,000 trials × 300 draws, three-run median times
on this machine were 1.163 seconds before and 0.0827 seconds after: **14.1× faster**.
Random streams differ, but the null law is the same. This is a component speedup,
not a claim that neural training or the whole report became 14 times faster.

The maximum of 5,000 random strategies is a useful illustration of selection
effects, not a replacement for a multiplicity-corrected hypothesis test.

## Research priorities after this change

| Priority | Work | Acceptance criterion |
|---|---|---|
| 1 | Use exact portfolio coverage for a fixed ticket budget | Every proposed portfolio reports exact any-prize and jackpot odds; compare the same row count |
| 2 | Freeze and record prospective forecasts, archive hash, model configuration, and prediction time | Predictions are immutable before outcomes; promote only on a predeclared metric and test |
| 3 | Compare uniform, shrunk frequency, and a small calibrated model on marginal Brier/log loss and strong log loss | Stable prospective improvement with uncertainty and family correction; do not select by the best historical fold |
| 4 | Calibrate the remaining randomness tests with complete simulated fair histories | Correct family false-positive behavior under dependent pairs, gaps, and temporal splits |
| 5 | Extend portfolio search across seeds and exchange/local-search moves | Exact coverage gain at equal budget, reported runtime, no claim of global optimality without a bound |
| 6 | Model prize sharing only if suitable player-choice and payout data can be obtained | Validate full-combination popularity and realized payout outcomes; keep this separate from draw prediction |

At 80% power, a rough normal approximation with the null variance 0.70197 needs
about 434 independent test draws to detect a true +0.10 mean-match improvement
with one predeclared one-sided alpha-0.05 test. A conservative 20-comparison
Bonferroni threshold raises that to about 935 draws. For +0.05, the corresponding
figures are approximately 1,736 and 3,739. These are planning approximations that
assume the alternative variance is near the null variance; simulation should
calibrate the final protocol. A short lucky streak is insufficient evidence.

Repeatedly inspecting an expanding or rolling test set also requires sequential
inference, or fixed predeclared decision times. Confidence sequences can maintain
coverage over time under their stated assumptions; see
[Howard et al. (2021)](https://arxiv.org/abs/1810.08240). This is a proposed next
step, not a feature implemented here.

Player choices are a more plausible source of payout differences than “hot” draw
numbers. An Israeli study of manually selected tickets found nonuniform player
preferences, including number-position effects and preferences among strong
numbers. It examined 2018 data, so those patterns are neither current full-ticket
popularity estimates nor evidence of biased winning draws.
[Polin, Ben Isaac, and Aharon (2021)](https://www.cambridge.org/core/journals/judgment-and-decision-making/article/patterns-in-manually-selected-numbers-in-the-israeli-lottery/F7167C1DD46E4876DAFCDDD6CE8F238C)
supports investigating prize sharing, but not assigning a quantified EV advantage
to the existing “unpopular numbers” heuristic. Avoiding birthdays alone is not a
validated return model.

The strongest next prediction research investment is better prospective evidence
and calibrated uncertainty. The present results do not justify scaling neural
architectures or a seed search as a route to better next-draw accuracy.

## Reproduction and validation

Use the committed archive matching the hash above. No archive refresh or remote
compute is needed for these commands:

```bash
uv run python scripts/research_prediction_probability.py --portfolio-seeds 5 --out /tmp/prediction-probability.json
uv run python scripts/build_report.py --n-test 300 --null-trials 5000 --out-dir /tmp/prediction-benchmark
uv run pytest -q
uv run ruff check .
uv run mkdocs build --strict
```

Numerical results from deterministic research experiments should reproduce with
the same archive and dependency versions. Runtime measurements depend on the
machine; neural results may vary across platforms and training kernels.

Regression coverage includes independent exhaustive small-game oracles, exact
single-ticket combinatorics, full-versus-main ticket duplication, prize-tier
partitioning, the covariance-based uniformity statistic, finite-sample match
probabilities, null moments, and target/future mutation checks for forecasts and
adaptive selection.

Local validation: **132 tests passed, two skipped**, Ruff passed over the whole
repository, and the strict MkDocs build passed. The skips were the live CUDA
test (no configured device binary) and the host CUDA-hash C++ check (the local
toolchain could not link its trivial probe). The new CLI reproduced the saved
seed-0 portfolio and all its exact odds fields. The complete 20-model report also
built successfully.
