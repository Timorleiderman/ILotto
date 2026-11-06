# Second review: exact coverage, performance, and reporting

2026-11-06. This follows the [prediction and probability research](prediction_probability_research.md).
The objective was to improve measured coverage and evaluation correctness without
mistaking either for a newly demonstrated ability to predict the draw.

## Measured coverage improvement

Compared 20 fixed seeds, 0–19, before and after two exact refinement passes.
Every initial portfolio used ten rows, 512 candidate tickets, and 30,000 simulated
fair draws. All probabilities below count the complete 2,324,784-outcome main-draw
universe. No draw history is used by this optimizer.

| Metric | Initial greedy search | Exact refinement |
|---|---:|---:|
| Mean probability of at least one prize | 38.3536% | **38.5779%** |
| Minimum across 20 portfolios | 38.1963% | 38.5635% |
| Maximum across 20 portfolios | 38.5419% | 38.5849% |

All 20 portfolios improved. The mean paired increase was **0.2243 percentage
points**. For seed 0, coverage increased from 892,764 to 896,832 main outcomes:
38.4020% to 38.5770%. Each portfolio still covers ten distinct jackpot outcomes
out of 16,273,488; jackpot probability is unchanged.

The previous 38.40% headline was an average over five initial portfolios. The
38.3536% baseline here uses 20 portfolios; this report compares each refined
portfolio with its own initial portfolio. These are exact probabilities for
the returned portfolios, not a claim about every possible seed or global optimality.
At least one prize is not necessarily net profit, and a larger-prize probability
can decline when optimizing the three-main-match objective.

[Raw results](coverage_refinement_2026_11_06.json) contain all before/after tickets,
full best-match histograms, exact swap gains, timings, and configuration.

## How refinement works

For each ticket, score all 6 × 31 possible one-number exchanges. Other tickets'
coverage determines which outcomes can matter. A swap can gain outcomes with
exactly two old matches or lose outcomes with exactly three. Single and joint
incidence counts compute every candidate's exact change in one pass through the
universe, instead of reevaluating 186 complete portfolios.

Only strictly positive integer gains are accepted. Ticket count and strong
numbers are preserved. The search stops after the configured pass budget or a
pass with no changes. The output audit records each removed/added number and
the exact increase in covered outcomes, then checks the accumulated gain against
the final union count. Bounded coordinate search need not reach even a local
optimum before its pass budget ends; no global optimum is claimed.

## Performance

The default exact evaluator now represents tickets and draws as bit masks and
counts intersections in compiled integer loops. The generic chunked Python/NumPy
reference remains available via `best_match_counts(..., engine="reference")`.
Numba was already a project dependency; no dependency was added.

For the same ten-ticket portfolio, three-run median evaluation times, measured
without tests running concurrently, were:

| Evaluator | Median time |
|---|---:|
| Chunked reference | 757.0 ms |
| Compiled evaluator, warm | 8.11 ms |
| Warm speedup | **93.3×** |

The benchmark process's first evaluation took 182.7 ms, including lazy import
and loading an existing JIT cache. First-ever compilation can take longer.
The compiled engine retains approximately 31 MiB of immutable outcome arrays.
These are component timings on the recorded machine, not a 93× speedup for the
whole command or neural training.

Both initial and refined portfolios for seeds 0–4 were independently verified
against the reference engine, including their entire seven-bin match histograms.
Separate regression tests compare every candidate exchange against an exhaustive
small-universe set oracle, including number 37, and test the coverage-count update.

## Review findings fixed

| Finding | Fix and verification |
|---|---|
| Simulated greedy selection stopped without checking local exact improvements | Two exact exchange passes by default, strict gain checks, full audit, and paired tests |
| Repeated Python tuple enumeration dominated exact evaluation | Compiled bit-mask engine with independent reference comparisons |
| Report generation used a different top-six tie rule from backtesting | Published picks now reuse `metrics.top_k`; regression checks tied scores and displayed shortlist |
| Ranking-only outputs were normalized and displayed as probabilities | JSON identifies `ranking_score` versus `normalized_marginal_probability`; rendering labels each correctly |
| HTML could display a significant-test badge alongside a hard-coded “no signal” conclusion | Headline and marginal-log-loss comparison now depend on actual results; tests exercise a deliberately significant fixture |
| Frequency chart claimed every bar lay within its band, regardless of data | Counts outlying bars and explains that the band is per number, not simultaneous |
| Pair tests used a normal approximation and an independence-based Šidák correction | Exact doubled-inclusive binomial tails plus a Bonferroni bound across 666 dependent pairs |
| Drift treated within-draw ball appearances as independent | A Wald statistic uses the fixed-size draw's covariance; verified against its full covariance pseudoinverse |
| Malformed match totals could be truncated into misleading p-values | Reject non-integer/out-of-range counts, misaligned outcomes, nonfinite losses/p-values, and invalid calibration settings |
| Duplicate calibration configurations could collapse names during correction | Reject duplicate grid values and non-integer model indices |
| Oversized search settings could allocate excessive memory before failing | Conservative pre-allocation budget, with Python-integer arithmetic to avoid overflow |

## Statistical changes and limits

For each pair, its appearance count over independent fair draws is binomial with
success probability C(35,4)/C(37,6). The reported pair p-value doubles the smaller
inclusive tail and applies the Bonferroni union bound. It is not SciPy's
probability-ordered two-sided `binomtest` definition, although the one-sided tails
agree. The test uses exact binomial probabilities and a conservative family bound;
see the [SciPy binomial-test reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

On the 1,661-draw archive through 2026-11-03, the corrected pair-family p-value is
0.4620 and the corrected drift p-value is 0.7483. Neither flags a deviation. The
drift reference remains asymptotic under independent uniform 6/37 draws; the
gap, pooled serial, and sum tests still have approximation limits. Family
correction within one run does not correct repeated researcher selection or
monitoring across rebuilds.

Prediction-model parameters were not tuned to this iteration's coverage results.
Coverage improvement supplies no evidence of better next-draw prediction. A
frozen prospective prediction protocol remains the next step for testing an
accuracy advantage. Documentation now states this distinction and removes
unsupported assertions that the popularity heuristic has a proven payout edge.

## Use and reproduction

```bash
uv run python scripts/optimize_ticket_coverage.py --count 10 --seed 0 --out /tmp/lotto-coverage.json
uv run python scripts/optimize_ticket_coverage.py --count 10 --seed 0 --refine-passes 0
uv run python scripts/benchmark_coverage.py --seeds 20 --refine-passes 2 --out /tmp/coverage-review.json
uv run python scripts/build_report.py --n-test 300 --null-trials 5000
uv run pytest -q
uv run ruff check .
uv run mkdocs build --strict
```

The first command includes two refinement passes. The second reproduces the
initial portfolio. JSON includes the pass budget and swap audit; timing depends
on the machine and JIT cache state. The benchmark uses no purchases or external
compute. The full report regenerates the site sources from the committed archive.

## Validation outcome

- **157 tests passed, two skipped.** Skips remain the unconfigured live CUDA
  device and the host C++ hash check whose local toolchain cannot link its probe.
- Ruff passed across the repository; `git diff --check` passed.
- The strict MkDocs build and the full 20-model report build passed.
- The regenerated benchmark uses 1,661 draws through 2026-11-03 and its latest
  300-draw evaluation window. No strategy passes the family-corrected accuracy test.
- The default CLI exactly reproduced the saved seed-0 refined portfolio, all
  odds fields, and the complete swap audit.
