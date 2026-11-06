# Algorithm research: learning structure and improving coverage

Research date: 2026-11-06. This follows the [progress report](progress.md) and investigates whether an algorithm can start from guesses, improve them iteratively, and produce a useful advantage. Literature was checked against primary papers, author/institution pages, and official documentation; experiments use the existing 1,661-draw archive through 2026-11-03.

**Two useful directions emerged.** A discrete basin-hopping search measurably improves fixed-budget ticket coverage beyond the previous local search. A new sparse/dense probability mixture learns planted biases and gives coherent full-draw scores, but does not establish a predictive edge on the real archive. No production forecasting default was changed.

## Decisions from the research

| Approach | Why it could help | What was done / decision |
| --- | --- | --- |
| Newton / quasi-Newton fitting | Fit smooth model parameters by following likelihood gradients | Already used by the date-conditioned model. A solver improves the specified objective; fitting historical outcomes alone does not identify future ones. [L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) |
| Sparse Bayesian models and sequential mixtures | Learn one persistent bias with fewer parameters; average over which ball might be biased | Implemented 37 single-ball and seven single-strong models, 12 broader frequency models, and uniform. Exact joint scoring respects sampling six distinct balls. [Conditional Bernoulli research](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n44.pdf) |
| Tracking experts / change detection | Adapt if different models work in different periods | Implemented a fixed-share alternative; reviewed Bayesian online changepoint detection. Fixed share did worse on the planted-bias tests here, so it is not promoted. [Herbster–Warmuth](https://doi.org/10.1023/A:1007424614876), [Adams–MacKay](https://arxiv.org/abs/0710.3742) |
| Sequential likelihood-ratio evidence | Evaluate forecasts as observations arrive without treating repeated monitoring as independent tests | Implemented log evidence paths, with synthetic null checks. Historical paths remain exploratory because this archive was already investigated. [Ramdas et al.](https://arxiv.org/abs/2210.01948) |
| Constraint solving for a known generator | Encode integer state transitions and observed outputs as constraints | Useful when generator equations are known and tractable. No evidence links the real draw to our artificial SHA-256 generator; no new brute-force campaign was run. [Z3 bit-vectors](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/) |
| Symbolic regression / physical system identification | Recover a compact relationship if measured inputs constrain it | The archive contains sorted outcomes, not measured physical states. All 1,661 modern rows are already sorted, so the extraction order cannot be recovered by changing our loader. More flexible historical fits alone are not a new signal. |
| Basin hopping / simulated annealing for coverage | Escape local optima through perturbation and occasional worse intermediate moves | Implemented and measured: all five tested portfolios improved. [SciPy basin hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html), [Nurmela–Östergård covering-design research](https://www.tcs.hut.fi/Publications/info/bibdb.HUT-TCS-B10.shtml) |

This is an evidence-based shortlist, not a claim to have exhausted all algorithms. The published covering-design work supports the search mechanism; our problem is maximum coverage with ten rows, not a guarantee that every draw wins. Its published constructions are not imported as solutions to our 6/37 instance.

## Experiment 1: a learner that starts uncertain

The new model starts with 50% of its weight on uniform draws. It assigns 20% to the 37 single-ball models, 10% to the seven single-strong models, and 20% to 12 dense models. Weight is uniform within each family. The protocol was written before running this new experiment, but the archive itself was already explored; this is not a fresh holdout or preregistration claim.

| Family | Count | Parameters learned from earlier draws |
| --- | --- | --- |
| Uniform | 1 | None; every complete draw has probability 1 / 16,273,488 |
| One distinguished main ball | 37 | One inclusion probability per model; Beta prior equivalent to 100 draws, centered at 6/37 |
| One distinguished strong number | 7 | One occurrence probability per model; Beta prior equivalent to 100 draws, centered at 1/7 |
| Dense frequency model | 12 | Windows 50, 100, 300, all; prior strengths 100, 500, 2,000; exact conditional-Bernoulli set probability plus strong-number probability |
| Static mixture | 1 | Update expert weights from complete-draw probability after observing each draw |
| Fixed-share mixture | 1 | Same update, then mix 1% of the weight back into the initial prior |

For a single-ball model with inclusion probability q, each set containing that ball has probability q / C(36,5); each set excluding it has probability (1−q) / C(36,6). Conditional on that event, the other balls are uniform. This is a normalized model for a six-ball set, not 37 independent Bernoulli draws. The corresponding strong-number model has probability q for its distinguished number and (1−q)/6 for each other number.

Dense models use smoothed historical inclusion odds as conditional-Bernoulli weights. We compute the exact normalizer and inclusion marginals using the existing dynamic program. The resulting marginals are not claimed to equal the original smoothed frequencies after conditioning on a set size of six. [Chen and Liu’s conditional-Bernoulli treatment](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n44.pdf) describes the distribution and recursion.

The prediction/update order is: form each model’s forecast from past draws; combine probabilities using the existing weights; observe the draw; update weights by its full-draw likelihood. The mixture of complete-draw probabilities is used for scoring. Multiplying separately mixed main and strong marginals would be a different distribution and is not used.

For the static mixture, the probability assigned to any complete observation sequence is the prior-weighted sum of the experts’ sequential probabilities. Since uniform starts with weight 1/2, total extra log loss versus uniform can never exceed ln(2) over the entire evaluated sequence. This algebraic bound is checked in regression tests. It does not bound loss on an arbitrary later subwindow, guarantee more matched numbers, or apply unchanged to fixed share.

## Complete-draw forecast results

The primary window is the latest 300 draws, 2024-04-13 through 2026-11-03. Model weights update over 1,461 forecasts after the initial 200 draws. The primary score is negative log probability of the complete unordered six-ball set and strong number, in nats per draw. Uniform gives ln(16,273,488) = 16.605047839. This has a different scale from the older marginal ball log loss ln(37).

| Model | Full-draw log loss ↓ | Loss − uniform ↓ | 95% exploratory interval | Mean main matches | Accuracy Holm |
| --- | --- | --- | --- | --- | --- |
| uniform | 16.605047839 | +0.000000000 | [0.000000, 0.000000] | 1.05000 | No |
| static_joint_mixture | 16.605210282 | +0.000162443 | [-0.000233, 0.000535] | 1.00333 | No |
| fixed_share_joint_mixture | 16.605396930 | +0.000349091 | [-0.001340, 0.001913] | 1.04000 | No |

Neither mixture improves the latest-window mean joint log loss over uniform. Both uncertainty intervals cross zero. All 59 methods, including the 57 component experts, fail the within-run Holm accuracy correction. Uniform’s observed 1.05 main matches is just the result of its deterministic tie-breaking on this window; the theoretical mean remains 36/37 = 0.972973. Intervals use 4,000 circular block-bootstrap replicates of length 20; they are exploratory and not family-adjusted.

| Model | Block start | Block end | Mean main matches | Full-draw loss − uniform |
| --- | --- | --- | --- | --- |
| static_joint_mixture | 2016-04-16 | 2018-12-04 | 1.02333 | -0.000597264 |
| static_joint_mixture | 2018-12-08 | 2021-08-21 | 0.97667 | +0.000559664 |
| static_joint_mixture | 2021-08-24 | 2024-04-11 | 1.05000 | +0.000141856 |
| static_joint_mixture | 2024-04-13 | 2026-11-03 | 1.00333 | +0.000162443 |
| fixed_share_joint_mixture | 2016-04-16 | 2018-12-04 | 1.04000 | -0.001828990 |
| fixed_share_joint_mixture | 2018-12-08 | 2021-08-21 | 0.96667 | +0.000106432 |
| fixed_share_joint_mixture | 2021-08-24 | 2024-04-11 | 1.05667 | +0.000848728 |
| fixed_share_joint_mixture | 2024-04-13 | 2026-11-03 | 1.04000 | +0.000349091 |

## Sequential evidence and synthetic controls

For a fully specified fair-draw null p₀ and a normalized forecast qₜ formed before observing draw t, accumulate Eₜ = product(qₜ(observed draw)/p₀(observed draw)). Under that null, this is a nonnegative test martingale. For one procedure fixed before future outcomes, the probability of ever reaching 20 is at most 5%. This is statistical evidence, not a ticket payout or a probability that the model is true. [Ramdas et al., sections 2–3](https://arxiv.org/abs/2210.01948) provide the sequential-inference framework.

| Model | Final historical E | Maximum historical E, including initial 1 | Threshold |
| --- | --- | --- | --- |
| static_joint_mixture | 0.738965 | 1.742528 | 20 |
| fixed_share_joint_mixture | 0.871630 | 2.325365 | 20 |

These historical values are diagnostics only. Choosing a model after inspecting this archive prevents treating the retrospective path as newly obtained confirmatory evidence. A future monitor must freeze the method and restart evidence at 1 after that freeze. Comparing many detectors also needs a declared combination or multiplicity control; 5% is not a family-wide guarantee for a menu of independently selected tests.

Each synthetic scenario has 64 independent histories of 1,661 draws. When biased, ball 1 is included with the specified multiple of 6/37; remaining balls are uniform conditional on that event, and the strong number stays uniform. The same forecast/update code is used. These scenarios test a particular learnable structure, not every possible bias or dependence.

| Scenario | Bias | Static mixture alerts | Static 95% Wilson interval | Fixed-share alerts |
| --- | --- | --- | --- | --- |
| fair | 1.0× inclusion; from index 0 | 0/64 (0.0%) | 0.0%–5.7% | 0/64 (0.0%) |
| weak_single_ball | 1.2× inclusion; from index 0 | 13/64 (20.3%) | 12.3%–31.7% | 6/64 (9.4%) |
| persistent_single_ball | 1.5× inclusion; from index 0 | 64/64 (100.0%) | 94.3%–100.0% | 63/64 (98.4%) |
| late_single_ball | 1.5× inclusion; from index 830 | 29/64 (45.3%) | 33.7%–57.4% | 19/64 (29.7%) |

The learner detects a persistent 50% inclusion increase in 64/64 trials, but only 13/64 for a 20% increase. Zero alerts in 64 fair trials does not prove a zero false-alarm rate: its 95% Wilson upper endpoint is 5.66%. These counts do not establish improved power over the earlier fixed-horizon uniformity test: the experiments use different decision rules and must be compared on matched simulations before making that claim.

Fixed share was less sensitive in all three planted-bias scenarios. It shares weights among existing experts; it does not reset the cumulative single-ball parameter estimates. A true changepoint model would maintain distributions over time since a change and reset appropriate sufficient statistics. [Adams and MacKay](https://arxiv.org/abs/0710.3742) derive that mechanism. It remains a distinct next experiment, not an implemented or validated feature here.

## Experiment 2: escaping a coverage local optimum

First, we allowed up to ten additional improving-swap passes on the previously refined portfolios for seeds 0–4. Every run stopped after one pass with zero improving exchanges. These five portfolios are local optima for the implemented one-number neighborhood; that does not prove global optimality.

The follow-up uses discrete basin hopping: perturb two numbers, run two exact local refinement passes, and accept better candidates or sometimes worse ones using exp(coverage change / temperature). Temperature decreases from 1,000 to 10 covered outcomes across 20 proposals. Always retain the best portfolio visited. This combines the perturbation/local-search/acceptance pattern described in [SciPy’s basin-hopping documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) with the covering-design application studied by [Nurmela and Östergård](https://www.tcs.hut.fi/Publications/info/bibdb.HUT-TCS-B10.shtml). Our implementation uses integer ticket moves and exact coverage, not a continuous relaxation.

| Seed | Before: any prize | After: any prize | Gain, percentage points | Additional covered outcomes | Time, seconds |
| --- | --- | --- | --- | --- | --- |
| 0 | 38.5770% | 38.5982% | 0.021206 | 493 | 3.93 |
| 1 | 38.5770% | 38.5982% | 0.021206 | 493 | 3.77 |
| 2 | 38.5849% | 38.5876% | 0.002753 | 64 | 3.76 |
| 3 | 38.5849% | 38.5876% | 0.002753 | 64 | 3.83 |
| 4 | 38.5849% | 38.5876% | 0.002753 | 64 | 3.79 |

All five improved. Mean coverage rose from **38.5817% to 38.5919%**, a gain of **0.010134 percentage points**. The best portfolio covers 897,325 main draws, or **38.59820955%**, versus 896,832 before its search. These are exact probabilities over all 2,324,784 main outcomes. Each returned histogram was independently reproduced by the reference evaluator.

This five-seed baseline differs from the previous 20-seed mean of 38.5779%; the table compares each portfolio with its own starting portfolio. It uses more computation with the same ten rows, not an equal-runtime comparison. Jackpot probability remains 10 / 16,273,488 for these ten-distinct-row portfolios. At least one prize is not net profit; this objective can trade off larger-prize probabilities. All odds and proposal audits are retained in the result JSON. Global optimality is not claimed.

## Why seed recovery remains a separate problem

Z3 can encode fixed-width arithmetic and logical constraints, making it relevant when a known software generator and enough of its outputs are available. [The official bit-vector guide](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/) documents those capabilities. This is not evidence that solving the full SHA-256-based candidate family is practical. Our seed mapping is deliberately hash-based, with no known smooth relationship between nearby seeds and output similarity. [NIST](https://csrc.nist.gov/Projects/Hash-Functions) describes SHA-256 and preimage resistance.

The operator’s [2022 activity report](https://www.pais.co.il/download/documents/Freeinfo2022.pdf) describes Lotto draws using ball machines; that dated description is not evidence for a recoverable software seed. The repository archive has dates, sorted outcomes and winner counts, but no measured ball trajectories or machine-state inputs. All 1,661 modern rows are already sorted in the downloaded data. A physical model would need relevant measurements, while a regime model would benefit from machine or ball-set identifiers. Their predictive value would still need testing.

## Recommendations supported by this iteration

| Priority | Action | Evidence / condition |
| --- | --- | --- |
| 1 | Use the optional basin-hopping coverage mode when the extra compute is acceptable | Measured small exact gain on all five tested portfolios at the same row count |
| 2 | Freeze the static sparse/dense mixture for a prospective evaluation | Learns the specified planted structure; no real forecasting advantage established; full joint scores and per-draw outputs are now available |
| 3 | Collect or locate machine/ball-set metadata and actual extraction order | Current CSV cannot support inference about unobserved physical states or extraction-order dependence |
| 4 | Test true change-point resets on matched synthetic scenarios | Fixed share alone did not handle the planted late bias as well as the static mixture |
| 5 | Use SAT/SMT only for a justified, known generator hypothesis | Encoding a hash-based artificial world does not make it the mechanism of the lottery |

## Reproduce and inspect the data

From the repository root:

```bash
# New probability learner: historical comparison + 256 synthetic histories + local-optimum probe
uv run python scripts/research_joint_mixture.py

# Five saved portfolios, twenty basin-hopping proposals each
uv run python scripts/research_coverage_search.py

# Generate a new ten-row portfolio with the optional additional search
uv run python scripts/optimize_ticket_coverage.py --count 10 --seed 0 --search-steps 20 --search-seed 70106 --out /tmp/lotto-coverage-search.json

uv run pytest -q
uv run ruff check .
uv run mkdocs build --strict
```

The coverage CLI keeps its previous two-pass default; `--search-steps 20` explicitly adds the experimental search. Its JSON includes initial refinement, every search proposal, the best returned tickets, and final exact odds. No archive refresh, remote compute, purchases, or seed-search campaign was used.

| Artifact | Contents |
| --- | --- |
| [Prediction protocol](https://github.com/Timorleiderman/ILotto/blob/main/experiments/algorithm_research_protocol_2026_11_06.json) | Fixed grid, family prior, primary/secondary methods and synthetic scenarios |
| [Prediction results](https://github.com/Timorleiderman/ILotto/blob/main/experiments/joint_mixture_research_2026_11_06.json) | All 59 summaries, four folds each, intervals, all 256 synthetic trials, next weights, and exact local-optimum probe |
| [Per-draw arrays](https://github.com/Timorleiderman/ILotto/blob/main/experiments/joint_mixture_research_2026_11_06.npz) | 59 × 1,461 full-draw log probabilities, marginal forecasts, matches, strong hits, dates, names and both mixture weight paths; no pickled objects |
| [Coverage follow-up protocol](https://github.com/Timorleiderman/ILotto/blob/main/experiments/coverage_search_protocol_2026_11_06.json) | Five seeds, fixed proposal budget, temperature schedule and validation rule |
| [Coverage search results](https://github.com/Timorleiderman/ILotto/blob/main/experiments/coverage_search_2026_11_06.json) | Before/after tickets and exact odds, all 100 proposal audits, independent reference verification and timings |
| [Prediction implementation](https://github.com/Timorleiderman/ILotto/blob/main/bench/joint_mixture.py) | Normalized sparse/dense forecasts and past-only mixture updates |
| [Coverage implementation](https://github.com/Timorleiderman/ILotto/blob/main/bench/portfolio.py) | Exact local search and discrete basin hopping |

Load the array artifact with `numpy.load(path, allow_pickle=False)`. Axes are method × draw × ball/strong where applicable; mixture weights are draw × component expert, aligned with the first 57 names. Probabilities are forecasts made before each target; stored log probabilities and match counts additionally use that target’s realized outcome for scoring. The JSON records archive/protocol hashes and the NPZ checksum. Individual run timings are descriptive, not isolated performance benchmarks.

## Full latest-window model table

| Method | Full-draw log loss ↓ | Mean main matches | Strong accuracy | Nominal accuracy p | Holm significant |
| --- | --- | --- | --- | --- | --- |
| uniform | 16.605048 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 1 | 16.603020 | 1.08333 | 11.3333% | 0.02602 | No |
| single ball 2 | 16.606368 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 3 | 16.605339 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 4 | 16.605465 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 5 | 16.605382 | 1.04333 | 11.3333% | 0.15714 | No |
| single ball 6 | 16.605432 | 1.04667 | 11.3333% | 0.13818 | No |
| single ball 7 | 16.606512 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 8 | 16.606067 | 1.07000 | 11.3333% | 0.05043 | No |
| single ball 9 | 16.605762 | 1.02000 | 11.3333% | 0.34801 | No |
| single ball 10 | 16.605347 | 1.08667 | 11.3333% | 0.02184 | No |
| single ball 11 | 16.604633 | 1.09667 | 11.3333% | 0.01260 | No |
| single ball 12 | 16.608156 | 1.01333 | 11.3333% | 0.42259 | No |
| single ball 13 | 16.607186 | 1.02333 | 11.3333% | 0.31410 | No |
| single ball 14 | 16.606305 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 15 | 16.602577 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 16 | 16.603625 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 17 | 16.604532 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 18 | 16.606878 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 19 | 16.605068 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 20 | 16.605514 | 1.00333 | 11.3333% | 0.55066 | No |
| single ball 21 | 16.604174 | 1.09000 | 11.3333% | 0.01825 | No |
| single ball 22 | 16.605391 | 1.01667 | 11.3333% | 0.38418 | No |
| single ball 23 | 16.605625 | 1.07667 | 11.3333% | 0.03651 | No |
| single ball 24 | 16.609463 | 1.04667 | 11.3333% | 0.13818 | No |
| single ball 25 | 16.603263 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 26 | 16.605382 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 27 | 16.603252 | 1.07667 | 11.3333% | 0.03651 | No |
| single ball 28 | 16.605237 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 29 | 16.605259 | 1.09000 | 11.3333% | 0.01825 | No |
| single ball 30 | 16.605417 | 1.02000 | 11.3333% | 0.34801 | No |
| single ball 31 | 16.604376 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 32 | 16.604804 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 33 | 16.606026 | 1.02667 | 11.3333% | 0.28247 | No |
| single ball 34 | 16.605400 | 1.06000 | 11.3333% | 0.07951 | No |
| single ball 35 | 16.603102 | 1.10000 | 11.3333% | 0.01041 | No |
| single ball 36 | 16.604243 | 1.05000 | 11.3333% | 0.12104 | No |
| single ball 37 | 16.605436 | 1.05000 | 11.3333% | 0.12104 | No |
| single strong 1 | 16.604776 | 1.05000 | 13.3333% | 0.12104 | No |
| single strong 2 | 16.605808 | 1.05000 | 11.3333% | 0.12104 | No |
| single strong 3 | 16.606813 | 1.05000 | 11.3333% | 0.12104 | No |
| single strong 4 | 16.605203 | 1.05000 | 11.3333% | 0.12104 | No |
| single strong 5 | 16.603340 | 1.05000 | 19.3333% | 0.12104 | No |
| single strong 6 | 16.604749 | 1.05000 | 15.6667% | 0.12104 | No |
| single strong 7 | 16.605202 | 1.05000 | 11.3333% | 0.12104 | No |
| dense frequency window=50 prior_draws=100 | 16.669407 | 0.95333 | 14.3333% | 0.71380 | No |
| dense frequency window=50 prior_draws=500 | 16.613471 | 0.94000 | 14.3333% | 0.51960 | No |
| dense frequency window=50 prior_draws=2000 | 16.606642 | 0.92333 | 14.3333% | 0.32133 | No |
| dense frequency window=100 prior_draws=100 | 16.647691 | 1.06333 | 13.3333% | 0.06858 | No |
| dense frequency window=100 prior_draws=500 | 16.607377 | 1.06333 | 13.3333% | 0.06858 | No |
| dense frequency window=100 prior_draws=2000 | 16.604509 | 1.03000 | 13.3333% | 0.25309 | No |
| dense frequency window=300 prior_draws=100 | 16.640183 | 1.03667 | 16.3333% | 0.20093 | No |
| dense frequency window=300 prior_draws=500 | 16.610311 | 1.04000 | 16.3333% | 0.17802 | No |
| dense frequency window=300 prior_draws=2000 | 16.604079 | 1.04000 | 16.3333% | 0.17802 | No |
| dense frequency window=all prior_draws=100 | 16.614128 | 1.02667 | 12.6667% | 0.28247 | No |
| dense frequency window=all prior_draws=500 | 16.610065 | 1.03667 | 12.6667% | 0.20093 | No |
| dense frequency window=all prior_draws=2000 | 16.605715 | 1.03000 | 12.6667% | 0.25309 | No |
| static_joint_mixture | 16.605210 | 1.00333 | 14.6667% | 0.55066 | No |
| fixed_share_joint_mixture | 16.605397 | 1.04000 | 16.3333% | 0.17802 | No |

## Validation

Code review corrected coverage settings that were recorded but ignored: the runner now passes the protocol's perturbation count and local-refinement budget to the search, and the audit records both. Prediction protocols now reject invalid evaluation windows, duplicate methods/grid entries, and invalid synthetic settings before computing results. Both runners stage completed output before replacing saved evidence, so computation or serialization failures preserve the previous files. Output paths cannot overwrite prediction inputs, and coverage invariants remain active when Python runs with optimization enabled. Regression tests cover these failure cases.

After these fixes, both experiments were rerun without retuning. All 12 per-draw arrays reproduced exactly, as did all 59 method summaries, four folds per method, 256 synthetic trials, five final portfolios, and 100 coverage proposal traces. Only execution timings, completion metadata, and the additional search-setting audit fields changed.

Regression checks cover exhaustive small-board normalization, marginal sums, exact uniform probabilities, future-data isolation, predict-before-update timing, the static mixture identity and ln(2) bound, fixed-share updates, malformed inputs, deterministic search reproduction, preservation of tickets/strong numbers, best-coverage retention, and independent exhaustive coverage counts. Final validation: **194 tests passed, two skipped** (unconfigured live CUDA and the local host C++ probe). Repository-wide Ruff, strict MkDocs, and whitespace checks passed. The optional coverage CLI exactly reproduced the saved seed-0 portfolio, odds, and search audit; its default still reproduced the previous seed-0 result. Per-draw results for all 59 methods, probability sums, 256 synthetic trials, and artifact/protocol hashes were checked against the JSON summaries.
