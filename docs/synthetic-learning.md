# Synthetic training and the full real archive

**Training on one million simulated draws produced a model that learns planted biases, but did not improve the real-draw win count over the fixed uniform baseline.** With ten tickets per draw, the mixed-simulation model won on **657 of 1,661 draws**, compared with **663** for the baseline and **641.12** expected under fair draws. None of the eight method/budget comparisons passes the statistical correction.

Experiment date: 2026-11-07. Real evaluation: **2012-01-03 through 2026-11-03**, including every modern-format draw in the frozen archive. This extends the [1,000-draw historical test](historical-strikes.md). It uses a different, simulation-trained procedure, so the earlier best-model counts are not directly comparable.

[Download the complete experiment bundle](data/synthetic-learning-2026-11-07.zip), including all 1.2 million synthetic draws, generator seeds, learned settings, every real forecast and score, and the source snapshots.

## What was generated

We generated independent 2,000-draw episodes. Each starts with an empty history. Training and validation use separate deterministic random-seed streams; no real outcome is used to fit the generator or mixture weights.

| Scenario | Training draws | Separate validation draws | Mechanism |
| --- | ---: | ---: | --- |
| Fair | 500,000 | 100,000 | Uniform six of 37, plus an independent strong number of seven |
| Persistent bias | 250,000 | 50,000 | One random main label and one random strong label have altered probabilities for the episode |
| Changing bias | 250,000 | 50,000 | Labels and bias strengths are redrawn every 250 draws |
| **Total** | **1,000,000** | **200,000** | **600 independent episodes** |

For biased scenarios, the distinguished main inclusion probability and strong probability are independently multiplied by 0.5, 1.5, or 2.0. All other outcomes remain uniform conditional on whether those distinguished labels appear. This keeps every draw valid: exactly six distinct main numbers and one strong number. Labels vary randomly across episodes, so there is no permanently privileged number to memorize. These mechanisms are hypotheses for controlled experiments; they were not inferred from the real lottery.

## What the model learned

The model combines 13 proper probability distributions: twelve frequency models and the exact uniform baseline. Frequency models use windows of 50, 100, 300, or all earlier draws, with prior strengths of 100, 500, or 2,000 fair draws. The main-number distribution is normalized over sets of exactly six; the strong-number distribution has seven outcomes.

Training minimizes average full-draw negative log likelihood over the synthetic training targets. A constrained optimizer learns nonnegative mixture weights summing to one, using an analytic gradient. It is a convex weight-fitting problem, not a search for a random-number seed or a future draw. Both fits converged. Validation results are reported without selecting or revising a model afterward.

Two learned controls were fixed in advance: one uses only the 500,000 fair training draws; the other uses all one million training draws. An equal-weight mixture shows what the same component models achieve before learning weights. All four procedures are evaluated, without selecting the real-data winner for deployment.

| Component | Fair-trained weight | Mixed-trained weight |
| --- | ---: | ---: |
| frequency window=50 prior_draws=100 | 0.0000% | 3.0543% |
| frequency window=50 prior_draws=500 | 0.0000% | 0.0000% |
| frequency window=50 prior_draws=2000 | 0.0000% | 0.0000% |
| frequency window=100 prior_draws=100 | 0.0000% | 5.2630% |
| frequency window=100 prior_draws=500 | 0.2550% | 11.5397% |
| frequency window=100 prior_draws=2000 | 6.1026% | 0.0000% |
| frequency window=300 prior_draws=100 | 0.0000% | 6.3711% |
| frequency window=300 prior_draws=500 | 0.0000% | 0.0000% |
| frequency window=300 prior_draws=2000 | 0.0000% | 0.0000% |
| frequency window=all prior_draws=100 | 0.0194% | 0.0000% |
| frequency window=all prior_draws=500 | 0.0000% | 35.4293% |
| frequency window=all prior_draws=2000 | 0.0000% | 38.3426% |
| uniform | 93.6230% | 0.0000% |

The fair-trained model places 93.62% of its weight on uniform. The mixed-trained model assigns its weight to frequency models. Small probability differences can still change which tickets are ranked highest; near-uniform probabilities do not imply the same realized ticket choices as a fixed uniform baseline.

## Independent synthetic validation

Values below are average full-draw log loss minus the uniform baseline, in nats per draw. **Negative is better.** The oracle knows the planted labels and probabilities; it is a diagnostic reference, not a deployable method.

| Method | Fair | Persistent bias | Changing bias |
| --- | ---: | ---: | ---: |
| Fixed uniform baseline | 0.000000 | 0.000000 | 0.000000 |
| Equal-weight mixture (untrained) | +0.006037 | -0.047182 | -0.014813 |
| Trained on fair simulations | +0.000005 | -0.000749 | -0.000462 |
| Trained on mixed simulations | +0.005634 | -0.058989 | -0.010822 |
| Oracle with known planted parameters | 0.000000 | -0.093205 | -0.078125 |

The mixed-trained model improves on uniform in both planted-bias validation scenarios. It improves on the untrained mixture for persistent bias, but is worse than that mixture for changing bias. On fair validation draws, its probability loss is worse than uniform. This shows both learning and a cost when the assumed biases are absent. These validation numbers measure probability quality, not prize counts.

## All 1,661 real draws

The training stage writes the learned settings and hashes before a separate evaluation stage opens the real archive. Evaluation checks those hashes and scores all 1,661 draws chronologically. The first forecast is uniform because no earlier real draws exist. Each later forecast uses only earlier real outcomes as inputs. Learned mixture weights stay fixed; history inputs naturally update after each draw. This is an online evaluation of a frozen procedure, not 1,661 forecasts with no real-history inputs.

A winning draw means at least one ticket matched three or more main numbers. Each method gets either one ticket per draw (1,661 tickets total) or ten (16,610 total). Multiple prizes in one draw count once. Winning a prize does not imply recovering ticket costs.

One-ticket construction retains the existing top-six selection rule. Ten-ticket construction relabels the existing coverage template using forecast ranks. It preserves the overlap structure and fair-draw coverage, so all methods have the same theoretical chance at the same budget. This ticket construction is a fixed heuristic, not a claimed optimum under the learned distribution.

| Ticket budget | Fair expected winning draws | Central 95% range for one fixed procedure |
| --- | ---: | --- |
| One | 69.35 | 54–86 |
| Ten | 641.12 | 602–680 |

| Method | One-ticket wins | Ten-ticket wins | Full-draw log loss minus uniform | Passes correction |
| --- | ---: | ---: | ---: | --- |
| Fixed uniform baseline | 85 | 663 | -0.00000000 | No |
| Equal-weight mixture (untrained) | 82 | 671 | +0.00437154 | No |
| Trained on fair simulations | 83 | 631 | -0.00001937 | No |
| Trained on mixed simulations | 72 | 657 | +0.00450549 | No |

The mixed-trained model is below the fixed baseline on both win counts and has worse probability loss. The fair-trained model has a very small descriptive probability-loss improvement, but lower win counts. This experiment does not establish a predictive advantage for either learned model. The untrained equal-weight mixture has the highest ten-ticket count here; choosing it because of this result would be further selection on the same archive.

| Method | One-ticket nominal p | Ten-ticket nominal p | Ten-ticket 4+ main wins | Ten-ticket 5+ main wins | Ten-ticket 3+ main and strong wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed uniform baseline | 0.034582 | 0.140634 | 67 | 0 | 99 |
| Equal-weight mixture (untrained) | 0.070777 | 0.069597 | 51 | 2 | 97 |
| Trained on fair simulations | 0.056341 | 0.703220 | 48 | 0 | 116 |
| Trained on mixed simulations | 0.389233 | 0.218842 | 52 | 1 | 98 |

The primary tests use upper exact binomial tails under independent fair draws, with Holm correction over all four methods at both budgets: **eight comparisons**, family level 0.05. None passes. The correction covers this declared experiment, not the entire history of research on this archive. Other metrics and block comparisons are descriptive.

## Chronological consistency

Each cell gives winning draws with **one / ten** tickets. Block sizes are 416, 415, 415, and 415.

| Method | Block 1 | Block 2 | Block 3 | Block 4 |
| --- | --- | --- | --- | --- |
| Fixed uniform baseline | 23 / 174 | 23 / 168 | 18 / 151 | 21 / 170 |
| Equal-weight mixture (untrained) | 23 / 160 | 24 / 162 | 17 / 163 | 18 / 186 |
| Trained on fair simulations | 22 / 158 | 22 / 158 | 17 / 164 | 22 / 151 |
| Trained on mixed simulations | 20 / 149 | 16 / 172 | 17 / 176 | 19 / 160 |

- Block 1: 2012-01-03–2015-11-07.
- Block 2: 2015-11-10–2019-07-04.
- Block 3: 2019-07-06–2023-03-23.
- Block 4: 2023-03-25–2026-11-03.

## Interpretation and limits

Generating more samples lets a learner estimate the simulated mechanisms more accurately. It does not establish that those mechanisms describe the real draw process. Here, we confirmed learning on independent simulated biases and measured its transfer to every modern real draw. The transfer did not produce a demonstrated improvement in winning chances.

The broader research literature also treats the gap between generated and real distributions as a central problem in synthetic training; for example, [Meta-Sim](https://arxiv.org/abs/1904.11621) explicitly optimizes the simulator to reduce that gap in a computer-vision setting. That paper is methodological context, not evidence for lottery predictability.

The real archive and model families were examined in earlier research. Freezing this experiment prevents new target leakage, but does not turn the archive into a fresh holdout of the overall development process. Fitting a generator to all real draws and then calling those same draws an independent test would also leak test information. [Scikit-learn’s data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) explains why model and preprocessing choices must be kept separate from test outcomes.

The next useful step would be to test a specifically supported mechanism or obtain genuinely new evaluation outcomes. Merely increasing the sample count from this same simulator would not resolve the missing evidence that its biases exist in reality. Production predictor defaults are unchanged.

## Reproduce and audit

```bash
uv run python scripts/research_synthetic_learning.py --stage train
uv run python scripts/research_synthetic_learning.py --stage evaluate
uv run pytest -q
uv run ruff check .
uv run mkdocs build --strict
```

Artifacts are in `experiments/synthetic_learning_*_2026_11_07.*`: protocol, frozen training result, synthetic dataset, real result JSON, per-draw real NPZ, and the audit record. Load NPZ files with `numpy.load(path, allow_pickle=False)`. The synthetic dataset has shape `(600, 2000, 7)`; the first 500 episodes train the model and the next 100 validate it. The frozen JSON contains every episode’s seed and generating parameters. The real NPZ includes dates, actual outcomes, probability marginals, full-draw log probabilities, every ticket, and row-level scores.

Independent checks reproduced all **1,200,000 synthetic draws** from the saved seeds, verified all **73,084 real ticket outcomes** using integer bit masks, and checked the main/strong incidence structure of all **6,644 ten-ticket portfolios**. Aggregate wins, chronological block totals, probability normalization, and source/input/artifact hashes agree.

**226 tests passed, two skipped** for the unconfigured live CUDA binary and unavailable host C++ probe. Tests include fair-generator calibration, known planted probabilities, recovery of an analytically known mixture, transfer to independent planted biases, and current/future outcome isolation. Ruff and strict documentation checks passed.
