# The 1,000-draw test

!!! abstract "In one minute"

    **The question.** If you had bought tickets ranked by a forecast for the last 1,000
    draws, would you have won more often than chance?

    **The answer.** No. The best of 59 forecasts won with ten tickets on 422 of 1,000
    draws against 386 expected, which does not survive correction for 118 comparisons;
    the forecast that led the first 500 draws (211 wins) fell to 187 in the second 500.

    **Why it matters.** It turns "forecast quality" into the thing a player cares about,
    winning draws, and shows what picking the best of many looks like.

## Run it

```bash
uv run lotto run thousand-draws
```

About three seconds. It rebuilds all 59 forecasts from the archive, checks them against the
saved ones, and writes `results/thousand-draws/`.

!!! note "One-ticket numbers differ slightly from the report below"

    The report was written before tied scores were given a fixed rule (lowest number first).
    Single tickets from tied forecasts now pick different numbers, so today's run gives a best
    one-ticket result of 53 wins where the report says 62. Ten-ticket results are identical,
    and no comparison is significant either way.

## The dated report

**The best one-ticket method won on 62 of 1,000 draws, versus 41.75 expected under fair draws. The best ten-ticket method won on 422, versus 385.98 expected. Neither survives the correction for comparing 59 models at two ticket budgets.** The ten-ticket model selected using the first 500 draws won 211 times there and 187 times in the next 500.

Experiment date: 2026-11-07. Test period: **2018-01-30 through 2026-11-03**. There are 661 earlier draws available before the first target. This follows the [probability-model research](algorithm-research.md) and [fixed-budget coverage experiment](ticket-chances.md).

[Download the test data and source snapshots](data/historical-strikes-2026-11-07.zip). It includes every generated ticket, realized result, match count, all model summaries, the protocol, and checksums.

## What counts as a strike

A winning draw means at least one ticket matched three or more main numbers. Multiple winning tickets on the same draw count once. Each method is evaluated separately with either one ticket per draw or ten tickets per draw. These are 1,000 versus 10,000 total tickets over the test period; the methods are not pooled into one giant portfolio. Winning any prize does not mean recovering the ticket cost.

| Ticket budget per method per draw | Expected winning draws out of 1000 | Central 95 percent fair-null range |
| --- | --- | --- |
| One | 41.75 | 30–55 |
| Ten with the fixed coverage design | 385.98 | 356–416 |

These ranges describe one fixed procedure, before searching across models. The null probabilities are exact combinatorial probabilities. A global number-label permutation preserves them on every draw, even when the permutation depends on earlier outcomes. Under independent fair draws, this gives the same conditional win probability at each step and an exact binomial null for the win count.

## How forecasts were kept separate from targets

All 59 existing probability models were included: uniform, 37 single-ball experts, seven single-strong experts, 12 dense frequency experts, and the two mixtures. No model was removed based on these results. Every original forecast path was recomputed from its saved settings and reproduced the existing artifact exactly before selecting the last 1,000 forecasts.

For each target, frequency estimates use only earlier draws. Mixture weights update after observing each outcome; their original warmup starts after 200 draws. Earlier draws in this test can therefore train later predictions, but a target never trains its own forecast. The algorithm and its update rules remain fixed.

One-ticket forecasts use the existing top-six and strong-number selection rules. Ten-ticket forecasts retain the selected coverage design and consistently relabel its numbers using the forecast ranks: numbers with higher scores occupy the more-used template labels. The fixed tie rule returns the original template for uniform scores. This preserves fair-draw coverage and ticket count, so improvements would reflect the chosen labels rather than extra coverage. It is a declared ticket-construction heuristic, not an optimum under each forecast distribution.

The archive has already been examined in earlier research. This is a retrospective walk-forward experiment, **not a fresh holdout of the model-development process**. Preventing target leakage is necessary, but choosing methods repeatedly after inspecting the same archive can still overstate future performance. [Scikit-learn’s guidance on data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) explains that distinction between training and test information.

## Selected results

| Method | Wins with one ticket | Wins with ten tickets | Nominal p for one ticket | Nominal p for ten tickets | Passes correction |
| --- | --- | --- | --- | --- | --- |
| uniform | 50 | 392 | 0.112205 | 0.359217 | No |
| static_joint_mixture | 49 | 389 | 0.143488 | 0.434100 | No |
| fixed_share_joint_mixture | 49 | 392 | 0.143488 | 0.359217 | No |
| single ball 21 | 62 | 396 | 0.001606 | 0.267712 | No |
| single ball 4 | 50 | 422 | 0.112205 | 0.010797 | No |
| single ball 37 | 56 | 398 | 0.018043 | 0.226859 | No |

The leaders were identified after examining all 59 methods. The comparison uses one-sided exact binomial tails and Holm correction across all **118 model/budget comparisons**, at family level 0.05. None passes. This correction is conservative and covers this declared comparison; it does not remove the effects of previous research choices on this archive. The other event counts and probability losses are descriptive.

The single-ball-21 expert had 62 winning draws with one ticket and used three distinct ticket combinations across the period. Its name identifies the model’s bias parameter; every prediction still contains six main numbers and a strong number. The ten-ticket leader, single-ball-4, returned exactly the same portfolio for all 1,000 draws. Its historical advantage therefore does not demonstrate an ability to time changing conditions.

The fixed uniform ten-ticket baseline itself had **eight consecutive winning draws**. Repeated strikes can occur without a learned prediction advantage. The static and fixed-share mixtures won on 389 and 392 draws, respectively, near the fair expectation of 385.98.

## Four chronological blocks

The one-ticket leader produced 12, 16, 14, and 20 winning draws in the four blocks, each above the fair expectation of 10.44. This makes it a candidate for further testing. It was chosen using the entire period, so these block counts are not independent validation of its selection.

| Block | Dates | Fair expected one-ticket wins | Ball 21 one-ticket wins | Fair expected ten-ticket wins | Ball 4 ten-ticket wins | Static mixture ten-ticket wins |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2018-01-30–2020-03-07 | 10.44 | 12 | 96.50 | 101 | 101 |
| 2 | 2020-03-10–2022-06-25 | 10.44 | 16 | 96.50 | 97 | 99 |
| 3 | 2022-06-28–2024-09-12 | 10.44 | 14 | 96.50 | 117 | 86 |
| 4 | 2024-09-14–2026-11-03 | 10.44 | 20 | 96.50 | 107 | 103 |

## Pick using the first half and check the second

The fixed protocol also selects the ten-ticket method with the most winning draws in the first 500, using original model order to break ties. That selects **single ball 37**. The model identity then stays fixed while its normal past-only parameter updates continue.

| Period | Winning draws | Fair expected wins | Win rate |
| --- | --- | --- | --- |
| First 500 used to select the method | 211 | 192.99 | 42.2% |
| Following 500 used to check that selection | 187 | 192.99 | 37.4% |

Its advantage did not persist into the second half; the second-half one-sided p-value is 0.723734. This selection check is distinct from naming the eventual full-period winner. Both halves remain retrospective parts of the already-examined archive.

## All model results

All counts below are out of 1,000 draws. The last column covers the two primary any-prize comparisons; no method passes either. Full JSON also contains six-main counts, jackpot counts, streak lengths, all four block summaries, full-draw log loss, and nominal p-values.

| Method | One-ticket wins | Ten-ticket wins | Ten-ticket four or more main | Ten-ticket three or more main and strong | Passes correction |
| --- | --- | --- | --- | --- | --- |
| uniform | 50 | 392 | 35 | 49 | No |
| single ball 1 | 50 | 390 | 36 | 49 | No |
| single ball 2 | 52 | 404 | 36 | 64 | No |
| single ball 3 | 43 | 384 | 31 | 60 | No |
| single ball 4 | 50 | 422 | 37 | 58 | No |
| single ball 5 | 51 | 403 | 34 | 66 | No |
| single ball 6 | 51 | 393 | 38 | 66 | No |
| single ball 7 | 50 | 392 | 37 | 61 | No |
| single ball 8 | 47 | 385 | 36 | 63 | No |
| single ball 9 | 50 | 400 | 27 | 48 | No |
| single ball 10 | 51 | 377 | 36 | 61 | No |
| single ball 11 | 52 | 390 | 29 | 52 | No |
| single ball 12 | 47 | 389 | 28 | 60 | No |
| single ball 13 | 56 | 391 | 35 | 64 | No |
| single ball 14 | 50 | 400 | 30 | 62 | No |
| single ball 15 | 50 | 407 | 32 | 63 | No |
| single ball 16 | 50 | 400 | 37 | 52 | No |
| single ball 17 | 50 | 402 | 33 | 64 | No |
| single ball 18 | 51 | 401 | 36 | 69 | No |
| single ball 19 | 50 | 405 | 33 | 70 | No |
| single ball 20 | 46 | 384 | 28 | 64 | No |
| single ball 21 | 62 | 396 | 39 | 60 | No |
| single ball 22 | 51 | 377 | 34 | 57 | No |
| single ball 23 | 45 | 393 | 39 | 52 | No |
| single ball 24 | 52 | 389 | 35 | 50 | No |
| single ball 25 | 50 | 382 | 29 | 66 | No |
| single ball 26 | 50 | 380 | 27 | 64 | No |
| single ball 27 | 54 | 361 | 32 | 61 | No |
| single ball 28 | 50 | 364 | 25 | 61 | No |
| single ball 29 | 49 | 375 | 23 | 59 | No |
| single ball 30 | 47 | 385 | 37 | 64 | No |
| single ball 31 | 50 | 410 | 33 | 70 | No |
| single ball 32 | 50 | 392 | 35 | 49 | No |
| single ball 33 | 56 | 373 | 22 | 56 | No |
| single ball 34 | 54 | 404 | 24 | 56 | No |
| single ball 35 | 57 | 377 | 25 | 64 | No |
| single ball 36 | 50 | 416 | 30 | 64 | No |
| single ball 37 | 56 | 398 | 37 | 70 | No |
| single strong 1 | 50 | 392 | 35 | 60 | No |
| single strong 2 | 50 | 392 | 35 | 61 | No |
| single strong 3 | 50 | 392 | 35 | 59 | No |
| single strong 4 | 50 | 392 | 35 | 47 | No |
| single strong 5 | 50 | 392 | 35 | 58 | No |
| single strong 6 | 50 | 392 | 35 | 64 | No |
| single strong 7 | 50 | 392 | 35 | 49 | No |
| dense frequency window=50 prior_draws=100 | 46 | 399 | 36 | 66 | No |
| dense frequency window=50 prior_draws=500 | 45 | 385 | 27 | 66 | No |
| dense frequency window=50 prior_draws=2000 | 42 | 381 | 33 | 74 | No |
| dense frequency window=100 prior_draws=100 | 47 | 369 | 21 | 62 | No |
| dense frequency window=100 prior_draws=500 | 49 | 377 | 23 | 69 | No |
| dense frequency window=100 prior_draws=2000 | 45 | 389 | 30 | 69 | No |
| dense frequency window=300 prior_draws=100 | 47 | 380 | 41 | 56 | No |
| dense frequency window=300 prior_draws=500 | 49 | 392 | 39 | 59 | No |
| dense frequency window=300 prior_draws=2000 | 47 | 355 | 32 | 62 | No |
| dense frequency window=all prior_draws=100 | 47 | 383 | 25 | 58 | No |
| dense frequency window=all prior_draws=500 | 47 | 373 | 20 | 55 | No |
| dense frequency window=all prior_draws=2000 | 48 | 374 | 30 | 55 | No |
| static_joint_mixture | 49 | 389 | 35 | 57 | No |
| fixed_share_joint_mixture | 49 | 392 | 32 | 55 | No |

## Reproduce and inspect

```bash
uv run lotto run thousand-draws
uv run pytest -q
uv run ruff check .
uv run mkdocs build --strict
```

The fixed protocol is `experiments/historical_strikes_protocol_2026_11_07.json`. Results and every ticket are in `experiments/historical_strikes_2026_11_07.json` and the matching `.npz` file. Load arrays with `numpy.load(path, allow_pickle=False)`. The arrays contain method × draw × row × ticket-number forecasts, row-level scores and strong hits, actual outcomes, dates, and full-draw log probabilities. One-ticket arrays have one row; portfolio arrays have ten. Actual results appear only for scoring and audit.

All **649,000 ticket outcomes** were independently checked using integer bit masks against the saved draws. Main- and strong-label incidence signatures were checked for all **59,000 ten-ticket portfolios**, establishing that their coverage structure matches the template. Every aggregate agrees with its per-draw arrays and four block totals. Input, protocol, source, and artifact checksums were verified. Existing tests check that current and future outcomes cannot change current forecasts; new tests check mapping independence across time and independent scoring.

**217 tests passed, two skipped** for the unconfigured live CUDA binary and unavailable host C++ probe. Ruff and strict documentation checks passed. Previous experiment files are preserved.
