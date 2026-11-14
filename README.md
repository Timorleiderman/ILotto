# ILotto

A reproducible study of the Israeli Lotto (6 of 37, plus a strong number 1–7): what can be
measured, what can be predicted, and what a ticket buyer can actually control. It is built to be
taught, from basic probability to neural networks and exact algorithms.

**[Read the documentation site →](https://timorleiderman.github.io/ILotto/)**

## What was found

| Finding | Number | Read more |
|---|---|---|
| The draw cannot be picked | 19 strategies, from hot numbers to a Transformer, each replayed over the latest 300 of 1,661 draws using only earlier draws: none beats a random ticket after correction | [Scoreboard](https://timorleiderman.github.io/ILotto/results/) |
| Looking impressive is cheap | A lookup table reconstructs every past draw (6.000 of 6) and scores chance on the next one | [Memorisation](https://timorleiderman.github.io/ILotto/approaches/date-conditioned/) |
| The ticket can be designed | One row wins a prize in 4.18% of draws; ten designed rows win in 38.60%, fourteen in 51.01%, counted exactly | [Coverage](https://timorleiderman.github.io/ILotto/approaches/ticket-generators/) |
| Three bugs once looked like skill | An earlier version reported 30% accuracy, all of it from data handling | [Measurement](https://timorleiderman.github.io/ILotto/evaluation/) |

## Run it

You need [uv](https://docs.astral.sh/uv/). It installs Python for you.

```bash
git clone https://github.com/Timorleiderman/ILotto.git
cd ILotto
uv sync
```

```bash
uv run lotto list
```

```bash
uv run lotto run coverage --count 10 --seed 0 --out results/coverage.json
```

```bash
uv run lotto report --quick --n-test 100 --null-trials 500
```

`lotto list` shows every experiment with the question it answers. The light install above runs
everything except the two neural predictors; add them with `uv sync --extra neural`, then
`uv run lotto report` builds the full benchmark in about two minutes.

## The story, from basics to advanced

| | Chapter | The question | Code |
|---|---|---|---|
| 1 | [The game](docs/start/the-game.md) | What are the odds? | `lotto/tickets/coverage.py` |
| 2 | [Measuring honestly](docs/evaluation.md) | How do you know a result is real? | `lotto/core/` |
| 3 | [Simple strategies](docs/approaches/index.md) | Do hot numbers, due numbers or patterns work? | `lotto/approaches/statistical.py` and neighbours |
| 4 | [Neural networks](docs/approaches/set-prediction.md) | Can a network find what counting misses? | `lotto/approaches/neural.py`, `date_conditioned.py` |
| 5 | [Probability forecasts](docs/algorithm-research.md) | Can we forecast the whole draw? | `lotto/forecasting/` |
| 6 | [Algorithms for the ticket](docs/levers/index.md) | Given a budget, which rows? | `lotto/tickets/` |
| 7 | [Side quest](docs/side-quests/seed-search.md) | What if the lottery had a seed? | `lotto/seedsearch/`, `native/` |

## Map of the repository

```text
lotto/            the package
  core/           the measuring apparatus: data, metrics, backtest, randomness tests
  approaches/     every prediction strategy, simplest first
  forecasting/    probability forecasts of the whole draw
  tickets/        coverage, strong numbers, the draw planner
  seedsearch/     the seed-search side quest
  report/         rendering of the published report
  experiments/    one runnable module per experiment, and the catalog
experiments/      frozen protocols and results of the dated research runs
input/            the draw archive
legacy/           the 2023–2025 first attempt, kept as an exhibit
native/           Rust and CUDA engines for the seed search
docs/             the documentation site
tests/            the test suite, with a pinned copy of the archive
```

## How it is checked

```bash
uv sync --extra neural --group docs
uv run pytest -q
uv run ruff check .
uv run mkdocs build --strict
```

CI runs all of these on every push. A scheduled workflow rebuilds the report and the site on the
mornings after draw days. The draw archive itself can only be refreshed from a machine that can
reach pais.co.il, which GitHub's runners cannot:

```bash
uv run lotto report --refresh --out-dir docs
```

Then commit `input/` and `docs/`.

## Disclaimer

This is an educational project. Lottery draws are designed to be random; nothing here predicts
them, and nothing here is gambling advice. Designing a ticket changes how often a small prize is
won, not whether playing is profitable: the expected return of a row stays below its price.
