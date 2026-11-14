# Run it yourself

Ten minutes from a fresh clone to a result you produced.

## 1. Install

You need [uv](https://docs.astral.sh/uv/) and nothing else. It installs Python for you.

```bash
git clone https://github.com/Timorleiderman/ILotto.git
cd ILotto
uv sync
```

This installs the light version: no TensorFlow, no GPU, no Rust. It runs everything except the
two neural predictors.

## 2. See what there is

```bash
uv run lotto list
```

Every experiment is listed with the question it answers and a command for a fast first run.

## 3. Run your first experiment

```bash
uv run lotto run coverage --count 10 --seed 0 --out results/coverage.json
```

It designs ten rows that overlap as little as possible and counts, over all 2,324,784 possible
draws, how often at least one of them wins. It prints the ten rows and a line ending
`at least one prize in 38.58% of fair draws`. Ten times the single-row chance would be 41.8%: the
gap is what overlap costs, and the design keeps it small.

## 4. Build the benchmark

```bash
uv run lotto report --quick --n-test 100 --null-trials 500
```

About ten seconds. It first checks pais.co.il for draws newer than the archive and adds them to
`input/` (add `--offline` to skip that). It then runs nine randomness tests and sixteen prediction strategies on the real
archive and writes the report to `results/report/`: open `results/report/benchmark/index.html`
in a browser. The published site is untouched; it is rebuilt with `--out-dir docs`.

## 5. Fill a form with every model

```bash
uv run lotto predict --quick
```

A few seconds. Every model's best ticket for the next draw, and a 10-row form from each, built
on the exact coverage design. See [every model's form](../levers/predict.md).

## 6. Open the dashboard

```bash
uv run lotto ui
```

Prediction, comparison, backtesting and a personal log, with controls for the models. See
[the dashboard](dashboard.md).

## 7. Add the neural networks

```bash
uv sync --extra neural
uv run lotto report
```

About two minutes. The extra installs TensorFlow for the GRU and Transformer. The full report
also runs the SciPy-based date-conditioned model (which needs no TensorFlow), for nineteen
strategies in all.

## 8. Check everything

```bash
uv sync --extra neural --group docs
uv run pytest -q            # the test suite, about 30 seconds
uv run ruff check .         # lint
uv run mkdocs serve         # this site, on http://localhost:8000
```

## Where things are

| Folder | What is in it |
|---|---|
| `lotto/core/` | the measuring apparatus: data, metrics, backtest, randomness tests |
| `lotto/approaches/` | every prediction strategy, simplest first |
| `lotto/forecasting/` | probability forecasts of the whole draw |
| `lotto/tickets/` | coverage, strong numbers, the draw planner |
| `lotto/seedsearch/` | the seed-search side quest |
| `lotto/experiments/` | one runnable module per experiment |
| `lotto/ui/` | the dashboard |
| `experiments/` | frozen protocols and results of the dated research runs |
| `legacy/` | the first attempt, kept as an exhibit |
| `tests/` | the test suite, including a pinned copy of the archive |

Next: [how results are measured →](../evaluation.md)
