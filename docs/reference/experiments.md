# Experiment catalog

Generated from `lotto/experiments/__init__.py` by `uv run lotto list --markdown`. Do not edit by hand.

Every experiment runs through one command. Arguments after the id go to the experiment; add `--help` to see them.

| Chapter | Experiment | Question | Command | Write-up |
|---|---|---|---|---|
| 1. The benchmark | `report` | Does any strategy beat a random ticket, and is the draw random? | `uv run lotto run report --quick --n-test 100 --null-trials 500` | [results.md](../results.md) |
| 2. Forecasting | `calibration` | Do smoothed frequency forecasts beat the uniform forecast? | `uv run lotto run calibration --out results/calibration.json --portfolio-seeds 2` | [reference/research-log/prediction-probability.md](../reference/research-log/prediction-probability.md) |
| 2. Forecasting | `joint-mixture` | Can a learner that starts uncertain find a bias in the draws? | `uv run lotto run joint-mixture` | [algorithm-research.md](../algorithm-research.md) |
| 2. Forecasting | `thousand-draws` | Over 1,000 draws, do forecast-ranked tickets win more often? | `uv run lotto run thousand-draws` | [historical-strikes.md](../historical-strikes.md) |
| 2. Forecasting | `synthetic-training` | Does training on a million simulated draws transfer to the real ones? | `uv run lotto run synthetic-training` | [synthetic-learning.md](../synthetic-learning.md) |
| 3. Algorithms | `coverage` | How much prize chance can a fixed number of rows buy? | `uv run lotto run coverage --count 10 --seed 0 --out results/coverage.json` | [approaches/ticket-generators.md](../approaches/ticket-generators.md) |
| 3. Algorithms | `coverage-benchmark` | How much does exact refinement add, and how fast is it? | `uv run lotto run coverage-benchmark --seeds 3 --out results/coverage-benchmark.json` | [approaches/ticket-generators.md](../approaches/ticket-generators.md) |
| 3. Algorithms | `coverage-search` | Can basin hopping escape a local optimum of the coverage design? | `uv run lotto run coverage-search` | [algorithm-research.md](../algorithm-research.md) |
| 3. Algorithms | `strong-numbers` | Which strong number should each row carry? | `uv run lotto run strong-numbers` | [ticket-chances.md](../ticket-chances.md) |
| 3. Algorithms | `plan` | Which form should I play on the next draw? | `uv run lotto run plan` | [levers/draw-planner.md](../levers/draw-planner.md) |
| 4. Side quest | `seed-search` | If draws came from a seeded generator, could the seed be found? | `uv run lotto run seed-search --mode toy` | [side-quests/seed-search.md](../side-quests/seed-search.md) |
| 4. Side quest | `seed-search-job` | How many seeds can one machine test in a fixed time? | `uv run lotto run seed-search-job` | [side-quests/seed-search.md](../side-quests/seed-search.md) |
| 4. Side quest | `seed-search-remote` | How is a search shared with a remote GPU without overlap? | `uv run lotto run seed-search-remote` | [side-quests/seed-search.md](../side-quests/seed-search.md) |
| 4. Side quest | `seed-search-cuda-benchmark` | How fast is the CUDA engine, and is it correct? | `uv run lotto run seed-search-cuda-benchmark` | [side-quests/seed-search.md](../side-quests/seed-search.md) |

