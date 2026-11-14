# Frozen evidence

This folder holds the protocols and saved results of the dated research runs: the inputs each
run fixed in advance and every number it produced. There is no code here; the code is in
`lotto/experiments/` and runs through `uv run lotto run <id>`.

| Files | Experiment | Write-up |
|---|---|---|
| `prediction_*_2026_11_06.json` | `calibration` | `docs/reference/research-log/prediction-probability.md` |
| `algorithm_research_protocol_*`, `joint_mixture_research_*` | `joint-mixture` | `docs/algorithm-research.md` |
| `coverage_*_2026_11_06.json` | `coverage-search`, `coverage-benchmark` | `docs/approaches/ticket-generators.md` |
| `ticket_chances_*` | `strong-numbers` | `docs/ticket-chances.md` |
| `historical_strikes_*` | `thousand-draws` | `docs/historical-strikes.md` |
| `synthetic_learning_*` | `synthetic-training` | `docs/synthetic-learning.md` |
| `archive_2026-11-03.csv` | the draw archive those runs used | read by `joint-mixture`, `thousand-draws`, `synthetic-training` |

These runs were produced at commit `e6a792c`, before the package was reorganised. The protocol
files name source files by their paths at that commit. Since then, `tests/test_golden.py` checks
that the reorganised code still produces the same numbers.

New runs write to `results/`, which is not tracked, so they cannot overwrite this evidence.
