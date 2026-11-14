# ILotto Educational Refactor & Polish — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn ILotto from three layered generations of research into one teachable project: one story, one way to run anything, one page per experiment, with every published number still reproducible.

**Architecture:** Three homes, one rule each. Code that computes lives in one package with four subject subpackages. Each experiment is a thin runner plus a folder of frozen evidence. Words live in the docs site, organised as a learning path. The 2023–2025 Keras stack becomes a labelled exhibit instead of the front door.

**Tech stack:** Python 3.12, uv, NumPy/SciPy/pandas, Numba (coverage kernels), TensorFlow (optional extra), pytest, ruff, MkDocs Material, Rust + CUDA (optional seed-search engines).

**Review date:** 2026-11-14, branch `ticket-research-and-draw-planner` at `e6a792c`. Status: proposal. Nothing in the repository was changed to produce it.

---

## How to read this plan

| Part | What it is | Read it if you want to |
|---|---|---|
| A | Review: what is good, what is wrong, with evidence | decide whether you agree with the diagnosis |
| B | Target shape: layout, story, experiment catalog, page template, style rules | see where this is going |
| C | Six decisions with a recommended default each | veto something before work starts |
| D | Ten phases of tasks with files, code, commands, expected output | execute |
| E | Definition of done, and how this differs from the other plan in `planning/` | check completeness |

Phases are ordered so each one leaves the repository working and green. Phases 0 and 1 do not depend on any decision in Part C and can start immediately.

---

# Part A — Review

## A1. What I think of the project

The measurement core is the best thing here and it is good. `bench/data.py`, `metrics.py`, `backtest.py`, `predictors.py`, `randomness.py` and `generative.py` read like teaching material already: every docstring says *why*, every baseline is an exact formula, every control has a test proving the control has teeth (a compressor test that must flag a planted generator, a memorising lookup table that must score 6.000 in-sample and chance out of sample). The approach pages written in July–August share a shape that works: a one-minute abstract, what the method believes, the mechanism by which it fails, the measurement.

The problem is arrangement, not quality. The repository holds three generations of work stacked in the order they were written:

| Generation | When | Where | State |
|---|---|---|---|
| 1. Keras sequence models | 2023 – Dec 2025 | repo root (`train.py`, `models.py`, `smart_generator.py`, …), `model/` | untested, still contains every data bug the project teaches about |
| 2. The benchmark and site | Jul – Aug 2026 | `bench/` (11 modules), `docs/approaches/` | clean, tested, documented |
| 3. Ticket and forecast research | Nov 2026 | `bench/` (15 more modules), `scripts/` (12 more), `experiments/`, `native/` | correct and heavily tested, but dense, and filed wherever there was room |

A newcomer meets them in the wrong order. The README opens with six dated research bulletins, lists features from generation 1, and its "How to run" leads with the legacy scripts. The site navigation lists five dated reports before "How it is measured". `bench/` is a flat folder of 26 modules about four unrelated subjects.

## A2. Baseline measured during this review

| Check | Result |
|---|---|
| `uv run pytest -q` | **232 passed, 2 skipped**, 25–32 s. Skips: one needs a CUDA binary, one needs a host C++ toolchain |
| `uv run ruff check .` | passes over the whole repository |
| `uv run mkdocs build --strict` | passes, 0.5 s |
| `scripts/build_report.py --quick --n-test 100 --null-trials 500` | works, 4 s (output sent to a scratch folder) |
| Archive | 1,661 modern draws, 2012-01-03 → 2026-11-03; no duplicate dates, no repeated balls, no non-integer values |
| Null calibration of the analytic randomness tests | 600 simulated fair histories of 1,661 draws: rejection rate at α = 0.05 is 4.2% (uniformity), 5.8% (strong), 4.2% (serial), 4.8% (drift), all inside the 3.3–6.7% band. Pair test is conservative by design (mean p 0.70). Gap test is slightly off (KS p = 0.014 against uniform) |
| Size | 163 tracked files, 14.1k lines of Python, 6.6k lines of Markdown, 52 MB of tracked binaries, `.git` 197 MB |

## A3. Findings

Severity: **High** = a reader can be misled about a result. **Medium** = a real defect with a bounded effect. **Low** = polish.

### Correctness and validity

| ID | Sev | Finding | Evidence | Fixed in |
|---|---|---|---|---|
| C1 | High | The legacy split pairs test windows with unrelated targets and trains on reversed time. With a 100-row fixture (`test_size=10, w=3`) the first test window is rows `[87, 88, 89]` and its target is row `0`; the first training window is rows `[10, 11, 12]` predicting row `13`, and the file is newest-first, so that is predicting an older draw from newer ones. The value filter is still there too. README line 293 says this handling "was corrected"; the correction lives only in `bench/data.py` | `helpers.py:55`, `helpers.py:121-133`; reproduced | 5.2, 8.1 |
| C2 | High | The legacy "test" set drives checkpoint selection, early stopping and learning-rate decay, so it is a validation set and there is no untouched test set | `train.py:230` | 5.2 |
| C3 | Med | The surrogate-determinism test can never be flagged. With `n_sim=40` its smallest possible p is 1/41 = 0.0244; Holm's bar for the most extreme of 9 tests is 0.05/9 = 0.0056. The headline verdict is blind to it. Incompressibility clears the bar by 0.0006 (floor 0.00498) and stops clearing it when the family reaches 11 tests | `bench/randomness.py:256`, `:196` | 1.1 |
| C4 | Med | Ticket selection among tied scores is left to `np.argpartition`, whose choice among ties depends on the CPU-specific kernel. The docstring claims ties are "broken deterministically". The repo's own test notes that "argpartition breaks ties platform-dependently … CI caught exactly that". A laptop and the CI runner can publish different tickets from the same history. Ties are common, so this matters: on a flat score vector the current code picks `[21, 22, 23, 24, 25, 27]`, and switching to a fixed lowest-index rule moves 7 of the 17 quick-suite rows by 1–5 matches over 100 draws | `bench/metrics.py:51`, `bench/chaos.py:108,110,155`, comment in `tests/test_pipeline.py` (surrogate test); measured | 1.2 |
| C5 | Med | A refresh writes the download over the committed archive before parsing it. Any HTTP-200 body over 10,000 bytes passes: an HTML error page destroys the working copy, and a truncated CSV that still parses would be committed by the scheduled workflow | `bench/data.py:90`, same pattern in `helpers.py` | 1.3 |
| C6 | Med | The backtest window and the Monte-Carlo null can have different lengths. `walk_forward(n_test=1600)` scores 1,461 draws because of the 200-draw warm-up, while the null is built for 1,600 | `bench/backtest.py:46`, `scripts/build_report.py:89`; reproduced | 1.4 |
| C7 | Med | Frozen experiments are pinned to live paths. The synthetic-learning evaluation refuses to run unless nine source files (including `bench/data.py` and `bench/metrics.py`) and the live archive match recorded SHA-256 hashes and the archive has exactly 1,661 draws. The next archive refresh, or any fix in this plan, disables it. Several tests also assert statistical properties of the live archive, so a refresh can turn CI red by chance | `scripts/research_synthetic_learning.py:32,144-151`; `tests/test_pipeline.py:353,447`, `tests/test_generative.py:230` | 0.1, 0.2 |
| C8 | Low | Draws are cast to integers before validation (3.7 becomes 3) and repeated balls in a draw are not rejected. The committed archive is clean; this is ingestion hardening | `bench/data.py:147` | 1.5 |
| C9 | Low | `build_report.py`: the comment says "predict with the top statistical strategy" but `suite[0]` is the random ticket; the date-conditioned model is walk-forward backtested twice in a full run; `report.build` defaults to `docs/index.html`, which is not where the report goes | `scripts/build_report.py:93`, `:77`; `bench/report.py:154` | 1.6 |
| C10 | Low | `sum_distribution` says "exact null" but simulates 100,000 draws and applies a KS test to discrete data (conservative). The exact distribution is a ten-line dynamic program | `bench/randomness.py:145` | 1.7 |
| C11 | Low | Legacy model code: the Bahdanau branch concatenates the attention context twice; a mutable default argument; importing `helpers` or `metrics` configures logging and creates `evaluate.log` in the working directory | `ilotto.py:143,148`, `:18`; `helpers.py:9` | 5.2 (documented, not fixed) |

### Structure, readability and running

| ID | Sev | Finding | Evidence | Fixed in |
|---|---|---|---|---|
| S1 | High | No front door. README is 436 lines: six dated bulletins, legacy features, legacy commands first, the training section twice. The site nav is chronological | `README.md:19-59,61-69,94-160,241-255`; `mkdocs.yml:75-97` | 6.1, 8.1 |
| S2 | High | `bench/` is a flat bag: 26 modules covering the benchmark, approach families, ticket mathematics, forecast research, a web scraper and a distributed Rust/CUDA search. `__all__` lists 11 of them. The name no longer describes the contents | `bench/__init__.py` | 2.1–2.4 |
| S3 | High | Experiments are not units. Twelve research scripts with hard-coded dated output paths; scripts import other scripts (`research_synthetic_learning` → `research_historical_strikes` → `research_joint_mixture`); one imports the legacy `smart_generator`, which loads TensorFlow and writes a log file; write-ups are split between `experiments/*.md` (not on the site) and `docs/*.md` (on the site) with no rule; raw artifacts (20 MB in `experiments/`) and zipped copies (8.8 MB in `docs/data/`) are both tracked | `scripts/research_*.py`; `scripts/research_prediction_probability.py:109` | 4.1–4.4 |
| S4 | Med | The November code is correct but hard to read next to the July code. Validation is written as multi-line boolean expressions; three copies of `_integer`; private names imported across modules (`_full_tickets`, `_ticket_masks`, `_prefix_hash`, `_validate_draws`); `continue_search` is one ~180-line function; return-type annotations cover 63% of `bench/` functions and 6% of `scripts/`; docstrings 45% and 11% | `bench/portfolio.py:20`, `bench/calibration.py:15`, `scripts/research_joint_mixture.py:29`; `bench/strong_assignment.py:13`, `bench/synthetic_learning.py:17`; `bench/seed_campaign.py:141` | 7.1–7.4 |
| S5 | Med | Running things is not uniform. Thirteen scripts with different conventions and 14 `sys.path.insert` hacks, although `uv sync` already installs the package in editable mode (verified: `import bench` works from `/tmp`). Default data paths are relative to the working directory, so everything fails outside the repo root | `scripts/*.py`; `bench/data.py:29-33` | 3.1–3.3 |
| S6 | Med | Runtime dependencies include developer tools (`black`, `flake8`, `ruff`, `ipykernel`) and packages no code imports (`pydot`, `pydotplus`, `pyyaml`, `tqdm`). TensorFlow is required even to run the statistical benchmark | `pyproject.toml:8-29` | 3.4 |
| S7 | Med | Documentation drift. `CLAUDE.md` says 47 tests in ~10 s (now 232 in ~30 s) and that ruff must pass on the whole repo while CI lints three folders; the approach pages say 1,629 draws, the research pages 1,661; `bench/data.py` cites a `docs/methodology` that does not exist; the README project tree omits 15 modules; `GUIDE.md` tells readers to start with `evaluate.py`; the report's meta description says "1,500+ draws" | `CLAUDE.md:21`, `.github/workflows/ci.yml:32`, `bench/data.py:8`, `bench/report.py:436` | 0.3, 8.2 |
| S8 | Med | Popularity and birthday logic is still on live surfaces, against your stated preference: a scoreboard strategy that also appears in the published next-draw picks, the whole of `smart_generator.py`, and a README section that claims it will "maximize expected payout IF you win" | `bench/predictors.py:228`, `scripts/build_report.py:97`, `README.md:395-410` | 5.3 (Decision D3) |
| S9 | Med | Two features have no page at all: the next-draw planner and the seed search. The November pages repeat the same caveat in most paragraphs, which buries the result | `docs/` | 6.2, 6.4 |
| S10 | Low | Weight: 28 MB of Keras models (several unreferenced), 20 MB of experiment arrays, 8.8 MB of zips, in a 197 MB `.git` | `model/`, `experiments/`, `docs/data/` | 9 (optional) |

---

# Part B — Target shape

## B1. The rule: three homes

| Kind of thing | Lives in | Rule |
|---|---|---|
| Code that computes | `lotto/` | importable, typed, tested; never writes files or configures logging on import |
| An experiment | `lotto/experiments/<id>.py` + `experiments/<id>/` | the module is a recipe that reads top to bottom; the folder holds the frozen protocol, input snapshot and results |
| Words | `docs/` | one page per experiment, one template; generated pages are never hand-edited |

Everything else is either infrastructure (`tests/`, `.github/`, `native/`) or the exhibit (`legacy/`).

## B2. Repository layout

```text
ILotto/
├── README.md                    one screen: what, three results, three commands, map
├── CLAUDE.md                    contributor rules, kept in step with every phase
├── pyproject.toml  uv.lock
├── lotto/                       the package (today: bench/)
│   ├── cli.py                   `uv run lotto …` — the only entry point
│   ├── core/                    the measuring apparatus
│   │   ├── data.py              archive → Draws, by date, chronological
│   │   ├── metrics.py           matches, log loss, exact nulls, Holm
│   │   ├── backtest.py          walk-forward harness, Monte-Carlo null
│   │   ├── predictor.py         the Predictor contract (base class only)
│   │   ├── randomness.py        the nine tests
│   │   └── validation.py        small named input checks, shared by everything
│   ├── approaches/              one file per idea that tries to predict
│   │   ├── statistical.py  neural.py  date_conditioned.py
│   │   ├── perfect_fit.py  chaos.py   golden_ratio.py
│   │   └── __init__.py          quick_suite(), full_suite()
│   ├── forecasting/             probability forecasts of the whole draw
│   │   ├── calibration.py  joint_mixture.py  synthetic_learning.py
│   ├── tickets/                 what a buyer controls
│   │   ├── coverage.py  _kernels.py  strong_assignment.py
│   │   ├── forecast_portfolios.py   draw_plan.py
│   ├── seedsearch/              the artificial-generator side quest
│   │   ├── generator.py  campaign.py  allocations.py
│   │   ├── rust_engine.py  cuda_engine.py
│   ├── report/                  rendering only
│   │   ├── html.py  markdown.py  json.py  charts.py
│   └── experiments/             thin runners + the catalog
│       ├── __init__.py          CATALOG: id, question, runtime, page
│       └── coverage.py  strong_numbers.py  …  (one per experiment)
├── experiments/                 frozen evidence, no code
│   ├── README.md                generated index
│   └── <id>/  protocol.json  inputs/  results/
├── data/                        today: input/
│   └── Orig_IL_lotto.csv  lotto_IL_modern.csv
├── legacy/                      the 2023–2025 stack, labelled as an exhibit
├── native/                      Rust and CUDA engines (unchanged)
├── docs/                        the site (file paths unchanged; nav regrouped)
├── tests/                       mirrors lotto/: tests/core, tests/tickets, …
└── planning/                    this plan
```

## B3. The story the project tells

Lead with what works, then show what does not and how we know. This is the order for the README, the site home page and any talk.

1. **A fixed budget can be spent better.** One row wins a prize in 4.18% of fair draws. Two designed rows reach 8.33%, ten reach 38.60%, fourteen reach 51.01%. Exact, by enumerating all 2,324,784 main draws.
2. **Strong numbers can be assigned exactly.** For ten fixed rows, the chance of three-or-more plus the strong number rises from 5.926% to 5.957%, proven optimal by dynamic programming.
3. **Some draws pay more.** A draw after an unclaimed 40M jackpot pays every small tier 1.5×; the planner reads the last official result and says which form to play.
4. **The next draw cannot be picked.** Twenty strategies, walk-forward, against exact nulls: none beats a random ticket after correction.
5. **How we know.** Three data bugs once manufactured a 30% "accuracy". Filtering by date, chronological order and set-valued targets removed it.
6. **Looking impressive is cheap.** A lookup table reconstructs every past draw (6.000 of 6) and is worth chance on the next one.
7. **Side quests.** Searching for a seed in an artificial generator; the first attempt, kept as a specimen.

State each limit once per page, in one "Limits" box, not in every paragraph.

## B4. Experiment catalog

Every piece of existing work, mapped to one experiment. Numbers are not changed by this plan; only where things live.

| ID | Part | Question | Library code today | Runner today | Write-up today |
|---|---|---|---|---|---|
| `coverage` | Levers | How much prize chance can a fixed number of rows buy? | `bench/portfolio.py`, `_portfolio_exact.py` | `optimize_ticket_coverage.py`, `benchmark_coverage.py`, `research_coverage_search.py` | `docs/approaches/ticket-generators.md`, `docs/algorithm-research.md` §2, `experiments/coverage_review_iteration2.md` |
| `strong-numbers` | Levers | Which strong number should each row carry? | `bench/strong_assignment.py` | `research_ticket_chances.py` | `docs/ticket-chances.md` |
| `draw-planner` | Levers | Which form should I play on the next draw? | `bench/draw_plan.py` | `plan_draw.py` | none |
| `data-eras` | Prediction | What is in the archive, and how can cleaning invent a pattern? | `bench/data.py` | part of `build_report.py` | `docs/evaluation.md` |
| `randomness` | Prediction | Is there anything to predict? | `bench/randomness.py` | part of `build_report.py`; power study in `research_prediction_probability.py` | `docs/results.md` (generated) |
| `scoreboard` | Prediction | Does any strategy beat a random ticket? | `bench/predictors.py`, `nn.py`, `chaos.py`, `golden.py` | `build_report.py` | `docs/results.md`, six approach pages |
| `memorisation` | Prediction | Can a perfect fit of the past predict the future? | `bench/generative.py`, `spacetime.py` | part of `build_report.py` | `docs/approaches/date-conditioned.md`, `perfect-fit.md` |
| `calibration` | Prediction | Do smoothed frequencies forecast better than uniform? | `bench/calibration.py` | `research_prediction_probability.py` | `experiments/prediction_probability_research.md` |
| `joint-mixture` | Prediction | Can a learner that starts uncertain find a bias? | `bench/joint_mixture.py` | `research_joint_mixture.py` | `docs/algorithm-research.md` §1 |
| `thousand-draws` | Prediction | Over 1,000 draws, do forecast-ranked tickets win more? | `bench/historical_strikes.py` | `research_historical_strikes.py` | `docs/historical-strikes.md` |
| `synthetic-training` | Prediction | Does training on a million simulated draws transfer to real ones? | `bench/synthetic_learning.py` | `research_synthetic_learning.py` | `docs/synthetic-learning.md` |
| `seed-search` | Side quest | If draws came from a seeded generator, could the seed be found? | `bench/seed_*.py`, `native_seed_search.py`, `cuda_seed_search.py`, `native/` | `search_seeds.py`, `search_remote_seeds.py`, `run_seed_search_job.py`, `benchmark_cuda_search.py` | four files in `experiments/`, `native/cuda-search/README.md` |
| (exhibit) | Side quest | What did the first attempt get wrong? | root `*.py` | `train.py`, `predict.py`, `evaluate.py` | `docs/approaches/legacy-models.md` |

## B5. One page template

Every experiment page uses these seven sections, in this order. The July approach pages already follow most of it.

```markdown
# <Title>

!!! abstract "In one minute"
    **The question.** One sentence.
    **The answer.** One sentence with the headline number.
    **Why it matters.** One sentence.

## The idea            plain words and one diagram; no formulas yet
## How it works        the method, the formula, and a link to the function that implements it
## Run it              the quick command, its measured runtime, the output you should see;
                       then the full command that reproduces the published number
## Result              one table or figure; generated from results where possible
## What it teaches     two or three sentences a reader should be able to repeat
## Limits              one box: assumptions and what the result does not show
## Go deeper           code, tests, frozen data, related experiments
```

## B6. Style rules for code a learner will read

1. A module opens with a docstring that says what question it answers and why it exists.
2. Public functions carry type hints and a docstring stating shapes, indexing (0- or 1-based) and units.
3. Validation is a named function call (`require_ticket(row)`), never a five-line boolean.
4. No module imports another module's underscore-prefixed name. If two modules need it, it is public.
5. A function that scrolls off one screen (about 60 lines) is split at its comment headings.
6. Computation returns values; only `lotto/report/` and `lotto/experiments/` write files.
7. Importing anything from `lotto` has no side effects: no logging configuration, no downloads, no files.
8. Comments explain why a line must be the way it is; they do not restate it.

---

# Part C — Decisions

Each has a default that the tasks below assume. Say so if you want the alternative; only the named phases change.

| # | Decision | Default (recommended) | Alternative | Affects |
|---|---|---|---|---|
| D1 | Package name | Rename `bench` → `lotto` in the same commit that adds subpackages. Every import changes in that commit anyway, and `bench` no longer describes a package containing a scraper and a GPU search | Keep `bench`, add subpackages under it | Phase 2 |
| D2 | Legacy stack | Move to `legacy/`, keep it runnable, label it as an exhibit, and pin its known bugs with tests that document them | Delete it and keep only the docs page; or repair it onto the benchmark data path | Phase 5 |
| D3 | Popularity and birthday logic | Remove from live surfaces: drop `UnpopularNumbers` from the scoreboard and next-draw picks, drop the README section, remove the smart-generator arm from the calibration runner. It survives only inside `legacy/` and the dated research log | Keep `UnpopularNumbers` as a labelled control row | 5.3 |
| D4 | TensorFlow | Move to a `neural` extra so the statistical benchmark, coverage tools and docs install in seconds. CI installs the extra | Leave it in the base dependencies | 3.4 |
| D5 | Site URLs | Do not move existing doc files. Regroup the nav and add new pages only, so no published link breaks | Move files into new folders and add a redirects plugin | Phase 6 |
| D6 | Heavy artifacts | Stop tracking two copies. Keep small JSON in git; leave history alone | Move large arrays and models to a GitHub Release; or rewrite history to shrink `.git` | Phase 9 |

---

# Part D — Tasks

Conventions for every task:

- Run commands from the repository root.
- After each task: `uv run pytest -q` and `uv run ruff check .` must pass, then commit.
- A task that changes a published number says so and records the before and after in its commit message.

## Phase 0 — Safety net (no behaviour change)

Purpose: make it impossible for the refactor to change a result without a test failing.

### Task 0.1: Tag the research snapshot

The November experiments pin source files by path and hash (finding C7). Moving files breaks those checks by design, so the commit that still satisfies them gets a name.

**Step 1: Tag**

```bash
git tag -a research-2026-11 -m "Source and archive state that reproduces the November 2026 frozen experiments"
```

**Step 2: Record it**

Create `experiments/README.md` with one paragraph: the frozen November results reproduce exactly at tag `research-2026-11`; later commits reproduce the same numbers through the golden tests in `tests/golden/`, not through source hashes.

**Step 3: Commit**

```bash
git add experiments/README.md
git commit -m "docs: name the commit that reproduces the frozen November experiments"
```

Pushing the tag is your call; the plan does not push.

### Task 0.2: Pin an archive snapshot for tests

**Files:**
- Create: `tests/golden/archive_2026-11-03.csv` (copy of `input/Orig_IL_lotto.csv`, 184 KB)
- Create: `tests/conftest.py`
- Modify: `tests/test_pipeline.py:15-17`, `tests/test_generative.py:194,230,239,304,314`, `tests/test_research_runners.py:57`

**Step 1: Copy the snapshot**

```bash
mkdir -p tests/golden
cp input/Orig_IL_lotto.csv tests/golden/archive_2026-11-03.csv
shasum -a 256 tests/golden/archive_2026-11-03.csv
```

Expected: the hash starts `0034509bdfa510c3`.

**Step 2: Add shared fixtures**

```python
# tests/conftest.py
"""Shared fixtures.

Tests read a pinned copy of the archive, never the live one. The live archive
grows after every draw; a test that asserts a statistical property of it would
then fail by chance about one refresh in twenty. Monitoring the live data is
the report's job, not the test suite's.
"""

from pathlib import Path

import pytest

from bench import data

SNAPSHOT = Path(__file__).parent / "golden" / "archive_2026-11-03.csv"


@pytest.fixture(scope="session")
def draws():
    """The 1,661 modern draws through 2026-11-03."""
    return data.load(raw_csv=str(SNAPSHOT), clean_csv=None)
```

**Step 3: Use it**

Delete the module-level `draws` fixture in `tests/test_pipeline.py:15-17`. In each test listed above replace `d = data.load(clean_csv=None)` with a `draws` parameter (`def test_x(draws): d = draws`).

**Step 4: Verify**

Run: `uv run pytest -q`
Expected: `232 passed, 2 skipped`.

**Step 5: Commit**

```bash
git add tests/conftest.py tests/golden tests/test_pipeline.py tests/test_generative.py tests/test_research_runners.py
git commit -m "test: read a pinned archive snapshot instead of the live archive"
```

### Task 0.3: Fix the drift that costs nothing

**Files:**
- Modify: `CLAUDE.md:21` — "Tests (232, ~30s)"
- Modify: `.github/workflows/ci.yml:32` and the Lint step of `report.yml` — `uv run ruff check .`
- Modify: `bench/data.py:8` — replace "(see `docs/methodology`)" with "(see `docs/evaluation.md`)"
- Modify: `bench/report.py:436` — "1,500+ draws" → "1,600+ draws"
- Modify: `bench/__init__.py` — list all 26 modules in `__all__`, grouped by subject with a comment per group
- Modify: `README.md:293` — "once the data handling was corrected in `bench/data.py`; the legacy scripts at the repository root still use the old handling"

**Verify:** `uv run ruff check . && uv run pytest -q && uv run mkdocs build --strict`

**Commit:** `docs: correct stale counts, paths and lint scope`

### Task 0.4: Golden numbers

Do this **after Phase 1**, because Tasks 1.1 and 1.2 deliberately change a few numbers. It is listed here because it belongs to the safety net.

**Files:**
- Create: `tests/golden/regenerate.py`
- Create: `tests/golden/quick_suite_n100.json`, `tests/golden/coverage.json`
- Create: `tests/test_golden.py`

**Step 1: Write the test**

```python
# tests/test_golden.py
"""Characterization tests: numbers that must not move while files do.

These pin behaviour, not correctness. If one fails during a mechanical move,
the move changed a result. If one fails after an intended fix, regenerate with
`uv run python -m tests.golden.regenerate` and put the before/after numbers in
the commit message.
"""

import json
from pathlib import Path

import pytest

from bench import backtest, predictors
from bench.chaos import chaos_suite
from bench.golden import golden_suite
from bench.portfolio import portfolio_odds
from bench.spacetime import spacetime_suite
from bench.strong_assignment import strong_prize_odds

GOLDEN = Path(__file__).parent / "golden"


def quick_suite_numbers(draws) -> dict:
    suite = predictors.statistical_suite() + spacetime_suite() + chaos_suite() + golden_suite()
    results = backtest.run_suite(suite, draws, n_test=100)
    return {
        r.name: {"total_matches": int(r.match_counts.sum()), "p_value": r.summary["p_value"]}
        for r in results
    }


def coverage_numbers() -> dict:
    saved = json.loads(Path("experiments/ticket_chances_2026_11_07.json").read_text())
    rows = saved["selected_portfolio"]
    odds, strong = portfolio_odds(rows), strong_prize_odds(rows)
    return {
        "best_match_counts": odds["best_match_counts"],
        "covered_with_strong": strong["covered_full_outcomes"],
    }


def test_quick_suite_is_unchanged(draws):
    want = json.loads((GOLDEN / "quick_suite_n100.json").read_text())
    got = quick_suite_numbers(draws)
    assert got.keys() == want.keys()
    for name, expected in want.items():
        assert got[name]["total_matches"] == expected["total_matches"], name
        assert got[name]["p_value"] == pytest.approx(expected["p_value"], rel=1e-12), name


def test_selected_portfolio_coverage_is_unchanged():
    assert coverage_numbers() == json.loads((GOLDEN / "coverage.json").read_text())
```

**Step 2: Write the regenerator**

```python
# tests/golden/regenerate.py
"""Rewrite the golden files from the current code. Run only after an intended change."""

import json
from pathlib import Path

from bench import data
from tests.test_golden import GOLDEN, coverage_numbers, quick_suite_numbers

draws = data.load(raw_csv=str(GOLDEN / "archive_2026-11-03.csv"), clean_csv=None)
(GOLDEN / "quick_suite_n100.json").write_text(json.dumps(quick_suite_numbers(draws), indent=2) + "\n")
(GOLDEN / "coverage.json").write_text(json.dumps(coverage_numbers(), indent=2) + "\n")
print("golden files rewritten")
```

**Step 3: Run the test to see it fail**

Run: `uv run pytest tests/test_golden.py -q`
Expected: FAIL, `FileNotFoundError: … quick_suite_n100.json`.

**Step 4: Generate and pass**

```bash
uv run python -m tests.golden.regenerate
uv run pytest tests/test_golden.py -q
```

Expected: `2 passed`. Check `coverage.json`: `covered_with_strong` must be `969420`, the figure published in `docs/ticket-chances.md` (verified during this review). When Phase 4 moves `experiments/ticket_chances_2026_11_07.json`, update the path in `coverage_numbers()` in the same commit.

**Step 5: Push the branch and watch CI.** If Linux disagrees with a golden recorded on macOS, that is a real finding: some selection still depends on the platform. Find the remaining tie before going on.

**Step 6: Commit** — `test: pin scoreboard and coverage numbers before restructuring`

Add one golden per experiment as it is migrated in Phase 4. Mark any that take more than five seconds with `@pytest.mark.slow`.

---

## Phase 1 — Correctness fixes (test first)

Each task is one behaviour change with its own failing test. Keep them out of the mechanical moves in Phase 2 so a reviewer can see what changed.

### Task 1.1: Every randomness test can reach significance

**Files:**
- Create: `tests/test_randomness_reachability.py`
- Modify: `bench/randomness.py:196` and `:256`

**Step 1: Write the failing test**

```python
# tests/test_randomness_reachability.py
"""A test that can never be flagged is not a test."""

import inspect

from bench import randomness

ALPHA = 0.05


def monte_carlo_floor(test) -> float | None:
    """Smallest p-value a Monte-Carlo test can report, 1 / (1 + n_sim). None if analytic."""
    n_sim = inspect.signature(test).parameters.get("n_sim")
    return None if n_sim is None else 1 / (1 + n_sim.default)


def test_every_monte_carlo_test_can_survive_holm():
    # Holm compares the smallest p-value of m tests with alpha / m. A simulated
    # test whose floor sits above that bar cannot be flagged whatever the data
    # do, and the report's headline verdict would be blind to it. Half the bar
    # leaves room for the family to double before this has to be revisited.
    bar = ALPHA / len(randomness.ALL_TESTS)
    for test in randomness.ALL_TESTS:
        floor = monte_carlo_floor(test)
        if floor is not None:
            assert floor <= bar / 2, f"{test.__name__}: floor {floor:.5f} vs bar {bar:.5f}"
```

**Step 2: Run it**

Run: `uv run pytest tests/test_randomness_reachability.py -q`
Expected: FAIL, `incompressibility: floor 0.00498 vs bar 0.00556` (the surrogate test fails the same way at 0.02439).

**Step 3: Fix**

In `bench/randomness.py` set `n_sim: int = 399` in both `incompressibility` and `surrogate_determinism` (floor 0.0025). Add one sentence to each docstring: "`n_sim` sets the smallest reportable p-value, 1/(1+n_sim); it must stay below alpha divided by the number of tests in `ALL_TESTS`."

**Step 4: Verify and measure**

```bash
uv run pytest -q
time uv run python scripts/build_report.py --quick --n-test 100 --null-trials 500 --out-dir /tmp/lotto-report
```

Expected: tests pass. Note the build time in the commit message; it was 4 s before this change and the two tests now run ten and two times as many simulations.

**Step 5: Commit** — `fix: give the two simulated randomness tests enough resolution to survive Holm`

### Task 1.2: Tie-breaking that is the same on every machine

**Files:**
- Modify: `bench/metrics.py:49-51`
- Modify: `bench/chaos.py:108,110,155`
- Test: `tests/test_pipeline.py`

**Step 1: Write the test**

```python
def test_top_k_breaks_ties_toward_the_lower_index():
    scores = np.array([1.0, 3.0, 3.0, 3.0, 2.0, 3.0])
    assert metrics.top_k(scores, k=2).tolist() == [1, 2]
    assert metrics.top_k(np.zeros(37)).tolist() == [0, 1, 2, 3, 4, 5]
```

**Step 2: Run it**

Run: `uv run pytest tests/test_pipeline.py::test_top_k_breaks_ties_toward_the_lower_index -q`
Expected on this Mac: FAIL on the second assertion; the current code returns `[21, 22, 23, 24, 25, 27]` for a flat vector. On another CPU it may return something else, which is the defect.

**Step 3: Implement**

```python
def top_k(scores: np.ndarray, k: int = N_DRAWN) -> np.ndarray:
    """Indices (0-based) of the k highest scores, in ascending order.

    Ties go to the lower index. A stable sort makes that true on every
    platform; `np.argpartition` leaves the choice among tied scores to a
    CPU-specific kernel, so the same history could produce one ticket on a
    laptop and another on the CI runner.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return np.sort(order[:k])
```

In `bench/chaos.py` add `from . import metrics`, replace both `np.argpartition(dist, k)[:k]` calls with `np.argsort(dist, kind="stable")[:k]`, and replace `np.argpartition(-pred, N_DRAWN)[:N_DRAWN]` with `metrics.top_k(pred)`.

**Step 4: Verify**

Run: `uv run pytest -q`
Expected: all pass. The surrogate test's three-seed workaround in `tests/test_pipeline.py` can stay; note in its comment that ties are now stable.

**Step 5: Commit** — `fix: break score ties by index so tickets are identical on every platform`

This changes published numbers. Measured on the 100-draw quick suite: Hot (last 100) 99 → 104 total matches, Overdue 91 → 96, Method of analogues 98 → 100, Hot (all-time) 104 → 105, Polynomial 105 → 106, Rank ensemble 77 → 78, Pairwise Markov 88 → 87. None of these is skill; a different arbitrary rule was replaced by a stated one. Put the table in the commit message.

### Task 1.3: A refresh can only extend the archive

**Files:**
- Modify: `bench/data.py:77-97`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing tests**

```python
class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


def _serve(monkeypatch, content: bytes) -> None:
    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: _Response(content))


def _local_archive(tmp_path):
    raw = tmp_path / "archive.csv"
    original = (Path(__file__).parent / "golden" / "archive_2026-11-03.csv").read_bytes()
    raw.write_bytes(original)
    return raw, original


def test_refresh_rejects_a_large_error_page(tmp_path, monkeypatch):
    raw, original = _local_archive(tmp_path)
    _serve(monkeypatch, b"<html>" + b"x" * 20_000 + b"</html>")
    assert data.download(str(raw)) is False
    assert raw.read_bytes() == original


def test_refresh_rejects_an_archive_that_lost_draws(tmp_path, monkeypatch):
    raw, original = _local_archive(tmp_path)
    lines = original.split(b"\n")
    _serve(monkeypatch, b"\n".join(lines[:1] + lines[200:]))  # header, minus the newest 199 draws
    assert data.download(str(raw)) is False
    assert raw.read_bytes() == original


def test_refresh_accepts_an_archive_that_keeps_every_draw(tmp_path, monkeypatch):
    raw, original = _local_archive(tmp_path)
    _serve(monkeypatch, original)
    assert data.download(str(raw)) is True
    assert not (tmp_path / "archive.csv.download").exists()
```

Add `from pathlib import Path` to the test module if it is missing.

**Step 2: Run them**

Expected: the first two FAIL (the file was overwritten).

**Step 3: Implement**

```python
def download(raw_csv: str = RAW_CSV, timeout: int = 15) -> bool:
    """Refresh the raw archive from pais.co.il. Returns True if it was replaced.

    The download replaces the local file only after it has been parsed and
    shown to contain every draw already on record: an archive only grows.
    Any failure keeps the committed copy, which is what keeps CI green when
    the site is unreachable.
    """
    os.makedirs(os.path.dirname(raw_csv) or ".", exist_ok=True)
    try:
        res = requests.get(CSV_URL, timeout=timeout)
        res.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not refresh archive (%s); using committed copy", exc)
        return False

    candidate = raw_csv + ".download"
    with open(candidate, "wb") as fh:
        fh.write(res.content)
    try:
        problem = _refresh_problem(candidate, raw_csv)
    except Exception as exc:  # an HTML error page, or a changed column layout
        problem = f"it could not be parsed ({exc})"
    if problem:
        os.remove(candidate)
        logger.warning("Downloaded archive rejected: %s; keeping local copy", problem)
        return False
    os.replace(candidate, raw_csv)
    logger.info("Downloaded %d bytes of draw history", len(res.content))
    return True


def _refresh_problem(candidate: str, current: str) -> str | None:
    """Why `candidate` must not replace `current`, or None if it may."""
    new = _read_raw(candidate)
    if new.empty:
        return "no draws were parsed from it"
    if not os.path.exists(current):
        return None
    missing = set(_read_raw(current)["draw_id"]) - set(new["draw_id"])
    if missing:
        return f"{len(missing)} draws already on record are missing from it"
    return None
```

Add `*.download` to `.gitignore`.

**Step 4: Verify** — `uv run pytest -q`, all pass.

**Step 5: Commit** — `fix: validate a downloaded archive before it replaces the committed one`

### Task 1.4: One evaluation window for the backtest and the null

**Files:**
- Modify: `bench/backtest.py:39-48`
- Modify: `scripts/build_report.py:55-56,89`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
def test_backtest_and_null_share_one_window():
    d = fake_draws(400)
    window = backtest.evaluation_window(len(d), n_test=1000)
    assert len(window) == 200  # 400 draws minus the 200-draw warm-up
    result = backtest.walk_forward(predictors.Frequency(), d, n_test=1000)
    assert result.summary["n_draws"] == len(window)
```

**Step 2: Run it** — FAIL, `AttributeError: module 'bench.backtest' has no attribute 'evaluation_window'`.

**Step 3: Implement**

```python
def evaluation_window(n_draws: int, n_test: int, min_history: int = 200) -> range:
    """Indices of the draws a walk-forward run will score.

    Resolved once and shared by the backtest, the Monte-Carlo null and the
    report, so they cannot describe windows of different lengths.
    """
    if n_test < 1:
        raise ValueError("n_test must be at least 1")
    start = max(n_draws - n_test, min_history)
    if start >= n_draws:
        raise ValueError(f"Not enough draws ({n_draws}) for min_history={min_history}")
    return range(start, n_draws)
```

In `walk_forward`, replace the `start_idx` block with `window = evaluation_window(len(draws), n_test, min_history)` and iterate `for i in window:`; replace `(i - start_idx)` with `(i - window.start)`.

In `scripts/build_report.py`, compute `window = backtest.evaluation_window(len(draws), args.n_test)` before the backtest, log a warning if `len(window) < args.n_test`, and call `random_ticket_null(draws, len(window), …)`.

**Step 4: Verify** — `uv run pytest -q`.

**Step 5: Commit** — `fix: build the Monte-Carlo null for the window the backtest actually scored`

### Task 1.5: Validate draws before casting

**Files:** Modify `bench/data.py:146-158`; test in `tests/test_pipeline.py`.

**Step 1: Failing tests** — write a three-row CSV in `tmp_path` in the raw layout (copy the header and two rows from the snapshot, then edit a third) and assert `data.load(raw_csv=…, clean_csv=None)` raises `ValueError` for (a) a ball value of `3.7`, (b) a row whose balls are `5,5,9,12,20,31`. Follow the pattern of `test_empty_era_fails_with_a_clear_message` at `tests/test_pipeline.py:206`.

**Step 2: Implement** — before the integer cast:

```python
    values = df[ALL_COLS].to_numpy(dtype=np.float64)
    if not np.array_equal(values, np.round(values)):
        raise ValueError(f"Non-integer ball values in {raw_csv}; the archive layout changed upstream.")
```

and after `balls = np.sort(balls, axis=1)`:

```python
    if (np.diff(balls, axis=1) == 0).any():
        raise ValueError(f"A draw in {raw_csv} repeats a ball; six distinct numbers are required.")
```

**Step 3: Verify, commit** — `fix: reject fractional and repeated balls instead of truncating them`

### Task 1.6: Three small report defects

**Files:** `scripts/build_report.py:69-101`, `bench/report.py:154`.

1. Correct the comment at `build_report.py:91-93` to say what the code does: "Show the random ticket, the top-ranked strategy, one neural model and the date-conditioned model".
2. Reuse the suite's result for the date-conditioned model in the memorisation table instead of a second `walk_forward`: look it up with `next(r for r in results if r.name == p.name)`. Keep the extra backtest only for `MemorisingDateLookup`, which is not in the suite.
3. Make `out_path` in `report.build` a required argument.

**Verify:** the existing `test_verdict_covers_randomness_tests_too` and `test_stale_archive_is_declared_in_generated_pages` already call `report.build` with `out_path`; `uv run pytest -q` passes. Run a full (non-quick) build once and confirm the memorisation table still has two rows.

**Commit:** `fix: stop backtesting the date-conditioned model twice; correct a misleading comment`

### Task 1.7: An exact null for the draw-sum test

**Files:** `bench/randomness.py:144-164`; test in `tests/test_pipeline.py`.

**Step 1: Failing test**

```python
def test_sum_null_is_exact():
    pmf = randomness.sum_pmf()
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.flatnonzero(pmf)[[0, -1]].tolist() == [21, 207]  # 1+…+6 and 32+…+37
    assert (np.arange(len(pmf)) * pmf).sum() == pytest.approx(6 * 38 / 2)  # mean 114
```

**Step 2: Implement** — a dynamic program over the 37 numbers counting subsets by size and sum:

```python
def sum_pmf() -> np.ndarray:
    """Exact distribution of the sum of six distinct numbers from 1..37."""
    max_sum = sum(range(N_NUMBERS - N_DRAWN + 1, N_NUMBERS + 1))
    ways = np.zeros((N_DRAWN + 1, max_sum + 1))
    ways[0, 0] = 1.0
    for number in range(1, N_NUMBERS + 1):
        for size in range(N_DRAWN, 0, -1):
            ways[size, number:] += ways[size - 1, :-number]
    return ways[N_DRAWN] / ways[N_DRAWN].sum()
```

Then replace the simulation in `sum_distribution` with a chi-square of observed sums against `sum_pmf()`, pooling sums into bins whose expected count is at least five. Update the test name string and docstring; the row label in `docs/results.md` changes on the next build.

**Step 3: Calibrate** — add `sum_distribution` to the calibration experiment in Task 4.3 and confirm its rejection rate at α = 0.05 sits near 5%.

**Step 4: Commit** — `fix: test draw sums against their exact distribution`

**After Phase 1, do Task 0.4** and record the golden numbers.

---

## Phase 2 — Restructure the package (mechanical, no behaviour change)

One commit per task. The golden tests from 0.4 must pass after each.

### Task 2.1: Move modules

**Step 1: Create the tree and move files with history**

```bash
git mv bench lotto
mkdir -p lotto/core lotto/approaches lotto/forecasting lotto/tickets lotto/seedsearch lotto/report
cd lotto
git mv data.py metrics.py backtest.py randomness.py core/
git mv predictors.py approaches/statistical.py
git mv nn.py approaches/neural.py
git mv generative.py approaches/date_conditioned.py
git mv spacetime.py approaches/perfect_fit.py
git mv chaos.py approaches/chaos.py
git mv golden.py approaches/golden_ratio.py
git mv calibration.py joint_mixture.py synthetic_learning.py forecasting/
git mv portfolio.py tickets/coverage.py
git mv _portfolio_exact.py tickets/_kernels.py
git mv strong_assignment.py draw_plan.py tickets/
git mv historical_strikes.py tickets/forecast_portfolios.py
git mv seed_search.py seedsearch/generator.py
git mv seed_campaign.py seedsearch/campaign.py
git mv seed_allocations.py seedsearch/allocations.py
git mv native_seed_search.py seedsearch/rust_engine.py
git mv cuda_seed_search.py seedsearch/cuda_engine.py
git mv report.py report/html.py
git mv charts.py report/charts.py
cd ..
```

Give each new folder an `__init__.py` whose docstring is one paragraph: what question this subpackage answers and which module to read first.

**Step 2: Rewrite imports**

The mapping (old dotted name → new):

| Old | New |
|---|---|
| `bench.data` `bench.metrics` `bench.backtest` `bench.randomness` | `lotto.core.<same>` |
| `bench.predictors` | `lotto.approaches.statistical` |
| `bench.nn` | `lotto.approaches.neural` |
| `bench.generative` | `lotto.approaches.date_conditioned` |
| `bench.spacetime` | `lotto.approaches.perfect_fit` |
| `bench.chaos` | `lotto.approaches.chaos` |
| `bench.golden` | `lotto.approaches.golden_ratio` |
| `bench.calibration` `bench.joint_mixture` `bench.synthetic_learning` | `lotto.forecasting.<same>` |
| `bench.portfolio` | `lotto.tickets.coverage` |
| `bench._portfolio_exact` | `lotto.tickets._kernels` |
| `bench.strong_assignment` `bench.draw_plan` | `lotto.tickets.<same>` |
| `bench.historical_strikes` | `lotto.tickets.forecast_portfolios` |
| `bench.seed_search` | `lotto.seedsearch.generator` |
| `bench.seed_campaign` | `lotto.seedsearch.campaign` |
| `bench.seed_allocations` | `lotto.seedsearch.allocations` |
| `bench.native_seed_search` | `lotto.seedsearch.rust_engine` |
| `bench.cuda_seed_search` | `lotto.seedsearch.cuda_engine` |
| `bench.report` | `lotto.report.html` |
| `bench.charts` | `lotto.report.charts` |

Inside the package, relative imports gain one level (`from .data import` in `lotto/approaches/chaos.py` becomes `from ..core.data import`). Update `scripts/`, `tests/`, `tests/golden/`, `native/cuda-search/bootstrap.py`, both workflows (the freshness snippet in `report.yml` imports `bench`), `Dockerfile`, `pyproject.toml` (`packages = ["lotto"]`).

Two path constants change depth and will fail silently if missed:
- `lotto/seedsearch/rust_engine.py`: `DEFAULT_BINARY` needs one more `.parent`.
- Any `Path(__file__).resolve().parent.parent` in scripts that referred to the repo root through `bench/`.

**Step 3: Verify**

```bash
uv sync
grep -rn "from bench\|import bench\|bench\." --include='*.py' --include='*.yml' --include='*.toml' . | grep -v '\.venv'
uv run pytest -q
uv run ruff check .
uv run python scripts/build_report.py --quick --n-test 100 --null-trials 500 --out-dir /tmp/lotto-report
```

Expected: the grep prints nothing; tests pass including `tests/test_golden.py`; the report builds.

**Step 4: Update prose** — replace `bench/<file>.py` references in `docs/`, `README.md`, `CLAUDE.md`, `GUIDE.md`, `ARCHITECTURES.md` using the same table. Run `uv run mkdocs build --strict`.

**Step 5: Commit** — `refactor: rename bench to lotto and group modules by subject`

### Task 2.2: Separate the contract from the strategies

Move `class Predictor` from `lotto/approaches/statistical.py` to a new `lotto/core/predictor.py`; re-import it where used. Move suite assembly out of `scripts/build_report.py` into `lotto/approaches/__init__.py`:

```python
def quick_suite() -> list[Predictor]:
    """Every strategy that needs no training: seconds to backtest."""

def full_suite() -> list[Predictor]:
    """quick_suite() plus the date-conditioned model and the two neural models."""
```

`full_suite` imports TensorFlow lazily, as `build_report.py` does today. Commit: `refactor: one contract module, one place that lists the strategies`.

### Task 2.3: Split the report module

`lotto/report/html.py` (651 lines) holds three writers. Move `dump_markdown` to `lotto/report/markdown.py` and `dump_json` to `lotto/report/json.py`; move `predict_next` to `lotto/core/backtest.py` (it is computation, not rendering). The staleness calculation is duplicated in the HTML and Markdown writers: extract `stale_days(draws) -> int` once. Commit: `refactor: one module per report format`.

### Task 2.4: Mirror the package in tests

| Today | Becomes |
|---|---|
| `tests/test_pipeline.py` (39 tests, 8 modules) | `tests/core/test_data.py`, `test_metrics.py`, `test_backtest.py`, `test_randomness.py`; `tests/approaches/test_statistical.py`, `test_neural.py`, `test_perfect_fit.py`, `test_chaos.py`, `test_golden_ratio.py`; `tests/report/test_report.py` |
| `tests/test_generative.py` | `tests/approaches/test_date_conditioned.py` |
| `tests/test_prediction_probability.py` (28 tests, mixed) | split by the module each test exercises: `tests/tickets/test_coverage.py`, `tests/forecasting/test_calibration.py`, `tests/core/test_metrics.py` |
| `tests/test_coverage_search.py`, `test_strong_assignment.py`, `test_draw_plan.py`, `test_historical_strikes.py` | `tests/tickets/` |
| `tests/test_joint_mixture.py`, `test_synthetic_learning.py` | `tests/forecasting/` |
| `tests/test_seed_*.py`, `test_native_seed_search.py`, `test_cuda_seed_search.py` | `tests/seedsearch/` |

Move `fake_draws` into `tests/conftest.py` as a fixture factory. Verify the count is unchanged: `uv run pytest -q` reports the same number of passed tests as before the split. Commit: `test: mirror the package layout`.

### Task 2.5: Rename `input/` to `data/`

`git mv input data`; update `RAW_CSV` and `CLEAN_CSV`, the `git add` line in `report.yml`, `.dockerignore`, the legacy `helpers.py` defaults, and `CLAUDE.md`. Resolve the two paths from the package location (`Path(__file__).resolve().parents[2] / "data"`) rather than the working directory, which fixes the "fails outside the repo root" half of S5. The protocol JSONs in `experiments/` that name `input/Orig_IL_lotto.csv` are historical records: leave them. Commit: `refactor: data/ holds data; paths no longer depend on the working directory`.

---

## Phase 3 — One way to run everything

### Task 3.1: The catalog

**Files:** Create `lotto/experiments/__init__.py`; test `tests/test_catalog.py`.

**Step 1: Failing test**

```python
# tests/test_catalog.py
import importlib
from pathlib import Path

from lotto.experiments import CATALOG


def test_every_experiment_is_complete():
    assert len({e.id for e in CATALOG}) == len(CATALOG)
    for e in CATALOG:
        assert e.question.endswith("?"), e.id
        assert callable(importlib.import_module(e.module).run), e.id
        assert (Path("docs") / e.page).is_file(), e.id
```

**Step 2: Implement**

```python
# lotto/experiments/__init__.py
"""The catalog: every experiment in the project, in the order a reader meets them.

Adding an experiment means adding one entry here and one module exposing
`run(full: bool, out: Path) -> dict`. The command line, the generated index
page and the completeness test all read this list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    id: str  #: what you type: `lotto run coverage`
    title: str
    question: str  #: one sentence ending in a question mark
    part: str  #: "levers", "prediction" or "side-quest"
    module: str  #: dotted path of the runner
    page: str  #: write-up, relative to docs/
    quick_seconds: int  #: measured on a laptop, shown by `lotto list`
    needs: tuple[str, ...] = ()  #: optional extras, e.g. ("neural",)


CATALOG: tuple[Experiment, ...] = ()  # filled one entry at a time in Phase 4
```

The test passes vacuously now and becomes meaningful as entries arrive. Commit.

### Task 3.2: The command

**Files:** Create `lotto/cli.py`; modify `pyproject.toml`; test `tests/test_cli.py`.

```toml
[project.scripts]
lotto = "lotto.cli:main"
```

```python
# lotto/cli.py
"""One command for the whole project.

    uv run lotto list                 every experiment: id, question, runtime
    uv run lotto run coverage         run one, quick version (under a minute)
    uv run lotto run coverage --full  reproduce the published numbers
    uv run lotto report --quick       rebuild the benchmark report
    uv run lotto plan                 which form to play on the next draw
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .experiments import CATALOG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lotto", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="show every experiment")

    run = commands.add_parser("run", help="run one experiment")
    run.add_argument("experiment", choices=[e.id for e in CATALOG])
    run.add_argument("--full", action="store_true", help="reproduce the published numbers (slow)")
    run.add_argument("--out", type=Path, help="where to write results (default: results/<id>/)")

    args = parser.parse_args(argv)
    if args.command == "list":
        for e in CATALOG:
            print(f"{e.id:20s} ~{e.quick_seconds:>3d}s  {e.question}")
        return 0

    experiment = next(e for e in CATALOG if e.id == args.experiment)
    out = args.out or Path("results") / experiment.id
    summary = importlib.import_module(experiment.module).run(full=args.full, out=out)
    print(json.dumps(summary, indent=2))
    print(f"\nWrite-up: docs/{experiment.page}\nFiles:    {out}/")
    return 0
```

Test: `main(["list"])` returns 0; an unknown id exits with code 2. Add `results/` to `.gitignore`: new runs never overwrite frozen evidence in `experiments/`.

`report` and `plan` subcommands arrive in 3.3.

### Task 3.3: Fold the two everyday scripts into the command

Move the body of `scripts/build_report.py` to `lotto/report/build.py::main(argv)` and of `scripts/plan_draw.py` to `lotto/tickets/draw_plan.py::main(argv)`; register `lotto report` and `lotto plan` as subcommands that forward their remaining arguments. Update `ci.yml`, `report.yml`, `Dockerfile` and `CLAUDE.md` to call `uv run lotto report …`. Delete the two scripts. Verify `uv run lotto report --quick --n-test 100 --null-trials 500` writes the same files as before.

### Task 3.4: Dependencies by purpose

```toml
dependencies = [
    "numba>=0.61.0",
    "numpy>=2.0",
    "pandas>=2.2.3",
    "requests>=2.32.0",
    "scipy>=1.15.2",
]

[project.optional-dependencies]
neural = [
    "tensorflow>=2.19.0; sys_platform != 'darwin' or platform_machine != 'x86_64'",
    "tensorflow>=2.16.2,<2.17; sys_platform == 'darwin' and platform_machine == 'x86_64'",
]
gpu = ["tensorflow[and-cuda]>=2.19.0; sys_platform == 'linux' and platform_machine == 'x86_64'"]
legacy = ["h5py>=3.13.0", "matplotlib>=3.10.1", "mdutils>=1.6.0", "pydot>=3.0.4", "scikit-learn>=1.6.1"]

[dependency-groups]
dev = ["pytest>=9.1.1", "ruff>=0.11.2"]
docs = ["mkdocs-material>=9.5"]
notebook = ["ipykernel>=6.29.5"]
```

Keep the comments that explain the TensorFlow markers; they move with the entries. Check the resolved NumPy version in `uv.lock` before choosing its lower bound. Drop `black`, `flake8`, `pydotplus`, `pyyaml`, `tqdm`.

Steps: edit; `uv lock`; `uv sync --extra neural --group docs`; add `pytest.importorskip("tensorflow")` at the top of the neural tests; change both workflows and the Dockerfile to `uv sync --frozen --extra neural --group docs`; then prove the light path in a clean environment:

```bash
uv sync --frozen            # no extras
uv run pytest -q            # neural tests skip, everything else passes
uv run lotto report --quick --n-test 100 --null-trials 500
```

Update the GPU section of `CLAUDE.md`. Commit: `build: TensorFlow is an extra; developer tools leave the runtime dependencies`.

---

## Phase 4 — Experiments as units

Migrate one experiment completely, review the pattern, then repeat. Order: `coverage` first (fast, exact, no randomness in the result), then `strong-numbers`, `draw-planner`, `randomness`, `scoreboard`, `memorisation`, `calibration`, `joint-mixture`, `thousand-draws`, `synthetic-training`, `seed-search`.

### Task 4.1: The pattern, on `coverage`

**Files:**
- Create: `lotto/experiments/coverage.py`
- Create: `experiments/coverage/README.md`, move `experiments/coverage_search_protocol_2026_11_06.json` → `experiments/coverage/protocol.json`, results → `experiments/coverage/results/`
- Modify: `lotto/experiments/__init__.py` (one catalog entry)
- Delete after the golden test passes: `scripts/optimize_ticket_coverage.py`, `scripts/benchmark_coverage.py`, `scripts/research_coverage_search.py`
- Test: `tests/experiments/test_coverage.py`

**Step 1: Failing test**

```python
def test_quick_run_reports_exact_odds(tmp_path):
    summary = coverage.run(full=False, out=tmp_path)
    assert summary["rows"] == 10
    assert 0.38 < summary["p_any_prize"] < 0.39
    assert summary["p_any_prize"] >= summary["p_any_prize_before_refinement"]
    assert (tmp_path / "portfolio.json").is_file()
```

**Step 2: The runner — a recipe, not a framework**

```python
# lotto/experiments/coverage.py
"""How much prize chance can a fixed number of rows buy?

One row wins a prize (three or more main numbers) in 4.18% of fair draws. Ten
rows chosen at random overlap, so they win less than ten times as often. This
experiment chooses rows that overlap as little as possible and counts, over all
2,324,784 possible main draws, how often at least one of them wins.

Write-up: docs/approaches/ticket-generators.md
"""

from __future__ import annotations

import json
from pathlib import Path

from ..tickets.coverage import greedy_coverage_tickets, portfolio_odds, refine_coverage_tickets, search_coverage_tickets


def run(full: bool, out: Path) -> dict:
    # 1. Start from a greedy design built on simulated draws.
    rows = greedy_coverage_tickets(count=10, seed=0)
    before = portfolio_odds(rows)

    # 2. Improve it with swaps that are checked against every possible draw.
    rows, _ = refine_coverage_tickets(rows, max_passes=2)

    # 3. The full run adds basin hopping to escape a local optimum.
    if full:
        rows, _ = search_coverage_tickets(rows, steps=20, seed=0)

    # 4. Report exact odds, never an estimate.
    after = portfolio_odds(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "portfolio.json").write_text(json.dumps({"rows": rows, "odds": after}, indent=2) + "\n")
    return {
        "rows": len(rows),
        "p_any_prize_before_refinement": before["p_any_prize"],
        "p_any_prize": after["p_any_prize"],
        "p_jackpot": after["p_jackpot"],
    }
```

**Step 3: Catalog entry, measured runtime**

Run `time uv run lotto run coverage` and put the measured seconds in `quick_seconds`. Do not guess. For reference, the three library calls above took 0.5 s on a warm Numba cache during this review and moved ten rows from 38.40% to 38.58%; the first run on a new machine also pays the Numba compile.

**Step 4: Golden** — add the full run's `best_match_counts` to `tests/golden/coverage.json` and confirm it equals the saved result in `experiments/coverage/results/`.

**Step 5: Page** — bring `docs/approaches/ticket-generators.md` to the B5 template: add "Run it" with the quick command and its real output; fold the still-useful parts of `experiments/coverage_review_iteration2.md` into "How it works"; move the remainder of that file to the research log (Task 6.3).

**Step 6: Commit** — `feat: coverage is a catalog experiment with a quick and a full run`

### Task 4.2: Repeat for each experiment

The same six steps. Per-experiment notes:

| Experiment | Notes |
|---|---|
| `strong-numbers` | Quick: optimise the saved ten-row portfolio (about 2 s). Golden: `covered_full_outcomes_after == 969420` |
| `draw-planner` | `run` reads a saved result page from `tests/` fixtures when offline; `lotto plan` does the live fetch. Needs a new page (Task 6.2) |
| `randomness` | Quick: the nine tests on the live archive. Full: adds the calibration and power study (Task 4.3) |
| `scoreboard` | Quick: `quick_suite()`, 100 draws. Full: `full_suite()`, 300 draws; `needs=("neural",)` |
| `memorisation` | The DKRR table, extracted from `build_report.py` |
| `calibration` | Move `prediction_experiment` and `uniformity_power` from `scripts/research_prediction_probability.py` into `lotto/forecasting/`. Drop the `SmartTicketGenerator` arm (D3); compare against random portfolios instead |
| `joint-mixture` | `build` and `mixtures` move from the script into `lotto/forecasting/joint_mixture.py`; this removes the script-imports-script chain. `tests/test_research_runners.py` follows the runner |
| `thousand-draws` | `summarize` moves into `lotto/tickets/forecast_portfolios.py`. Golden: 62 and 422 winning draws for the best one- and ten-ticket models |
| `synthetic-training` | Quick: 20,000 simulated draws. Full: the 1M-draw protocol. Replace the source-hash check with the golden test (657 vs 663 ten-ticket winning draws); keep the protocol and input-archive hashes, pointed at `experiments/synthetic-training/inputs/` |
| `seed-search` | Quick: recover a planted seed in the toy generator with the Python engine. The Rust and CUDA engines stay opt-in flags. The four seed scripts become functions in `lotto/seedsearch/` behind `lotto run seed-search` options |

When the last script is gone, delete `scripts/` and remove the `sys.path.insert` lines with it.

### Task 4.3: A new experiment — testing the tests

The calibration run from this review becomes part of `randomness --full`: simulate fair histories, run each analytic test, and report the rejection rate at α = 0.05 with its binomial band. It answers a question a careful reader will ask ("how do you know the p-values mean what they say?") and it is the cheapest experiment in the project (600 histories in 3 s). Include the gap test's result honestly; if it stays off at 5,000 histories, say so on the page and fix the test (it ignores right-censored gaps).

### Task 4.4: Generated index

Add `lotto list --markdown` and write its output to `docs/reference/experiments.md` and `experiments/README.md` (a table: id, question, command, runtime, page). CI adds:

```bash
uv run lotto list --markdown > /tmp/experiments.md && diff /tmp/experiments.md docs/reference/experiments.md
```

so the index cannot drift from the catalog.

---

## Phase 5 — The legacy stack becomes an exhibit

### Task 5.1: Move

```bash
mkdir legacy
git mv train.py predict.py evaluate.py models.py ilotto.py helpers.py metrics.py logger.py \
       smart_generator.py visualizations.py model_viz.py train.ipynb model legacy/
git mv MODEL_DESIGN_PLAN.md docs/reference/research-log/
```

Scripts are run as `uv run --extra neural --extra legacy python legacy/train.py …` from the repository root, so their relative `data/` paths keep working. Update the `model/` path strings in the moved files to `legacy/model/`. `git rm data/lotto_IL_filtered.csv` and ignore it: `helpers.fetch_dataset` regenerates it on every run, and the committed copy is stale (it ends 2025-12-16).

Once `ilotto.py` is no longer at the root, the naming hazard in `CLAUDE.md` is gone; rewrite that section.

### Task 5.2: Label it and pin its bugs

Create `legacy/README.md`:

- What this is: the 2023–2025 first attempt, kept because its mistakes are the project's best teaching material.
- What is wrong with it, each with file and line: the value filter (C1), reversed time (C1), misaligned test windows (C1), test set used for model selection (C2), sorted targets, double attention concatenation (C11).
- Where each is done properly: links into `lotto/core/`.
- How to run it, and that nothing it prints is evidence.

Create `tests/test_legacy_exhibit.py` that documents the bugs on purpose:

```python
"""The legacy split is wrong, and this test proves it stays visibly wrong.

These tests document defects. If one fails, someone repaired the exhibit;
update legacy/README.md and docs/approaches/legacy-models.md to match.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("requests")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "legacy"))


def test_legacy_test_windows_do_not_precede_their_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # importing helpers creates a log file
    import helpers

    row_ids = pd.DataFrame(np.arange(100).repeat(7).reshape(100, 7) + 1)
    _, _, x_test, y_test = helpers.train_test_split(row_ids, test_size=10, w=3)
    assert x_test[0][:, 0].tolist() == [87, 88, 89]
    assert int(y_test[0][0]) == 0
```

### Task 5.3: Remove popularity logic from live surfaces (Decision D3)

- Remove `UnpopularNumbers` from `statistical_suite()` and the `startswith("Unpopular")` line in the report builder. The scoreboard goes from 20 strategies to 19; update the "Eleven counting rules" sentence in `docs/approaches/statistical.md` and regenerate the goldens with the count change in the commit message.
- Remove the "Smart Tickets" and "Hybrid Generator" sections from the README.
- In `docs/approaches/ticket-generators.md`, move the "Prize sharing", "unpopularity score" and "hybrid generator" sections to the legacy page.
- Edit the `FibonacciTicket` docstring and description in `lotto/approaches/golden_ratio.py` to drop the popularity argument; it stays as a fixed-ticket control.

---

## Phase 6 — The site as a learning path

### Task 6.1: Regroup the navigation (no file moves)

```yaml
nav:
  - Home: index.md
  - Start here:
      - The game in five minutes: start/the-game.md
      - Run it yourself: start/run.md
      - How results are measured: evaluation.md
      - Glossary: start/glossary.md
  - What you can control:
      - levers/index.md
      - Coverage at a fixed budget: approaches/ticket-generators.md
      - Strong-number assignment: ticket-chances.md
      - Which draw to play: levers/draw-planner.md
  - Can the draw be predicted?:
      - approaches/index.md
      - Live scoreboard: results.md
      - Statistical strategies: approaches/statistical.md
      - Neural set prediction: approaches/set-prediction.md
      - Date-conditioned generative: approaches/date-conditioned.md
      - The perfect-fit function: approaches/perfect-fit.md
      - Chaos theory: approaches/chaos.md
      - The golden ratio: approaches/golden-ratio.md
      - A mixture of 57 experts: algorithm-research.md
      - The 1,000-draw test: historical-strikes.md
      - Training on a million simulated draws: synthetic-learning.md
  - Side quests:
      - Searching for a seed: side-quests/seed-search.md
      - The first attempt: approaches/legacy-models.md
  - Benchmark report: benchmark/index.html
  - Reference:
      - Experiment catalog: reference/experiments.md
      - Architecture deep dive: reference/architectures.md
      - User guide: reference/guide.md
      - Research log: reference/research-log/index.md
```

Verify: `uv run mkdocs build --strict`.

### Task 6.2: New pages

| Page | Content |
|---|---|
| `start/the-game.md` | Rules: 6 of 37 plus strong 1–7, the eight prize tiers with exact single-row probabilities (generated from `single_ticket_prize_odds()`), 3 NIS per row, draw days, what a bonus draw is |
| `start/run.md` | Install (`uv sync`), `uv run lotto list`, one quick experiment with its real output, the report, the tests. Ten minutes from clone to a result |
| `start/glossary.md` | Walk-forward, leakage, null distribution, hypergeometric, p-value, Holm–Bonferroni, log loss, proper scoring rule, calibration, coverage, DKRR, conditional Bernoulli. Two sentences each, with a link to where it is used |
| `levers/index.md` | The three levers with their headline numbers, B3 items 1–3 |
| `levers/draw-planner.md` | New. The 40M rule (24 of 24 draws after an unclaimed 40M were bonus draws; 7 of 7 first-40M draws were not), the two forms and their exact odds, `lotto plan`. B5 template |
| `side-quests/seed-search.md` | New. Consolidates the four `experiments/seed_search_*.md` files and `parallel_search_plan.md` into one page: the artificial generator, why it says nothing about the real machine, search rates, how to recover a planted seed |

### Task 6.3: Research log

Move the dated reports to `docs/reference/research-log/` with an index that lists them by date: `progress.md`, `MODEL_DESIGN_PLAN.md`, and the six `experiments/*.md` files. They are history: copy them verbatim, add a one-line banner ("Dated report, 2026-11-06. Current numbers: Results"). `progress.md` keeps its URL (D5); the index links to it.

### Task 6.4: Bring every page to the template and the voice

One commit per page. Checklist for each:

- [ ] Opens with the "In one minute" box: question, answer with a number, why it matters.
- [ ] Has a "Run it" section whose command was executed and whose output is pasted from that run.
- [ ] States each limit once, in the "Limits" box. Delete repeats of "this does not establish…" from the body.
- [ ] Every number carries its denominator or unit and matches the generated results or a frozen artifact.
- [ ] Draw counts say "1,661 draws through 2026-11-03" where a dated run is described, and are generated where the page describes the live benchmark.
- [ ] Links to the implementing function and its test.

The four November pages (`algorithm-research.md`, `ticket-chances.md`, `historical-strikes.md`, `synthetic-learning.md`) need the most work: they lead with protocol detail and have no diagram. The six July approach pages need only "Run it" and "Go deeper".

### Task 6.5: Fold the two root guides into the site

`GUIDE.md` (776 lines) and `ARCHITECTURES.md` (647 lines) describe the legacy stack. Move both into `legacy/`, repoint the snippet includes in `docs/reference/*.md`, and retitle the nav entries "Legacy user guide" and "Legacy architecture deep dive". Update the `paths:` filter in `report.yml`.

---

## Phase 7 — Readability pass on the November modules

No behaviour change; golden tests guard it. One module per commit.

### Task 7.1: Named validation

Create `lotto/core/validation.py` and replace the boolean blocks.

```python
"""Small input checks with names, so callers read as sentences.

Every check returns the validated value in its canonical form or raises
ValueError with a message that says what was expected.
"""

from __future__ import annotations

import numpy as np

from .data import N_DRAWN, N_NUMBERS, N_STRONG


def require_int(value, name: str, *, low: int | None = None, high: int | None = None) -> int:
    """An integer (not a bool) inside [low, high]."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must be in {low}..{high}, got {value}")
    return int(value)


def require_main_numbers(row, *, n_numbers: int = N_NUMBERS, n_drawn: int = N_DRAWN) -> tuple[int, ...]:
    """Six distinct numbers from the pool, returned sorted."""
    numbers = tuple(require_int(x, "ball", low=1, high=n_numbers) for x in row)
    if len(numbers) != n_drawn or len(set(numbers)) != n_drawn:
        raise ValueError(f"A row needs {n_drawn} distinct numbers in 1..{n_numbers}, got {row!r}")
    return tuple(sorted(numbers))


def require_row(row) -> list[int]:
    """Six main numbers followed by one strong number."""
    row = tuple(row)
    if len(row) != N_DRAWN + 1:
        raise ValueError(f"A row is six main numbers and one strong number, got {row!r}")
    return [*require_main_numbers(row[:N_DRAWN]), require_int(row[-1], "strong number", low=1, high=N_STRONG)]
```

Write tests first (`tests/core/test_validation.py`): booleans rejected, NumPy integers accepted, out-of-range and duplicate rows rejected with the expected message. Then replace the three `_integer` helpers and `_main_tickets` / `_full_tickets` in `lotto/tickets/coverage.py`, and make `ticket_masks` public. Existing tests assert on `ValueError`, not message text; confirm with `grep -rn "match=" tests/tickets`.

### Task 7.2: No private imports across modules

For each of `_full_tickets`, `_ticket_masks`, `_prefix_hash`, `_validate_draws`: either make it public with a docstring, or replace it with a `validation` function. Then prove it:

```bash
grep -rnE "from \.+[a-z_.]* import .*\b_[a-z]" lotto/ ; echo "exit $?"
```

Expected: no output.

### Task 7.3: Split the long functions

| Function | Split into |
|---|---|
| `seedsearch/campaign.py::continue_search` (~180 lines) | `_load_ledger`, `_import_checkpoints`, `_next_range`, `_totals`, and a short loop that calls them |
| `seedsearch/rust_engine.py::RustSearch.__call__` | `_load_or_start`, `_run_batch`, `_write_progress` |
| `tickets/coverage.py::search_coverage_tickets` | `_perturb`, `_metropolis_accept`, the loop |
| `report/html.py::build` | one function per report section, returning an HTML string |

The seed-search modules have 49 tests around them; run `uv run pytest tests/seedsearch -q` after each split.

### Task 7.4: Types and docstrings

Add return annotations and one-line docstrings to every public function in `lotto/tickets/`, `lotto/forecasting/`, `lotto/seedsearch/` and `lotto/experiments/`. Then turn the rule on so it stays true:

```toml
[tool.ruff.lint]
extend-select = ["I", "B", "UP", "ANN201", "D103"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["ANN", "D"]
"legacy/**" = ["ANN", "D", "B", "UP", "I"]
```

Fix what it reports in `lotto/`; do not blanket-ignore.

---

## Phase 8 — Front door

### Task 8.1: README, one screen

Target: under 120 lines. Sections in order:

1. Title and one sentence: a reproducible study of what can and cannot be done with the Israeli Lotto.
2. **Three results**, each one line with its number and a link (B3 items 1, 3, 4).
3. **Run it** — three commands:

   ```bash
   uv sync
   uv run lotto list
   uv run lotto run coverage
   ```

4. **Learn it** — links to Start here, the levers, the prediction benchmark, the side quests.
5. **Map** — the B2 tree trimmed to top-level folders, one phrase each.
6. **How it is checked** — test count, CI, the scheduled rebuild, and the fact that new draws arrive only through a local refresh.
7. Disclaimer, one paragraph.

Everything else in today's README moves to the site: bulletins to the research log, legacy commands to `legacy/README.md`, parameter tables to the reference.

### Task 8.2: CLAUDE.md

Rewrite the sections the refactor invalidated: commands (`lotto` CLI), "Two parallel stacks" (now "The package and the exhibit"), the naming hazard (gone), data file ownership, "Adding a predictor" (new paths), plus two new sections: "Adding an experiment" (catalog entry, runner, page, golden) and "Three homes". Keep every invariant; they are the most valuable part of the file.

### Task 8.3: The talk

Create `docs/reference/presentation.md`: the seven steps of B3 as a slide outline, each with the one figure to show and the command that produces it. This is the script for presenting the project.

---

## Phase 9 — Optional: slim the repository (Decision D6)

Not required for anything above. Do it last, in this order of increasing risk:

1. Delete model files nothing references: `model/set_prediction_functional.keras`, `model/original_functional.keras`, the `*_history.csv` files. Confirm with `grep -rn` first.
2. Stop tracking two copies of research data: keep the raw JSON under `experiments/<id>/results/`, and build the `docs/data/*.zip` bundles in the docs workflow instead of committing them.
3. Move arrays over 1 MB (`*.npz`) and the Keras models to a GitHub Release, with a checksum manifest in the tree. This needs you to create the release.
4. Rewriting history to shrink `.git` is destructive and breaks every clone and open branch. Not recommended.

---

# Part E — Finish line

## E1. Definition of done

- [ ] `git clone`, `uv sync`, `uv run lotto list`, `uv run lotto run coverage` works on a machine without TensorFlow, Rust or a GPU, in under two minutes after install.
- [ ] Every experiment in the catalog has a quick run under a minute, a full run that matches its golden, and a page on the B5 template.
- [ ] `grep -rn "sys.path.insert" lotto/ tests/` finds only the legacy exhibit test.
- [ ] No module in `lotto/` imports another module's private name.
- [ ] `uv run pytest -q`, `uv run ruff check .`, `uv run mkdocs build --strict` pass locally and in CI on Linux.
- [ ] The golden tests pass on both macOS and Linux.
- [ ] README is under 120 lines; every number in it comes from a generated file or a golden.
- [ ] No published URL from before the refactor returns 404.
- [ ] `CLAUDE.md` describes the repository as it is.

## E2. Things considered and left out

- **A `src/` layout.** The flat layout already installs in editable mode, and `lotto/` at the root is easier for a reader to find.
- **Compatibility shims for `bench`.** Nothing outside this repository imports it.
- **A YAML experiment registry and per-run manifests with schema versions.** A Python tuple that a test checks is enough; the frozen runs already carry protocols and hashes.
- **A separate `lessons/` tree.** The site is the learning path, and every experiment has a quick run. A fourth home for content would drift from the other three.
- **Unifying `Predictor.scores` so every strategy takes a target date.** It would remove two branches at the cost of twenty signature edits and a changed public contract.
- **Repairing the legacy models onto the benchmark data path.** The benchmark's neural predictors are that repair.

## E3. Relation to `planning/EDUCATIONAL_REFACTOR_PLAN.md`

That file appeared in the working tree at 01:35 on 2026-11-14 while this review was running. This session did not write it and has not changed it.

I read it and checked its findings against the code. Four of them were things I had not yet pinned down and are now in this plan with my own reproduction: the misaligned legacy test windows (C1), the unvalidated refresh (C5), the window mismatch between backtest and null (C6), and the fragility of source-hash freezes (C7). Its draw-validation point is C8.

Where this plan differs:

| Topic | That plan | This plan |
|---|---|---|
| Level | Findings, a target layout and phase descriptions | The same, plus tasks with files, code, commands and expected output |
| Findings it does not have | — | The unreachable surrogate test (C3), platform-dependent ties (C4), measured null calibration, the double backtest (C9), popularity logic against your stated preference (S8), missing pages (S9) |
| Package | `src/ilotto_lab/`, with compatibility wrappers | `lotto/`, flat, no wrappers |
| Teaching material | A separate `lessons/` tree | The docs site plus a quick run per experiment |
| Experiment records | `registry.yaml`, run manifests, schema validation | A Python catalog checked by one test; golden tests for numbers |
| Legacy | Repair the training path onto validated data | Keep as a labelled exhibit with tests that document the bugs |
| Story order | Starts from the false prediction claim | Starts from what a buyer can control, then prediction |

The two agree on the diagnosis, on fixing correctness before moving files, on migrating one experiment end to end before the rest, and on not rewriting history.
