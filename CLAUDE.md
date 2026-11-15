# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A benchmark of lottery-prediction methods against the Israeli Lotto (6/37 + strong 1–7). The
conclusion is that the draw is unpredictable; the deliverable is the *measurement* — exact
nulls, leak-free backtesting, and multiple-testing correction — published to GitHub Pages.

Treat "no strategy beat chance" as the expected result, not a failure to fix. If a change
makes a strategy look good, suspect the change.

## Commands

```bash
# Setup. The light install has no TensorFlow; `neural` adds it, `docs` adds the site generator.
uv sync --extra neural --group docs

uv run pytest -q             # neural tests skip without the extra
uv run pytest -k "log_loss"  # by pattern
uv run ruff check .          # must pass over the whole repo
node --check lotto/ui/static/app.js
node --test tests/test_ui_frontend.cjs  # built-in Node test runner; no npm dependencies

# Everything runs through one command.
uv run lotto list                                   # the experiment catalog
uv run lotto report --quick                         # checks for new draws, report into results/report/, no neural models
uv run lotto report --out-dir docs                  # checks for new draws, full run (~2 min), rebuilds the site pages
uv run lotto report --offline --quick               # same, without contacting pais.co.il
uv run lotto run coverage --count 10 --seed 0 --out results/coverage.json
uv run lotto plan                                   # which form to play next
uv run lotto predict --quick                        # every model's best ticket and 10-row form for the next draw
uv run lotto ui                                     # the dashboard at http://localhost:6037

uv run mkdocs serve                # live preview
uv run mkdocs build --strict       # what CI runs; broken links fail the build

# Legacy exhibit
uv sync --extra neural --extra legacy
uv run python legacy/train.py --model multi_output --epochs 5
```

**There is a `.dockerignore` and it matters.** Docker does not read `.gitignore`, and the Dockerfile creates a Linux virtualenv at `/app/.venv` before copying the source over it — without the exclusion a host `.venv` (1.9 GB of arm64 wheels on this Mac) would overwrite it and `uv run` would fail inside the image.

**Dependencies live in `pyproject.toml` + `uv.lock` only.** The Dockerfile installs from that pair too; a parallel `requirements.txt` used to exist and had silently drifted out of sync (it was missing `requests`, which `lotto/core/data.py` imports).

**TensorFlow is the `neural` extra**, so the light install takes seconds. CUDA is the `gpu` extra:

```bash
uv sync --extra gpu      # x86_64 Linux with an NVIDIA GPU
```

It is opt-in rather than default because the CI runner has no GPU — the ~3 GB of `nvidia-*`
wheels would be downloaded and never used — and because the models here are small enough
(GRU 96/64, 4-head attention over a 20x74 input, ~1600 samples) that kernel-launch overhead
makes a GPU no faster than CPU. The extra carries a `sys_platform == 'linux' and
platform_machine == 'x86_64'` marker, so requesting it elsewhere is a no-op rather than a
resolution failure. Do not move it back into the base dependencies: that is what used to
make `uv sync` and `uv run` fail outright on macOS.

**TensorFlow version is platform-split.** Upstream stopped publishing Intel-macOS wheels
after 2.16.2 — 2.17, 2.18 and 2.19 have no `macosx x86_64` artifact at all — so an
unconditional `>=2.19` would fail to resolve on an Intel Mac in exactly the way
`tensorflow[and-cuda]` used to fail on Apple Silicon. The `neural` extra therefore carries
two markered entries: `>=2.19` everywhere, `>=2.16.2,<2.17` on Intel macOS. That path is
resolvable but **untested** — nobody working on this has an Intel Mac — and it pins Keras
3.0 rather than 3.x-latest, so treat a bug report from that platform as plausible rather
than surprising.

## Operational fact: CI cannot reach pais.co.il

Every `lotto report` checks pais.co.il for new draws first (`--offline` skips it), but from GitHub's runners the download times out
(geo/cloud-IP blocking) and falls back to the committed archive — so **new draws only land
when someone runs `uv run lotto report --out-dir docs` locally and
commits `input/` and `docs/`**. The workflow prints archive freshness to the job summary and warns past 7 days;
the generated pages carry the same warning. Do not "fix" the fallback by making the
download fatal: the site must still rebuild when the source is down.

## Layout: three homes

| Kind of thing | Lives in | Rule |
|---|---|---|
| Code that computes | `lotto/` (`core`, `approaches`, `forecasting`, `tickets`, `seedsearch`, `report`) | importable and tested; no side effects on import |
| An experiment | `lotto/experiments/<name>.py` + frozen evidence in `experiments/` | one catalog entry in `lotto/experiments/__init__.py`; exposes `main()`; writes to `results/` (untracked), never to `experiments/` — except the seed-search ledger, below |
| Words | `docs/` | dated reports go under `docs/reference/research-log/` and are not rewritten |

`legacy/` is the 2023–2025 Keras stack, kept as an exhibit. It is untested on purpose except
for `tests/test_legacy_exhibit.py`, which pins its split defect; do not repair it without
updating `legacy/README.md` and `docs/approaches/legacy-models.md`.

### The dashboard (`lotto/ui/`)

Presentation only, standard library only. Rules:

- No statistics in `lotto/ui`: `service.py` calls `lotto.core`, `lotto.approaches`, `lotto.tickets`.
- `registry.py` is the only list of models and editable settings. Its presets must equal
  `full_suite()`; `tests/test_ui_service.py` compares them score for score.
- A fresh predictor per job; reads use `clean_csv=None`; writes go only under `results/ui/`.
- The server binds to 127.0.0.1, checks the Host header, and requires the page's Origin and
  token for anything that changes state. Do not add a `--host` option or relax these.
- Log entries are append-only in the UI. Lab verdicts are labelled exploratory, never "signal".
- The Lab currently rejects sampled date-conditioned tickets: the core harness evaluates
  top-six tickets, and must not silently measure a different selection rule.
  Legacy notebook entries with that mismatch are retained as invalid, outside the correction.
- Acquire `dashboard_lock()` before constructing the CLI's Service: startup recovers the
  notebook. One dashboard owns `results/ui/`, even across different ports. Keep its lock
  until the worker finishes shutdown; never delete the lock file while a process owns it.
- Completed predictions are immutable and cached by archive/config/date/design plus source
  and runtime fingerprints. Publish through `Job.finish()` so cancellation cannot save a
  result that the UI calls cancelled. Cache hits preserve the original generation timestamp.
- A published scoreboard row is a dated preset reference, not a verified exact-settings
  match: the report currently lacks a full configuration manifest.

### The seed-search ledger is the one exception

`experiments/seed_search_runs/` (git-ignored) holds the live seed-search ledger: every
range ever searched, so no seed is tested twice across runs and machines. The seed-search
commands append to it by default. Never move or delete it; pass `--ledger-dir` to search
somewhere else.

### Adding an experiment

Add a module with `main()` to `lotto/experiments/`, one `Experiment` entry to the catalog, a
docs page, then regenerate the index: `uv run lotto list --markdown` into
`docs/reference/experiments.md` (keep its four header lines). `tests/test_cli.py` fails if the
page is missing or the index is stale.

### Tests read a pinned archive

`tests/golden/archive_2026-11-03.csv` via `tests.SNAPSHOT`, never the live `input/` file.
`tests/test_golden.py` pins scoreboard and coverage numbers; after an intended change run
`uv run python -m tests.golden.regenerate` and report the before/after.

### No popularity or birthday logic

Removed deliberately. Do not add ticket scoring based on what other players pick.

### Data files

`lotto.core.data.load` reads `input/Orig_IL_lotto.csv` (committed so builds survive pais.co.il
being unreachable) and writes `input/lotto_IL_modern.csv`. The check for new draws (`data.download`, run by every
`lotto report` unless `--offline`) replaces the archive only
if the download parses and still contains every draw already on record.

## Invariants that make the results valid

Each corresponds to a bug that made the old pipeline report ~30% top-10 "accuracy" against a
~27% random baseline. All are enforced by tests in `tests/test_pipeline.py`.
Ties between equal scores go to the lower number (`metrics.top_k`), so tickets are the same on every machine.

1. **Filter draws by date, never by ball value.** The archive spans several games (6/49-ish to
   2004, 6/34 through 2008, 6/37 from 2009, strong 1–7 only from 2012). Dropping rows whose
   balls exceed 37 is a *selection on the outcome*: it deletes exactly the draws containing
   high numbers and biases the surviving frequency table downward.
2. **Chronological order.** The archive ships newest-first; consuming it as-is trains models to
   predict the past from the future.
3. **Order-invariant targets.** Balls are published ascending, so `Ball_1` is the minimum of six
   draws and `Ball_6` the maximum. Per-position top-k accuracy mostly measures order statistics.
   `lotto/approaches/neural.py` predicts the drawn *set* (37-way multi-hot, BCE).
4. **Baselines are stated, not implied.** `BASELINE_MATCHES = 6*6/37 = 0.9730` (hypergeometric
   mean), `BASELINE_LOGLOSS = ln 37 = 3.6109`.
5. **Holm–Bonferroni over every family**, both the strategy table and the randomness tests. The
   report's headline verdict reads both.

## The date-conditioned model is a special case

`lotto/approaches/date_conditioned.py` is conditioned on the draw's *date* rather than on preceding draws,
which makes it the one model here that can cheat spectacularly: there is exactly one draw
per calendar date, so the date is a unique key and enough capacity turns the model into a
lookup table that "predicts the past" perfectly.

Two invariants keep it honest, both enforced by tests:

- **The head must stay linear in the date basis.** The basis includes a time trend, so it
  is injective over all 1,629 dates — a unique key in a smooth costume. Only linearity
  bounds capacity: fitting one date necessarily moves all the others. Adding a hidden layer
  turns this into a memoriser and breaks `test_head_is_linear`.
- **DKRR is never reported alone.** Date-Keyed Reconstruction Rate is in-sample matched
  numbers; it measures storage, not skill. `MemorisingDateLookup` ships as a deliberate
  control that scores a perfect 6.000/6 in-sample and chance out of sample.

`Predictor.wants_target_date` makes the harness pass `draws.dates[i]`. That is not
lookahead — the next draw's date is public in advance — but `scores` raises `LeakageError`
if the history ever reaches the target date.

## Adding a predictor

Subclass `lotto.approaches.statistical.Predictor` and implement:

```python
def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:  # (37,), (7,)
```

`history` contains only draws strictly before the one being predicted — that is what makes the
backtest leak-free. Then register it in `statistical_suite()`.

Set `emits_probabilities = True` **only** if the returned values are proportional to a genuine
per-number inclusion probability. Log loss is a proper scoring rule and is withheld otherwise:
normalising ranks or gaps into a distribution is an arbitrary transform, and for all-negative
scores (e.g. the cold-frequency predictor) it collapses to the uniform prior and would report
exactly the baseline no matter how the numbers were ranked.

Set `refit_every` for models with a `fit` step; the harness refits on history only.

## Docs site

MkDocs Material. `docs/` is the *source* tree, `site/` is the build output and the Pages
artifact.

Generated — never hand-edit, `lotto report --out-dir docs` overwrites them (without
`--out-dir` the command writes to `results/report/` instead):

- `docs/benchmark/index.html` and `docs/benchmark/results.json` (standalone report, copied
  through verbatim by MkDocs)
- `docs/results.md` (the scoreboard page)

`docs/reference/architectures.md` pulls `legacy/ARCHITECTURES.md` in via `pymdownx.snippets`.

### Mermaid constraints

- **Diagram colours cannot come from a stylesheet.** Mermaid inlines its own styles at render
  time, so every diagram carries a `classDef` block. `docs/stylesheets/extra.css` only styles
  the legend chips, and the two must be kept in step by hand.
- **`navigation.instant` is deliberately off.** It re-injects pages via XHR and leaves the
  Mermaid containers empty, so diagrams vanish on the second page a visitor opens.
- Node roles are consistent site-wide: gray outline = tensors, blue = learned weights, dashed =
  fixed operations, green = outputs, red = a step that leaks sort order.

## CI

- `ci.yml` — ruff, Python and JavaScript tests, Rust engine checks, a quick report build,
  and `mkdocs build --strict`, on pushes to main and PRs.
- `report.yml` — full rebuild on a cron, commits the refreshed archive, deploys `site/` to Pages.

The cron is `0 6 * * 0,3,5`. Draw days since 2012 are Tuesday (707) and Saturday (703) with an
intermittent Thursday (161), and draws are in the evening Asia/Jerusalem — so the rebuild runs
the *following* morning UTC. Changing this schedule means re-deriving it from the archive.

## Chart rendering

`lotto/report/charts.py` emits hand-rolled inline SVG (no plotting library) so the standalone report
stays a single self-contained file. Colours come in as CSS custom properties so the page's
light/dark theming applies without regenerating anything.
