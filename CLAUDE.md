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
# Setup. The docs group is separate so library users and the test job don't pull
# in a site generator. CUDA is an opt-in extra — see "GPU" below.
uv sync --group docs

# Tests (47, ~10s). testpaths=tests and pythonpath=["."] come from pyproject.
uv run pytest -q
uv run pytest tests/test_pipeline.py::test_walk_forward_never_sees_the_future  # single test
uv run pytest -k "log_loss"                                                    # by pattern

uv run ruff check .          # must pass over the whole repo, both stacks

# Benchmark + report. Writes docs/benchmark/index.html and docs/results.md.
uv run python scripts/build_report.py --quick            # skip neural models (fast iteration)
uv run python scripts/build_report.py --refresh          # fresh draws, full run
uv run python scripts/build_report.py --n-test 300 --null-trials 5000

# Docs site
uv run mkdocs serve                # live preview
uv run mkdocs build --strict       # what CI runs; broken links fail the build

# Legacy stack
uv run python train.py --model multi_output --epochs 50   # original|multi_output|transformer|set_prediction
uv run python predict.py                                  # writes PREDICTION.md
uv run python smart_generator.py --model multi_output --count 5
```

**There is a `.dockerignore` and it matters.** Docker does not read `.gitignore`, and the Dockerfile creates a Linux virtualenv at `/app/.venv` before copying the source over it — without the exclusion a host `.venv` (1.9 GB of arm64 wheels on this Mac) would overwrite it and `uv run` would fail inside the image.

**Dependencies live in `pyproject.toml` + `uv.lock` only.** The Dockerfile installs from that pair too; a parallel `requirements.txt` used to exist and had silently drifted out of sync (it was missing `requests`, which both `helpers.py` and `bench/data.py` import).

**GPU.** The default install is CPU TensorFlow, on every platform. CUDA is the `gpu` extra:

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
`tensorflow[and-cuda]` used to fail on Apple Silicon. The base dependency therefore carries
two markered entries: `>=2.19` everywhere, `>=2.16.2,<2.17` on Intel macOS. That path is
resolvable but **untested** — nobody working on this has an Intel Mac — and it pins Keras
3.0 rather than 3.x-latest, so treat a bug report from that platform as plausible rather
than surprising.

## Operational fact: CI cannot reach pais.co.il

The scheduled publish runs `--refresh`, but the download times out from GitHub's runners
(geo/cloud-IP blocking) and falls back to the committed archive — so **new draws only land
when someone runs `uv run python scripts/build_report.py --refresh` locally and commits
`input/`**. The workflow prints archive freshness to the job summary and warns past 7 days;
the generated pages carry the same warning. Do not "fix" the fallback by making the
download fatal: the site must still rebuild when the source is down.

## Two parallel stacks

This is the single most important structural fact. They share the `input/` data and nothing
else, and both are live.

| | `bench/` + `scripts/` + `tests/` | Repo root (`models.py`, `train.py`, `predict.py`, …) |
|---|---|---|
| Purpose | measure whether anything beats chance | train and sample from Keras models |
| Data path | `bench/data.py` | `helpers.py` |
| Metrics | `bench/metrics.py` | `metrics.py` (different file, similar name) |
| Covered by tests | yes | no |

`bench.metrics` and root `metrics.py` do not collide only because `bench/` uses relative
imports internally and legacy code imports the top-level module.

### Naming hazard

`ilotto.py` is a top-level module holding the original Seq2Seq model. **Never create a package
directory named `ilotto/`** — it shadows the module and breaks `train.py`, `predict.py` and
`evaluate.py` with `ImportError: cannot import name 'ILotto'`. This happened once; the
benchmark package is called `bench/` for exactly this reason.

### Data file ownership

Two writers with incompatible semantics — do not point them at the same file:

- `helpers.ILottoCSV` regenerates `input/lotto_IL_filtered.csv` using the legacy value filter.
- `bench.data.load` writes `input/lotto_IL_modern.csv` (2012+, one game format).

Both read `input/Orig_IL_lotto.csv`, the raw archive, which is committed so builds survive
pais.co.il being unreachable.

## Invariants that make the results valid

Each corresponds to a bug that made the old pipeline report ~30% top-10 "accuracy" against a
~27% random baseline. All are enforced by tests in `tests/test_pipeline.py`.

1. **Filter draws by date, never by ball value.** The archive spans several games (6/49-ish to
   2004, 6/34 through 2008, 6/37 from 2009, strong 1–7 only from 2012). Dropping rows whose
   balls exceed 37 is a *selection on the outcome*: it deletes exactly the draws containing
   high numbers and biases the surviving frequency table downward.
2. **Chronological order.** The archive ships newest-first; consuming it as-is trains models to
   predict the past from the future.
3. **Order-invariant targets.** Balls are published ascending, so `Ball_1` is the minimum of six
   draws and `Ball_6` the maximum. Per-position top-k accuracy mostly measures order statistics.
   `bench/` predicts the drawn *set* (37-way multi-hot, BCE).
4. **Baselines are stated, not implied.** `BASELINE_MATCHES = 6*6/37 = 0.9730` (hypergeometric
   mean), `BASELINE_LOGLOSS = ln 37 = 3.6109`.
5. **Holm–Bonferroni over every family**, both the strategy table and the randomness tests. The
   report's headline verdict reads both.

## The date-conditioned model is a special case

`bench/generative.py` is conditioned on the draw's *date* rather than on preceding draws,
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

Subclass `bench.predictors.Predictor` and implement:

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

Generated — never hand-edit, `scripts/build_report.py` overwrites them:

- `docs/benchmark/index.html` and `docs/benchmark/results.json` (standalone report, copied
  through verbatim by MkDocs)
- `docs/results.md` (the scoreboard page)

`docs/reference/*.md` pull `ARCHITECTURES.md` and `GUIDE.md` from the repo root via
`pymdownx.snippets`, so those files stay where the README links to them.

### Mermaid constraints

- **Diagram colours cannot come from a stylesheet.** Mermaid inlines its own styles at render
  time, so every diagram carries a `classDef` block. `docs/stylesheets/extra.css` only styles
  the legend chips, and the two must be kept in step by hand.
- **`navigation.instant` is deliberately off.** It re-injects pages via XHR and leaves the
  Mermaid containers empty, so diagrams vanish on the second page a visitor opens.
- Node roles are consistent site-wide: gray outline = tensors, blue = learned weights, dashed =
  fixed operations, green = outputs, red = a step that leaks sort order.

## CI

- `ci.yml` — ruff, pytest, a quick report build, and `mkdocs build --strict`, on push and PR.
- `report.yml` — full rebuild on a cron, commits the refreshed archive, deploys `site/` to Pages.

The cron is `0 6 * * 0,3,5`. Draw days since 2012 are Tuesday (707) and Saturday (703) with an
intermittent Thursday (161), and draws are in the evening Asia/Jerusalem — so the rebuild runs
the *following* morning UTC. Changing this schedule means re-deriving it from the archive.

## Chart rendering

`bench/charts.py` emits hand-rolled inline SVG (no plotting library) so the standalone report
stays a single self-contained file. Colours come in as CSS custom properties so the page's
light/dark theming applies without regenerating anything.
