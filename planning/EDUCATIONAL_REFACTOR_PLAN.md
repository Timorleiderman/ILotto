# ILotto: educational review and refactor plan

Review date: 2026-11-14. Status: proposal, not an implemented migration.

## 1. Assessment

This project has a strong educational thesis: an impressive-looking prediction can disappear when we fix the data, choose the right baseline, and evaluate honestly. Its most valuable result is the measurement process, including negative results. Present it as **a reproducible laboratory for randomness, forecasting, and optimization**.

The repository already contains substantial engineering: chronological backtesting, exact finite-sample match nulls, multiplicity corrections, synthetic controls, frozen protocols, saved per-draw evidence, exact portfolio enumeration, and Python/Rust/CUDA seed-search work. Preserve those investments. The next major improvement should be coherence rather than another model.

Today, it reads like a research workspace accumulated over several stages. The README emphasizes training and generating tickets, while the strongest scientific work lives in `bench/`. Legacy code remains executable with known data problems. Research outputs, protocols, historical notes, downloadable bundles, and runnable scripts require readers to reconstruct the relationships themselves.

A successful presentation should let someone answer four questions immediately:

1. What question does this experiment ask?
2. Which assumptions make its answer valid?
3. How do I reproduce a small version locally?
4. What does the result establish, and what does it leave unknown?

## 2. Review scope and limits

Directly inspected the README, contributor guidance, dependency configuration, docs navigation, CI, data loader, backtest and scoring implementation, legacy data/training path, historical and synthetic research runners, research-runner tests, portfolio reference implementation, and seed-generator interface. Surveyed Python module sizes and test organization.

This is a targeted architectural and correctness review, not a line-by-line audit of every model, notebook, native kernel, or saved artifact. Native-engine equivalence, remote execution, GPU execution, saved model provenance, and complete historical reruns need dedicated validation. Do not interpret this plan as certification of all published results.

## 3. Findings, ordered by importance

### F1 — High: the recommended legacy workflow still has invalid data semantics

Evidence: `helpers.py:ILottoCSV.filter_csv`, `helpers.py:train_test_split`, and `train.py:main`.

The legacy loader filters by observed ball values and preserves archive order. The split takes the first rows as test targets, but builds test windows from the end of the complete dataset. Thus test features and targets are drawn from different parts of the archive. Training windows also follow the archive's newest-first order instead of chronological forecasting order. The README describes these issues as corrected, but the correction belongs to the separate benchmark pipeline.

Action: immediately mark these commands and historical checkpoints as legacy demonstrations. Then move supported training to the validated chronological dataset and an explicit window builder. Retain the broken behavior only as a named teaching fixture with a small synthetic example showing why it fails.

Acceptance: a row-ID fixture proves every input window precedes and belongs to its target; train, validation, and test targets occupy disjoint chronological blocks; the README never implies legacy checkpoint scores are valid forecast evidence.

### F2 — High: legacy validation is named and used as test data

Evidence: `train.py:train_model` passes `X_test, y_test` as `validation_data`; callbacks monitor `val_loss` for checkpoint selection, early stopping, and learning-rate changes.

This block participates in model selection. It cannot also be described as an untouched test set. This issue is separate from the window alignment defect.

Action: define training, validation, and final evaluation roles explicitly. Select checkpoints on validation only. Report final evaluation once, or use the established walk-forward protocol with an explicit refit policy.

Acceptance: tests verify that final evaluation targets never reach fitting or checkpoint selection. CLI and report labels use the correct split names.

### F3 — High: refresh can destroy a valid local archive before parsing a replacement

Evidence: `bench/data.py:download` and `helpers.py:fetch_dataset` accept a successful response longer than 10,000 bytes and write it directly over the archive.

A large HTML error page or malformed CSV can pass this check. A subsequent parse failure then occurs after the last known good data has been replaced. The documented fallback only protects against request failures and short responses.

Action: download to a temporary file; parse and validate its schema, dates, integer values, uniqueness, and modern-era constraints; then atomically replace the archive. Preserve the old archive on any failure and report refresh status separately from analysis status.

Acceptance: tests supply a large HTML response, malformed CSV, duplicate rows, invalid values, and interrupted writes; each preserves the existing valid archive. Valid downloads replace it successfully.

### F4 — Medium: draw validity is assumed beyond the range checks

Evidence: `bench/data.py:load` casts numeric columns to integers before checking their range, does not reject repeated balls within a draw, and does not check duplicate draw identities. `Draws` itself accepts arbitrary arrays.

Fractional numbers can be truncated; duplicate balls violate the six-distinct-number assumptions used by the metrics. Duplicate archive rows can also create invalid forecasting histories. This is an ingestion robustness defect, not evidence that the committed archive contains these errors.

Action: validate the source values before casting. Retain draw IDs for identity checks. Define whether duplicate dates are permitted by the domain; fail with a useful diagnostic instead of silently deduplicating. Validate once at public construction/ingestion boundaries rather than rescanning every history slice.

Acceptance: malformed-input tests cover fractional, missing, repeated, out-of-range, unordered, and duplicate-identity cases.

### F5 — Medium: benchmark and null can use different sample lengths

Evidence: `bench/backtest.py:walk_forward` limits the evaluation start using `min_history=200`; `scripts/build_report.py` supplies the requested `n_test` independently to `random_ticket_null`.

For requests larger than the available evaluation history but no larger than the archive, forecasts use fewer observations than the Monte Carlo comparison. If the request exceeds archive length, the null rejects it after the benchmark has already run.

Action: resolve and validate one evaluation interval before starting any models. Supply its actual length and dates to every scorer, null, and report.

Acceptance: a small fixture and oversized `n_test` either fail before computation or consistently use the same explicitly reported shortened interval.

### F6 — Medium: published and contributor-facing claims have drifted

Evidence: README says the legacy data handling was corrected and that smart tickets maximize expected payout conditional on winning; elsewhere it correctly describes payout gains as unverified. `CLAUDE.md` describes 47 tests and whole-repository linting, while CI only lints `bench scripts tests`. README says automatic rebuilding happens after every draw while contributor guidance documents stale upstream data in CI.

Action: replace claims with scoped statements grounded in artifacts. Distinguish a scheduled rebuild from successful data refresh. Describe popularity scoring as an unvalidated heuristic unless supported by player-choice and payout data. Keep historical findings tied to a dated dataset and protocol.

Acceptance: one claim-to-evidence table maps every headline number to its artifact, metric, evaluation interval, and limitation. Instructions match actual CI commands.

### F7 — Medium: research reproducibility is strong but tightly coupled to historical filenames

Evidence: historical and synthetic runners embed dated protocol/output paths and import helpers from other executable scripts. Frozen synthetic evaluation hashes source files and requires exactly 1,661 draws.

The freezes are valuable safeguards. Moving files or editing shared code, however, can invalidate reproduction even if behavior is unchanged. A general-purpose rerun currently risks conflicting with historical outputs.

Action: preserve the original snapshot and hashes. Extract reusable experiment logic into library modules, leaving thin command wrappers. Create new run directories with versioned schemas, configuration, input hashes, source revision, environment, metrics, and completion manifest. Give migrations their own provenance; never rewrite old hashes to make new code appear original.

Acceptance: an archived run remains retrievable, a new run cannot overwrite it by default, and migrated implementations reproduce agreed numerical outputs under a documented tolerance or exact comparison.

### F8 — Medium: installation and import structure obscure the lightweight path

Evidence: `pyproject.toml` installs TensorFlow, notebook tooling, plotting, multiple lint/format tools, and native-related dependencies as base requirements. The wheel packages `bench` only, while default data paths depend on the working directory. Scripts insert the checkout root into `sys.path`.

Action: separate core, neural, plotting/notebook, native acceleration, development, and documentation dependencies. Provide an installed CLI with explicit data/config paths. Decide whether a small sample is packaged or supplied separately; do not silently require a repository-relative archive.

Acceptance: the first lesson runs offline without TensorFlow, a GPU, Rust, or a docs generator; the built wheel runs outside the checkout with explicit data input.

### F9 — Design concern: statistical meaning needs clearer public contracts

Evidence: `bench/metrics.py:log_loss` normalizes inclusion scores into a distribution over numbers; the research code also reports joint outcome loss. `run_suite` corrects the strategy family, while the report separately corrects randomness tests and combines their flags into a headline.

These quantities answer different questions. Marginal loss is not a full-ticket probability or joint likelihood. Separately controlled families do not imply one global family-wise error guarantee for their union. Repeated exploratory iterations are also not corrected by a single run's Holm procedure.

Action: use explicit labels such as normalized marginal log loss and joint outcome log loss. State each correction family and endpoint. Either scope the headline to separate families or predeclare a combined family if a global guarantee is desired. Keep exploratory selection distinct from prospective confirmation. Distinguish below-chance deviations, above-chance performance, and distributional anomalies.

Acceptance: metric definitions include the predicted object, units, baseline, direction, assumptions, and a tiny worked example. Report tests cover below-chance strategies and randomness-only flags without implying profitable prediction.

## 4. Proposed repository layout

Use `ilotto_lab` as the future import package. Do not create `ilotto/` while root `ilotto.py` is still active: it can shadow the existing module. Preserve compatibility wrappers during migration.

```text
README.md                       # short entry point and first successful command
CONTRIBUTING.md                  # development, checks, adding a lesson/experiment
pyproject.toml / uv.lock
src/ilotto_lab/
  data/                         # archive parsing, validation, immutable draw model
  forecasting/                  # predictor contracts and statistical/neural models
  evaluation/                   # windows, backtests, nulls, scoring, correction
  coverage/                     # portfolio objectives, reference and fast evaluators
  synthetic/                    # controlled generators and learning experiments
  seed_search/                  # artificial generator, campaigns, engine adapters
  experiments/                  # reusable experiment runners and artifact schemas
  reporting/                    # presentation models, templates, charts
  cli.py                        # small command dispatcher
lessons/
  01_randomness_and_baselines/
  02_data_and_leakage/
  03_forecasting_honestly/
  04_memorization_and_generalization/
  05_learning_planted_structure/
  06_ticket_coverage/
  07_seed_search_and_performance/
experiments/
  registry.yaml                 # discoverable inventory, status, evidence links
  <experiment-id>/
    README.md                   # question, methods, interpretation, commands
    protocol.json               # versioned inputs, endpoints, budgets, seeds
    runs/<run-id>/               # manifest + small metrics; large files by checksum
legacy/                         # historical architectures and explicit limitations
native/                         # Rust/CUDA implementations and build documentation
scripts/                        # compatibility and maintenance wrappers only
examples/                       # short runnable API examples
tests/                          # unit, integration, scientific, CLI, native
docs/                           # rendered learning path and reference
data/                           # raw snapshots, manifests, tiny teaching fixtures
```

Folder names are proposed, not existing commands. Avoid splitting every short module into many tiny files: group by concept first, then split when a module has multiple independent responsibilities.

Keep original raw archives immutable. Put derived output outside source directories. Track small canonical JSON results in Git; use release assets or another versioned artifact store for large datasets, models, and bundles when their growth warrants it. The current local directories are modest (approximately 25 MB models, 22 MB experiments, 8.4 MB downloadable bundles); no urgent destructive history rewrite is justified.

## 5. Learning path

Each lesson should take the reader from intuition to a small executable example, then optionally into the full research implementation.

| Lesson | Central question | Small exercise | Evidence to inspect |
|---|---|---|---|
| Randomness and baselines | What does a random ticket achieve? | Enumerate a toy lottery and compare simulation | Mean overlap and distribution |
| Data and leakage | How can cleaning manufacture a pattern? | Compare value filtering with era filtering; trace row IDs through windows | Before/after counts and causal timeline |
| Honest forecasting | How do we evaluate an adaptive strategy? | Compare uniform and rolling frequency on a frozen small dataset | Per-draw predictions, effect size, corrected tests |
| Memorization | Can perfect fit fail to predict? | Fit lookup and smooth/linear controls | Training reconstruction beside future score |
| Synthetic learning | Does the method detect information when it exists? | Fair, persistent-bias, and changing-bias generators | Recovery/power and fair-null false positives |
| Coverage | Can ticket overlap change the chance of any small prize? | Exhaustively compare portfolios in a toy game | Union coverage, duplicates, objective tradeoffs |
| Search and performance | What can recovering a known artificial seed teach? | Recover a planted seed in a bounded range | Python oracle and native equivalence |

For every lesson provide: prerequisites, learning objectives, a plain-language question, a diagram or worked example where helpful, the smallest runnable script, expected output, interpretation, failure modes, one exercise, a solution, and links to the full experiment and implementation.

Do not start beginners with the million-draw run or a GPU campaign. Advanced acceleration belongs after the reference algorithm is understood. Explain that seed recovery concerns an explicitly artificial generator, not evidence about the real lottery mechanism.

## 6. Standard experiment page and run manifest

Every existing research family needs one landing page, including the statistical benchmark, neural set models, date-conditioned controls, perfect-fit interpolation, chaos, golden-ratio schedules, probability calibration, joint mixtures, coverage refinement, strong assignment, historical strikes, synthetic learning, and seed campaigns.

Use the same page structure:

1. **Question and motivation:** one concrete question, why it is useful to learn.
2. **Status:** exploratory, frozen retrospective, prospective, superseded, or historical demonstration.
3. **Hypothesis and endpoint:** what is predicted or optimized; what counts as success.
4. **Data:** source snapshot, hash, era, chronological splits, exclusions.
5. **Method:** intuition, notation table, worked example, implementation entry point.
6. **Controls:** uniform baseline, planted signal, memorization or other relevant negative control.
7. **Protocol:** seeds, hyperparameters, search budget, refits, selection policy, correction family.
8. **Run:** fast teaching configuration and exact historical reproduction, with measured runtime/memory and prerequisites.
9. **Results:** generated table, uncertainty, sample size, effect size, and artifact links.
10. **Interpretation:** what follows, what does not follow, plausible failure modes.
11. **Reproduction checks:** input/output hashes, environment, source revision, comparison tolerances.
12. **Exercise and next step:** one targeted modification with a prediction to make before running it.

A manifest should record schema version, experiment/run IDs, creation time, status, source commit and dirty state, configuration/hash, dataset/hash, dependency lock hash, seeds, backend/hardware, actual evaluation interval, metrics, artifact hashes, and completion status. Timing and host details are not expected to reproduce bit-for-bit. Report numerical tolerances explicitly, especially for neural/GPU runs.

Publish a multi-file run by writing everything into a staging directory, verifying it, then exposing a completed manifest/directory. Atomic replacement of each file individually does not make an entire run atomic.

## 7. Human-readable code rules

Prefer domain names (`target_date`, `history_draws`, `main_inclusion_probabilities`) over abbreviations at public boundaries. Keep mathematical notation where it makes an algorithm easier to compare with its derivation, and explain it once.

Public functions should document input shapes, indexing conventions, units, invariants, return values, and failure cases. Comments should explain the reason for a constraint, not repeat the next line. Add examples for subtle concepts such as normalized marginal probabilities and sampling without replacement.

Introduce small typed result objects for forecasts, evaluation plans, and experiment manifests where they prevent ambiguous dictionary keys. Avoid a large plugin framework or deep inheritance tree. A simple predictor protocol and explicit registry are sufficient.

Separate computation from rendering and file writes. Library imports should not configure logging, fetch data, or write output. Move reusable helpers out of runnable research scripts. Keep reference algorithms next to accelerated implementations, with shared test vectors and independent small-game oracles.

## 8. Delivery sequence and acceptance gates

### Phase 0 — Preserve evidence and establish the baseline

Inventory all experiments, artifacts, command entry points, generated pages, and model checkpoints. Record current tests, lint, docs build, and representative output hashes. Create the claim-to-evidence table. Mark unsupported legacy metrics immediately.

Done when every published headline has an evidence link and every file family has a declared owner/purpose. No historical outputs are regenerated merely to make their names consistent.

### Phase 1 — Fix correctness boundaries

Address F1–F5 in separate focused changes. Start with synthetic fixtures that expose each defect. Keep corrections separate from mechanical moves so reviewers can see behavior changes clearly.

Done when causal windows, split roles, ingestion fallback, draw validation, and shared evaluation intervals have regression coverage. Reassess affected legacy claims and checkpoint provenance.

### Phase 2 — Make a first lesson easy to run

Create the lightweight dependency path, installed CLI, sample data, and first three lessons. Rewrite README around purpose, five-minute run, expected output, learning map, evidence, and development links. Keep exhaustive flags in reference documentation.

Proposed interface, not yet implemented:

```bash
uv sync --frozen
uv run ilotto lesson baselines
uv run ilotto benchmark --config experiments/baseline/protocol.json --offline
uv run ilotto experiment list
```

Done when a fresh environment can run the baseline offline without optional frameworks and from outside the checkout. Measure runtime before promising a specific duration.

### Phase 3 — Migrate modules without changing results

Move shared data/evaluation code first, then forecasting, coverage, synthetic and seed-search code, then reporting. Keep `bench` and old script wrappers temporarily. Move legacy modules last, updating imports together. Test the wheel, not just source-tree imports.

Done when representative deterministic outputs match the baseline and old documented commands either still work or provide explicit migration guidance. Original frozen experiments retain their original reproducible source snapshot.

### Phase 4 — Standardize experiments and artifacts

Add registry, schema validation, immutable run directories, manifest-based reporting, and consistent smoke/full configurations. Migrate one experiment end to end before generalizing the abstraction. Then convert the rest.

Done when a reader can navigate question → protocol → command → metrics → per-draw evidence without searching filenames. Interrupted runs cannot look complete.

### Phase 5 — Teaching and presentation polish

Complete lessons, glossary, worked examples, architecture diagrams, exercises and solutions. Move long historical progress reports into a research-history section. Build a short presentation narrative from verified results.

Done when a new reader can explain leakage, baselines, memorization, coverage, and the limits of the result after following the learning path. Review rendered pages and execute documented commands in a clean environment.

## 9. Verification strategy

Fast tests: data validation, causal window boundaries, metric identities, tiny exact enumerations, artifact schemas, and CLI behavior. These should be deterministic and offline.

Scientific tests: uniform-null behavior, planted-signal recovery, memorization controls, invariance to ball ordering, and explicit family correction. Prefer analytical or bounded checks over brittle assertions on random p-values.

Integration tests: frozen small benchmark, report rendering, failed-run preservation, lightweight installation, and installed-wheel execution outside the repository.

Optional jobs: TensorFlow smoke test, Rust parity, host CUDA hash checks, real GPU equivalence where hardware exists, and longer research reproduction. Report skipped capabilities visibly. Keep compile correctness, algorithm equivalence, and hardware performance claims separate.

Documentation checks: strict site build, links, runnable examples, and manual rendered-page review. A strict build cannot validate the scientific meaning of prose or whether a lesson is understandable.

## 10. Presentation narrative

1. Start with the tempting claim: a model appears to predict lottery numbers.
2. Show how ordering, filtering, and an inappropriate metric can create that impression.
3. Introduce the fair baseline and chronological evaluation.
4. Demonstrate perfect memorization with no future advantage.
5. Show that the method can learn a planted synthetic signal.
6. Explain why coverage optimization is a different and legitimate mathematical objective.
7. End with the evidence and limits: these evaluations have not established predictive advantage; negative results are useful when the measurement is trustworthy.

The best portfolio piece here is not a large catalog of architectures. It is a clear, executable explanation of how to test a persuasive idea, discover mistakes, preserve evidence, and report what the experiments actually support.

## 11. Validation performed during this review

- `.venv/bin/python -m pytest -q`: **232 passed, 2 skipped**, 38.18 seconds. Skips are not evidence that the corresponding capabilities work.
- `.venv/bin/ruff check . --output-format concise`: **passed** across the repository.
- Executed a synthetic 100-row identifier fixture through the existing legacy split with `test_size=10, w=3`: its first test window contains row IDs `[87, 88, 89]`, while the paired target is row ID `0` (after the function's internal subtraction). This directly reproduces F1's misalignment.
- No production code or historical research artifacts were changed. No full research experiments, docs build, or GPU jobs were run for this review.

Passing the existing suite demonstrates useful regression protection in the covered paths; it does not resolve the uncovered legacy defects above.
