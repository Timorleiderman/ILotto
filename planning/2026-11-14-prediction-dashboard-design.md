# Prediction dashboard — design

Date: 2026-11-14. Status: implemented and reviewed in `lotto/ui/` (see `docs/start/dashboard.md`).
Custom form sizes, MLP, epoch-level progress and sampled-ticket backtests remain deferred.
Mockup: open `planning/dashboard-mockup.html` in a browser. It shows four screens with
frozen examples from the archive through 10 November 2026. Controls explore the design;
they do not run models. The historical log examples are retrospective, not saved predictions.

This revision resolves gaps between the original design and the existing code, separates
settings from completed results, and defines the checks needed before each screen ships.

## 1. What it is

A local browser dashboard for exploring the project's prediction methods, designing forms,
and measuring results against chance.

```bash
uv run lotto ui
```

It starts a server at `http://localhost:6037` (6 of 37), bound to `127.0.0.1`, opens the browser,
and stops with Ctrl-C. Supports `--no-browser`, `--port`, and `--offline`. Starts from the local archive;
checking pais.co.il is an explicit action, disabled in offline mode.

1. **Predict.** Choose a model, settings and form size; see its ticket, form and exact fair-draw odds.
2. **Compare.** Compare available benchmark presets and see which numbers they agree on.
3. **Lab.** Replay configurations on past draws and keep a record of the settings tried.
4. **Log.** Save completed predictions and score them when the corresponding result arrives.
5. **Report.** Export a completed result as printable HTML or Markdown, with its provenance.

### Implementation notes after review

The sections below retain the design rationale and acceptance criteria. The current API uses
one persistent `/api/notebook`; Compare runs all benchmark presets. Historical generation
uses past-only history rather than reconstruction from a fit containing the target draw.
The Lab rejects sampled mode because the core harness evaluates top-six tickets. Prediction
results use schema 2, source/runtime fingerprints and immutable caching. Comparisons receive
a new run ID so transient failures cannot overwrite a prior report. Legacy scoreboard records
remain dated references with unverified configuration matches. Fine-grained neural progress
and full configuration manifests in the published report are still follow-up work.

## 2. Implementation choice

| | A. Standard library + plain JavaScript | B. FastAPI + plain JavaScript | C. Streamlit |
|---|---|---|---|
| Dependencies | none added | web stack in an optional extra | larger UI dependency |
| Offline, no frontend build | yes | yes | yes |
| Look and feel | fully ours; reuse report tokens and charts | fully ours | framework conventions |
| Main responsibility | explicit routing, validation, jobs and storage | routing/validation supplied; jobs/storage still ours | script reruns and session state |
| Testing | service, HTTP and interaction checks | service, HTTP and interaction checks | more framework-specific |

**Recommendation: A**, retaining the original lightweight direction. Revisit B if routing and
validation become hard to maintain. No fixed line-count target: correctness and readable
boundaries matter more than fitting the server into 300 lines.

## 3. Architecture and integration

```text
Browser: plain HTML, CSS and JavaScript; no CDN or build step
   | JSON requests                  ^ job status while active
   v                                |
lotto/ui/server.py     loopback ThreadingHTTPServer, routing and request protection
lotto/ui/registry.py   stable model/preset IDs, parameter metadata and constructors
lotto/ui/service.py    validation and orchestration of existing computation
lotto/ui/jobs.py       one worker, bounded queue, progress and cooperative cancellation
lotto/ui/storage.py    immutable result/log records and persistent Lab notebooks
lotto/ui/static/       index.html, app.css, app.js
   |
   v
lotto/core, lotto/approaches, lotto/tickets, lotto/report/charts.py
```

- **No statistical algorithms in the UI.** Reuse `next_draw_scores`, `best_ticket`,
  `model_form`, `portfolio_odds`, `strong_prize_odds`, `walk_forward`, `holm_bonferroni`,
  `metrics.prize_tier` and the coverage/strong-assignment algorithms.
- **The UI registry owns editable parameter metadata.** The existing suites remain the
  reference for benchmark presets. Test that their model configurations match; avoid an
  independent, drifting copy of the model catalog. Build controls and validate requests
  from the same metadata. A model ID and a benchmark preset ID are different concepts.
- **Server-rendered charts** reuse `lotto/report/charts.py`. The browser draws the balls and
  number grid. Do not call ranking scores probabilities; respect `emits_probabilities`.
- **One worker**, including refresh and log scoring. Create a fresh predictor for every job;
  mutable trained models and RNG state must not leak between prediction and backtest jobs.

### Existing contracts that need adapters or small core changes

| Existing code | Required treatment |
|---|---|
| `next_draw_scores()` chooses its own date and returns `None` for most predictors | Resolve a target date for every result. Add an optional explicit target to the core helper, preserving old callers and leakage checks. |
| `DateConditionedCB.scores()` rejects dates at or before the training cutoff | Keep that protection. Historical generation needs a separate core helper/result mode; never catch `LeakageError` and silently retry on training data. |
| `DateConditionedCB.sample()` supplies an `in_sample` flag | Retain it alongside training cutoff and mode. Even a date absent from the archive is retrospective if it precedes training data. |
| `walk_forward()` and neural `fit()` expose no progress/cancellation callback | Initially report truthful stage progress. Add optional per-draw/per-epoch callbacks with unchanged numerical defaults before promising fine-grained progress. |
| `scoreboard_record()` joins on display name; the report has no complete config manifest | Show legacy scores as dated reference results. Exact correspondence requires a versioned configuration manifest emitted by the report. |
| `data.load()` writes the clean archive by default | Read with `clean_csv=None`; only explicit refresh may replace archive files. Freeze loaded draws per job. |
| Some models use `-inf` for excluded numbers | Preserve ranking semantics in core code; serialize a documented `null` plus exclusion mask, never JSON `Infinity`/`NaN`. |
| CLI dispatch currently supports `report`, `plan`, `predict` | Add a dedicated `ui` branch and help entry. Importing the package must never start a server. |

### Result identity

A versioned result envelope carries: result ID, model/preset IDs, resolved parameters and
seeds, target date and mode, archive SHA-256 and training cutoff, algorithm/code fingerprint,
generation time in UTC, exact ticket/form rows, design ID/hash, and record provenance.
The screen, cache, report and log use this same immutable envelope. Display dates and times
in `Asia/Jerusalem`; compute the local day explicitly when calling `next_draw_date(today=...)`.

## 4. API

| Method and path | Contract |
|---|---|
| `GET /`, `/static/*` | page and a fixed static-file allowlist |
| `GET /api/state` | archive identity/cutoff, estimated next date, refresh status, registry, availability, designs and active job |
| `POST /api/refresh` | queue validated refresh; distinguish updated, unchanged and failed |
| `POST /api/predict` | model/preset + resolved settings + seeds + rows + date/mode → job |
| `POST /api/compare` | all benchmark presets + one common archive/date → job |
| `POST /api/backtest` | frozen top-six config + evaluation window → job in the persistent notebook |
| `GET /api/jobs/<id>` | state, progress, result or structured error |
| `DELETE /api/jobs/<id>` | request cancellation; return actual cancellation state |
| `GET /api/notebook` | persisted attempts and recomputed exploratory verdicts |
| `GET /api/log`, `POST /api/log` | read records; save a server-held completed result by ID |
| `GET /report/<id>` | printable HTML; `?format=md` returns Markdown |

Job creation returns `202`. Invalid fields return `400` with field-level messages, busy or
unavailable actions `409`, oversized bodies `413`, and unknown/expired IDs `404`. Set finite
body, queue and cache limits. Idempotency keys prevent duplicate jobs and log saves from
retries/double clicks. All JSON is strict; do not accept arbitrary browser-supplied rows as
saved predictions. Persist exported/saved results before acknowledging success.

## 5. Shared interaction and visual rules

The top bar shows estimated next draw, archive cutoff, last successful check, refresh, and
theme. `next_draw_date()` infers cadence; it is not an official schedule or a cutoff time.
Never infer “up to date” merely from the newest row. Distinguish not checked, checking,
checked, failed using cached data, and stale. Failed refresh leaves the dashboard usable.

Each screen has a heading and short explanation. Use native labelled controls, visible
keyboard focus, keyboard-operable tabs, status announcements, and text alongside colour.
At 320px controls stack above results, number grids fit, and only wide tables scroll.
Theme follows the system until explicitly changed and is remembered where storage works.

**Settings and results are separate state.** Changing a control labels a displayed previous
result “Settings changed — generate to update.” Its header retains the settings that made it.
An older job finishing must not overwrite a newer selection. Save and export refer to the
visible completed result, not current draft controls. Cover empty, queued, running,
cancel-requested, cancelled, failed, and complete states. Explain disabled actions nearby.

## 6. Predict

Controls on the left, results on the right; stack them on narrow screens.

| Control | Behaviour |
|---|---|
| Model/preset | grouped by family; separate benchmark presets where settings differ |
| Parameters | typed inputs, allowed range, help and reset; invalid fields block submission |
| Prediction seed | only for stochastic modes; default belongs to the preset |
| Form | single illustrative ticket, or stored 2, 10, 14-row designs initially |
| Design seed | separate from prediction seed, only for later custom-size optimization |
| Draw date | estimated next date by default; historical generation is an explicit mode |
| Generate | submit once; stage progress initially, epoch progress only when implemented |

Results:

- The six main numbers and a text-labelled strong number. Sampled tickets say “sampled”,
  not “best”. Explain that the best single ticket need not be a row in the coverage form.
- Odds tiles for **this form**: any prize, three or more with strong, first prize, base cost.
  A prize is not necessarily profit; these tiles do not estimate money returned.
- Numbered form rows with a copy action and confirmation. Never change rows on tab switches.
- A 1–37 score grid and separate 1–7 strong grid, chosen numbers ringed. Explain units and
  expose values on focus/touch; negative scores and ties must remain understandable.
- Detailed whole-form odds for at least one row matching 3, 4, 5 or 6, plus first prize.
- A dated scoreboard reference with its evaluation window and archive cutoff. Only the
  manifest can establish a configuration match; changed parameters/seed/sampling/protocol
  show “These settings are untested” and a Lab action. Never join evidence by name alone.
- Save to log and download report, tied to the completed result and its eligibility.

## 7. Compare

Show benchmark presets with ticket, labelled strong number, reference record, and
queued/running/complete/unavailable/failed state. A real button opens the matching preset
in Predict, including settings and seed. Partial failures preserve completed rows.

There are currently **19 benchmark presets**: 16 quick, one SciPy-only date-conditioned,
and two neural. Counts come from registry/results, not constants. The consensus board
counts successful included tickets, gives the denominator, and says agreement is descriptive,
not confidence. Shared history makes these methods correlated. Seed world is a separate
opt-in toy and excluded from benchmark consensus. Do not silently add it as a twentieth vote.

## 8. Lab

Replay a frozen configuration from Predict over 100–1,000 past draws, bounded by available
history. Show the actual start/end dates and scored count from `evaluation_window()`, which
can truncate a request. Each forecast sees only earlier draws.

- Show mean matches versus 0.973, and the existing **two-sided finite-sample** `p_value`
  from `metrics.summarise()`. A significant decrease is not predictive skill.
- Plot observed match counts against chance, including all 0–6 outcomes or an explicit tail.
- Persist the notebook under `results/ui/notebooks/`, including attempts recorded before
  execution, cancelled jobs and failures. Reloading must not erase the history of tuning.
- A cached rerun of the same config/archive/window is the same comparison. A new seed,
  window or config is a new attempt. Recompute all completed verdicts with the existing
  `holm_bonferroni()` whenever a comparison is added; use stable attempt IDs as keys.
- Holm is step-down, not a universal `0.05 / number_of_tries` threshold. Show raw p-values
  and adjusted verdicts, with the comparison family and number of completed tests.

Label the entire notebook **exploratory**. Selecting new settings after seeing evaluation
results is adaptive selection; applying Holm to a count of trials does not make that process
confirmatory or account for trials elsewhere. Validation requires a frozen configuration on
untouched future draws. Do not display “signal found” or imply a new notebook resets evidence.

## 9. Log

Save immutable server-held results, with both generation and save timestamps. Saving a
cached result uses the current save time. Deduplicate retries, reject reconstructed results
and already observed target dates. Entries are not editable/deletable in the dashboard;
plain files remain under `results/ui/log/` for the owner to manage.

- Waiting entries say “Waiting for a published result”. Score by the actual target date/draw
  ID, not by the next row of a refreshed archive. Keep scoring separate from original data.
- A local timestamp is a personal record, not independently verified preregistration.
  A date-only estimate cannot establish the cutoff for same-day saves. Without a verified
  cutoff, mark timing **unverified** and exclude the entry from prospective summaries.
- Show every row's main matches, strong hit and `metrics.prize_tier`; the displayed best
  single ticket and the form are distinct objects with distinct scores.
- Summary: **forms with a prize / forms scored**, with expectation `sum(p_any_prize)` for
  those exact forms. Multiple forms on the same draw are correlated: no independent-binomial
  intervals. A distinct-draw summary would need union odds over all entries for that draw.
- Retrospective demo entries stay outside prospective totals. Four forms across two draw
  dates are four forms, not four draws, and are not proof of anything saved in advance.

Use atomic writes (`core.files.write_json`) and a single writer. Immutable entry creation
must also check identity/collisions: atomic replacement alone does not enforce immutability.

## 10. Models and parameters

Presets follow `quick_suite()` and `full_suite()`, not merely constructor defaults. In
particular: random baseline seed **12345**, sum-targeted seed **0**, neural seeds **42**.

| Model | Parameters and initial UI bounds | Seed |
|---|---|---|
| Random ticket | — | yes |
| Hot numbers | all-time and last-100 presets; window `null` = all, or 10–1,500 | no |
| Cold numbers | window all by default; same range | no |
| EWMA | half-life 50; 5–500 | no |
| Overdue | — | no |
| Pairwise co-occurrence | smoothing 1.0; 0–10 | no |
| Lag-1 persistence | smoothing 2.0; 0.1–10 | no |
| Sum-targeted | target 114; 21–207 | yes |
| Rank ensemble | nonempty subset of the seven suite members, including both hot windows | no |
| Perfect fit | three distinct DFT, spline, polynomial presets | no |
| Method of analogues | window 3 (1–6), neighbours 20 (5–100), prior 5 (0–50) | no |
| Golden ratio / Fibonacci | two distinct presets | no |
| Date-conditioned CB | regularisation 10 (0.1–1,000), date, top-six or sampled mode | sampling only |
| GRU / Transformer | history 20 (5–60), epochs 60 (1–200); refit 50, validation fraction 0.15 in provenance | yes |
| Seed world | seed through `2⁶⁴−1`, date; artificial generator, no benchmark record | seed is the input |

No recursive ensembles. MLP exists in the class but is not in the published suite; defer it
or label it an untested additional preset. Neural availability must cover both missing and
broken TensorFlow installs; do not import TensorFlow just to render `/api/state`.

Transport large seeds as **decimal strings**, using text inputs with numeric keyboards.
JavaScript numbers cannot exactly represent the full uint64 range. Validate in Python;
reject booleans, fractions, unknown keys and out-of-range values. Individual models may
require narrower seed bounds. A dice action must respect the selected model's range.

Historical date-conditioned generation remains explicitly labelled, with training cutoff
and actual result where available; no retrospective output is eligible for the forecast log.

## 11. Forms, odds and reports

- Stored forms come from `draw_plan.FORMS`: 2, 10 and 14 rows, with approximately 8.33%,
  38.60%, 51.01% chance of any prize under uniform fair draws.
- Model relabelling preserves exact odds **for the same design**, including strong-number
  assignment. Different custom designs of the same size need their own counts.
- A one-row result is the single ticket. It is an illustrative calculation, not a claim
  about the operator's minimum purchase. Base cost uses `draw_plan.ROW_PRICE` (currently
  3 NIS), labelled an estimate rather than a live price quote.
- Add other even sizes 4, 6, 8, 12 as a separate bounded job after measuring runtime/memory:
  greedy candidate selection, exact refinement, strong-number optimization. Do not promise
  global optimality or a one-second runtime. Design seed is independent of prediction seed.

Reports contain the immutable configuration, date/mode, ticket/form, odds, reference record,
archive fingerprint and timestamps. Escape content and include the same sample/reconstruction
labels as the screen. Print legibly with repeated table headers; allow more than one page
for a 14-row form or comparison. Markdown contains the same data. Export must work offline.

## 12. Jobs, cache and performance

Initial estimates from the original design: statistical prediction/backtest is fast,
date-conditioned prediction around a second, neural prediction seconds, and neural backtest
around a minute. **Measure cold and warm runs before publishing promises.** Custom design
search may be substantially heavier; imported compiled counters may also need warm-up.

Cache by normalized configuration, prediction/design seeds, design hash, target date/mode,
archive SHA-256, algorithm and schema version. Cache odds by exact design and rules, not
just row count. Keep generation time on cache hits. Archive refresh changes the cache key;
previous results remain accessible with their original cutoff. Do not cache mutable models.

State machine: `queued → running → succeeded / failed`, with
`running → cancel_requested → cancelled`. Queued cancellation is immediate. Running work
stops at safe callback boundaries. During an uninterruptible fit/count show “Cancellation
requested — finishing current stage”; do not publish cancelled results or start another job
meanwhile. Hard cancellation would require a subprocess and is outside the first release.

Poll while jobs are active, back off when the page is hidden, and reconnect by job ID after
reload. On shutdown, mark persisted interrupted attempts and discard incomplete output.
Refresh is serialized with computation; every compare job sees one fixed archive snapshot.

## 13. Local server protection and storage

- Bind only to `127.0.0.1`. Remote use is via SSH forwarding. No `--host` option in the
  first release: it contradicts the loopback-only access model and there is no login.
- Validate the exact Host/port. Mutations require the matching Origin and a per-launch
  anti-CSRF token. No permissive CORS, no form-encoded writes, no missing-Origin mutations.
- Fixed static paths and opaque result IDs; reject encoded traversal and arbitrary paths.
  Bound requests/queue/cache, validate every field, and return safe structured errors.
- Writes stay under `results/ui/`, except explicit validated refresh of `input/`. Reads use
  `clean_csv=None`. Never alter frozen `experiments/` or generated `docs/` from the dashboard.
- No popularity or birthday logic, matching the rest of the project.

## 14. Validation and build order

| Step | Deliverable and acceptance checks |
|---|---|
| 1 | Registry, result schema and service. Presets match suites, tickets valid, seeds exact, leakage rejected, odds preserved and seeded runs repeat from fresh instances. |
| 2 | CLI, server, worker and page shell. HTTP request validation, origin/token protection, static traversal rejection, queue limits, cancellation and offline startup. |
| 3 | Predict and exports. Success, invalid settings, failure, pending settings, late completion, cancelled output, and report provenance all behave correctly. |
| 4 | Compare. Common archive/date, partial failure, correct consensus denominator, and preset navigation. This completes the first usable version. |
| 5 | Lab. Actual evaluation window, persisted attempts, reload, cached reruns and Holm results equal core functions; explanatory limits remain visible. |
| 6 | Log. Atomic/idempotent saves, timing eligibility, hand-checked scoring, same-draw correlation, and archive refresh without rewriting originals. |
| 7 | Docs/run guide/README/CLAUDE.md; keyboard, mobile, both themes, print and light-install checks on Mac and Linux. |

Keep numerical tests on the pinned archive. Exercise parameter boundaries with bounded
fixtures; do not train every large network at its maximum just to test input validation.
Test refresh/cache invalidation, and prove read-only requests do not write `input/`.
Also check legacy notebook migration, competing dashboard launches, safe worker shutdown,
malformed browser preferences, unfinished numeric inputs and failed cancellation retries.
The JavaScript regression checks run with Node's built-in test runner, without a build step.

Page checks cover 320/768/1280px widths, labels, focus, contrast, table scrolling, disabled
features, empty state, failed refresh, neural unavailable, job reconnect and print. An HTTP
200 or a JavaScript line-count target is not a substitute for interaction verification.

Documentation adds “The dashboard” under Start here with a screenshot and command examples
**when the command exists**. Explain the UI boundaries and result storage in `CLAUDE.md`.
Each step is reviewable with relevant tests. Custom-size search, MLP and historical
reconstruction can follow the first usable prediction/compare loop.

## 15. Scope and defaults

Retain all four planned screens, plain JavaScript, a standard-library server, port 6037,
seed world as a separate toy, and append-only log behaviour in the dashboard.

Defer long research runs, saved named presets, a bonus-draw planner panel, row-count charts,
expected payout, accounts and public hosting. Remember last draft settings in the browser
without treating them as a completed result. No change to prediction algorithms is implied
by this UI review; optional progress/date adapters must preserve existing tested invariants.
