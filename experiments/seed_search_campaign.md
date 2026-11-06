# Larger seed search and non-repetition — 2026-11-06

The second real-history experiment searched **100,000,000 new seeds** in approximately **75.99 seconds**, about **1.32 million candidates/second** on the same Mac. It imported the initial million-seed pilot without searching those seeds again.

| Quantity | Result |
|---|---:|
| Previously tested | 1,000,000 |
| Newly tested | 100,000,000 |
| Total distinct seeds tested | 101,000,000 |
| Completed interval, inclusive start / exclusive end | [0, 101,000,000) |
| Candidates matching the first known draw | 4 |
| Candidates matching the first two known draws | 0 |
| Candidates matching all 1,628 search draws | 0 |
| Next untested seed | 101,000,000 |

The four one-draw matches are retrospective coincidences, not predictions made before a draw. All four failed the second draw, so none produced a withheld continuation to score. This experiment still tests only the artificial `calendar-sha256-v1` generator and provides no demonstrated real predictive advantage.

The source is the unchanged local archive of 1,632 modern-format draws ending 2026-08-04, with the last four reserved for retrospective evaluation. No download, ticket purchase, neural training, or background job occurred. The speed predominantly measures rejection at the first draw, not generation of entire histories.

## Subsequent official archive refresh

After that measured run, the existing `bench.data.download` function successfully fetched the official archive on 2026-11-06. Both `input/Orig_IL_lotto.csv` and `input/lotto_IL_modern.csv` gained **29 draws**, taking the modern-era total to **1,661** and the latest result to **2026-11-03**. The fetched data was validated before replacement, including chronology, distinct balls, ranges, draw IDs, and preservation of the previous history. The previous raw snapshot is retained locally under `experiments/seed_search_runs/archive_snapshots/`.

All previous dates and outcomes were unchanged. The new default search prefix has 1,657 draws, with the last four again reserved. Every previously tested seed already failed the earlier prefix, so the new ledger inherits **101,000,000 proven rejections without rerunning them**. This is logical reuse of earlier failures, not a new 101-million-seed execution. The next untested seed remains **101,000,000**.

`bench.seed_campaign.inherit_rejections` performs this migration only for an exactly unchanged prefix followed by additional observations, and only when the old campaign has no surviving candidates. It preserves the source ledger and records the inheritance in the destination. Changed history, survivors needing new evaluation, or an existing destination ledger cause a refusal rather than silent reuse or overwrite. The migration was invoked for this refresh; future downloads still require an explicit verified migration.

## Continue without repeating completed seeds

From the repository root:

```bash
.venv/bin/python scripts/search_seeds.py --mode real --count 100000000 --max-seconds 120
```

`--count` requests that many **new** candidates, within the time limit. It uses the existing shared ledger automatically and generates a separate report for each invocation. Repeating this command advances through additional ranges; it does not restart at zero. After a timeout, the next invocation first resumes the unfinished range, then allocates further work if needed to meet its new-candidate budget.

## How the record works

- **Deterministic enumeration:** visit integer seeds in order; no random seed sampling or probabilistic duplicate filter.
- **Compact ranges:** record reserved intervals and their completed cursors, not one database row per seed. The ledger after this run is about 1 KB; checkpoints also store match-count histograms and complete-history survivors.
- **Persistent identity:** records bind the generator version and a hash of the matching dates and outcomes. Changing only the report directory does not reset coverage.
- **Import of earlier work:** matching checkpoints directly under `experiments/seed_search_runs/` are validated and copied into the ledger. Other checkpoints can be supplied with `--import-checkpoint PATH`. Conflicting overlapping imports fail rather than inflate totals.
- **Exclusive execution:** an operating-system file lock permits one process at a time for this history and ledger. A second cooperating process is rejected before searching; a process crash releases the lock automatically.
- **Crash recovery:** completed batches are saved atomically. A reserved but not-yet-started range can resume safely. An interrupted, uncommitted batch may replay; its work is never counted twice. Python uses 4,096-candidate batches by default; the optional Rust engine uses adaptive batches targeting about one second, capped at one billion candidates. Exactly-once physical execution across arbitrary crashes is not claimed.
- **Validation:** inconsistent counts, regressed checkpoints, missing known-completed work, and malformed or overlapping registered ranges stop the run instead of silently repeating work.
- **Preserved reports:** each continuation gets its own output directory. The ledger's latest summary is a convenience; the separate reports retain earlier run results.

All cooperating runs must use the **same ledger directory**. The default is `experiments/seed_search_runs/campaigns/`. Choosing a different `--ledger-dir`, deleting the ledger, or using the older independent `--start`/`--stop` mode bypasses that shared coverage. Use `--count` for ongoing searches. Keep ledger files and their checkpoints together when backing up or moving the project; current ledger paths are absolute.

A changed matching history receives a distinct ledger, because it defines a different test. We do not silently carry rejections across corrected or replaced data. The verified append-only migration described above supports the no-survivor case; migration with survivors still requires additional implementation to evaluate them against the suffix. Changing only withheld outcomes does not change the search prefix's identity, although reports record the source archive hash.

The local record prevents cooperating processes from repeating completed work under normal restart conditions. It is not a distributed work scheduler, and preservation after deletion or storage corruption depends on backups.

The later Rust implementation parallelizes disjoint intervals **inside** each batch while the single Python coordinator retains the ledger lock. See `experiments/parallel_search_plan.md` for measured worker scaling and the supervised one-hour run.

## Files and verification

`bench/seed_campaign.py` implements the shared range ledger and lock. `bench/seed_search.py` now supports a per-invocation candidate budget while retaining the original generator. `scripts/search_seeds.py --count` connects them to the real and toy experiments.

The targeted tests exercise imported coverage, repeated continuation, gap filling, partial-range budgets, competing locks, overlapping imports, invalid counts, different histories, missing checkpoints, crash recovery, original seed recovery, and stable generator outputs. Full-project test validation remains limited by the previously observed local SciPy binary-loading error; the seed-search tests do not depend on that binary.

Generated reports and checkpoints are local, ignored artifacts under `experiments/seed_search_runs/`. The report for this run is `campaign_reports/real_next_100m/report.json`. This document preserves the measured result in the repository.
