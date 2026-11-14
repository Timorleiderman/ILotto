# Parallel Rust search and one-hour run

Requested on 2026-11-06. Hardware: M3 Max, 10 performance cores, 4 efficiency cores, 36 GiB memory.

1. Keep `calendar-sha256-v1` bit-for-bit compatible, preserving the previous 101 million rejections against the refreshed archive.
2. Move the candidate loop to Rust. Prebuild the fixed SHA-256 input block, use RustCrypto's ARM SHA backend, avoid per-seed allocation, and reject at the first mismatch.
3. Divide each finite batch into disjoint contiguous intervals across CPU threads. Keep the single Python coordinator and shared ledger lock; workers never write the ledger.
4. Commit the completed batch atomically before advancing. Store boundaries, aggregate counts, and rare surviving seeds. A crashed unfinished batch may replay; completed batches do not.
5. Cross-check Rust against Python, including known seeds, rejection mapping, uneven partitions, and the unsigned 64-bit boundary.
6. Benchmark 1, 4, 8, 10, 12, and 14 workers on new, ledger-recorded ranges. Compare the best configurations again after warm-up. Select from measured throughput, not core count alone.
7. Start a bounded 3,600-second local run, counting from the actual launch after benchmarking. Record start/end times, selected configuration, archive/checkpoint hashes, process IDs, and progress. Stop automatically at the deadline, after the last small in-flight batch.
8. Keep the public output to at most 10 distinct candidate combinations if any seeds survive the historical criteria. Do not fill the list with rejected candidates or describe historical matches as future predictions.

Ten distinct complete combinations cover ten outcomes of one fair draw; their combined exact-hit probability is 10/16,273,488. Ten model seeds need not generate ten distinct combinations. Deduplicate any eventual ticket output separately from seed deduplication.

The refreshed archive has 1,661 modern-era draws through 2026-11-03. Search uses the first 1,657 and reserves the final four for retrospective scoring. CPU search is the first optimization; GPU implementation would require a separately validated kernel and measured benefit. No ticket purchase is part of this run.

RustCrypto's [SHA-2 implementation](https://docs.rs/sha2/0.10.9/sha2/) supplies the hash primitive. The existing cached version requires its `asm` feature for its AArch64 accelerated path; the local source was inspected before selecting build features.

## Measured configuration selection

All trials searched new intervals through the same ledger. Short trials used approximately five seconds each; the two finalists were then measured for approximately 15 seconds each.

| Workers | Short-trial checks/second | Warmed finalist checks/second |
|---|---:|---:|
| 1 | 18.29 million | — |
| 4 | 66.81 million | — |
| 8 | 130.17 million | — |
| 10 | 160.41 million | — |
| 12 | 174.13 million | 174.57 million |
| 14 | 185.12 million | **181.47 million** |

Selected **14 workers**. Relative to the earlier measured Python rate of 1.316 million/second, the warmed Rust rate is about **138× faster**. Extrapolation gives approximately **653 billion checks/hour** if throughput stays constant. These measurements mostly reflect first-draw rejection; they are not complete simulated histories or evidence of predictive skill.

The benchmark added 9,025,365,746 new candidates, raising cumulative coverage to 9,126,365,746. A two-second supervisor smoke test added another 96,907,081, for a starting cursor of **9,223,272,827** before the hour-long run. None survived the full history. Exact trial records are stored under `experiments/seed_search_runs/rust_benchmark_20261106/`.

## One-hour job launched

- Actual launch: **2026-11-06 17:06:06 Asia/Jerusalem**.
- Planned end: approximately **18:06:07**, after 3,600 seconds of searching and final checkpoint/report writes.
- Configuration: Rust, 14 threads, one coordinating Python process, automatic batches targeting about one second each.
- Dataset: a frozen copy of the official archive through 2026-11-03; 1,657 matching draws and four retrospective withheld draws.
- Initial cursor: **9,223,272,827**. The shared ledger retains earlier rejected ranges and all benchmark work.
- Executable: copied into the job directory and hashed before launch, so later rebuilds cannot change this running job.
- Idle sleep: `/usr/bin/caffeinate -i` wraps this finite job only. It does not override shutdown, closing the laptop lid, or manual system sleep.

This section records a launch, not a completed one-hour result. Read the status/report artifacts for the final outcome.

Job directory: `experiments/seed_search_runs/rust_hour_20261106/`.

| Artifact | Purpose |
|---|---|
| `job.json` | Supervisor state, timestamps, process identifiers, final status |
| `search/progress.json` | Live committed progress and throughput |
| `run.log` | Periodic updates and errors |
| `search/report.json` | Final results, written when the search exits |
| `candidate_summary.json` | At most ten surviving seeds, if any; explicitly labels retrospective results |
| `archive.csv`, `ilotto-seed-search` | Frozen inputs and executable for this run |

For a graceful stop, create `search/STOP` in this directory. The current batch completes and commits, then the search exits. A failed in-flight batch is not counted as completed; it may be replayed after restart. The supervisor records failures and enforces an additional bounded shutdown fallback if the search does not exit normally.

## Build and future runs

```bash
cargo build --release --locked --manifest-path native/seed-search/Cargo.toml
cargo test --release --locked --manifest-path native/seed-search/Cargo.toml
.venv/bin/python -m pytest tests/test_seed_search.py tests/test_seed_campaign.py tests/test_native_seed_search.py -q
```

The engine is optional; existing Python search remains available. `Cargo.lock` pins the native dependencies. New counted searches can use `--engine rust --workers 14`; they share the existing ledger and cannot run concurrently against the same history. Do not launch another search while this job holds that lock.

For a later supervised run, choose a fresh output directory:

```bash
caffeinate -i .venv/bin/python -u scripts/run_seed_search_job.py \
  --seconds 3600 --workers 14 \
  --out-dir experiments/seed_search_runs/CHOOSE_A_NEW_JOB_DIRECTORY
```

The currently launched job runs independently of this chat and automatically writes its final report. No automatic chat notification has been configured.

Validation before launch: **29 targeted Python tests, five Rust tests, Ruff, and Rust Clippy with warnings denied passed**. The supervisor's finite-duration smoke test also passed and produced final status, results, and candidate-summary files. The pre-existing SciPy binary issue still limits full-project test collection.

## RTX 3090 CUDA implementation and measured pilot

Implemented on 2026-11-06 after the user supplied the Ubuntu machine at
`192.168.50.225` and agreed to C++ with CUDA. The machine has an RTX 3090 with
24 GiB VRAM, an i7-9700K with eight CPU cores, and about 32 GiB system memory.
The pilot originally ran at `/home/timor/ILotto-cuda` on that host. The working
Git checkout is now `/home/timor/Developer/ILotto`, with a local copy of the
toolchain and saved GPU results; the original pilot directory is retained.
A pinned CUDA 12.8
compiler/runtime/header installation is local to that directory; no system
driver or system package was changed. Build and operation instructions are in
`native/cuda-search/README.md`.

The GPU hashes the first date in parallel. Its rare first-date matches return
to the CPU for verification against the remaining 1,656 dates. A persistent
process keeps the CUDA context alive across batches. Python verifies result
structure and independently rechecks any full survivor before committing it.

Actual hardware validation passed **1,003 independent SHA-256 outcome vectors
and 36 batch comparisons**. Local regression checks passed **34 tests**, with
one GPU-only pytest skipped on the Mac; that GPU validation was run directly on
the Ubuntu machine. Ruff passed. The previously reported full-suite SciPy
limitation remains.

All real-data benchmark batches used disjoint, explicitly assigned intervals
beyond the Mac's entire reservation ending at `1,000,000,101,000,000`. Both
machines worked concurrently without modifying the running Mac job. Correctness
fixtures are separate from counted real-data search coverage.

The first allocation contained 100 billion seeds and finished after three
roughly five-second trials. A second, larger allocation allowed all launch
configurations and an additional 15-second confirmation:

| CUDA threads/block | Blocks | Billion seeds/second |
|---|---:|---:|
| 64 | 656 | 6.510 |
| 128 | 656 | 6.631 |
| 256 | 656 | 6.765 |
| 512 | 656 | **6.811** |
| 128 | 1,312 | 6.717 |
| 512, confirmation | 656 | **6.818** |

The confirmation measured 103,084,721,522 seeds in 15.119 seconds, including
Python coordination and checkpoint writes. That is about 38 times the Mac's
earlier warmed rate and extrapolates to **24.55 trillion seeds/hour** if the rate
holds. This is a short confirmation, not a completed hour-long GPU measurement.

The GPU completed **372,494,505,781 unique real-history seed checks** across
both allocations. Of those, 22,954 matched the first draw and **none matched the
second**, so no full-history candidates or future tickets were produced. This
remains a search of an artificial generator, with no demonstrated predictive
advantage.

Compact checkpoints and reports were copied back to:

- `experiments/seed_search_runs/cuda_benchmark_20261106/`
- `experiments/seed_search_runs/cuda_warm_benchmark_20261106/`
- Combined summary: `experiments/seed_search_runs/cuda_summary_20261106.json`

Both GPU benchmark processes have finished. The Mac's original one-hour job
continues independently. The GPU checkpoints are ready for automatic import by
the next `search_seeds.py --count` invocation after the Mac releases its ledger
lock; they are not yet included in the running Mac job's totals. Do not restart
those allocations from a different output directory. The second checkpoint
retains the remainder of its finite reservation for safe continuation.

The initial cross-machine pilot used explicit static assignments. New remote
work now uses `scripts/search_remote_seeds.py`: the Mac reserves ranges under a
shared allocation lock, runs the Ubuntu CUDA executable over SSH, and commits
all progress locally. It excludes entire existing CPU and GPU reservations;
updated CPU campaigns also skip remote reservations. Resume uses an allocation
ID and one canonical checkpoint, with per-allocation locking and a durable
cursor to detect missing/regressed state. The original running Mac job remains
bounded by its already reserved interval. See `native/cuda-search/README.md`
for the new-run, resume, and graceful-stop commands. Arbitrary independent
checkouts are not coordinators; keep allocation state on the original Mac.
