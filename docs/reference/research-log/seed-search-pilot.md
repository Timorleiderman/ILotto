# Local seed-search pilot — 2026-11-06

The inspected Mac has an Apple M3 Max, 36 GiB of unified memory, and 14 logical CPU cores. The pilot ran in a single Python process without a GPU, extra packages, network access, or archive refresh. This machine is sufficient for this initial bounded experiment. Large neural-model training is a separate, unmeasured workload.

## What was implemented

`bench/seed_search.py` defines an artificial generator, `calendar-sha256-v1`. A fixed domain identifier, an unsigned 64-bit seed, and a canonical calendar date map through SHA-256 to one of the 16,273,488 valid complete lottery outcomes. Rejection mapping avoids modulo bias under a uniform-hash assumption. A combinatorial rank maps each outcome to six distinct numbers and a strong number.

This is a reproducible candidate-world family, not a physical model of the lottery or evidence that the lottery uses SHA-256. Unlike searching independent seed/state assignments per draw, the same seed must match every date in the matching prefix.

Search rejects a candidate at its first mismatch, keeps all full-prefix survivors up to a configured cap, and checkpoints completed batches. Every run has a seed-range bound and time budget. Resuming requires the same generator, snapshot, split, and range. The program does not place tickets or start background work.

## Measured runs

| Run | Seeds searched | Matching history | Withheld draws | Survivors | Search-loop time |
|---|---:|---:|---:|---:|---:|
| Known artificial generator | 65,536 | 8 | 4 | 1 | 0.0433 seconds |
| Real archive, retrospective | 1,000,000 | 1,628 | 4 | 0 | 0.7646 seconds |

The artificial run recovered seed 45,123; its search function received only observed dates and outcomes, not that seed. The surviving candidate correctly reproduced all four withheld artificial draws. The fixture seed is public in the script for reproducibility; this is an engineering control, not a blinded discovery claim.

The real-data snapshot contains 1,632 modern-format draws ending on 2026-08-04. Search used the first 1,628 and reserved the last four. Zero seeds survived even the first matching draw in this searched range, so there were no candidate continuation predictions to score. There is no demonstrated real predictive advantage and no upcoming-draw forecast.

The real run measured approximately **1.31 million candidate seeds per second**, including batch checkpoint writes but excluding Python startup, archive loading, and report generation. This short measurement is dominated by first-draw rejection: it does not mean 1.31 million complete 1,628-draw worlds per second. Sustained performance, other generators, longer surviving prefixes, and parallel scaling require separate measurements.

At that observed short-run rate, one billion candidates would take about 13 minutes if throughput stayed unchanged. This is an extrapolation, not a measured billion-candidate run. A 64-bit space would take roughly 447,000 years at the same rate; buying a faster machine would not make arbitrary exhaustive search practical.

The initial recommendation is to continue bounded CPU experiments on this Mac, measure candidate-family quality and sustained speed, and delay any hardware purchase. Streaming candidates avoids a giant stored-world dataset. No extra hardware is needed to reproduce these results.

## Reproduce and resume

For ongoing work, use the shared non-repetition ledger described in `experiments/seed_search_campaign.md`: `--mode real --count 100000000 --max-seconds 120`. It imports the original pilot and skips its completed seeds. The explicit-range commands below document the original pilot; they do not themselves coordinate with that shared ledger.

From the repository root, using its existing environment:

```bash
.venv/bin/python scripts/search_seeds.py --mode toy --out-dir experiments/seed_search_runs/toy_65536
.venv/bin/python scripts/search_seeds.py --mode real --stop 1000000 --max-seconds 20 --out-dir experiments/seed_search_runs/real_1000000
```

Use a different output directory for a different range or snapshot. Repeating a completed experiment resumes at its completed boundary and does not search the same seeds again. To continue with a new finite range, set both `--start` and `--stop` and choose a new directory. A timeout or keyboard interruption preserves completed batches; an interrupted partial batch is safely replayed.

Each directory contains a source manifest, checkpoint, withheld prediction file, and report. Those generated files are excluded from version control. The saved withheld predictions are retrospective artifacts; writing them before the local scoring step does not turn already-known historical outcomes into prospective evidence.

## Validation

The new tests cover ticket ranking, versioned generator reference outputs, included-seed recovery and continuation, absent seeds, timeout/resumption, interrupted-batch replay, checkpoint identity, input validation, and completed-run idempotence. Lint checks cover `bench`, `scripts`, and `tests`.

The full project test suite was attempted but could not collect its existing SciPy-dependent tests. macOS rejected the installed `_spropack.cpython-312-darwin.so` with a Mach-O `__DATA/__thread_bss` section error. The pilot tests and search do not depend on that binary. The local dependency environment has not been modified to repair this separate issue.

The broader design still requires wrong-generator controls, repeated null experiments, multiple candidate families, formal chronological model selection, and predictions committed before genuinely future draws. These pilot results do not satisfy those research milestones.
