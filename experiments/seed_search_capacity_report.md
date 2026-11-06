# Seed-search capacity report — 2026-11-06

Baseline numbers for the `calendar-sha256-v1` seed search, so later speed-ups
and hardware changes can be measured against them. All rates are measured
unless marked otherwise.

## Target

| Item | Value |
|---|---:|
| Real draws in the archive (2012-01-03 to 2026-11-03) | 1,661 |
| Draws a seed must match (matching prefix) | **1,657** |
| Draws held back to check any survivor | 4 |
| Possible outcomes per draw, `N` = C(37,6) × 7 | 16,273,488 (23.96 bits) |
| Seed space, `S` | 2⁶⁴ = 1.845 × 10¹⁹ |

Only the 1,657 modern-format draws matter; the older-format archive is not used.

## Current speed

| Engine | Seeds/second | Per hour | Per day | Per year | Source |
|---|---:|---:|---:|---:|---|
| Mac, Rust, 14 threads, before 2026-11-06 fix | 182 million | 0.66 trillion | 15.7 trillion | 5.7 quadrillion | A/B run, 1e9 seeds |
| **Mac, Rust, 14 threads, current build** | **237 million** | 0.85 trillion | 20 trillion | 7.5 quadrillion | A/B run, 1e9 seeds |
| **RTX 3090, CUDA, old build, end-to-end over SSH** | **6.66 billion** | 24 trillion | 575 trillion | 210 quadrillion | last 3 remote runs |
| RTX 3090, CUDA, kernel only, old build | 6.82 billion | — | — | — | 103e9 seeds in 15.1 s |
| RTX 3090, CUDA, current build | *not yet measured* | | | | rebuild and benchmark |
| **Both machines together** | **6.9 billion** | 25 trillion | 596 trillion | 218 quadrillion | sum of the two |

Mac hardware: Apple M3 Max, 14 cores, 36 GiB. GPU host: RTX 3090 24 GiB,
i7-9700K, 32 GiB. Mature SHA-256 GPU code reaches roughly 9–10 billion
hashes/s on a 3090, so the GPU has some headroom left.

## Coverage so far

| Item | Value |
|---|---:|
| Distinct seeds tested | 952,605,319,763 |
| On the Mac | 490,110,813,982 |
| On the 3090 | 462,494,505,781 |
| Share of 2⁶⁴ | 5.2 × 10⁻⁸ (0.0000052%) |
| Seeds matching draw 1 | consistent with chance (~58,500 expected) |
| Seeds matching draws 1–2 | 0 (0.0036 expected) |
| Seeds matching all 1,657 | 0 |

## Time to search the whole 64-bit space

| Engine | Coverage per year | Time for all 2⁶⁴ seeds |
|---|---:|---:|
| Mac alone | 0.041% | 2,466 years |
| 3090 alone | 1.14% | 88 years |
| Both | 1.18% | about 85 years |

## Time to find one seed matching the first k draws

At 6.9 billion seeds/second (both machines):

| Draws matched | Seeds to check per match (`N`ᵏ) | Time per match | Matching seeds expected in all of 2⁶⁴ |
|---:|---:|---:|---:|
| 1 | 1.6 × 10⁷ | 0.0024 seconds | 1.1 × 10¹² |
| 2 | 2.6 × 10¹⁴ | 10.7 hours | about 70,000 |
| 3 | 4.3 × 10²¹ | about 20,000 years | 0.004 |
| 5 | 1.1 × 10³⁶ | 5.2 × 10¹⁸ years | 10⁻¹⁷ |
| 10 | 1.3 × 10⁷² | 6.0 × 10⁵⁴ years | 10⁻⁵³ |
| **1,657** | 10^11,949 | **10^11,932 years** | **10^-11,930** |

The universe is about 1.4 × 10¹⁰ years old.

## Formulas for updating these numbers

With `r` = seeds/second, `N` = 16,273,488, `S` = 2⁶⁴, `k` = draws matched:

```text
seeds per match       = N^k
seconds per match     = N^k / r
matches in all of S   = S / N^k
years to exhaust S    = (S - seeds_done) / r / 31,557,600
log10(N)              = 7.2115     (each matched draw adds 7.2 zeros)
```

A speed-up of `x` subtracts `log10(x)` zeros from every time above. Ten times
faster removes one zero; a million times faster removes six.

## What improvements change

| Improvement | Effect on time per match | Effect on the 1,657-draw row |
|---|---|---|
| Faster code on the same machines (up to ~1.5×) | divide by ~1.5 | 10^11,932 years |
| Ten RTX 3090s | divide by 10 | 10^11,931 years |
| Every computer on Earth (~10¹² × this setup) | divide by ~10¹² | 10^11,920 years |
| Fewer draws required | multiplies by `N` less per draw | 3 draws is already beyond the 64-bit space |
| Larger seed (e.g. 40,000 bits) | none | matches exist but still take `N`¹⁶⁵⁷ hashes to find |
| Invertible generator (seed encodes the draws) | instant | matches all 1,657 in 0.13 s; every next draw equally covered |

The last two rows were demonstrated on 2026-11-06. An invertible generator with
a 39,791-bit seed reproduced all 1,661 draws without search, and 16,273,488
different extensions of that seed matched every draw while each predicted a
different next draw.

## Engineering changes in this baseline (uncommitted as of writing)

- Rust: one 64-bit division per check instead of sixteen; compile-time
  constants; strict calendar-date validation. +30% at 14 threads, identical results.
- CUDA: single-division reduction; SHA-256 rounds 0–3 precomputed per date;
  hash code moved to `native/cuda-search/sha_outcome.h` and cross-checked on the
  host against Python (2,000 seeds). Batches up to 2 × 10¹⁰ seeds, ~2 s target.
- CUDA worker answers a capability query (`C`); an old, unrebuilt binary is
  restarted and limited to 10⁹-seed batches instead of failing.
- Campaign: the allocation lock waits up to 30 s instead of failing at once,
  and the time budget covers lock waits.
- CI builds and tests the Rust engine; SciPy 1.15.2 → 1.17.1 so the full suite
  runs on macOS 27.

## Next measurement

Rebuild the 3090 worker (`bash native/cuda-search/build.sh`), run the
validation and benchmark from `native/cuda-search/README.md`, and fill in the
"current build" row of the speed table.
