# CUDA seed-search worker

C++17/CUDA implementation of the existing artificial `calendar-sha256-v1`
generator, targeted at the RTX 3090 (`sm_86`). This is an experimental search
engine, not a demonstrated predictor of lottery results.

The working Ubuntu checkout is `/home/timor/Developer/ILotto`. Its local CUDA
toolchain lives under `.cuda-12.8/`, and saved search runs live under
`experiments/seed_search_runs/`. Both are ignored by Git. The original pilot
directory `/home/timor/ILotto-cuda` is retained as a backup.

The GPU computes the exact SHA-256 outcome for the first observed date. Only
matching seeds return to the CPU, which verifies all remaining dates. Most
seeds reject on the first date. A persistent worker avoids recreating the CUDA
context between batches. Each GPU thread visits a disjoint sequence of seeds;
only first-draw matches perform an atomic append. The bounded output buffer
fails closed on overflow. SHA-256 rejection sampling matches Python and Rust,
including the unsigned 64-bit boundary.

## Build on the NVIDIA Linux machine

The bootstrap downloads about 81 MB of NVIDIA compiler/runtime/header archives,
checks pinned SHA-256 digests, and extracts them into a project-local directory.
It requires Python 3.12+ and an installed host C++ compiler. It does not install
drivers or change system packages. The files and hashes come from NVIDIA's
[CUDA 12.8.1 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
The component license files are retained with the installation.

```bash
python3 native/cuda-search/bootstrap.py .cuda-12.8
bash native/cuda-search/build.sh
uv run lotto run seed-search-cuda-benchmark \
  --binary native/cuda-search/ilotto-cuda-search \
  --validate-only --out-dir results/validation
```

For an existing toolkit, set `ILOTTO_CUDA_ROOT` to its root directory before
building. The build uses the existing NVIDIA driver. NVIDIA documents the
distinction between the driver-supported CUDA version reported by `nvidia-smi`
and the installed toolkit in its [installation prerequisites](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html).

## Correctness and finite benchmarks

The validation driver checks 1,003 GPU outcomes against Python's independent
`hashlib` implementation, plus 36 complete batch comparisons. These exercise
four launch sizes, uneven intervals, seeds above 2^53 and at 2^64−1, partial
historical matches, and full survivors. The native executable also checks
known hashes and the rejection-mapping boundary at startup. Every full
survivor is independently rechecked in Python before committing coverage.

```bash
uv run lotto run seed-search-cuda-benchmark \
  --binary native/cuda-search/ilotto-cuda-search \
  --manifest results/allocation.json --out-dir results/benchmark \
  --seconds-per-config 5 --warm-seconds 15
```

The allocation manifest supplies `generator`, `prefix_sha256`, the chronological
`prefix` list of date/outcome pairs, integer `start`/`stop` bounds, and source
provenance. The coordinator must reserve that interval before running this
command. The driver does **not** allocate ranges or coordinate across machines.
Do not reuse an allocation in another output directory. Reusing an unfinished
output directory resumes its completed cursor; a finished output is refused.
Create `STOP` inside the output directory for a graceful stop between batches.

The initial pilot uses explicit, finite machine assignments beyond the Mac's
entire reserved interval. A local file lock prevents competing writers to the
same remote output. Atomic checkpoints save the completed cursor, counts, and
rare survivors, with no per-seed history. An interrupted uncommitted batch can
replay. After the worker stops, copy its checkpoint back and import it through
the existing campaign ledger. Do not import a checkpoint while its remote
worker is still advancing. Do not run two machines against copies of one local
ledger: its lock is not a distributed allocator.

The Python benchmark entry point runs without third-party Python packages. The
normal `lotto run seed-search --engine rust` path remains CPU-only. The benchmark
driver remains useful for explicitly allocated offline pilots. For new shared
work, use the coordinator below.

## Run new GPU work without overlapping the Mac

Run this **on the Mac, from the original checkout with its existing history**:

```bash
uv run lotto run seed-search-remote \
  --host timor@192.168.50.225 \
  --remote-dir /home/timor/Developer/ILotto \
  --count 30000000000 --seconds 10
```

It reserves 30 billion new seeds beyond every Mac reservation, both earlier
GPU pilot reservations, and every new remote allocation. Ubuntu executes the
GPU batches over one persistent SSH connection. The Mac owns the checkpoints
and the registry at `experiments/seed_search_runs/campaigns/remote_allocations.json`.
Git synchronizes code; it does not synchronize this ignored runtime state.
Keep this Mac as the sole coordinator rather than launching the command from a
fresh checkout or using a different `--run-root` to bypass history.

The command prints an allocation ID. If the time limit or a connection failure
interrupts it, resume the same allocation:

```bash
uv run lotto run seed-search-remote --resume ALLOCATION_ID --seconds 10
```

Resuming starts at the next committed seed. A completed allocation returns
`already_complete` without contacting the GPU. Locks reject concurrent runs of
the same allocation. Missing or regressed checkpoints fail rather than silently
starting over. Future Mac CPU campaigns skip the entire remote reservation,
including any unfinished tail. Remote checkpoints remain coordinator-owned and
must not be manually imported into the CPU ledger.

Progress lives in `experiments/seed_search_runs/remote_workers/ALLOCATION_ID/`:
`status.json`, `checkpoint.json`, `committed.json`, and the frozen `manifest.json`.
Create `STOP` in that directory to stop after the current batch; remove it before
resuming. The time budget applies to searching after correctness validation.
Only a batch interrupted before its result is committed can replay. Synthetic
correctness fixtures intentionally repeat and are not counted as new historical
search coverage. Explicit `--start/--stop` legacy tools do not consult this registry;
use the coordinated runner and normal `--count` CPU campaigns for shared work.
