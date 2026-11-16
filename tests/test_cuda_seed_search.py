"""Fail closed on bad GPU output; exercise the live device when explicitly set."""

import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from lotto.seedsearch.cuda_engine import CudaWorker
from lotto.seedsearch.generator import GENERATOR, Observation, world_outcome
from lotto.seedsearch.rust_engine import validate_native_result


def fixture():
    history = [Observation("2020-01-01", world_outcome(123, "2020-01-01"))]
    result = {"generator": GENERATOR, "start": "0", "stop": "1003", "tested": 1003,
              "matched_prefix_counts": [1002, 1], "survivors": [123], "survivor_overflow": False}
    return history, result


@pytest.mark.parametrize("corruption", ["counts", "survivor", "overflow", "range"])
def test_bad_cuda_results_are_not_committed(corruption):
    history, result = fixture()
    bad = deepcopy(result)
    if corruption == "counts":
        bad["matched_prefix_counts"][0] -= 1
    elif corruption == "survivor":
        bad["survivors"] = [124]
    elif corruption == "overflow":
        bad["survivor_overflow"] = True
    else:
        bad["stop"] = "1004"
    with pytest.raises(RuntimeError):
        validate_native_result(bad, history, 0, 1003)


def test_stalled_worker_is_killed(tmp_path):
    fake = tmp_path / "stalled-worker"
    fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    fake.chmod(0o700)
    with CudaWorker(fake, timeout=0.05) as worker:
        with pytest.raises(TimeoutError, match="not committed"):
            worker.request("V 1\n0 2020-01-01\n")
        assert worker.process.poll() is not None


@pytest.mark.skipif(not os.environ.get("ILOTTO_CUDA_BINARY"), reason="Requires an explicit CUDA device binary")
def test_live_gpu_matches_independent_reference():
    from lotto.experiments.seed_search_cuda_benchmark import validate_worker

    with CudaWorker(Path(os.environ["ILOTTO_CUDA_BINARY"])) as worker:
        result = validate_worker(worker)
    assert result["independent_hash_vectors"] == 1003
    assert result["batch_comparisons"] == 36


def test_cuda_hash_code_matches_python_on_the_host(tmp_path):
    """Compile the kernel's own hash header as plain C++ and compare with hashlib.

    Covers the midstate precomputation and single-division reduction without a
    GPU. The device build is still checked at worker startup and by
    `validate_worker` on the 3090.
    """
    import random
    import shutil
    import subprocess

    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("No host C++ compiler")
    probe = tmp_path / "probe.cpp"
    probe.write_text("int main() { return 0; }\n")
    if subprocess.run([compiler, str(probe), "-o", str(tmp_path / "probe")],
                      capture_output=True).returncode:
        # e.g. Command Line Tools older than the active macOS SDK; try SDKROOT.
        pytest.skip("Host C++ toolchain cannot link a trivial program")
    source = Path(__file__).resolve().parent.parent / "native/cuda-search/host_check.cpp"
    binary = tmp_path / "host_check"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)

    rng = random.Random(20261106)
    seeds = [0, 1, 45_123, 2**53 + 1, 2**64 - 1] + [rng.getrandbits(64) for _ in range(1995)]
    dates = ["2012-01-03", "2020-02-29", "2026-11-03", "9999-12-31"]
    pairs = [(seed, dates[i % len(dates)]) for i, seed in enumerate(seeds)]
    run = subprocess.run([str(binary)], input="".join(f"{s} {d}\n" for s, d in pairs),
                         capture_output=True, text=True, check=True)
    assert [int(x) for x in run.stdout.split()] == [world_outcome(s, d) for s, d in pairs]

    bad = subprocess.run([str(binary)], input="0 2021-02-29\n", capture_output=True, text=True)
    assert bad.returncode == 2


def _fake_worker(tmp_path, legacy):
    """A stand-in worker speaking the stdin protocol; legacy builds reject "C"."""
    script = tmp_path / ("legacy-worker" if legacy else "new-worker")
    capability = ("        sys.stderr.write('Unknown command\\n'); sys.exit(2)" if legacy else
                  "        print(json.dumps({'generator': 'calendar-sha256-v1', 'protocol': 2,"
                  " 'max_batch': 20_000_000_000}), flush=True)")
    script.write_text(f"""#!{sys.executable}
import json, sys
for line in sys.stdin:
    word = line.split()[0]
    if word == 'C':
{capability}
    elif word == 'V':
        sys.stdin.readline()
        print(json.dumps({{'outcomes': [7]}}), flush=True)
""")
    script.chmod(0o700)
    return script


def test_legacy_worker_is_restarted_and_limited_to_its_old_batch_size(tmp_path):
    from lotto.seedsearch.cuda_engine import LEGACY_MAX_BATCH

    history = [Observation("2020-01-01", world_outcome(123, "2020-01-01"))]
    with CudaWorker(_fake_worker(tmp_path, legacy=True), timeout=5) as worker:
        assert worker.max_batch == LEGACY_MAX_BATCH
        assert worker.vectors([(0, "2020-01-01")]) == [7]  # restarted and usable
        with pytest.raises(ValueError):
            worker.batch(history, 0, LEGACY_MAX_BATCH + 1)


def test_current_worker_reports_its_batch_limit(tmp_path):
    from lotto.seedsearch.cuda_engine import MAX_BATCH

    with CudaWorker(_fake_worker(tmp_path, legacy=False), timeout=5) as worker:
        assert worker.max_batch == MAX_BATCH
        assert worker.capabilities["protocol"] == 2
