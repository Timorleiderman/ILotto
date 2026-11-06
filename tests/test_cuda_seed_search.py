"""Fail closed on bad GPU output; exercise the live device when explicitly set."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import pytest

from bench.cuda_seed_search import CudaWorker
from bench.native_seed_search import validate_native_result
from bench.seed_search import GENERATOR, Observation, world_outcome


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
    from scripts.benchmark_cuda_search import validate_worker

    with CudaWorker(Path(os.environ["ILOTTO_CUDA_BINARY"])) as worker:
        result = validate_worker(worker)
    assert result["independent_hash_vectors"] == 1003
    assert result["batch_comparisons"] == 36
