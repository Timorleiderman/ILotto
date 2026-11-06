"""Independent Python/Rust agreement before ledger coverage can be reused."""

import json

import pytest

from bench.native_seed_search import DEFAULT_BINARY, RustSearch, native_batch
from bench.seed_campaign import continue_search
from bench.seed_search import Observation, search, world_outcome

pytestmark = pytest.mark.skipif(not DEFAULT_BINARY.is_file(), reason="Build Rust release binary first")


def prefix():
    return [Observation(day, world_outcome(123, day))
            for day in ("2020-01-01", "2020-01-04", "2020-01-07")]


@pytest.mark.parametrize("workers", [1, 4, 14])
def test_rust_matches_python_for_uneven_range(tmp_path, workers):
    history = prefix()
    python = search(history, start=0, stop=1003, checkpoint=tmp_path / "python.json")
    rust = native_batch(history, 0, 1003, workers)
    assert rust["matched_prefix_counts"] == python["matched_prefix_counts"]
    assert rust["survivors"] == python["survivors"]


def test_rust_and_python_resume_the_same_checkpoint(tmp_path):
    path = tmp_path / "state.json"
    history = prefix()
    search(history, start=0, stop=500, max_candidates=100, checkpoint=path)
    rust = RustSearch(4)(history, start=0, stop=500, max_candidates=200, checkpoint=path)
    assert rust["next_seed"] == 300
    result = search(history, start=0, stop=500, checkpoint=path)
    full = search(history, start=0, stop=500, checkpoint=tmp_path / "full.json")
    assert result["matched_prefix_counts"] == full["matched_prefix_counts"]
    assert result["survivors"] == [123]


def test_native_continuation_preserves_unique_coverage(tmp_path):
    history = prefix()
    continue_search(history, root=tmp_path, candidates=100, max_seconds=5)
    result = continue_search(history, root=tmp_path, candidates=200, max_seconds=5, search_impl=RustSearch(4))
    assert result["previously_tested"] == 100
    assert result["new_seeds_tested"] == 200
    assert result["completed_ranges"] == [[0, 300]]
    assert result["survivors"] == [123]


def test_stop_file_prevents_any_new_work(tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    path = tmp_path / "state.json"
    result = RustSearch(4, stop_file=stop)(prefix(), start=0, stop=500, checkpoint=path)
    assert result["stop_reason"] == "user_stop"
    assert json.loads(path.read_text())["next_seed"] == 0


def test_rust_supports_last_unsigned_seed():
    day = "2026-11-06"
    result = native_batch([Observation(day, world_outcome(2**64 - 1, day))], 2**64 - 1, 2**64, 4)
    assert result["survivors"] == [2**64 - 1]
