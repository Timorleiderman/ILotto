"""Cross-machine reservations and crash-safe continuation on one coordinator."""

import json

import pytest

from bench.seed_allocations import get_remote, reserve_remote
from bench import seed_campaign
from bench.seed_campaign import campaign_lock, continue_search
from bench.seed_search import Observation, _prefix_hash, world_outcome, write_json
from scripts.search_remote_seeds import run_allocation, ssh_command


def history():
    return [Observation("2020-01-01", world_outcome(123, "2020-01-01"))]


def reserve(tmp_path, count=10):
    return reserve_remote(history(), root=tmp_path / "campaigns", run_root=tmp_path,
                          candidates=count, worker={"host": "user@gpu", "directory": "/project"}, source={})


def test_reservation_excludes_whole_mac_and_prior_gpu_ranges(tmp_path):
    root = tmp_path / "campaigns"
    continue_search(history(), root=root, candidates=50, max_seconds=10)
    ledger_path = root / _prefix_hash(history()) / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["ranges"].append({"start": 50, "stop": 500, "checkpoint": "reserved-not-completed"})
    write_json(ledger_path, ledger)
    write_json(tmp_path / "old_gpu/allocation.json", {"generator": "calendar-sha256-v1",
                                                     "start": 500, "stop": 700})
    first, second = reserve(tmp_path), reserve(tmp_path)
    assert (first["start"], first["stop"]) == (700, 710)
    assert (second["start"], second["stop"]) == (710, 720)
    assert get_remote(root, first["id"]) == first


def test_cpu_skips_remote_reservation_including_unfinished_tail(tmp_path):
    root = tmp_path / "campaigns"
    continue_search(history(), root=root, candidates=100, max_seconds=10)
    allocation = reserve(tmp_path, 100)
    assert (allocation["start"], allocation["stop"]) == (100, 200)
    result = continue_search(history(), root=root, candidates=100, max_seconds=10)
    assert result["completed_ranges"] == [[0, 100], [200, 300]]
    assert result["new_seeds_tested"] == 100


def test_allocation_lock_refuses_racing_reservations(tmp_path, monkeypatch):
    monkeypatch.setattr(seed_campaign, "ALLOCATION_WAIT_SECONDS", 0.2)
    with campaign_lock(tmp_path / "campaigns/allocation.lock"):
        with pytest.raises(ValueError, match="already using"):
            reserve(tmp_path)


def test_allocation_lock_waits_for_a_momentary_holder(tmp_path):
    """A reservation racing a brief holder must wait, not crash a long search."""
    import threading

    lock = tmp_path / "campaigns/allocation.lock"
    held, release = threading.Event(), threading.Event()

    def hold():
        with campaign_lock(lock):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    threading.Timer(0.3, release.set).start()
    allocation = reserve(tmp_path)  # blocks ~0.3 s, then succeeds
    holder.join()
    assert allocation["stop"] > allocation["start"]


class FakeWorker:
    def __init__(self, calls, fail_after=None):
        self.calls, self.fail_after = calls, fail_after

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        pass

    def batch(self, prefix, start, stop, **unused):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("Lost connection")
        self.calls.append((start, stop))
        # Coordinator bookkeeping fixture; real CUDA/hash agreement is tested separately.
        return {"matched_prefix_counts": [stop-start, 0], "survivors": []}


def test_disconnect_resume_and_completed_rerun_never_repeat_committed_work(tmp_path):
    allocation = reserve(tmp_path, 1_000_010)
    calls = []
    with pytest.raises(RuntimeError, match="Lost connection"):
        run_allocation(allocation, run_root=tmp_path, seconds=10,
                       worker_factory=lambda: FakeWorker(calls, fail_after=1), validator=lambda worker: {})
    assert calls == [(0, 1_000_000)]
    result = run_allocation(allocation, run_root=tmp_path, seconds=10,
                            worker_factory=lambda: FakeWorker(calls), validator=lambda worker: {})
    assert calls == [(0, 1_000_000), (1_000_000, 1_000_010)]
    assert result["new_seeds_tested"] == 10
    result = run_allocation(allocation, run_root=tmp_path, seconds=10,
                            worker_factory=lambda: pytest.fail("Completed work contacted GPU"))
    assert result["new_seeds_tested"] == 0
    assert result["status"] == "already_complete"


def test_deleted_checkpoint_fails_instead_of_restarting(tmp_path):
    allocation = reserve(tmp_path)
    run_allocation(allocation, run_root=tmp_path, seconds=10,
                   worker_factory=lambda: FakeWorker([]), validator=lambda worker: {})
    (tmp_path / "remote_workers" / allocation["id"] / "checkpoint.json").unlink()
    with pytest.raises(ValueError, match="missing or regressed"):
        run_allocation(allocation, run_root=tmp_path, seconds=10,
                       worker_factory=lambda: pytest.fail("Must fail before starting GPU"))


def test_one_writer_per_remote_allocation(tmp_path):
    allocation = reserve(tmp_path)
    with campaign_lock(tmp_path / "remote_workers" / allocation["id"] / "worker.lock"):
        with pytest.raises(ValueError, match="already using"):
            run_allocation(allocation, run_root=tmp_path, seconds=10)


def test_ssh_host_cannot_inject_options_and_path_is_quoted():
    with pytest.raises(ValueError):
        ssh_command({"host": "-oProxyCommand=anything", "directory": "/tmp"})
    command = ssh_command({"host": "user@gpu", "directory": "/tmp/path with spaces"})
    assert command[-1] == "exec '/tmp/path with spaces/native/cuda-search/ilotto-cuda-search'"


def test_deleted_registry_cannot_allocate_over_existing_remote_history(tmp_path):
    allocation = reserve(tmp_path)
    run_allocation(allocation, run_root=tmp_path, seconds=10,
                   worker_factory=lambda: FakeWorker([]), validator=lambda worker: {})
    (tmp_path / "campaigns/remote_allocations.json").unlink()
    with pytest.raises(ValueError, match="registry is missing"):
        reserve(tmp_path)
