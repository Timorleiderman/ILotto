"""Durable range coverage, history identity, and exclusive local execution."""

import json
from pathlib import Path

import pytest

from bench.seed_campaign import campaign_lock, continue_search, inherit_rejections
from bench.seed_search import Observation, search, world_outcome


def history():
    return [Observation(day, world_outcome(123, day))
            for day in ("2020-01-01", "2020-01-04", "2020-01-07")]


def test_import_then_continue_does_not_retest_any_seed(tmp_path, monkeypatch):
    import bench.seed_search as module

    prefix = history()
    old = tmp_path / "legacy.json"
    search(prefix, start=0, stop=100, checkpoint=old)
    original = module._outcome
    seeds = []

    def track(seed, day):
        seeds.append(int.from_bytes(seed, "big"))
        return original(seed, day)

    monkeypatch.setattr(module, "_outcome", track)
    first = continue_search(prefix, root=tmp_path / "ledger", candidates=100, max_seconds=5, imports=[old])
    assert min(seeds) == 100 and max(seeds) == 199
    assert first["total_unique_seeds_tested"] == 200
    assert first["completed_ranges"] == [[0, 200]]
    assert first["survivors"] == [123]
    seeds.clear()
    second = continue_search(prefix, root=tmp_path / "ledger", candidates=100, max_seconds=5, imports=[old])
    assert min(seeds) == 200 and max(seeds) == 299
    assert second["previously_tested"] == 200
    assert second["completed_ranges"] == [[0, 300]]


def test_noncontiguous_imports_fill_gap_without_repeating(tmp_path):
    prefix = history()
    old = tmp_path / "legacy.json"
    search(prefix, start=100, stop=200, checkpoint=old)
    result = continue_search(prefix, root=tmp_path / "ledger", candidates=150, max_seconds=5, imports=[old])
    assert result["previously_tested"] == 100
    assert result["new_seeds_tested"] == 150
    assert result["completed_ranges"] == [[0, 250]]
    assert sum(result["matched_prefix_counts"]) == 250


def test_partial_import_resumes_only_requested_number(tmp_path):
    prefix = history()
    old = tmp_path / "legacy.json"
    search(prefix, start=0, stop=500, checkpoint=old, max_candidates=75)
    result = continue_search(prefix, root=tmp_path / "ledger", candidates=25, max_seconds=5, imports=[old])
    assert result["previously_tested"] == 75
    assert result["new_seeds_tested"] == 25
    assert result["completed_ranges"] == [[0, 100]]
    assert json.loads(old.read_text())["next_seed"] == 75


def test_concurrent_runner_is_rejected(tmp_path):
    path = tmp_path / "search.lock"
    with campaign_lock(path):
        with pytest.raises(ValueError, match="already using"):
            with campaign_lock(path):
                pytest.fail("Second writer acquired an occupied lock")
    with campaign_lock(path):
        pass


def test_overlapping_imports_are_rejected(tmp_path):
    prefix = history()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    search(prefix, start=0, stop=100, checkpoint=a)
    search(prefix, start=50, stop=150, checkpoint=b)
    with pytest.raises(ValueError, match="Overlapping"):
        continue_search(prefix, root=tmp_path / "ledger", candidates=1, max_seconds=5, imports=[a, b])


def test_wrong_history_import_rejected(tmp_path):
    prefix = history()
    old = tmp_path / "legacy.json"
    search(prefix, start=0, stop=100, checkpoint=old)
    with pytest.raises(ValueError, match="different generator or observation prefix"):
        continue_search(prefix[1:], root=tmp_path / "ledger", candidates=1, max_seconds=5, imports=[old])


def test_missing_completed_checkpoint_is_not_silently_researched(tmp_path):
    prefix = history()
    result = continue_search(prefix, root=tmp_path / "ledger", candidates=50, max_seconds=5)
    ledger_path = Path(result["ledger"])
    ledger = json.loads(ledger_path.read_text())
    Path(ledger["ranges"][0]["checkpoint"]).unlink()
    with pytest.raises(ValueError, match="completed work is missing"):
        continue_search(prefix, root=tmp_path / "ledger", candidates=50, max_seconds=5)


def test_corrupt_counts_are_not_imported(tmp_path):
    prefix = history()
    old = tmp_path / "legacy.json"
    search(prefix, start=0, stop=100, checkpoint=old)
    state = json.loads(old.read_text())
    state["matched_prefix_counts"][0] -= 1
    old.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Inconsistent checkpoint"):
        continue_search(prefix, root=tmp_path / "ledger", candidates=50, max_seconds=5, imports=[old])


def test_different_prefix_uses_a_separate_ledger(tmp_path):
    first = continue_search(history(), root=tmp_path / "ledger", candidates=100, max_seconds=5)
    second = continue_search(history()[1:], root=tmp_path / "ledger", candidates=100, max_seconds=5)
    assert first["ledger"] != second["ledger"]
    assert second["previously_tested"] == 0


def test_reserved_range_without_a_checkpoint_can_recover(tmp_path):
    prefix = history()
    result = continue_search(prefix, root=tmp_path / "ledger", candidates=50, max_seconds=5)
    ledger_path = Path(result["ledger"])
    ledger = json.loads(ledger_path.read_text())
    ledger["ranges"].append({
        "start": 50, "stop": 100,
        "checkpoint": str(ledger_path.parent / "range_50_100.json"),
    })
    ledger_path.write_text(json.dumps(ledger))
    resumed = continue_search(prefix, root=tmp_path / "ledger", candidates=25, max_seconds=5)
    assert resumed["previously_tested"] == 50
    assert resumed["completed_ranges"] == [[0, 75]]


def test_append_only_refresh_inherits_rejections_without_repeating(tmp_path):
    old = history()
    new = [*old, Observation("2020-01-10", world_outcome(123, "2020-01-10"))]
    continue_search(old, root=tmp_path, candidates=100, max_seconds=5)
    migration = inherit_rejections(old, new, root=tmp_path)
    assert migration["reused_rejections"] == 100
    resumed = continue_search(new, root=tmp_path, candidates=10, max_seconds=5)
    assert resumed["previously_tested"] == 100
    assert resumed["new_seeds_tested"] == 10
    assert resumed["completed_ranges"] == [[0, 110]]
    # Original history's evidence remains unchanged.
    source = Path(migration["source_ledger"])
    original = json.loads(source.read_text())
    assert original["ranges"][0]["committed_until"] == 100
    with pytest.raises(ValueError, match="already exists"):
        inherit_rejections(old, new, root=tmp_path)


def test_inheritance_rejects_changed_history(tmp_path):
    old = history()
    changed = [Observation(old[0].date, (old[0].outcome + 1) % 16273488), *old[1:],
               Observation("2020-01-10", 0)]
    with pytest.raises(ValueError, match="unchanged, strictly extended"):
        inherit_rejections(old, changed, root=tmp_path)


def test_inheritance_never_silently_discards_survivors(tmp_path):
    old = history()
    continue_search(old, root=tmp_path, candidates=200, max_seconds=5)
    new = [*old, Observation("2020-01-10", 0)]
    with pytest.raises(ValueError, match="Surviving seeds"):
        inherit_rejections(old, new, root=tmp_path)
