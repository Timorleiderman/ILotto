"""Seed recovery, bounded resumption, and separation of search from evaluation."""

from dataclasses import replace

import pytest

from lotto.seedsearch.generator import (
    OUTCOMES, Observation, index_ticket, search, ticket_index, world_outcome,
)


def prefix(seed=123):
    return [Observation(day, world_outcome(seed, day))
            for day in ("2020-01-01", "2020-01-04", "2020-01-07")]


def test_ticket_rank_round_trip_and_unordered_input():
    for value in (0, 6, 7, 12345, OUTCOMES // 2, OUTCOMES - 1):
        ticket = index_ticket(value)
        assert ticket_index(ticket["balls"][::-1], ticket["strong"]) == value
    with pytest.raises(ValueError):
        ticket_index([1, 1, 2, 3, 4, 5], 1)


def test_generator_version_has_stable_reference_outputs():
    assert world_outcome(0, "2020-01-01") == 15_611_497
    assert world_outcome(45_123, "2020-01-01") == 6_083_258
    assert world_outcome(2**64 - 1, "2026-11-06") == 9_078_729


def test_recovers_included_seed_and_predicts_unseen_dates(tmp_path):
    result = search(prefix(), start=0, stop=256, checkpoint=tmp_path / "state.json")
    assert result["stop_reason"] == "range_complete"
    assert 123 in result["survivors"]
    assert sum(result["matched_prefix_counts"]) == 256
    for seed in result["survivors"]:
        assert world_outcome(seed, "2020-01-10") == world_outcome(123, "2020-01-10")


def test_absent_seed_is_not_reported_as_recovered(tmp_path):
    result = search(prefix(), start=0, stop=100, checkpoint=tmp_path / "state.json")
    assert result["survivors"] == []


def test_resume_matches_uninterrupted_search(tmp_path, monkeypatch):
    import lotto.seedsearch.generator as module

    clock = iter(range(100))
    real_clock = module.time.perf_counter
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    partial = search(prefix(), start=0, stop=256, checkpoint=tmp_path / "resume.json",
                     max_seconds=2, batch_size=64)
    assert partial["stop_reason"] == "time_limit"
    assert partial["next_seed"] == 64
    monkeypatch.setattr(module.time, "perf_counter", real_clock)
    resumed = search(prefix(), start=0, stop=256, checkpoint=tmp_path / "resume.json")
    full = search(prefix(), start=0, stop=256, checkpoint=tmp_path / "full.json")
    for field in ("next_seed", "survivors", "matched_prefix_counts"):
        assert resumed[field] == full[field]


def test_changed_search_history_cannot_reuse_checkpoint(tmp_path):
    path = tmp_path / "state.json"
    history = prefix()
    search(history, start=0, stop=256, checkpoint=path)
    changed = [replace(history[0], outcome=(history[0].outcome + 1) % OUTCOMES), *history[1:]]
    with pytest.raises(ValueError, match="different generator, prefix, or seed range"):
        search(changed, start=0, stop=256, checkpoint=path)


def test_date_order_and_budget_validation(tmp_path):
    with pytest.raises(ValueError, match="chronological"):
        search(prefix()[::-1], start=0, stop=256, checkpoint=tmp_path / "bad.json")
    with pytest.raises(ValueError, match="finite and positive"):
        search(prefix(), start=0, stop=256, checkpoint=tmp_path / "bad.json", max_seconds=float("inf"))


def test_completed_resume_does_not_repeat_seeds(tmp_path):
    path = tmp_path / "state.json"
    first = search(prefix(), start=0, stop=256, checkpoint=path)
    second = search(prefix(), start=0, stop=256, checkpoint=path)
    assert second["matched_prefix_counts"] == first["matched_prefix_counts"]
    assert second["survivors"] == first["survivors"]


def test_interrupted_batch_replays_without_lost_or_duplicate_candidates(tmp_path, monkeypatch):
    import lotto.seedsearch.generator as module

    history = prefix()
    original = module._outcome
    calls = 0

    def interrupted(seed_bytes, date_bytes):
        nonlocal calls
        calls += 1
        if calls == 70:
            raise KeyboardInterrupt
        return original(seed_bytes, date_bytes)

    monkeypatch.setattr(module, "_outcome", interrupted)
    partial = search(history, start=0, stop=256, checkpoint=tmp_path / "resume.json", batch_size=64)
    assert partial["stop_reason"] == "interrupted"
    assert partial["next_seed"] == 64
    monkeypatch.setattr(module, "_outcome", original)
    resumed = search(history, start=0, stop=256, checkpoint=tmp_path / "resume.json")
    full = search(history, start=0, stop=256, checkpoint=tmp_path / "full.json")
    assert resumed["matched_prefix_counts"] == full["matched_prefix_counts"]
    assert resumed["survivors"] == full["survivors"]
