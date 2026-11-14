"""Local, single-writer range ledger for non-overlapping seed searches.

Completed work lives in validated checkpoints; the ledger reserves their ranges.
Only the current uncommitted search batch can replay after a process crash.
"""

from __future__ import annotations

import fcntl
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

from lotto.seedsearch.allocations import remote_ranges
from lotto.seedsearch.generator import (
    GENERATOR,
    MAX_SEED,
    Observation,
    prefix_hash,
    search,
    validate_checkpoint,
    write_json,
)

#: How long to wait for the short-lived allocation lock. Holders keep it only
#: while reading or writing small range files, so contention is momentary; a
#: long-running search must not crash because another machine reserved a range
#: at the same moment. Search and worker locks still fail immediately.
ALLOCATION_WAIT_SECONDS = 30.0


class CampaignLockTimeout(ValueError):
    """The ledger lock remained occupied for its entire wait budget."""


@contextmanager
def campaign_lock(path: Path, wait: float = 0.0):
    """Exclusive advisory lock. `wait=0` refuses at once; otherwise poll until `wait` s."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CampaignLockTimeout(
                        "Another search is already using this ledger; no work was started") from exc
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read(path: Path, prefix: list[Observation]) -> dict:
    state = json.loads(path.read_text())
    validate_checkpoint(state, prefix)
    return state


def _reserve(ledger: dict, path: Path, start: int, stop: int) -> bool:
    path = str(path.resolve())
    for record in ledger["ranges"]:
        if record["checkpoint"] == path:
            if (record["start"], record["stop"]) != (start, stop):
                raise ValueError("Registered checkpoint range has changed")
            return False
        if max(start, record["start"]) < min(stop, record["stop"]):
            raise ValueError("Overlapping historical ranges found; refusing to double-count or overwrite them")
    ledger["ranges"].append({"start": start, "stop": stop, "checkpoint": path})
    ledger["ranges"].sort(key=lambda record: record["start"])
    return True


def inherit_rejections(
    old_prefix: list[Observation], new_prefix: list[Observation], *, root: Path,
) -> dict:
    """Carry rejected seeds into a strictly append-only history without replay.

    This narrow migration refuses surviving candidates: they need explicit
    evaluation against the suffix, not blind reuse. Both ledgers remain auditable.
    """
    old_hash, new_hash = prefix_hash(old_prefix), prefix_hash(new_prefix)
    if len(new_prefix) <= len(old_prefix) or new_prefix[:len(old_prefix)] != old_prefix:
        raise ValueError("Rejection inheritance requires an unchanged, strictly extended prefix")
    root = Path(root).resolve()
    source, destination = root / old_hash, root / new_hash
    # Lock order follows increasing history length; extension cannot form a cycle.
    with campaign_lock(source / "search.lock"), campaign_lock(destination / "search.lock"):
        source_ledger_path = source / "ledger.json"
        old = json.loads(source_ledger_path.read_text())
        if old["identity"] != {"schema": 1, "generator": GENERATOR, "prefix_sha256": old_hash}:
            raise ValueError("Source ledger identity mismatch")
        target = destination / "ledger.json"
        if target.exists():
            raise ValueError("Destination ledger already exists; refusing to overwrite coverage")
        prepared = []
        reused = 0
        for record in old["ranges"]:
            path = Path(record["checkpoint"])
            if not path.exists() and record.get("committed_until", record["start"]) == record["start"]:
                continue
            state = _read(path, old_prefix)
            if ((state["identity"]["start"], state["identity"]["stop"]) != (record["start"], record["stop"])
                    or state["next_seed"] < record.get("committed_until", record["start"])):
                raise ValueError("Source checkpoint range has changed or regressed")
            if state["survivors"]:
                raise ValueError("Surviving seeds must be evaluated against new draws before inheritance")
            imported = {
                **state,
                "identity": {**state["identity"], "prefix_sha256": new_hash, "prefix_draws": len(new_prefix)},
                "matched_prefix_counts": state["matched_prefix_counts"] + [0] * (len(new_prefix) - len(old_prefix)),
                "inherited_rejections": {
                    "source_checkpoint": str(path), "source_prefix_sha256": old_hash,
                    "reason": "Every tested seed already mismatched the unchanged earlier prefix",
                },
            }
            validate_checkpoint(imported, new_prefix)
            prepared.append((record, imported))
            reused += state["next_seed"] - record["start"]
        ledger = {
            "identity": {"schema": 1, "generator": GENERATOR, "prefix_sha256": new_hash},
            "ranges": [], "imports": {},
            "inheritance": {"source_ledger": str(source_ledger_path), "reused_rejections": reused,
                            "old_prefix_draws": len(old_prefix), "new_prefix_draws": len(new_prefix)},
        }
        for record, state in prepared:
            start, stop = record["start"], record["stop"]
            path = destination / f"range_{start}_{stop}.json"
            _reserve(ledger, path, start, stop)
            next(r for r in ledger["ranges"] if r["checkpoint"] == str(path))[
                "committed_until"
            ] = state["next_seed"]
            if path.exists() and json.loads(path.read_text()) != state:
                raise ValueError("Unregistered destination file differs; refusing to overwrite it")
            write_json(path, state)
        write_json(target, ledger)
        return {**ledger["inheritance"], "ledger": str(target)}


def _load_ledger(ledger_path: Path, directory: Path, identity: dict) -> dict:
    """Read this prefix's ledger (or start an empty one) and check its ranges are sane."""
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "identity": identity, "ranges": [], "imports": {},
    }
    if ledger["identity"] != identity:
        raise ValueError("Ledger identity mismatch")
    previous_stop = 0
    for record in ledger["ranges"]:
        start, stop = record["start"], record["stop"]
        expected_path = directory / f"range_{start}_{stop}.json"
        if (not isinstance(start, int) or not isinstance(stop, int)
                or not previous_stop <= start < stop <= MAX_SEED
                or record["checkpoint"] != str(expected_path)
                or not start <= record.get("committed_until", start) <= stop):
            raise ValueError("Invalid or overlapping ledger ranges")
        previous_stop = stop
    return ledger


def _import_checkpoints(ledger: dict, ledger_path: Path, directory: Path, root: Path,
                        prefix: list[Observation], imports, out_of_time) -> None:
    """Copy finished checkpoints from elsewhere into this ledger, refusing any overlap."""
    try:
        with campaign_lock(Path(root) / "allocation.lock",
                           wait=min(ALLOCATION_WAIT_SECONDS, max(0, out_of_time.remaining()))):
            for source in imports:
                if out_of_time():
                    break
                source = Path(source).resolve()
                if str(source) in ledger["imports"]:
                    continue
                state = _read(source, prefix)
                start, stop = state["identity"]["start"], state["identity"]["stop"]
                destination = directory / f"range_{start}_{stop}.json"
                # Check overlap before writing; never overwrite an already-advanced range.
                if any(r["checkpoint"] == str(destination) for r in ledger["ranges"]):
                    raise ValueError("Import duplicates an existing range from another source")
                if any(max(start, r["start"]) < min(stop, r["stop"]) for r in remote_ranges(root)):
                    raise ValueError("Remote reservations remain owned by the coordinator; do not import them")
                _reserve(ledger, destination, start, stop)
                _record(ledger, destination)["committed_until"] = state["next_seed"]
                if destination.exists():
                    if _read(destination, prefix) != state:
                        raise ValueError("Unregistered import file differs; manual inspection required")
                else:
                    write_json(destination, state)
                ledger["imports"][str(source)] = str(destination)
                write_json(ledger_path, ledger)
    except CampaignLockTimeout:
        if not out_of_time():
            raise
        # Out of time while waiting: the caller returns the normal time-limit summary.


def _load_states(ledger: dict, prefix: list[Observation]) -> dict[str, dict]:
    """Every registered checkpoint, keyed by path, checked against the ledger's cursors.

    A missing checkpoint is fine if no work was committed to it (it was reserved
    just before a crash); a missing or regressed one with committed work is not.
    """
    states = {}
    for record in ledger["ranges"]:
        path = Path(record["checkpoint"])
        if path.exists():
            state = _read(path, prefix)
            if (state["identity"]["start"], state["identity"]["stop"]) != (
                record["start"], record["stop"]
            ) or state["next_seed"] < record.get("committed_until", record["start"]):
                raise ValueError("Registered checkpoint has changed or regressed")
            states[str(path)] = state
        elif record.get("committed_until", record["start"]) > record["start"]:
            raise ValueError("A checkpoint containing completed work is missing")
    return states


def _totals(states: dict[str, dict], n_prefix: int) -> tuple[list[int], list[int], list[list[int]]]:
    """Summed match counts, all survivors, and the merged list of completed seed ranges."""
    counts = [0] * (n_prefix + 1)
    survivors: list[int] = []
    completed: list[list[int]] = []
    for state in sorted(states.values(), key=lambda item: item["identity"]["start"]):
        counts = [a + b for a, b in zip(counts, state["matched_prefix_counts"])]
        survivors.extend(state["survivors"])
        start, cursor = state["identity"]["start"], state["next_seed"]
        if cursor > start:
            if completed and completed[-1][1] == start:
                completed[-1][1] = cursor
            else:
                completed.append([start, cursor])
    return counts, survivors, completed


def _record(ledger: dict, checkpoint) -> dict:
    """The ledger entry for a checkpoint path."""
    return next(r for r in ledger["ranges"] if r["checkpoint"] == str(checkpoint))


def _next_range(ledger: dict, ledger_path: Path, directory: Path, root: Path,
                states: dict[str, dict], budget: int) -> dict | None:
    """The lowest unfinished local range, or a newly reserved one; None if 2**64 is used up.

    Walks the seed line from 0 over local and remote ranges in order. The first
    local range with seeds left is resumed. If everything up to the first gap is
    finished, a new range of up to `budget` seeds is reserved in that gap. Must be
    called while holding the allocation lock.
    """
    coverage = [dict(r, remote=False) for r in ledger["ranges"]]
    coverage.extend(dict(r, remote=True) for r in remote_ranges(root))
    coverage.sort(key=lambda r: (r["start"], not r["remote"]))
    frontier = 0
    for record in coverage:
        if record["stop"] <= frontier:
            continue
        if frontier < record["start"]:
            break
        if not record["remote"]:
            state = states.get(record["checkpoint"])
            cursor = state["next_seed"] if state else record["start"]
            if cursor < record["stop"]:
                return _record(ledger, record["checkpoint"])
        frontier = record["stop"]
    if frontier == MAX_SEED:
        return None
    next_start = min((r["start"] for r in coverage if r["start"] > frontier), default=MAX_SEED)
    stop = min(frontier + budget, next_start)
    path = directory / f"range_{frontier}_{stop}.json"
    _reserve(ledger, path, frontier, stop)
    # Serialize this write with remote allocations, not remote work.
    write_json(ledger_path, ledger)
    return _record(ledger, path)


class _Deadline:
    """Calling it says whether the time budget is spent; `remaining()` gives seconds left."""

    def __init__(self, seconds: float):
        self.began, self.seconds = time.perf_counter(), seconds

    def remaining(self) -> float:
        return self.seconds - (time.perf_counter() - self.began)

    def __call__(self) -> bool:
        return self.remaining() <= 0


def continue_search(
    prefix: list[Observation], *, root: Path, candidates: int, max_seconds: float,
    imports=(), batch_size: int = 4096, search_impl=None,
) -> dict:
    """Search at most `candidates` NEW seeds, filling gaps and resuming ranges.

    All cooperating runners must use this same root. Explicit imports must match
    the generator and prefix, and are copied so legacy output stays unchanged.

    The steps: load the ledger, import finished checkpoints, then repeatedly
    pick the next range (resume or reserve) and search it until the candidate
    or time budget runs out. The time budget covers the whole call, lock waits
    included.
    """
    if not isinstance(candidates, int) or candidates <= 0:
        raise ValueError("candidates must be a positive integer")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    out_of_time = _Deadline(max_seconds)
    identity = {"schema": 1, "generator": GENERATOR, "prefix_sha256": prefix_hash(prefix)}
    directory = Path(root).resolve() / identity["prefix_sha256"]
    ledger_path = directory / "ledger.json"
    runner = search if search_impl is None else search_impl

    with campaign_lock(directory / "search.lock"):
        ledger = _load_ledger(ledger_path, directory, identity)
        _import_checkpoints(ledger, ledger_path, directory, root, prefix, imports, out_of_time)
        states = _load_states(ledger, prefix)
        initial = sum(_totals(states, len(prefix))[0])

        tested, reason = 0, "candidate_limit"
        while tested < candidates:
            if out_of_time():
                reason = "time_limit"
                break
            try:
                with campaign_lock(Path(root) / "allocation.lock",
                                   wait=min(ALLOCATION_WAIT_SECONDS, max(0, out_of_time.remaining()))):
                    # Waiting for the lock spends budget: re-check before reserving work.
                    if out_of_time():
                        reason = "time_limit"
                        break
                    selected = _next_range(ledger, ledger_path, directory, root, states,
                                           candidates - tested)
                    if selected is None:
                        reason = "seed_space_exhausted"
                        break
            except CampaignLockTimeout:
                if not out_of_time():
                    raise
                reason = "time_limit"
                break
            old = states.get(selected["checkpoint"])
            cursor = old["next_seed"] if old else selected["start"]
            # Lock waits and ledger writes spent budget. A range reserved above
            # stays registered and resumes on the next run. Read the clock once:
            # the search rejects a budget that is not positive.
            seconds_left = out_of_time.remaining()
            if seconds_left <= 0:
                reason = "time_limit"
                break
            state = runner(
                prefix, start=selected["start"], stop=selected["stop"],
                checkpoint=Path(selected["checkpoint"]), max_seconds=seconds_left,
                max_candidates=candidates - tested, batch_size=batch_size,
            )
            states[selected["checkpoint"]] = state
            selected["committed_until"] = state["next_seed"]
            write_json(ledger_path, ledger)
            tested += state["next_seed"] - cursor
            if state["stop_reason"] in ("interrupted", "survivor_limit", "time_limit", "user_stop"):
                reason = state["stop_reason"]
                break

        counts, survivors, completed = _totals(states, len(prefix))
        elapsed = time.perf_counter() - out_of_time.began
        result = {
            "identity": identity, "previously_tested": initial, "new_seeds_tested": tested,
            "total_unique_seeds_tested": sum(counts), "completed_ranges": completed,
            "matched_prefix_counts": counts, "survivors": survivors, "stop_reason": reason,
            "elapsed_seconds": elapsed, "new_seeds_per_second": tested / elapsed if elapsed else 0,
            "ledger": str(ledger_path),
        }
        write_json(directory / "latest_summary.json", result)
        return result
