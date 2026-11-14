"""Bounded search over an explicitly artificial date/seed lottery generator.

This tests seed recovery when the generator is known. It makes no assumption
that the real lottery uses a hash, a software PRNG, or a recoverable seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from lotto.core.files import write_json

GENERATOR = "calendar-sha256-v1"
DOMAIN = (GENERATOR + ":").encode("ascii")
OUTCOMES = math.comb(37, 6) * 7
HASH_LIMIT = (1 << 256) // OUTCOMES * OUTCOMES
MAX_SEED = 1 << 64


def ticket_index(balls, strong: int) -> int:
    """Bijective lexicographic set rank, followed by the strong number."""
    balls = sorted(balls)
    if (len(balls) != 6 or len(set(balls)) != 6
            or any(not isinstance(x, int) or not 1 <= x <= 37 for x in balls)
            or not isinstance(strong, int) or not 1 <= strong <= 7):
        raise ValueError("Expected six distinct integers in 1..37 and strong in 1..7")
    rank, previous = 0, 0
    for position, ball in enumerate(balls):
        for skipped in range(previous + 1, ball):
            rank += math.comb(37 - skipped, 5 - position)
        previous = ball
    return rank * 7 + strong - 1


def index_ticket(outcome: int) -> dict:
    """Inverse of ``ticket_index``: the six balls and strong number for an outcome index."""
    if not isinstance(outcome, int) or not 0 <= outcome < OUTCOMES:
        raise ValueError("Outcome index outside the lottery support")
    rank, strong = divmod(outcome, 7)
    balls, previous = [], 0
    for position in range(6):
        for ball in range(previous + 1, 38):
            block = math.comb(37 - ball, 5 - position)
            if rank < block:
                balls.append(ball)
                previous = ball
                break
            rank -= block
    return {"balls": balls, "strong": strong + 1}


def _date_bytes(value: str) -> bytes:
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError("Dates must be canonical YYYY-MM-DD")
    return value.encode("ascii")


def _outcome(seed_bytes: bytes, date_bytes: bytes) -> int:
    counter = 0
    while True:
        payload = DOMAIN + seed_bytes + date_bytes + counter.to_bytes(4, "big")
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        # Rejection avoids modulo bias in mapping a uniform hash to outcomes.
        if value < HASH_LIMIT:
            return value % OUTCOMES
        counter += 1


def world_outcome(seed: int, target_date: str) -> int:
    """The draw the artificial generator produces for this seed on this date, as an outcome index."""
    if not isinstance(seed, int) or not 0 <= seed < MAX_SEED:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    return _outcome(seed.to_bytes(8, "big"), _date_bytes(target_date))


@dataclass(frozen=True)
class Observation:
    date: str
    outcome: int


def prefix_hash(prefix: list[Observation]) -> str:
    """SHA-256 identifying an observed history; also validates that it is chronological."""
    if not prefix:
        raise ValueError("An empty matching prefix would retain every seed")
    previous = ""
    for observation in prefix:
        _date_bytes(observation.date)
        if observation.date <= previous:
            raise ValueError("Observation dates must be unique and chronological")
        if not isinstance(observation.outcome, int) or not 0 <= observation.outcome < OUTCOMES:
            raise ValueError("Invalid observed outcome")
        previous = observation.date
    payload = json.dumps([asdict(x) for x in prefix], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_checkpoint(state: dict, prefix: list[Observation]) -> None:
    """Validate persisted counts before resuming or importing completed work."""
    identity = state["identity"]
    start, stop = identity["start"], identity["stop"]
    expected = {
        "schema": 1, "generator": GENERATOR, "prefix_sha256": prefix_hash(prefix),
        "prefix_draws": len(prefix), "start": start, "stop": stop,
    }
    if identity != expected or not 0 <= start < stop <= MAX_SEED:
        raise ValueError("Checkpoint belongs to a different generator or observation prefix")
    cursor, survivors, counts = state["next_seed"], state["survivors"], state["matched_prefix_counts"]
    if (not isinstance(cursor, int) or not start <= cursor <= stop or len(counts) != len(prefix) + 1
            or any(not isinstance(n, int) or n < 0 for n in counts)
            or sum(counts) != cursor - start or counts[-1] != len(survivors)
            or len(set(survivors)) != len(survivors)
            or any(not isinstance(seed, int) or not start <= seed < cursor for seed in survivors)
            or not math.isfinite(state["elapsed_seconds"]) or state["elapsed_seconds"] < 0):
        raise ValueError("Inconsistent checkpoint")


def search(
    prefix: list[Observation],
    *,
    start: int,
    stop: int,
    checkpoint: Path,
    max_seconds: float = 10,
    batch_size: int = 4096,
    survivor_limit: int = 10_000,
    max_candidates: int | None = None,
) -> dict:
    """Search [start, stop), resuming a checkpoint only for the same experiment.

    Time limits are checked between batches. An interrupted batch is discarded
    and safely replayed; only whole batches enter the persisted search counts.
    Evaluation outcomes are deliberately absent from this interface.
    """
    if not 0 <= start < stop <= MAX_SEED:
        raise ValueError("Require 0 <= start < stop <= 2**64")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if not 1 <= batch_size <= 65_536 or survivor_limit < 1:
        raise ValueError("Require batch_size in 1..65536 and positive survivor_limit")
    if max_candidates is not None and (not isinstance(max_candidates, int) or max_candidates < 1):
        raise ValueError("max_candidates must be a positive integer")
    identity = {
        "schema": 1, "generator": GENERATOR, "prefix_sha256": prefix_hash(prefix),
        "prefix_draws": len(prefix), "start": start, "stop": stop,
    }
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("identity") != identity:
            raise ValueError("Checkpoint belongs to a different generator, prefix, or seed range")
        validate_checkpoint(state, prefix)
    else:
        state = {
            "identity": identity, "next_seed": start, "survivors": [],
            "matched_prefix_counts": [0] * (len(prefix) + 1), "elapsed_seconds": 0.0,
        }
    prepared = [(_date_bytes(x.date), x.outcome) for x in prefix]
    began = time.perf_counter()
    initial_elapsed = state["elapsed_seconds"]
    run_stop = min(stop, state["next_seed"] + max_candidates) if max_candidates is not None else stop
    reason = "time_limit"
    try:
        while state["next_seed"] < run_stop:
            if len(state["survivors"]) >= survivor_limit:
                reason = "survivor_limit"
                break
            if time.perf_counter() - began >= max_seconds:
                break
            # Publish all cursor/count changes together after a complete batch.
            counts = state["matched_prefix_counts"].copy()
            survivors = state["survivors"].copy()
            next_seed = state["next_seed"]
            for seed in range(next_seed, min(next_seed + batch_size, run_stop)):
                matched = 0
                seed_bytes = seed.to_bytes(8, "big")
                for day, outcome in prepared:
                    if _outcome(seed_bytes, day) != outcome:
                        break
                    matched += 1
                counts[matched] += 1
                if matched == len(prefix):
                    survivors.append(seed)
                next_seed = seed + 1
                if len(survivors) >= survivor_limit:
                    break
            state = {
                **state, "next_seed": next_seed, "survivors": survivors,
                "matched_prefix_counts": counts,
                "elapsed_seconds": initial_elapsed + time.perf_counter() - began,
            }
            write_json(checkpoint, state)
        if state["next_seed"] == stop:
            reason = "range_complete"
        elif state["next_seed"] == run_stop:
            reason = "candidate_limit"
    except KeyboardInterrupt:
        reason = "interrupted"
    state["elapsed_seconds"] = initial_elapsed + time.perf_counter() - began
    state["stop_reason"] = reason
    write_json(checkpoint, state)
    return state
