"""Rust batch adapter with Python-owned checkpoint and run lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from lotto.core import paths
from lotto.seedsearch.generator import (
    GENERATOR,
    MAX_SEED,
    Observation,
    prefix_hash,
    validate_checkpoint,
    world_outcome,
    write_json,
)

DEFAULT_BINARY = paths.REPO_ROOT / "native/seed-search/target/release/ilotto-seed-search"


def native_batch(prefix, start, stop, workers, binary=DEFAULT_BINARY):
    """Search seeds ``[start, stop)`` with the Rust binary and validate its result."""
    request = {"generator": GENERATOR, "start": str(start), "stop": str(stop),
               "workers": workers, "prefix": [asdict(x) for x in prefix]}
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        output, error = process.communicate(json.dumps(request), timeout=120)
    except BaseException:
        process.kill()
        process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"Rust batch failed ({process.returncode}): {error[:2000]}")
    result = json.loads(output)
    return validate_native_result(result, prefix, start, stop)


def validate_native_result(result, prefix, start, stop):
    """Reject incomplete native results and independently recheck all survivors."""
    counts, survivors = result["matched_prefix_counts"], result["survivors"]
    if (result["generator"] != GENERATOR or result["start"] != str(start) or result["stop"] != str(stop)
            or result["tested"] != stop - start or len(counts) != len(prefix) + 1
            or any(not isinstance(n, int) or n < 0 for n in counts) or sum(counts) != stop - start
            or result["survivor_overflow"] or counts[-1] != len(survivors)
            or len(set(survivors)) != len(survivors)
            or any(not isinstance(seed, int) or not start <= seed < stop for seed in survivors)):
        raise RuntimeError("Native batch failed validation; range was not committed")
    # Recheck rare full survivors in the independent Python reference before saving.
    for seed in survivors:
        if any(world_outcome(seed, item.date) != item.outcome for item in prefix):
            raise RuntimeError("Native survivor disagrees with Python reference; range was not committed")
    return result


class RustSearch:
    def __init__(self, workers: int, *, binary=DEFAULT_BINARY, progress: Path | None = None,
                 stop_file: Path | None = None):
        if not 1 <= workers <= 64:
            raise ValueError("workers must be in 1..64")
        self.workers, self.binary = workers, Path(binary).resolve()
        if not self.binary.is_file():
            raise ValueError("Rust binary missing; build native/seed-search in release mode first")
        self.binary_sha256 = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.progress, self.stop_file = progress, stop_file
        self.new_tested = 0
        self.started = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_log = 0.0

    def __call__(self, prefix: list[Observation], *, start, stop, checkpoint,
                 max_seconds=10, batch_size=4096, survivor_limit=10_000, max_candidates=None):
        if not 0 <= start < stop <= MAX_SEED or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Invalid finite range or time budget")
        if max_candidates is not None and (not isinstance(max_candidates, int) or max_candidates < 1):
            raise ValueError("max_candidates must be positive")
        identity = {"schema": 1, "generator": GENERATOR, "prefix_sha256": prefix_hash(prefix),
                    "prefix_draws": len(prefix), "start": start, "stop": stop}
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state["identity"] != identity:
                raise ValueError("Checkpoint belongs to another experiment")
            validate_checkpoint(state, prefix)
        else:
            state = {"identity": identity, "next_seed": start, "survivors": [],
                     "matched_prefix_counts": [0] * (len(prefix) + 1), "elapsed_seconds": 0.0}
        began, elapsed_before = time.perf_counter(), state["elapsed_seconds"]
        run_stop = min(stop, state["next_seed"] + max_candidates) if max_candidates is not None else stop
        reason, chunk = "time_limit", self.workers * 1_000_000
        try:
            while state["next_seed"] < run_stop:
                if self.stop_file and self.stop_file.exists():
                    reason = "user_stop"
                    break
                remaining = max_seconds - (time.perf_counter() - began)
                if remaining <= 0.02:
                    break
                if len(state["survivors"]) >= survivor_limit:
                    reason = "survivor_limit"
                    break
                a, b = state["next_seed"], min(state["next_seed"] + chunk, run_stop)
                tick = time.perf_counter()
                result = native_batch(prefix, a, b, self.workers, self.binary)
                seconds = time.perf_counter() - tick
                # Persist a complete disjoint batch as one unit. No per-seed disk I/O.
                state = {
                    **state, "next_seed": b,
                    "matched_prefix_counts": [x + y for x, y in zip(
                        state["matched_prefix_counts"], result["matched_prefix_counts"])],
                    "survivors": state["survivors"] + result["survivors"],
                    "elapsed_seconds": elapsed_before + time.perf_counter() - began,
                    "last_engine": {"language": "rust", "workers": self.workers,
                                    "binary_sha256": self.binary_sha256},
                }
                write_json(checkpoint, state)
                self.new_tested += b - a
                now = time.perf_counter()
                progress = {"status": "running", "pid": os.getpid(), "engine": "rust",
                            "workers": self.workers, "started_at_utc": self.started_at,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "new_seeds_tested": self.new_tested, "next_seed": b,
                            "elapsed_seconds": now - self.started,
                            "seeds_per_second": self.new_tested / (now - self.started),
                            "full_history_survivors": len(state["survivors"]),
                            "matched_prefix_counts": state["matched_prefix_counts"],
                            "checkpoint": str(checkpoint)}
                if self.progress:
                    write_json(self.progress, progress)
                if now - self.last_log >= 30:
                    print(f"Rust/{self.workers}: {self.new_tested:,} new seeds; next {b:,}; "
                          f"{progress['seeds_per_second']:,.0f}/s; "
                          f"{len(state['survivors'])} full survivors", flush=True)
                    self.last_log = now
                remaining = max_seconds - (time.perf_counter() - began)
                # Target about one second per atomic batch; shorten near deadline.
                chunk = max(1, min(1_000_000_000, int((b - a) / max(seconds, 1e-6)
                                                    * min(1.0, max(0.001, remaining)))))
            if state["next_seed"] == stop:
                reason = "range_complete"
            elif state["next_seed"] == run_stop:
                reason = "candidate_limit"
        except KeyboardInterrupt:
            reason = "interrupted"
        state["elapsed_seconds"] = elapsed_before + time.perf_counter() - began
        state["stop_reason"] = reason
        write_json(checkpoint, state)
        if self.progress and self.progress.exists():
            progress = json.loads(self.progress.read_text())
            progress.update(status="batch_search_stopped", stop_reason=reason)
            write_json(self.progress, progress)
        return state
