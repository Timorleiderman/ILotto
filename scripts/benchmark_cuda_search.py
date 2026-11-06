#!/usr/bin/env python3
"""Validate a CUDA worker, then benchmark an explicitly allocated finite range.

Use a frozen manifest containing prefix, prefix_sha256, start, stop, and source.
This does not allocate ranges across machines. The coordinator must reserve a
disjoint range before transferring the manifest. Checkpoints are importable by
the existing campaign ledger after the remote worker has stopped.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.cuda_seed_search import CudaWorker  # noqa: E402
from bench.seed_campaign import campaign_lock  # noqa: E402
from bench.seed_search import (  # noqa: E402
    GENERATOR, MAX_SEED, Observation, _prefix_hash, validate_checkpoint, world_outcome, write_json,
)


def validate_worker(worker):
    rng = random.Random(120893)
    pairs = [(0, "2020-01-01"), (45123, "2020-01-01"), (MAX_SEED - 1, "2026-11-06")]
    for _ in range(1000):
        pairs.append((rng.randrange(MAX_SEED),
                      (date(2000, 1, 1) + timedelta(days=rng.randrange(15000))).isoformat()))
    if worker.vectors(pairs) != [world_outcome(seed, day) for seed, day in pairs]:
        raise RuntimeError("CUDA vectors disagree with independent hashlib reference")
    cases = 0
    for start, stop, seed in [(0, 1003, 123), (MAX_SEED - 333, MAX_SEED, MAX_SEED - 1),
                              (1 << 53, (1 << 53) + 333, (1 << 53) + 201)]:
        prefix = [Observation(day, world_outcome(seed, day))
                  for day in ("2020-01-01", "2020-01-04", "2020-01-07")]
        for history in (prefix, prefix[:1], prefix[:2] + [Observation(prefix[2].date,
                                                                     (prefix[2].outcome + 1) % 16273488)]):
            expected, survivors = [0] * (len(history) + 1), []
            for candidate in range(start, stop):
                matched = 0
                for observation in history:
                    if world_outcome(candidate, observation.date) != observation.outcome:
                        break
                    matched += 1
                expected[matched] += 1
                if matched == len(history):
                    survivors.append(candidate)
            for threads in (64, 128, 256, 512):
                result = worker.batch(history, start, stop, threads=threads, blocks=7)
                if result["matched_prefix_counts"] != expected or result["survivors"] != survivors:
                    raise RuntimeError("CUDA histogram/survivors disagree with independent reference")
                cases += 1
    return {"independent_hash_vectors": len(pairs), "batch_comparisons": cases,
            "includes": ["all launch sizes", "uneven ranges", "unsigned maximum seed",
                         "integers above 2^53", "partial matches", "full survivors"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seconds-per-config", type=float, default=5)
    parser.add_argument("--warm-seconds", type=float, default=15,
                        help="Additional sustained trial of the fastest configuration")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.seconds_per_config <= 30:
        parser.error("seconds-per-config must be in 0.1..30")
    if not 0 <= args.warm_seconds <= 60:
        parser.error("warm-seconds must be in 0..60")
    with campaign_lock(args.out_dir / "worker.lock"), CudaWorker(args.binary) as worker:
        validation = validate_worker(worker)
        write_json(args.out_dir / "validation.json", validation)
        print(json.dumps(validation), flush=True)
        if args.validate_only:
            return
        if args.manifest is None:
            parser.error("--manifest is required for a real benchmark")
        manifest = json.loads(args.manifest.read_text())
        prefix = [Observation(**item) for item in manifest["prefix"]]
        start, stop = manifest["start"], manifest["stop"]
        if (manifest["generator"] != GENERATOR or manifest["prefix_sha256"] != _prefix_hash(prefix)
                or not 0 <= start < stop <= MAX_SEED):
            raise ValueError("Invalid allocation manifest")
        frozen = args.out_dir / "manifest.json"
        if frozen.exists() and json.loads(frozen.read_text()) != manifest:
            raise ValueError("Output directory belongs to another allocation")
        write_json(frozen, manifest)
        binary_hash = hashlib.sha256(args.binary.read_bytes()).hexdigest()
        checkpoint = args.out_dir / "checkpoint.json"
        identity = {"schema": 1, "generator": GENERATOR, "prefix_sha256": _prefix_hash(prefix),
                    "prefix_draws": len(prefix), "start": start, "stop": stop}
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
            "identity": identity, "next_seed": start, "matched_prefix_counts": [0] * (len(prefix) + 1),
            "survivors": [], "elapsed_seconds": 0.0}
        validate_checkpoint(state, prefix)
        if state["identity"] != identity:
            raise ValueError("Checkpoint allocation mismatch")
        report_path = args.out_dir / "report.json"
        if report_path.exists():
            raise ValueError("Benchmark already finished; choose a new allocation/output directory")
        write_json(checkpoint, state)
        trials_path = args.out_dir / "trials.json"
        trials = json.loads(trials_path.read_text()) if trials_path.exists() else []
        configurations = [(64, 656), (128, 656), (256, 656), (512, 656), (128, 1312)]
        for index in range(len(configurations) + bool(args.warm_seconds)):
            if state["next_seed"] == stop:
                break
            if index == len(configurations):
                best = max(trials, key=lambda trial: trial["seeds_per_second"])
                threads, blocks = best["threads"], best["blocks"]
                duration = args.warm_seconds
            else:
                threads, blocks = configurations[index]
                duration = args.seconds_per_config
            began, trial_start, chunk = time.perf_counter(), state["next_seed"], 1_000_000
            while time.perf_counter() - began < duration and state["next_seed"] < stop:
                if (args.out_dir / "STOP").exists():
                    state["stop_reason"] = "user_stop"
                    write_json(checkpoint, state)
                    return
                a, b = state["next_seed"], min(stop, state["next_seed"] + chunk)
                tick = time.perf_counter()
                result = worker.batch(prefix, a, b, threads=threads, blocks=blocks)
                seconds = time.perf_counter() - tick
                state["next_seed"] = b
                state["matched_prefix_counts"] = [x + y for x, y in zip(
                    state["matched_prefix_counts"], result["matched_prefix_counts"])]
                state["survivors"].extend(result["survivors"])
                state["elapsed_seconds"] += seconds
                state["last_engine"] = {"language": "cuda", "threads": threads, "blocks": blocks,
                                        "binary_sha256": binary_hash}
                validate_checkpoint(state, prefix)
                write_json(checkpoint, state)
                chunk = max(1, min(worker.max_batch, int((b-a) / max(seconds, 1e-6) * 0.5)))
            elapsed = time.perf_counter() - began
            trial = {"threads": threads, "blocks": blocks, "start": trial_start,
                     "stop": state["next_seed"], "tested": state["next_seed"] - trial_start,
                     "kind": "sustained" if index == len(configurations) else "configuration",
                     "wall_seconds": elapsed, "seeds_per_second": (state["next_seed"]-trial_start) / elapsed}
            trials.append(trial)
            write_json(trials_path, trials)
            print(json.dumps(trial), flush=True)
        state["stop_reason"] = "range_complete" if state["next_seed"] == stop else "benchmark_complete"
        write_json(checkpoint, state)
        write_json(report_path, {"status": "finished", "validation": validation, "trials": trials,
                                "completed_range": [start, state["next_seed"]],
                                "reserved_range": [start, stop], "binary_sha256": binary_hash,
                                "full_history_survivors": len(state["survivors"]),
                                "interpretation": "Artificial generator search; no established predictive advantage"})


if __name__ == "__main__":
    main()
