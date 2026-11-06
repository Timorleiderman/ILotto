#!/usr/bin/env python3
"""Reserve fresh work on the Mac, execute CUDA over SSH, checkpoint on the Mac.

Run this from the original coordinator checkout, not from a copied campaign.
Resume with --resume ID. No separate remote ledger or manual seed bounds needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.cuda_seed_search import MAX_BATCH, CudaWorker  # noqa: E402
from bench.seed_allocations import get_remote, reserve_remote  # noqa: E402
from bench.seed_campaign import campaign_lock  # noqa: E402
from bench.seed_search import (  # noqa: E402
    GENERATOR, Observation, _prefix_hash, validate_checkpoint, write_json,
)
from scripts.benchmark_cuda_search import validate_worker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "experiments/seed_search_runs"


def ssh_command(worker):
    host, directory = worker["host"], worker["directory"]
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", host) or not directory.startswith("/"):
        raise ValueError("Expected an SSH host and an absolute remote checkout path")
    binary = directory.rstrip("/") + "/native/cuda-search/ilotto-cuda-search"
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
            host, "exec " + shlex.quote(binary)]


def run_allocation(manifest, *, run_root, seconds, worker_factory=None, validator=validate_worker):
    if not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise ValueError("Require a finite time limit in (0, 86400]")
    if not re.fullmatch(r"[a-f0-9]{32}", manifest["id"]):
        raise ValueError("Invalid allocation ID")
    prefix = [Observation(**item) for item in manifest["prefix"]]
    if manifest["generator"] != GENERATOR or manifest["prefix_sha256"] != _prefix_hash(prefix):
        raise ValueError("Allocation prefix identity mismatch")
    out = Path(run_root) / "remote_workers" / manifest["id"]
    identity = {"schema": 1, "generator": GENERATOR, "prefix_sha256": manifest["prefix_sha256"],
                "prefix_draws": len(prefix), "start": manifest["start"], "stop": manifest["stop"]}
    with campaign_lock(out / "worker.lock"):
        manifest_path, checkpoint = out / "manifest.json", out / "checkpoint.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Stored allocation has changed")
        write_json(manifest_path, manifest)
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
            "identity": identity, "next_seed": manifest["start"], "survivors": [],
            "matched_prefix_counts": [0] * (len(prefix) + 1), "elapsed_seconds": 0.0}
        validate_checkpoint(state, prefix)
        if state["identity"] != identity:
            raise ValueError("Checkpoint range identity mismatch")
        # A separate durable cursor detects deletion/regression of the checkpoint.
        receipt_path = out / "committed.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if state["next_seed"] < receipt.get("next_seed", manifest["start"]):
            raise ValueError("Checkpoint containing completed work is missing or regressed")
        write_json(checkpoint, state)
        initial = state["next_seed"]
        status = {"allocation_id": manifest["id"], "worker": manifest["worker"],
                  "reserved_range": [manifest["start"], manifest["stop"]],
                  "checkpoint": str(checkpoint.resolve())}
        if initial == manifest["stop"]:
            return {**status, "status": "already_complete", "new_seeds_tested": 0,
                    "next_seed": initial}
        if (out / "STOP").exists():
            return {**status, "status": "user_stop", "new_seeds_tested": 0, "next_seed": initial}
        write_json(out / "status.json", {**status, "status": "starting", "next_seed": initial})
        factory = worker_factory or (lambda: CudaWorker(
            Path(manifest["worker"]["directory"]), command=ssh_command(manifest["worker"])))
        try:
            with factory() as worker:
                validation = validator(worker)
                write_json(out / "validation.json", validation)
                began, chunk, reason, last_log = time.perf_counter(), 1_000_000, "time_limit", 0.0
                while state["next_seed"] < manifest["stop"]:
                    remaining = seconds - (time.perf_counter() - began)
                    if remaining <= 0:
                        break
                    if (out / "STOP").exists():
                        reason = "user_stop"
                        break
                    a, b = state["next_seed"], min(state["next_seed"] + chunk, manifest["stop"])
                    tick = time.perf_counter()
                    result = worker.batch(prefix, a, b, threads=512, blocks=656)
                    elapsed = time.perf_counter() - tick
                    state = {**state, "next_seed": b,
                             "matched_prefix_counts": [x + y for x, y in zip(
                                 state["matched_prefix_counts"], result["matched_prefix_counts"])],
                             "survivors": state["survivors"] + result["survivors"],
                             "elapsed_seconds": state["elapsed_seconds"] + elapsed,
                             "last_engine": {"language": "cuda", "worker": manifest["worker"]}}
                    validate_checkpoint(state, prefix)
                    write_json(checkpoint, state)
                    write_json(receipt_path, {"next_seed": b})
                    wall = time.perf_counter() - began
                    status.update(status="running", next_seed=b, new_seeds_tested=b-initial,
                                  seeds_per_second=(b-initial)/wall,
                                  full_history_survivors=len(state["survivors"]))
                    write_json(out / "status.json", status)
                    if wall-last_log >= 5:
                        print(json.dumps(status), flush=True)
                        last_log = wall
                    # ~2 s per batch keeps the GPU busy relative to the SSH round trip and
                    # the three fsync'd writes above; a crash replays at most one batch.
                    chunk = max(1, min(getattr(worker, "max_batch", MAX_BATCH), int((b-a)/max(elapsed, 1e-6)
                                                  * min(2.0, max(0.001, seconds-wall)))))
                if state["next_seed"] == manifest["stop"]:
                    reason = "range_complete"
                state["stop_reason"] = reason
                write_json(checkpoint, state)
                status.update(status="finished", stop_reason=reason, next_seed=state["next_seed"],
                              new_seeds_tested=state["next_seed"]-initial,
                              full_history_survivors=len(state["survivors"]))
                write_json(out / "status.json", status)
                return status
        except BaseException as exc:
            write_json(out / "status.json", {**status, "status": "failed", "error": str(exc),
                                             "resume": manifest["id"]})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="For example timor@192.168.50.225")
    parser.add_argument("--remote-dir", default="/home/timor/Developer/ILotto")
    parser.add_argument("--count", type=int, help="Number of NEW seeds to reserve")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--resume", help="Resume an existing allocation ID")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 86400:
        parser.error("seconds must be in (0, 86400]")
    root = args.run_root / "campaigns"
    if args.resume:
        if args.count is not None or args.host is not None:
            parser.error("Resume uses the original host and range; omit --count and --host")
        manifest = get_remote(root, args.resume)
    else:
        if not args.host or args.count is None:
            parser.error("A new allocation requires --host and --count")
        worker = {"host": args.host, "directory": args.remote_dir}
        ssh_command(worker)
        from bench.data import RAW_CSV, load
        from bench.seed_search import ticket_index

        archive = Path(RAW_CSV)
        draws = load(raw_csv=str(archive), clean_csv=None, refresh=False)
        prefix = [Observation(str(day.astype("datetime64[D]")), ticket_index(balls.tolist(), int(strong)))
                  for day, balls, strong in zip(draws.dates, draws.balls, draws.strong)][:-4]
        manifest = reserve_remote(prefix, root=root, run_root=args.run_root, candidates=args.count,
                                  worker=worker, source={"archive_sha256": hashlib.sha256(
                                      archive.read_bytes()).hexdigest(), "withheld_draws": 4})
    print(json.dumps({k: manifest[k] for k in ("id", "start", "stop", "worker")}), flush=True)
    result = run_allocation(manifest, run_root=args.run_root, seconds=args.seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
