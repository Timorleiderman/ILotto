#!/usr/bin/env python3
"""Supervise one finite Rust search, preserving inputs, executable, and status.

Run under `caffeinate -i` on macOS to inhibit idle sleep for this command only.
The search itself enforces the duration; this supervisor adds failure reporting
and a bounded shutdown fallback. Create OUT_DIR/search/STOP for a graceful stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from lotto.seedsearch.generator import write_json
from lotto.seedsearch.rust_engine import DEFAULT_BINARY

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, default=ROOT / "experiments/seed_search_runs/campaigns")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 86400 or not 1 <= args.workers <= 64:
        parser.error("Require a finite duration in (0, 86400] and 1..64 workers")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "job.json"
    if status_path.exists():
        parser.error("Job directory already contains a recorded run; use a new directory")
    executable = out / "ilotto-seed-search"
    archive = out / "archive.csv"
    shutil.copy2(DEFAULT_BINARY, executable)
    shutil.copyfile(ROOT / "input/Orig_IL_lotto.csv", archive)
    start = datetime.now(timezone.utc)
    status = {
        "status": "starting", "supervisor_pid": os.getpid(), "workers": args.workers,
        "duration_seconds": args.seconds, "started_at_utc": start.isoformat(),
        "started_at_israel": start.astimezone(ZoneInfo("Asia/Jerusalem")).isoformat(),
        "estimated_end_israel": (start + timedelta(seconds=args.seconds)).astimezone(
            ZoneInfo("Asia/Jerusalem")).isoformat(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "progress": str(out / "search/progress.json"), "stop_file": str(out / "search/STOP"),
        "report": str(out / "search/report.json"), "ledger_dir": str(args.ledger_dir.resolve()),
    }
    write_json(status_path, status)
    command = [sys.executable, "-u", "-m", "lotto.experiments.seed_search", "--mode", "real",
               "--count", "1000000000000000", "--max-seconds", str(args.seconds),
               "--engine", "rust", "--workers", str(args.workers),
               "--rust-binary", str(executable), "--raw-csv", str(archive),
               "--ledger-dir", str(args.ledger_dir.resolve()), "--out-dir", str(out / "search")]
    tick = time.perf_counter()
    process = None
    try:
        process = subprocess.Popen(command, cwd=ROOT)
        status.update(status="running", search_pid=process.pid)
        write_json(status_path, status)
        try:
            code = process.wait(timeout=args.seconds + 30)
        except subprocess.TimeoutExpired:
            stop = out / "search/STOP"
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.touch()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
        status.update(status="finished" if code == 0 else "failed", exit_code=code)
        report_path = out / "search/report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            status.update(new_seeds_tested=report["seeds_tested"], stop_reason=report["stop_reason"],
                          total_unique_seeds=report["campaign"]["total_unique_seeds_tested"],
                          full_history_survivors=len(report["survivors"]))
            # An evidence summary, not an invented ten-ticket recommendation.
            write_json(out / "candidate_summary.json", {
                "count": min(10, len(report["survivors"])),
                "candidates": report["survivors"][:10],
                "meaning": "Seeds matching the search prefix; listed tickets are retrospective withheld draws",
                "future_ticket_recommendations": [],
            })
    except BaseException as exc:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter() - tick)
        write_json(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
