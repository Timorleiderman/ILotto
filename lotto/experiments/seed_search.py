#!/usr/bin/env python3
"""Run a bounded toy-seed recovery or retrospective real-history search.

Examples:
    uv run lotto run seed-search --mode toy
    uv run lotto run seed-search --mode real --stop 1000000
    uv run lotto run seed-search --mode real --count 100000000 --max-seconds 120

Reusing the same output directory resumes the same finite experiment. There is
no archive refresh, background process, or prospective prediction in this CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lotto.core import paths
from lotto.seedsearch.campaign import continue_search
from lotto.seedsearch.generator import (
    GENERATOR,
    Observation,
    index_ticket,
    prefix_hash,
    search,
    ticket_index,
    world_outcome,
    write_json,
)

RUN_ROOT = paths.REPO_ROOT / "experiments/seed_search_runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["toy", "real"], default="toy")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int, help="exclusive seed bound (legacy explicit-range mode)")
    parser.add_argument("--count", type=int, help="search this many NEW seeds using the shared range ledger")
    parser.add_argument("--engine", choices=["python", "rust"], default="python")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rust-binary", type=Path, help="optional frozen executable for a recorded run")
    parser.add_argument("--raw-csv", type=Path, help="optional immutable real-data snapshot")
    parser.add_argument("--ledger-dir", type=Path, default=RUN_ROOT / "campaigns")
    parser.add_argument("--import-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--max-seconds", type=float, default=10)
    parser.add_argument("--prefix-draws", type=int, help="default: 8 toy; all but final 4 real")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.count is not None and (args.start is not None or args.stop is not None):
        parser.error("Use --count for automatic continuation OR --start/--stop for a legacy explicit range")
    args.start = 0 if args.start is None else args.start
    args.stop = 65_536 if args.stop is None else args.stop
    if args.count is not None:
        out = args.out_dir or RUN_ROOT / "campaign_reports" / f"{args.mode}_{uuid.uuid4().hex}"
        if (out / "report.json").exists():
            parser.error("Use a new output directory to preserve the previous campaign report")
    else:
        out = args.out_dir or RUN_ROOT / args.mode
    if args.mode == "toy":
        # Fixture creation is separate from search; the search only gets observations.
        observations = []
        first = datetime(2020, 1, 1)
        for i in range(12):
            day = (first + timedelta(days=3 * i)).date().isoformat()
            observations.append(Observation(day, world_outcome(45_123, day)))
        prefix_size = args.prefix_draws if args.prefix_draws is not None else 8
        source = {"kind": "artificial_known_generator", "generator": GENERATOR}
    else:
        from lotto.core.data import RAW_CSV, load

        raw_path = args.raw_csv or Path(RAW_CSV)
        draws = load(raw_csv=str(raw_path), clean_csv=None, refresh=False)
        observations = [
            Observation(str(day.astype("datetime64[D]")), ticket_index(balls.tolist(), int(strong)))
            for day, balls, strong in zip(draws.dates, draws.balls, draws.strong)
        ]
        prefix_size = args.prefix_draws if args.prefix_draws is not None else len(observations) - 4
        source = {
            "kind": "retrospective_local_archive", "archive_path": str(raw_path),
            "archive_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "draws": len(observations), "last_date": observations[-1].date,
            "note": "Known historical outcomes; not prospective evidence or a current forecast.",
        }
    if not 1 <= prefix_size < len(observations):
        parser.error("prefix-draws must leave at least one search draw and one withheld draw")
    prefix, withheld = observations[:prefix_size], observations[prefix_size:]
    # Bind both the source snapshot and split before resuming this experiment.
    manifest = {"source": source, "prefix_draws": prefix_size}
    native = None
    if args.engine == "rust":
        from lotto.seedsearch.rust_engine import DEFAULT_BINARY, RustSearch

        native = RustSearch(args.workers, binary=args.rust_binary or DEFAULT_BINARY,
                            progress=out / "progress.json", stop_file=out / "STOP")
        manifest["engine"] = {"language": "rust", "workers": args.workers,
                              "binary_sha256": native.binary_sha256}
    if args.count is not None:
        manifest.update({"new_candidates_requested": args.count, "ledger_dir": str(args.ledger_dir.resolve())})
    else:
        manifest.update({"start": args.start, "stop": args.stop})
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error("Output directory belongs to another experiment; choose a new --out-dir")
    write_json(manifest_path, manifest)
    campaign = None
    try:
        if args.count is not None:
            history_hash = prefix_hash(prefix)
            imports = list(args.import_checkpoint)
            for path in sorted(RUN_ROOT.glob("*/checkpoint.json")):
                identity = json.loads(path.read_text())["identity"]
                if identity["generator"] == GENERATOR and identity["prefix_sha256"] == history_hash:
                    imports.append(path)
            campaign = continue_search(
                prefix, root=args.ledger_dir, candidates=args.count, max_seconds=args.max_seconds,
                imports=imports, search_impl=native,
            )
            result = campaign
            tested = campaign["new_seeds_tested"]
        else:
            runner = search if native is None else native
            result = runner(prefix, start=args.start, stop=args.stop,
                            checkpoint=out / "checkpoint.json", max_seconds=args.max_seconds)
            tested = result["next_seed"] - args.start
    except ValueError as exc:
        parser.error(str(exc))
    # Forecasts are materialized independently of the withheld outcome values.
    forecasts = [
        {"seed": seed, "outcomes": [world_outcome(seed, x.date) for x in withheld]}
        for seed in result["survivors"]
    ]
    write_json(out / "withheld_predictions.json", {
        "evaluation_kind": "toy_withheld" if args.mode == "toy" else "retrospective_withheld",
        "dates": [x.date for x in withheld], "forecasts": forecasts,
    })
    actual = [x.outcome for x in withheld]
    for forecast in forecasts:
        forecast["exact_hits"] = sum(a == b for a, b in zip(forecast["outcomes"], actual))
        forecast["tickets"] = [index_ticket(x) for x in forecast["outcomes"]]
    elapsed = result["elapsed_seconds"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "generator": GENERATOR,
        "source": source, "prefix_draws": prefix_size, "withheld_draws": len(withheld),
        "seeds_tested": tested, "stop_reason": result["stop_reason"],
        "elapsed_seconds": elapsed, "seeds_per_second": tested / elapsed if elapsed else 0,
        "matched_prefix_counts": result["matched_prefix_counts"],
        "survivors": forecasts,
        "interpretation": "Artificial generator search only; no demonstrated real predictive advantage.",
    }
    if campaign is not None:
        report["campaign"] = {key: campaign[key] for key in (
            "previously_tested", "new_seeds_tested", "total_unique_seeds_tested", "completed_ranges", "ledger",
        )}
        report["counts_scope"] = "all unique seeds in this campaign; timing is for newly searched seeds only"
    write_json(out / "report.json", report)
    if native is not None and (out / "progress.json").exists():
        progress = json.loads((out / "progress.json").read_text())
        progress.update(status="finished", report=str((out / "report.json").resolve()),
                        stop_reason=result["stop_reason"])
        write_json(out / "progress.json", progress)
    summary = {key: report[key] for key in (
        "generator", "seeds_tested", "stop_reason", "elapsed_seconds", "seeds_per_second",
        "prefix_draws", "withheld_draws",
    )}
    summary["survivors"] = len(forecasts)
    summary["withheld_exact_hits_per_survivor"] = [x["exact_hits"] for x in forecasts]
    summary["report"] = str((out / "report.json").resolve())
    if campaign is not None:
        summary["campaign"] = report["campaign"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
