#!/usr/bin/env python3
"""Measure paired exact coverage before/after refinement and evaluator runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np


from lotto.tickets.coverage import (
    best_match_counts, greedy_coverage_tickets, portfolio_odds, refine_coverage_tickets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--refine-passes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 100 or not 0 <= args.refine_passes <= 20:
        parser.error("Require seeds in 1..100 and refine-passes in 0..20")
    results = []
    for seed in range(args.seeds):
        tick = time.perf_counter()
        initial = greedy_coverage_tickets(seed=seed)
        generate_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        before = portfolio_odds(initial)
        evaluate_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        improved, audit = refine_coverage_tickets(initial, max_passes=args.refine_passes)
        refine_seconds = time.perf_counter() - tick
        after = portfolio_odds(improved)
        assert after["p_any_prize"] >= before["p_any_prize"]
        assert after["p_jackpot"] == before["p_jackpot"]
        if seed < 5:
            for tickets, odds in ((initial, before), (improved, after)):
                reference = best_match_counts([r[:6] for r in tickets], engine="reference")
                np.testing.assert_array_equal(reference, odds["best_match_counts"])
        results.append({"seed": seed, "initial_portfolio": initial, "refined_portfolio": improved,
                        "before": before, "after": after, "refinement": audit,
                        "generate_seconds": generate_seconds, "refine_seconds": refine_seconds,
                        "evaluate_seconds": evaluate_seconds, "reference_verified": seed < 5})
        print(f"Seed {seed}: {before['p_any_prize']:.6%} -> {after['p_any_prize']:.6%}", flush=True)
    # Same portfolio for both engines, after warmup; first-call time is separate.
    ticket_mains = [row[:6] for row in results[0]["refined_portfolio"]]
    timing = {}
    for engine in ("reference", "auto"):
        seconds = []
        for _ in range(3):
            tick = time.perf_counter()
            best_match_counts(ticket_mains, engine=engine)
            seconds.append(time.perf_counter() - tick)
        timing[engine] = {"seconds": seconds, "median_seconds": float(np.median(seconds))}
    before = np.asarray([row["before"]["p_any_prize"] for row in results])
    after = np.asarray([row["after"]["p_any_prize"] for row in results])
    output = {
        "schema": 1, "configuration": {"seeds": list(range(args.seeds)), "count": 10,
                                         "candidates": 512, "simulations": 30_000,
                                         "refine_passes": args.refine_passes},
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__},
        "scope": "Exact fair-draw probabilities for fixed portfolios; not prediction or expected profit",
        "summary": {"mean_before": float(before.mean()), "mean_after": float(after.mean()),
                    "mean_gain_percentage_points": float(100 * (after - before).mean()),
                    "min_before": float(before.min()), "max_before": float(before.max()),
                    "min_after": float(after.min()), "max_after": float(after.max()),
                    "improved_portfolios": int(np.count_nonzero(after > before)),
                    "warm_evaluator_speedup": timing["reference"]["median_seconds"]
                                              / timing["auto"]["median_seconds"],
                    "first_evaluation_seconds": results[0]["evaluate_seconds"]},
        "timing_note": "First call includes lazy import/JIT cache load or compilation; medians are warm",
        "evaluator_timings": timing, "rows": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
