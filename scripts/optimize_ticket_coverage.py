#!/usr/bin/env python3
"""Generate a fixed-size portfolio and verify its fair-draw odds exhaustively.

The objective is at least one prize (three or more main matches). This does
not improve jackpot odds over the same number of distinct full combinations,
and a prize does not necessarily cover the total ticket cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.portfolio import (  # noqa: E402
    greedy_coverage_tickets, portfolio_odds, refine_coverage_tickets, search_coverage_tickets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, choices=range(2, 15, 2),
                        help="Rows in an ordinary Lotto form (default: 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--simulations", type=int, default=30_000)
    parser.add_argument("--refine-passes", type=int, default=2,
                        help="Exact one-ball exchange passes (default: 2; 0 reproduces the initial search)")
    parser.add_argument("--search-steps", type=int, default=0,
                        help="Optional basin-hopping proposals after refinement (0 disables; maximum 500)")
    parser.add_argument("--search-seed", type=int, default=70106,
                        help="Seed for the optional basin-hopping search (default: 70106)")
    parser.add_argument("--out", type=Path, help="Optional JSON destination; otherwise print JSON")
    args = parser.parse_args()
    if args.refine_passes < 0:
        parser.error("refine-passes must be nonnegative")
    if not 0 <= args.search_steps <= 500 or args.search_seed < 0:
        parser.error("search-steps must be in 0..500 and search-seed must be nonnegative")
    try:
        tickets = greedy_coverage_tickets(args.count, candidates=args.candidates,
                                          simulations=args.simulations, seed=args.seed)
        tickets, refinement = refine_coverage_tickets(tickets, max_passes=args.refine_passes)
        search = None
        if args.search_steps:
            tickets, search = search_coverage_tickets(tickets, steps=args.search_steps, seed=args.search_seed)
    except ValueError as exc:
        parser.error(str(exc))
    result = {
        "objective": "At least one prize, not net profit or higher jackpot probability per ticket",
        "optimizer": ("Greedy simulation, exact improving exchanges, optional basin hopping; "
                      "no global optimum claimed"),
        "configuration": {"count": args.count, "seed": args.seed, "candidates": args.candidates,
                          "simulations": args.simulations, "min_matches": 3,
                          "refine_passes": args.refine_passes,
                          "search_steps": args.search_steps, "search_seed": args.search_seed},
        "refinement": refinement,
        "search": search,
        "portfolio": tickets,
        "exact_odds": portfolio_odds(tickets),
    }
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(f"Saved {args.out}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
