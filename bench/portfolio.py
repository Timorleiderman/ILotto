"""Exact fair-draw coverage for fixed portfolios; exploratory coverage search.

Coverage is P(at least one ticket reaches a match threshold), not the sum of
individual ticket probabilities. The optimizer uses simulated draws, then the
evaluator enumerates every possible main draw to verify its result exactly.
Neither assumes that historical frequencies predict the next draw.
"""

from __future__ import annotations

from itertools import batched, combinations
from math import comb

import numpy as np

from .data import N_DRAWN, N_NUMBERS, N_STRONG
from .metrics import prize_tier


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _ticket_masks(rows):
    return np.asarray([sum(1 << (int(x) - 1) for x in row) for row in rows], dtype=np.uint64)


def _main_tickets(tickets, n_numbers, n_drawn):
    if not _integer(n_numbers) or not _integer(n_drawn) or not 1 <= n_drawn <= n_numbers:
        raise ValueError("Require 1 <= n_drawn <= n_numbers")
    rows = []
    for ticket in tickets:
        row = tuple(ticket)
        if (len(row) != n_drawn or any(not _integer(x) or not 1 <= x <= n_numbers for x in row)
                or len(set(row)) != n_drawn):
            raise ValueError("Each ticket needs distinct integers inside the number pool")
        rows.append(tuple(sorted(int(x) for x in row)))
    if not rows:
        raise ValueError("Provide at least one ticket")
    return list(dict.fromkeys(rows))


def best_match_counts(tickets, *, n_numbers=N_NUMBERS, n_drawn=N_DRAWN, chunk_size=50_000,
                      engine="auto"):
    """Integer histogram of the best match count over all C(N,k) possible draws.

    Optional small games let tests compare with an independent exhaustive oracle.
    The fixed 6/37 game uses compiled bit-mask counting. The chunked reference
    engine remains available for independent verification and other game sizes.
    """
    rows = _main_tickets(tickets, n_numbers, n_drawn)
    if not _integer(chunk_size) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if engine not in ("auto", "reference"):
        raise ValueError("engine must be 'auto' or 'reference'")
    if engine == "auto" and (n_numbers, n_drawn) == (N_NUMBERS, N_DRAWN):
        from ._portfolio_exact import match_histogram, outcomes

        return match_histogram(outcomes()[1], _ticket_masks(rows))
    lookups = np.zeros((len(rows), n_numbers + 1), dtype=np.int8)
    for lookup, row in zip(lookups, rows):
        lookup[list(row)] = 1
    histogram = np.zeros(n_drawn + 1, dtype=np.int64)
    for chunk in batched(combinations(range(1, n_numbers + 1), n_drawn), chunk_size):
        draws = np.asarray(chunk, dtype=np.int64)
        best = np.zeros(len(draws), dtype=np.int64)
        for lookup in lookups:
            best = np.maximum(best, lookup[draws].sum(axis=1))
        histogram += np.bincount(best, minlength=n_drawn + 1)
    return histogram


def _full_tickets(tickets):
    tickets = [tuple(row) for row in tickets]
    if any(len(row) != N_DRAWN + 1 for row in tickets):
        raise ValueError("Each row must contain six main numbers and one strong number")
    _main_tickets([row[:N_DRAWN] for row in tickets], N_NUMBERS, N_DRAWN)
    if any(not _integer(row[-1]) or not 1 <= row[-1] <= N_STRONG for row in tickets):
        raise ValueError("Strong numbers must be integers in 1..7")
    return [[*sorted(int(x) for x in row[:N_DRAWN]), int(row[-1])] for row in tickets]


def portfolio_odds(tickets):
    """Fair-draw odds for rows of six main numbers followed by one strong number.

    Any prize includes 3 main matches without the strong number (eighth tier).
    The jackpot numerator is the number of distinct complete combinations.
    Duplicate rows do not improve the probability of at least one win.
    """
    tickets = _full_tickets(tickets)
    mains = _main_tickets([row[:N_DRAWN] for row in tickets], N_NUMBERS, N_DRAWN)
    full = set(tuple(row) for row in tickets)
    counts = best_match_counts(mains)
    total = comb(N_NUMBERS, N_DRAWN)
    return {
        "assumption": "Independent uniform fair draws; fixed portfolio for one draw",
        "tickets": len(tickets), "distinct_full_tickets": len(full), "distinct_main_tickets": len(mains),
        "total_main_outcomes": total, "best_match_counts": counts.tolist(),
        "p_at_least_matches": {str(k): float(counts[k:].sum() / total) for k in range(1, N_DRAWN + 1)},
        "p_any_prize": float(counts[3:].sum() / total),
        "jackpot_covered_outcomes": len(full), "total_full_outcomes": total * N_STRONG,
        "p_jackpot": len(full) / (total * N_STRONG),
    }


def single_ticket_prize_odds():
    """Exact disjoint prize-tier probabilities, including no prize."""
    total = comb(N_NUMBERS, N_DRAWN) * N_STRONG
    counts = {}
    for matched in range(N_DRAWN + 1):
        main_ways = comb(N_DRAWN, matched) * comb(N_NUMBERS - N_DRAWN, N_DRAWN - matched)
        for strong_hit, strong_ways in ((True, 1), (False, N_STRONG - 1)):
            tier = prize_tier(matched, strong_hit)
            counts[tier] = counts.get(tier, 0) + main_ways * strong_ways
    return {tier: {"outcomes": count, "probability": count / total} for tier, count in counts.items()}


def greedy_coverage_tickets(count=10, *, candidates=512, simulations=30_000, min_matches=3, seed=0,
                            max_working_bytes=256 * 1024**2):
    """Greedily cover simulated fair draws with a finite pool of distinct tickets.

    This is a heuristic, not a globally optimal covering design. Verify returned
    tickets with portfolio_odds; choose thresholds before comparing portfolios.
    Strong numbers are spread across 1..7 and do not affect the >=3 main objective.
    A conservative working-memory estimate rejects oversized requests before
    candidate generation. Callers may explicitly raise max_working_bytes.
    """
    if (any(not _integer(x) or x < 1 for x in (count, candidates, simulations))
            or not count <= candidates <= comb(N_NUMBERS, N_DRAWN)
            or not _integer(min_matches) or not 1 <= min_matches <= N_DRAWN):
        raise ValueError("Require positive counts, count <= candidates, and a match threshold in 1..6")
    if not _integer(max_working_bytes) or max_working_bytes < 1:
        raise ValueError("max_working_bytes must be a positive integer")
    # Python integers keep the pre-allocation estimate from overflowing when a
    # caller supplies very large NumPy integer configuration values.
    count, candidates, simulations = int(count), int(candidates), int(simulations)
    working_bytes = (2 * candidates * simulations + simulations * (16 * N_NUMBERS + 8 * N_DRAWN)
                     + candidates * 1024)
    if working_bytes > max_working_bytes:
        raise ValueError("Estimated optimizer memory exceeds max_working_bytes; reduce candidates or simulations")
    rng = np.random.default_rng(seed)
    pool = []
    seen = set()
    while len(pool) < candidates:
        row = tuple(sorted(rng.choice(N_NUMBERS, N_DRAWN, replace=False) + 1))
        if row not in seen:
            seen.add(row)
            pool.append(row)
    draws = np.argpartition(rng.random((simulations, N_NUMBERS)), N_DRAWN - 1, axis=1)[:, :N_DRAWN] + 1
    covered_by = np.zeros((candidates, simulations), dtype=bool)
    for i, row in enumerate(pool):
        lookup = np.zeros(N_NUMBERS + 1, dtype=np.int8)
        lookup[list(row)] = 1
        covered_by[i] = lookup[draws].sum(axis=1) >= min_matches
    uncovered = np.ones(simulations, dtype=bool)
    selected = []
    for _ in range(count):
        gains = covered_by[:, uncovered].sum(axis=1).astype(np.int64)
        gains[selected] = -1
        best = int(np.argmax(gains))
        selected.append(best)
        uncovered &= ~covered_by[best]
    strong_order = rng.permutation(np.arange(1, N_STRONG + 1))
    return [[*(int(x) for x in pool[index]), int(strong_order[i % N_STRONG])]
            for i, index in enumerate(selected)]


def refine_coverage_tickets(tickets, *, max_passes=2):
    """Improve >=3-main coverage using exact one-ball coordinate exchanges.

    Every accepted swap strictly increases the number of covered fair outcomes.
    Strong numbers and row count are preserved. This is bounded local search,
    not a global optimum, and need not improve the probability of a larger prize.
    Returns (copied tickets, audit metadata), including each exact integer gain.
    """
    rows = _full_tickets(tickets)
    if not _integer(max_passes) or max_passes < 0:
        raise ValueError("max_passes must be a nonnegative integer")
    from ._portfolio_exact import best_exchange, coverage_counts, outcomes, update_coverage

    draws, masks = outcomes()
    ticket_masks = _ticket_masks([row[:N_DRAWN] for row in rows])
    counts = coverage_counts(masks, ticket_masks)
    before = int(np.count_nonzero(counts))
    audit = {"objective": "At least three main matches", "max_passes": int(max_passes),
             "passes": 0, "covered_outcomes_before": before, "swaps": []}
    for pass_index in range(max_passes):
        changed = False
        for row_index, row in enumerate(rows):
            main = np.asarray(row[:N_DRAWN], dtype=np.int64) - 1
            gain, position, ball = best_exchange(draws, masks, counts, main, ticket_masks[row_index])
            if gain <= 0:
                continue
            old_ball = row[position]
            row[position] = int(ball) + 1
            row[:N_DRAWN] = sorted(row[:N_DRAWN])
            new_mask = _ticket_masks([row[:N_DRAWN]])[0]
            update_coverage(masks, counts, ticket_masks[row_index], new_mask)
            ticket_masks[row_index] = new_mask
            audit["swaps"].append({"pass": pass_index + 1, "row": row_index,
                                  "removed": old_ball, "added": int(ball) + 1,
                                  "newly_covered_outcomes": int(gain)})
            changed = True
        audit["passes"] += 1
        if not changed:
            break
    audit["covered_outcomes_after"] = int(np.count_nonzero(counts))
    if audit["covered_outcomes_after"] != before + sum(s["newly_covered_outcomes"] for s in audit["swaps"]):
        raise RuntimeError("Exact coverage audit failed")
    return rows, audit
