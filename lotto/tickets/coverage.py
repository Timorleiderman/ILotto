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

from lotto.core.data import N_DRAWN, N_NUMBERS, N_STRONG
from lotto.core.metrics import prize_tier
from lotto.core.validation import is_integer


def tickets_to_masks(rows):
    """Each row of main numbers as a 37-bit mask: bit n-1 is set when number n is in the row."""
    return np.asarray([sum(1 << (int(x) - 1) for x in row) for row in rows], dtype=np.uint64)


def main_tickets(tickets, n_numbers, n_drawn):
    """Validated, sorted, de-duplicated rows of main numbers; ValueError if any row is malformed."""
    if not is_integer(n_numbers) or not is_integer(n_drawn) or not 1 <= n_drawn <= n_numbers:
        raise ValueError("Require 1 <= n_drawn <= n_numbers")
    rows = []
    for ticket in tickets:
        row = tuple(ticket)
        if (len(row) != n_drawn or any(not is_integer(x) or not 1 <= x <= n_numbers for x in row)
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
    rows = main_tickets(tickets, n_numbers, n_drawn)
    if not is_integer(chunk_size) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if engine not in ("auto", "reference"):
        raise ValueError("engine must be 'auto' or 'reference'")
    if engine == "auto" and len(rows) == 1:
        return np.array(
            [comb(n_drawn, k) * comb(n_numbers - n_drawn, n_drawn - k) for k in range(n_drawn + 1)],
            dtype=np.int64,
        )
    if engine == "auto" and (n_numbers, n_drawn) == (N_NUMBERS, N_DRAWN):
        from lotto.tickets._kernels import match_histogram, outcomes

        return match_histogram(outcomes()[1], tickets_to_masks(rows))
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


def full_tickets(tickets):
    """Validated rows of six sorted main numbers followed by a strong number."""
    tickets = [tuple(row) for row in tickets]
    if any(len(row) != N_DRAWN + 1 for row in tickets):
        raise ValueError("Each row must contain six main numbers and one strong number")
    main_tickets([row[:N_DRAWN] for row in tickets], N_NUMBERS, N_DRAWN)
    if any(not is_integer(row[-1]) or not 1 <= row[-1] <= N_STRONG for row in tickets):
        raise ValueError("Strong numbers must be integers in 1..7")
    return [[*sorted(int(x) for x in row[:N_DRAWN]), int(row[-1])] for row in tickets]


def portfolio_odds(tickets):
    """Fair-draw odds for rows of six main numbers followed by one strong number.

    Any prize includes 3 main matches without the strong number (eighth tier).
    The jackpot numerator is the number of distinct complete combinations.
    Duplicate rows do not improve the probability of at least one win.
    """
    tickets = full_tickets(tickets)
    mains = main_tickets([row[:N_DRAWN] for row in tickets], N_NUMBERS, N_DRAWN)
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
    if (any(not is_integer(x) or x < 1 for x in (count, candidates, simulations))
            or not count <= candidates <= comb(N_NUMBERS, N_DRAWN)
            or not is_integer(min_matches) or not 1 <= min_matches <= N_DRAWN):
        raise ValueError("Require positive counts, count <= candidates, and a match threshold in 1..6")
    if not is_integer(max_working_bytes) or max_working_bytes < 1:
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
    pool_hot = np.zeros((candidates, N_NUMBERS), dtype=np.float32)
    pool_arr = np.asarray(pool, dtype=np.int64) - 1
    pool_hot[np.arange(candidates)[:, None], pool_arr] = 1.0
    draws_hot = np.zeros((simulations, N_NUMBERS), dtype=np.float32)
    draws_hot[np.arange(simulations)[:, None], draws - 1] = 1.0
    covered_by = (pool_hot @ draws_hot.T) >= (min_matches - 0.5)
    uncovered = np.ones(simulations, dtype=bool)
    gains = covered_by.sum(axis=1).astype(np.int64)
    selected = []
    for _ in range(count):
        best = int(np.argmax(gains))
        selected.append(best)
        gains[best] = -1
        newly = uncovered & covered_by[best]
        if newly.any():
            uncovered &= ~newly
            gains -= covered_by[:, newly].sum(axis=1, dtype=np.int64)
            gains[selected] = -1
    strong_order = rng.permutation(np.arange(1, N_STRONG + 1))
    return [[*(int(x) for x in pool[index]), int(strong_order[i % N_STRONG])]
            for i, index in enumerate(selected)]


def search_coverage_tickets(tickets, *, steps=20, seed=0, start_temperature=1000.0, end_temperature=10.0,
                            perturbations=2, local_passes=2):
    """Experimental discrete basin hopping with exact coverage and best retention.

    By default perturb two numbers, locally refine for two passes, then accept by a
    Metropolis rule. Temperatures are measured in covered main outcomes.
    Occasional worse intermediate states allow escape from one-exchange optima;
    only the best observed portfolio is returned. No history or forecasts enter
    this fixed-ticket-budget optimization, and global optimality is not claimed.
    """
    rows = full_tickets(tickets)
    if (not is_integer(steps) or not 1 <= steps <= 500 or not is_integer(seed) or seed < 0
            or not is_integer(perturbations) or not 1 <= perturbations <= 100
            or not is_integer(local_passes) or not 0 <= local_passes <= 20
            or any(isinstance(t, (bool, np.bool_)) or not np.isfinite(t) or t <= 0
                   for t in (start_temperature, end_temperature))
            or start_temperature < end_temperature):
        raise ValueError("Require steps in 1..500, perturbations in 1..100, local_passes in 0..20, "
                         "nonnegative seed, and finite decreasing positive temperatures")
    from lotto.tickets._kernels import coverage_counts, outcomes, update_coverage

    _, masks = outcomes()
    current = [list(row) for row in rows]
    best = [row.copy() for row in current]
    current_masks = tickets_to_masks([row[:N_DRAWN] for row in current])
    current_counts = coverage_counts(masks, current_masks)

    def count(portfolio):
        histogram = best_match_counts([row[:N_DRAWN] for row in portfolio])
        return int(histogram[3:].sum())

    current_count = best_count = initial_count = int(np.count_nonzero(current_counts))
    rng = np.random.default_rng(seed)
    proposals = []
    for step, temperature in enumerate(np.geomspace(start_temperature, end_temperature, steps)):
        candidate = [row.copy() for row in current]
        cand_masks = current_masks.copy()
        cand_counts = current_counts.copy()
        changes = []
        touched = set()
        for _ in range(perturbations):
            row_index = int(rng.integers(len(candidate)))
            position = int(rng.integers(N_DRAWN))
            old = candidate[row_index][position]
            available = sorted(set(range(1, N_NUMBERS + 1)) - set(candidate[row_index][:N_DRAWN]))
            new = int(rng.choice(available))
            candidate[row_index][position] = new
            candidate[row_index][:N_DRAWN] = sorted(candidate[row_index][:N_DRAWN])
            changes.append({"row": row_index, "removed": old, "added": new})
            touched.add(row_index)
        for row_index in touched:
            new_mask = tickets_to_masks([candidate[row_index][:N_DRAWN]])[0]
            update_coverage(masks, cand_counts, cand_masks[row_index], new_mask)
            cand_masks[row_index] = new_mask
        candidate, local_audit = _refine_with_counts(
            candidate, cand_masks, cand_counts, max_passes=local_passes
        )
        candidate_count = local_audit["covered_outcomes_after"]
        delta = candidate_count - current_count
        accepted = delta >= 0 or rng.random() < np.exp(delta / temperature)
        if accepted:
            current, current_count = candidate, candidate_count
            current_masks, current_counts = cand_masks, cand_counts
        if candidate_count > best_count:
            best, best_count = [row.copy() for row in candidate], candidate_count
        proposals.append({"step": step, "temperature": float(temperature), "perturbations": changes,
                          "local_refinement": local_audit, "candidate_covered_outcomes": candidate_count,
                          "change_from_current": delta, "accepted": bool(accepted),
                          "best_covered_outcomes": best_count})
    if count(best) != best_count or best_count < initial_count:
        raise RuntimeError("Exact coverage audit did not reproduce the retained best portfolio")
    return best, {"objective": "At least three main matches", "seed": int(seed), "steps": int(steps),
                  "perturbations": int(perturbations), "local_passes": int(local_passes),
                  "start_temperature": float(start_temperature), "end_temperature": float(end_temperature),
                  "covered_outcomes_before": initial_count, "covered_outcomes_after": best_count,
                  "proposals": proposals}


def _refine_with_counts(rows, ticket_masks, counts, *, max_passes=2):
    from lotto.tickets._kernels import best_exchange, outcomes, update_coverage

    draws, masks = outcomes()
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
            new_mask = tickets_to_masks([row[:N_DRAWN]])[0]
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


def refine_coverage_tickets(tickets, *, max_passes=2):
    """Improve >=3-main coverage using exact one-ball coordinate exchanges.

    Every accepted swap strictly increases the number of covered fair outcomes.
    Strong numbers and row count are preserved. This is bounded local search,
    not a global optimum, and need not improve the probability of a larger prize.
    Returns (copied tickets, audit metadata), including each exact integer gain.
    """
    rows = full_tickets(tickets)
    if not is_integer(max_passes) or max_passes < 0:
        raise ValueError("max_passes must be a nonnegative integer")
    from lotto.tickets._kernels import coverage_counts, outcomes

    _, masks = outcomes()
    ticket_masks = tickets_to_masks([row[:N_DRAWN] for row in rows])
    counts = coverage_counts(masks, ticket_masks)
    return _refine_with_counts(rows, ticket_masks, counts, max_passes=max_passes)
