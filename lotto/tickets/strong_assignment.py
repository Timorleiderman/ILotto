"""Exact strong-number assignment for fixed main tickets under uniform draws.

The objective is at least one ticket with >=k main matches AND the strong
number. Main-number coverage and the jackpot probability of distinct main
tickets are unchanged. This does not optimize cash return or prediction.
"""

from math import comb

import numpy as np

from lotto.core.data import N_DRAWN, N_NUMBERS, N_STRONG
from lotto.tickets.coverage import full_tickets, is_integer, tickets_to_masks, best_match_counts


def _threshold(value):
    if not is_integer(value) or not 1 <= value <= N_DRAWN:
        raise ValueError("min_matches must be an integer in 1..6")


def subset_union_counts(signatures):
    """Union counts for every row subset, via a subset-sum transform."""
    counts = np.asarray(signatures)
    if (counts.ndim != 1 or len(counts) < 2 or len(counts) > 1 << 14
            or len(counts) & (len(counts) - 1) or counts.dtype.kind not in 'iu'
            or np.any(counts < 0) or sum(int(v) for v in counts) > np.iinfo(np.int64).max):
        raise ValueError("Require 2^n nonnegative integer counts for 1..14 rows with an int64 total")
    subset_sums = counts.astype(np.int64, copy=True)
    stride = 1
    while stride < len(counts):
        blocks = subset_sums.reshape(-1, 2 * stride)
        blocks[:, stride:] += blocks[:, :stride]
        stride *= 2
    return int(counts.sum()) - subset_sums[(len(counts) - 1) ^ np.arange(len(counts))]


def strong_prize_odds(tickets, *, min_matches=3, engine='auto'):
    """Evaluate P(at least one ticket has >=k main matches and strong).

    Summing the seven conditional main-draw unions is exact: their strong
    outcomes are mutually exclusive. engine='reference' uses independent sets
    and exhaustive chunked enumeration through best_match_counts.
    """
    rows = full_tickets(tickets)
    _threshold(min_matches)
    counts = []
    for strong in range(1, N_STRONG + 1):
        group = [row[:N_DRAWN] for row in rows if row[-1] == strong]
        counts.append(int(best_match_counts(group, engine=engine)[min_matches:].sum()) if group else 0)
    total = comb(N_NUMBERS, N_DRAWN) * N_STRONG
    return {'min_matches': int(min_matches), 'covered_full_outcomes': sum(counts),
            'total_full_outcomes': total, 'p_at_least_matches_and_strong': sum(counts) / total,
            'covered_main_outcomes_by_strong': counts}


def optimize_strong_numbers(tickets, *, min_matches=3):
    """Find an exact optimal assignment for 1..14 distinct fixed main rows.

    Each row goes to one of seven strong outcomes. Splitting a group cannot
    reduce the objective, so it suffices to consider min(7,n) nonempty groups.
    The dynamic program considers all such partitions, including unbalanced
    ones; it does not just optimize pairs or spread labels evenly.
    """
    rows = full_tickets(tickets)
    _threshold(min_matches)
    if len(rows) > 14 or len({tuple(row[:N_DRAWN]) for row in rows}) != len(rows):
        raise ValueError("Require 1..14 distinct main tickets")
    from lotto.tickets._kernels import best_strong_partition, coverage_signatures, outcomes

    signatures = coverage_signatures(outcomes()[1], tickets_to_masks([r[:N_DRAWN] for r in rows]), min_matches)
    coverage = subset_union_counts(signatures)
    initial_masks = [sum(1 << i for i, row in enumerate(rows) if row[-1] == s)
                     for s in range(1, N_STRONG + 1)]
    before = sum(int(coverage[mask]) for mask in initial_masks)
    groups = min(N_STRONG, len(rows))
    score, choices = best_strong_partition(coverage, groups)
    remaining = (1 << len(rows)) - 1
    result = [row.copy() for row in rows]
    partitions = []
    for group in range(groups, 0, -1):
        subset = int(choices[group, remaining])
        if not subset:
            raise RuntimeError("Missing optimal partition")
        members = [i for i in range(len(rows)) if subset & (1 << i)]
        for i in members:
            result[i][-1] = groups - group + 1
        partitions.append(members)
        remaining ^= subset
    if remaining or score < before or sum(int(coverage[sum(1 << i for i in p)]) for p in partitions) != score:
        raise RuntimeError("Strong-number assignment failed its exact coverage audit")
    total = comb(N_NUMBERS, N_DRAWN) * N_STRONG
    return result, {'objective': f'At least {min_matches} main matches and the strong number on one ticket',
                    'min_matches': int(min_matches), 'optimal_for_fixed_main_tickets': True,
                    'groups': partitions, 'covered_full_outcomes_before': before,
                    'covered_full_outcomes_after': int(score), 'total_full_outcomes': total,
                    'probability_before': before / total, 'probability_after': int(score) / total,
                    'all_subset_main_union_counts': coverage.tolist()}
