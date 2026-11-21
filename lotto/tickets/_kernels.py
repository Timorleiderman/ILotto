"""Compiled integer kernels for the fixed 6/37 coverage universe.

Kept separate so ordinary benchmark imports do not initialize Numba. The
outcome arrays use about 31 MiB, are built once per process, and are read-only.
All counts are exact integers; there is no sampled acceptance criterion.
"""

from functools import lru_cache
from math import comb

import numpy as np
from numba import njit


@njit(cache=True)
def _enumerate_draws(total):
    draws = np.empty((total, 6), dtype=np.uint8)
    masks = np.empty(total, dtype=np.uint64)
    i = 0
    for a in range(32):
        for b in range(a + 1, 33):
            for c in range(b + 1, 34):
                for d in range(c + 1, 35):
                    for e in range(d + 1, 36):
                        for f in range(e + 1, 37):
                            draws[i] = (a, b, c, d, e, f)
                            masks[i] = ((np.uint64(1) << a) | (np.uint64(1) << b)
                                        | (np.uint64(1) << c) | (np.uint64(1) << d)
                                        | (np.uint64(1) << e) | (np.uint64(1) << f))
                            i += 1
    return draws, masks


@lru_cache(maxsize=1)
def outcomes():
    """All 2,324,784 main draws as ball arrays and bit masks, built once and cached."""
    draws, masks = _enumerate_draws(comb(37, 6))
    draws.setflags(write=False)
    masks.setflags(write=False)
    return draws, masks


@njit(cache=True, inline="always")
def _popcount(value):
    count = 0
    while value:
        value &= value - np.uint64(1)
        count += 1
    return count


@njit(cache=True, inline="always")
def _at_least_3(value):
    v1 = value & (value - np.uint64(1))
    return (v1 & (v1 - np.uint64(1))) != np.uint64(0)


@njit(cache=True)
def match_histogram(draw_masks, ticket_masks):
    """How many draws have each best-row match count (0-6) for this portfolio."""
    histogram = np.zeros(7, dtype=np.int64)
    for draw in draw_masks:
        best = 0
        for ticket in ticket_masks:
            best = max(best, _popcount(draw & ticket))
        histogram[best] += 1
    return histogram


@njit(cache=True)
def coverage_signatures(draw_masks, ticket_masks, min_matches):
    """Count which ticket subsets cover each draw (callers cap at 14 rows)."""
    histogram = np.zeros(1 << len(ticket_masks), dtype=np.int64)
    if min_matches == 3:
        for draw in draw_masks:
            signature = 0
            for i, ticket in enumerate(ticket_masks):
                if _at_least_3(draw & ticket):
                    signature |= 1 << i
            histogram[signature] += 1
    else:
        for draw in draw_masks:
            signature = 0
            for i, ticket in enumerate(ticket_masks):
                if _popcount(draw & ticket) >= min_matches:
                    signature |= 1 << i
            histogram[signature] += 1
    return histogram


@njit(cache=True)
def best_strong_partition(coverage, groups):
    """Exact subset DP for a partition into nonempty, unlabeled groups.

    Different strong outcomes are disjoint, so the objective is the sum of
    main-draw union coverage within groups. Anchoring the next group at the
    lowest remaining row removes label permutations without removing a partition.
    """
    size = len(coverage)
    n_rows = 0
    tmp = size - 1
    while tmp:
        n_rows += 1
        tmp >>= 1
    scores = np.full((groups + 1, size), -1, dtype=np.int64)
    choices = np.zeros((groups + 1, size), dtype=np.int64)
    scores[0, 0] = 0
    for mask in range(1, size):
        scores[1, mask] = coverage[mask]
        choices[1, mask] = mask
    for group in range(2, groups + 1):
        max_pop = n_rows - (groups - group)
        start_mask = size - 1 if group == groups else 1
        for mask in range(start_mask, size):
            pop = _popcount(np.uint64(mask))
            if pop < group or pop > max_pop:
                continue
            anchor = mask & -mask
            rem = mask ^ anchor
            sub = rem
            best_score = -1
            best_sub = 0
            while True:
                subset = sub | anchor
                rest = mask ^ subset
                prev = scores[group - 1, rest]
                if prev >= 0:
                    score = coverage[subset] + prev
                    if score > best_score:
                        best_score = score
                        best_sub = subset
                if sub == 0:
                    break
                sub = (sub - 1) & rem
            scores[group, mask] = best_score
            choices[group, mask] = best_sub
    return scores[groups, size - 1], choices


@njit(cache=True)
def _coverage_counts_into(draw_masks, ticket_masks, counts):
    for i in range(len(draw_masks)):
        draw = draw_masks[i]
        c = 0
        for ticket in ticket_masks:
            if _at_least_3(draw & ticket):
                c += 1
        counts[i] = c
    return counts


def coverage_counts(draw_masks, ticket_masks):
    """For each draw, how many rows match it in three or more numbers."""
    dtype = np.uint8 if len(ticket_masks) <= 255 else np.int32
    counts = np.zeros(len(draw_masks), dtype=dtype)
    return _coverage_counts_into(draw_masks, ticket_masks, counts)


@njit(cache=True)
def update_coverage(draw_masks, counts, old, new):
    """Update ``counts`` in place after one row's mask changes from ``old`` to ``new``."""
    diff = old ^ new
    for i in range(len(draw_masks)):
        draw = draw_masks[i]
        changed = draw & diff
        if changed == np.uint64(0) or changed == diff:
            continue
        delta = int(_at_least_3(draw & new)) - int(_at_least_3(draw & old))
        if delta:
            counts[i] += delta


@njit(cache=True)
def best_exchange(draws, masks, counts, ticket, ticket_mask):
    """Score all 6×31 one-ball swaps in one pass over the exact universe.

    Only draws uncovered by the other tickets can change union coverage. A
    swap can gain those with two old matches, or lose those with three. Count
    the required single/joint incidences once instead of evaluating 186 full
    portfolios. Strictly positive gains only; ties use ticket/number order.
    """
    positions = np.full(37, -1, dtype=np.int64)
    for i, ball in enumerate(ticket):
        positions[ball] = i
    added = np.zeros(37, dtype=np.int64)
    removed = np.zeros(6, dtype=np.int64)
    joint_two = np.zeros((6, 37), dtype=np.int64)
    joint_three = np.zeros((6, 37), dtype=np.int64)
    inside = np.empty(6, dtype=np.int64)
    outside = np.empty(6, dtype=np.int64)
    for i in range(len(masks)):
        c = counts[i]
        if c > 1:
            continue
        mask = masks[i]
        overlap = mask & ticket_mask
        o1 = overlap & (overlap - np.uint64(1))
        if o1 == np.uint64(0):
            continue
        o2 = o1 & (o1 - np.uint64(1))
        if c == 0:
            if o2 != np.uint64(0):
                continue
            hits = 2
        else:
            if o2 == np.uint64(0) or (o2 & (o2 - np.uint64(1))) != np.uint64(0):
                continue
            hits = 3
        n_in = n_out = 0
        for ball in draws[i]:
            position = positions[ball]
            if position >= 0:
                inside[n_in] = position
                n_in += 1
            else:
                outside[n_out] = ball
                n_out += 1
        if hits == 2:
            for j in range(n_out):
                ball = outside[j]
                added[ball] += 1
                for k in range(n_in):
                    joint_two[inside[k], ball] += 1
        else:
            for k in range(n_in):
                position = inside[k]
                removed[position] += 1
                for j in range(n_out):
                    joint_three[position, outside[j]] += 1
    best_gain, remove_position, add_ball = 0, -1, -1
    for position in range(6):
        for ball in range(37):
            if positions[ball] >= 0:
                continue
            gain = added[ball] - joint_two[position, ball] - removed[position] + joint_three[position, ball]
            if gain > best_gain:
                best_gain, remove_position, add_ball = gain, position, ball
    return best_gain, remove_position, add_ball
