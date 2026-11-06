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


@njit(cache=True)
def match_histogram(draw_masks, ticket_masks):
    histogram = np.zeros(7, dtype=np.int64)
    for draw in draw_masks:
        best = 0
        for ticket in ticket_masks:
            best = max(best, _popcount(draw & ticket))
        histogram[best] += 1
    return histogram


@njit(cache=True)
def coverage_counts(draw_masks, ticket_masks):
    counts = np.zeros(len(draw_masks), dtype=np.int32)
    for i, draw in enumerate(draw_masks):
        for ticket in ticket_masks:
            counts[i] += _popcount(draw & ticket) >= 3
    return counts


@njit(cache=True)
def update_coverage(draw_masks, counts, old, new):
    for i, draw in enumerate(draw_masks):
        counts[i] += int(_popcount(draw & new) >= 3) - int(_popcount(draw & old) >= 3)


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
    for i, mask in enumerate(masks):
        hits = _popcount(mask & ticket_mask)
        if counts[i] - int(hits >= 3) or (hits != 2 and hits != 3):
            continue
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
