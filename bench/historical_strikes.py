"""Past-forecast ticket construction and realized win scoring.

A global relabeling preserves the complete overlap structure and all fair-draw
coverage probabilities. Assigning more-used labels to higher forecast scores
therefore tests number prediction at an unchanged ticket count and null coverage.
"""

import numpy as np

from .data import N_DRAWN, N_NUMBERS, N_STRONG
from .portfolio import _full_tickets


def ranked_portfolios(template, ball_scores, strong_scores):
    """Relabel a fixed portfolio independently for each past-only forecast.

    Labels are ordered by decreasing occurrence count, then numeric label.
    Forecast ranks use that same order to break ties. Uniform forecasts thus
    return the template unchanged. This is a fixed heuristic, not a claim of
    optimal portfolio selection under the forecast distribution.
    """
    rows = np.asarray(_full_tickets(template), dtype=np.int64)
    balls, strong = np.asarray(ball_scores, dtype=float), np.asarray(strong_scores, dtype=float)
    if (balls.ndim != 2 or balls.shape[1] != N_NUMBERS or not len(balls)
            or strong.shape != (len(balls), N_STRONG)
            or not np.isfinite(balls).all() or not np.isfinite(strong).all()
            or np.any(balls < 0) or np.any(strong < 0)):
        raise ValueError('Require aligned, nonempty finite nonnegative forecast matrices with 37 and 7 columns')

    def mapping(values, size, scores):
        counts = np.bincount(values.ravel(), minlength=size + 1)[1:]
        order = np.lexsort((np.arange(size), -counts))
        ranked = order[np.argsort(-scores[:, order], axis=1, kind='stable')]
        relabel = np.empty_like(ranked)
        relabel[:, order] = ranked + 1
        return relabel

    ball_map = mapping(rows[:, :N_DRAWN], N_NUMBERS, balls)
    strong_map = mapping(rows[:, -1], N_STRONG, strong)
    mains = np.sort(ball_map[:, rows[:, :N_DRAWN] - 1], axis=2)
    bonus = strong_map[:, rows[:, -1] - 1]
    return np.concatenate([mains, bonus[:, :, None]], axis=2)


def score_portfolios(tickets, actual_balls, actual_strong):
    """Count row-level matches after forecasts have been fixed."""
    tickets, actual = np.asarray(tickets), np.asarray(actual_balls)
    strong = np.asarray(actual_strong)
    if (tickets.ndim != 3 or not tickets.shape[0] or not tickets.shape[1] or tickets.shape[2] != 7
            or actual.shape != (len(tickets), N_DRAWN) or strong.shape != (len(tickets),)
            or any(a.dtype.kind not in 'iu' for a in (tickets, actual, strong))
            or np.any((tickets[:, :, :6] < 1) | (tickets[:, :, :6] > N_NUMBERS))
            or np.any((tickets[:, :, -1] < 1) | (tickets[:, :, -1] > N_STRONG))
            or np.any((actual < 1) | (actual > N_NUMBERS)) or np.any((strong < 1) | (strong > N_STRONG))
            or np.any(np.diff(np.sort(tickets[:, :, :6], axis=2), axis=2) == 0)
            or np.any(np.diff(np.sort(actual, axis=1), axis=1) == 0)):
        raise ValueError('Require valid integer tickets and one valid realized draw per forecast')
    matches = (tickets[:, :, :6, None] == actual[:, None, None, :]).sum(axis=(2, 3))
    strong_hit = tickets[:, :, -1] == strong[:, None]
    return matches, strong_hit


def longest_run(events):
    longest = current = 0
    for event in events:
        current = current + 1 if event else 0
        longest = max(longest, current)
    return longest
