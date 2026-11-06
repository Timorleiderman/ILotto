"""Reproducible, past-only frequency forecasts for probability research.

Shrinkage improves calibration when a short history is noisy. It does not
change the ranking within one frequency window or prove a predictive edge.
The grid is experimental; the production predictor defaults stay unchanged.
"""

from __future__ import annotations

import numpy as np

from .data import Draws, N_DRAWN, N_NUMBERS, N_STRONG


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def frequency_forecasts(draws: Draws, *, min_history=200,
                        windows=(50, 100, 300, None), strengths=(0, 100, 500, 2000)):
    """Forecast before seeing each target, using prefix sums over earlier rows.

    strength is an equivalent number of uniform-prior draws, not a fitted
    parameter. Ball inclusion probabilities sum to six; strong probabilities
    sum to one. The last grid member is the exact uniform baseline.
    """
    if not _integer(min_history) or not 1 <= min_history < len(draws):
        raise ValueError("Require 1 <= min_history < number of draws")
    if (not len(windows) or not len(strengths)
            or any(w is not None and (not _integer(w) or w < 1) for w in windows)
            or any(isinstance(s, (bool, np.bool_)) or not np.isfinite(s) or s < 0 for s in strengths)):
        raise ValueError("Require positive windows and finite nonnegative prior strengths")
    if len(set(windows)) != len(windows) or len(set(strengths)) != len(strengths):
        raise ValueError("Duplicate grid values would collapse model names in multiple-testing correction")
    targets = np.arange(min_history, len(draws))
    balls = np.vstack([np.zeros(N_NUMBERS), np.cumsum(draws.multi_hot, axis=0, dtype=float)])
    strong_hot = np.eye(N_STRONG)[draws.strong - 1]
    strong = np.vstack([np.zeros(N_STRONG), np.cumsum(strong_hot, axis=0)])
    names, ball_p, strong_p = [], [], []
    for window in windows:
        starts = np.zeros_like(targets) if window is None else np.maximum(0, targets - window)
        n = (targets - starts)[:, None]
        for strength in strengths:
            names.append(f"frequency window={window or 'all'} prior_draws={strength}")
            ball_p.append((balls[targets] - balls[starts] + strength * N_DRAWN / N_NUMBERS) / (n + strength))
            strong_p.append((strong[targets] - strong[starts] + strength / N_STRONG) / (n + strength))
    names.append("uniform")
    ball_p.append(np.full((len(targets), N_NUMBERS), N_DRAWN / N_NUMBERS))
    strong_p.append(np.full((len(targets), N_STRONG), 1 / N_STRONG))
    return names, targets, np.asarray(ball_p), np.asarray(strong_p)


def past_loss_choices(losses, *, baseline, warmup=200, lookback=300, refit_every=50):
    """Choose a grid member using only loss rows strictly before the target.

    The fixed schedule and window form part of the evaluated algorithm. Never
    pick these settings by maximizing the same reported test result.
    """
    losses = np.asarray(losses, dtype=float)
    if (losses.ndim != 2 or not np.isfinite(losses).all()
            or not _integer(baseline) or not 0 <= baseline < len(losses)
            or any(not _integer(x) or x < 1 for x in (warmup, lookback, refit_every))):
        raise ValueError("Require a finite model-by-time loss matrix and positive schedule parameters")
    choices = np.full(losses.shape[1], baseline, dtype=int)
    chosen = baseline
    for i in range(warmup, losses.shape[1]):
        if (i - warmup) % refit_every == 0:
            chosen = int(np.argmin(losses[:, max(0, i - lookback):i].mean(axis=1)))
        choices[i] = chosen
    return choices
