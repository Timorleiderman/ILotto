"""Simulation-only training of a frozen mixture of past-frequency forecasts.

The generator does not accept a real archive. Randomized biased labels prevent
learning a privileged number from the simulator. Mixture weights are learned
from synthetic targets; on real data only the past-history inputs change.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from lotto.forecasting.calibration import frequency_forecasts
from lotto.core.data import Draws, N_DRAWN, N_NUMBERS, N_STRONG
from lotto.approaches.date_conditioned import log_partition, marginals
from lotto.forecasting.joint_mixture import LOG_FULL_OUTCOMES, validate_draws, event_log_ratio

SCENARIOS = ('fair', 'persistent_bias', 'changing_bias')


def simulate_episode(n, rng, scenario, *, change_every=250, multipliers=(0.5, 1.5, 2.0)):
    """Generate proper six-without-replacement draws and known oracle scores.

    A biased block chooses a random main and strong label. Their event
    probabilities are multiplied; other outcomes are uniform conditional on
    those events. Bias directions and labels are redrawn at block boundaries.
    These are testable hypothetical mechanisms, not inferred real mechanisms.
    """
    if (type(n) is not int or n < 2 or scenario not in SCENARIOS
            or type(change_every) is not int or change_every < 1
            or not len(multipliers) or any(not np.isfinite(m) or not 0 < m < N_NUMBERS / N_DRAWN
                                           for m in multipliers)):
        raise ValueError('Require a known scenario, at least two draws, and valid inclusion multipliers')
    balls = np.empty((n, N_DRAWN), dtype=np.int64)
    strong = np.empty(n, dtype=np.int64)
    oracle = np.full(n, -LOG_FULL_OUTCOMES)
    specifications = []
    block = change_every if scenario == 'changing_bias' else n
    for start in range(0, n, block):
        stop = min(start + block, n)
        count = stop - start
        ball_label, strong_label = int(rng.integers(1, 38)), int(rng.integers(1, 8))
        ball_multiplier, strong_multiplier = (1., 1.) if scenario == 'fair' else rng.choice(multipliers, 2)
        qb, qs = N_DRAWN / N_NUMBERS * ball_multiplier, strong_multiplier / N_STRONG
        other_balls = np.delete(np.arange(1, 38), ball_label - 1)
        other_strong = np.delete(np.arange(1, 8), strong_label - 1)
        selected = other_balls[np.argsort(rng.random((count, 36)), axis=1)[:, :6]]
        present = rng.random(count) < qb
        selected[present, -1] = ball_label
        balls[start:stop] = np.sort(selected, axis=1)
        bonus = other_strong[rng.integers(0, 6, count)]
        bonus_present = rng.random(count) < qs
        bonus[bonus_present] = strong_label
        strong[start:stop] = bonus
        oracle[start:stop] += (event_log_ratio(qb, present, N_DRAWN / N_NUMBERS)
                               + event_log_ratio(qs, bonus_present, 1 / N_STRONG))
        specifications.append({'start': start, 'stop': stop, 'ball': ball_label, 'strong': strong_label,
                               'ball_multiplier': float(ball_multiplier),
                               'strong_multiplier': float(strong_multiplier)})
    dates = np.datetime64('2000-01-01') + np.arange(n)
    return Draws(dates, balls, strong), oracle, specifications


def components(draws, *, windows=(50, 100, 300, None), strengths=(100, 500, 2000),
               include_marginals=False):
    """Return all forecasts from the first draw, with a uniform cold start.

    History resets at episode boundaries. Outcome t influences only forecasts
    after t. Full-set probabilities are exactly normalized to six main balls.
    """
    validate_draws(draws)
    if any(isinstance(s, (bool, np.bool_)) or not np.isfinite(s) or s <= 0 for s in strengths):
        raise ValueError('Cold-start forecasts require strictly positive prior strengths')
    names, _, bp, sp = frequency_forecasts(draws, min_history=1, windows=windows, strengths=strengths)
    bp = np.concatenate([np.full((len(names), 1, N_NUMBERS), N_DRAWN / N_NUMBERS), bp], axis=1)
    sp = np.concatenate([np.full((len(names), 1, N_STRONG), 1 / N_STRONG), sp], axis=1)
    actual = draws.multi_hot
    lp = np.empty((len(names), len(draws)))
    for i in range(len(names) - 1):
        theta = np.log(bp[i]) - np.log1p(-bp[i])
        lp[i] = ((theta * actual).sum(axis=1) - log_partition(theta)
                 + np.log(sp[i, np.arange(len(draws)), draws.strong - 1]))
        if include_marginals:
            bp[i] = marginals(theta)
    lp[-1] = -LOG_FULL_OUTCOMES
    # Avoid numerical label preferences on an exactly uniform cold start.
    lp[:, 0] = -LOG_FULL_OUTCOMES
    if include_marginals:
        bp[:, 0] = N_DRAWN / N_NUMBERS
    return {'names': names, 'log_prob': lp,
            'ball_p': bp if include_marginals else None,
            'strong_p': sp if include_marginals else None}


def fit_mixture(log_prob, *, max_iterations=500, tolerance=1e-10):
    """Convex maximum-likelihood fit of simplex weights on synthetic examples.

    The same weights combine proper component distributions for every target.
    Every component assigns positive probability to every draw, so simplex
    boundaries are safe. Real and validation outcomes are not training inputs.
    """
    lp = np.asarray(log_prob, dtype=float)
    if (lp.ndim != 2 or not lp.shape[0] or not lp.shape[1] or not np.isfinite(lp).all()
            or np.any(lp > 1e-12) or type(max_iterations) is not int or max_iterations < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Require finite component log probabilities and positive optimizer settings')
    # Subtract a per-example constant: stable probabilities with the same optimum.
    offset = lp.max(axis=0)
    relative = np.exp(lp.T - offset[:, None]).copy(order='C')
    initial = np.full(lp.shape[0], 1 / lp.shape[0])

    def objective(weights):
        mixture = relative @ weights
        return -np.log(mixture).mean(), -(relative.T @ (1 / mixture)) / len(mixture)

    fit = minimize(objective, initial, method='SLSQP', jac=True,
                   bounds=[(0, 1)] * len(initial),
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1,
                                'jac': lambda w: np.ones_like(w)},
                   options={'maxiter': max_iterations, 'ftol': tolerance})
    if not fit.success or not np.isfinite(fit.x).all() or np.any(fit.x < 0):
        raise RuntimeError(f'Synthetic mixture fit failed: {fit.message}')
    weights = np.where(fit.x < 1e-8, 0, fit.x)
    weights /= weights.sum()
    loss = float(objective(weights)[0] - offset.mean())
    initial_loss = float(objective(initial)[0] - offset.mean())
    if loss > initial_loss + 1e-9:
        raise RuntimeError('Fit made the training objective worse')
    return {'weights': weights.tolist(), 'iterations': int(fit.nit), 'converged': bool(fit.success),
            'mean_training_log_loss': loss, 'initial_equal_weight_log_loss': initial_loss}


def frozen_forecast(component, weights):
    """Combine components using frozen training weights, with no target update."""
    w = np.asarray(weights, dtype=float)
    if (w.shape != (len(component['names']),) or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError('Require nonnegative unit-sum weights matching the components')
    positive = w > 0
    lp = logsumexp(np.log(w[positive])[:, None] + component['log_prob'][positive], axis=0)
    return {'log_prob': lp,
            'ball_p': None if component['ball_p'] is None else np.einsum('e,etb->tb', w, component['ball_p']),
            'strong_p': None if component['strong_p'] is None else np.einsum('e,ets->ts', w, component['strong_p'])}
