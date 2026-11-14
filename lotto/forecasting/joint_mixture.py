"""Past-only, normalized full-draw forecasts and sequential expert mixtures.

Experimental research API, not a replacement for the production predictors.
Historical likelihood-ratio paths are exploratory when the archive was already
used to choose the model. Prospective evidence requires a frozen procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log

import numpy as np
from scipy.special import logsumexp

from lotto.approaches.date_conditioned import log_partition, marginals
from lotto.core.data import N_DRAWN, N_NUMBERS, N_STRONG, Draws
from lotto.forecasting.calibration import frequency_forecasts

LOG_FULL_OUTCOMES = log(comb(N_NUMBERS, N_DRAWN) * N_STRONG)
FAMILY_PRIOR = {"uniform": 0.5, "single_ball": 0.2, "single_strong": 0.1, "dense": 0.2}


@dataclass
class JointExperts:
    names: list[str]
    families: list[str]
    targets: np.ndarray
    prior: np.ndarray
    log_prob: np.ndarray  # expert by time: probability of the complete observed draw
    ball_p: np.ndarray | None  # expert by time by ball, inclusion probabilities
    strong_p: np.ndarray | None


def validate_draws(draws):
    """Raise ValueError unless draws are valid 6/37 + strong rows on strictly increasing dates."""
    balls, strong, dates = np.asarray(draws.balls), np.asarray(draws.strong), np.asarray(draws.dates)
    if (balls.shape != (len(draws), N_DRAWN) or strong.shape != (len(draws),)
            or not np.issubdtype(balls.dtype, np.integer)
            or not np.issubdtype(strong.dtype, np.integer)
            or np.any((balls < 1) | (balls > N_NUMBERS))
            or np.any((strong < 1) | (strong > N_STRONG))
            or np.any(np.diff(np.sort(balls, axis=1), axis=1) == 0)):
        raise ValueError("Require six distinct integer balls in 1..37 and an integer strong number in 1..7")
    if (dates.shape != (len(draws),) or not np.issubdtype(dates.dtype, np.datetime64)
            or np.isnat(dates).any() or np.any(dates[1:] <= dates[:-1])):
        raise ValueError("Require finite, strictly increasing draw dates")


def event_log_ratio(probability, observed, null_probability):
    """Likelihood ratio for one event, leaving its conditional outcomes uniform.

    For a distinguished ball, the event is inclusion in the six-ball set.
    This produces a proper full-set model despite dependence between balls.
    """
    q, hit = np.asarray(probability, dtype=float), np.asarray(observed)
    if (not np.isfinite(q).all() or np.any((q <= 0) | (q >= 1))
            or not np.isfinite(null_probability) or not 0 < null_probability < 1
            or not np.isin(hit, [0, 1]).all()):
        raise ValueError("Require interior probabilities and binary observations")
    return np.where(hit, np.log(q / null_probability), np.log1p(-q) - np.log1p(-null_probability))


def build_experts(draws: Draws, *, min_history=200, sparse_prior=100,
                  windows=(50, 100, 300, None), strengths=(100, 500, 2000),
                  include_marginals=True):
    """Uniform + 37 single-ball + 7 single-strong + dense frequency experts.

    Sparse experts use a Beta posterior mean with prior sample size 100 centered
    on the fair inclusion rate. Each has just one bias parameter. Other outcomes
    are uniform conditional on that event. Dense experts turn smoothed frequency
    odds into conditional-Bernoulli weights; those are *not* claimed to match the
    original frequencies exactly after conditioning on a set size of six.
    """
    validate_draws(draws)
    if (isinstance(sparse_prior, (bool, np.bool_)) or not np.isscalar(sparse_prior)
            or not np.isfinite(sparse_prior) or sparse_prior <= 0
            or any(isinstance(s, (bool, np.bool_)) or not np.isfinite(s) or s <= 0 for s in strengths)):
        raise ValueError("Require positive finite prior strengths")
    dense_names, targets, freq_b, freq_s = frequency_forecasts(
        draws, min_history=min_history, windows=windows, strengths=strengths)
    n = len(targets)
    hot = draws.multi_hot
    strong_hot = np.eye(N_STRONG)[draws.strong - 1]
    counts_b = np.vstack([np.zeros(N_NUMBERS), hot.cumsum(axis=0, dtype=float)])[targets]
    counts_s = np.vstack([np.zeros(N_STRONG), strong_hot.cumsum(axis=0)])[targets]
    q_b = (counts_b + sparse_prior * N_DRAWN / N_NUMBERS) / (targets[:, None] + sparse_prior)
    q_s = (counts_s + sparse_prior / N_STRONG) / (targets[:, None] + sparse_prior)
    names, families = ["uniform"], ["uniform"]
    lp = [np.full(n, -LOG_FULL_OUTCOMES)]
    bp = [np.full((n, N_NUMBERS), N_DRAWN / N_NUMBERS)] if include_marginals else None
    sp = [np.full((n, N_STRONG), 1 / N_STRONG)] if include_marginals else None
    for i in range(N_NUMBERS):
        names.append(f"single ball {i + 1}")
        families.append("single_ball")
        lp.append(-LOG_FULL_OUTCOMES + event_log_ratio(q_b[:, i], hot[targets, i], N_DRAWN / N_NUMBERS))
        if include_marginals:
            p = np.repeat(((N_DRAWN - q_b[:, i]) / (N_NUMBERS - 1))[:, None], N_NUMBERS, axis=1)
            p[:, i] = q_b[:, i]
            bp.append(p)
            sp.append(np.full((n, N_STRONG), 1 / N_STRONG))
    for i in range(N_STRONG):
        names.append(f"single strong {i + 1}")
        families.append("single_strong")
        lp.append(-LOG_FULL_OUTCOMES + event_log_ratio(q_s[:, i], strong_hot[targets, i], 1 / N_STRONG))
        if include_marginals:
            bp.append(np.full((n, N_NUMBERS), N_DRAWN / N_NUMBERS))
            p = np.repeat(((1 - q_s[:, i]) / (N_STRONG - 1))[:, None], N_STRONG, axis=1)
            p[:, i] = q_s[:, i]
            sp.append(p)
    for name, p, strong in zip(dense_names[:-1], freq_b[:-1], freq_s[:-1]):
        names.append(f"dense {name}")
        families.append("dense")
        theta = np.log(p) - np.log1p(-p)
        lp.append((theta * hot[targets]).sum(axis=1) - log_partition(theta)
                  + np.log(strong[np.arange(n), draws.strong[targets] - 1]))
        if include_marginals:
            bp.append(marginals(theta))
            sp.append(strong)
    prior = np.array([FAMILY_PRIOR[f] / families.count(f) for f in families])
    return JointExperts(names, families, targets, prior, np.asarray(lp),
                        np.asarray(bp) if include_marginals else None,
                        np.asarray(sp) if include_marginals else None)


def mix_experts(log_prob, prior, *, share=0.0):
    """Log-loss mixture: predict, observe, update, then optionally share weights.

    weights[t] contains no information from outcome t or later. With share=0,
    the cumulative probability equals the prior-weighted sum of each expert's
    sequential probability, so total excess log loss over uniform is <= ln(2)
    when uniform has prior weight 1/2. This is a scoring bound, not a win bound.
    Fixed sharing has a different bound and must not inherit the ln(2) claim.
    """
    lp, p = np.asarray(log_prob, dtype=float), np.asarray(prior, dtype=float)
    if (lp.ndim != 2 or not lp.shape[0] or not lp.shape[1]
            or not np.isfinite(lp).all() or np.any(lp > 1e-12)
            or p.shape != (lp.shape[0],) or not np.isfinite(p).all() or np.any(p <= 0)
            or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
            or isinstance(share, (bool, np.bool_)) or not np.isscalar(share)
            or not np.isfinite(share) or not 0 <= share <= 1):
        raise ValueError("Require finite log probabilities, a positive unit-sum prior, and share in [0,1]")
    log_w = np.log(p)
    weights = np.empty((lp.shape[1] + 1, len(p)))
    forecast = np.empty(lp.shape[1])
    for t in range(lp.shape[1]):
        weights[t] = np.exp(log_w)
        forecast[t] = logsumexp(log_w + lp[:, t])
        posterior = log_w + lp[:, t] - forecast[t]
        if share == 1:
            log_w = np.log(p)
        elif share > 0:
            log_w = np.logaddexp(np.log1p(-share) + posterior, np.log(share) + np.log(p))
        else:
            log_w = posterior
    weights[-1] = np.exp(log_w)
    return {"log_prob": forecast, "weights_before_draw": weights[:-1], "next_weights": weights[-1],
            "log_e_path": np.cumsum(forecast + LOG_FULL_OUTCOMES)}
