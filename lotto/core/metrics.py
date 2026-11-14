"""Scoring rules for set-valued lottery prediction.

The original evaluation measured per-position top-k accuracy against balls
that are *sorted ascending*. That is not a prediction task: `Ball_1` is the
minimum of six draws from 1-37 and is small almost by definition, so a model
that memorises the marginal distribution of each order statistic scores well
while knowing nothing. Every metric here is invariant to ordering and is
compared against the exact combinatorial baseline.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats

from lotto.core.data import N_DRAWN, N_NUMBERS, N_STRONG

#: E[matches] for a uniformly random ticket: 6 picks x P(any pick is drawn).
BASELINE_MATCHES = N_DRAWN * N_DRAWN / N_NUMBERS  # 0.9730
#: Var[matches] for a random ticket (hypergeometric with N=37, K=6, n=6).
BASELINE_MATCH_VAR = stats.hypergeom(N_NUMBERS, N_DRAWN, N_DRAWN).var()
#: E[log p] for a uniform predictive distribution over the 37 numbers.
BASELINE_LOGLOSS = float(np.log(N_NUMBERS))
BASELINE_STRONG_ACC = 1.0 / N_STRONG


@lru_cache(maxsize=8)
def match_sum_distribution(n_draws: int) -> np.ndarray:
    """Finite-sample null for total matches over independent fair draws.

    A ticket chosen only from past data still has the same conditional
    hypergeometric match law. Repeated convolution therefore also covers
    adaptive past-only strategies. It does not excuse selection on test results.
    """
    if (not isinstance(n_draws, (int, np.integer)) or isinstance(n_draws, (bool, np.bool_))
            or n_draws < 1):
        raise ValueError("n_draws must be a positive integer")
    single = stats.hypergeom.pmf(np.arange(N_DRAWN + 1), N_NUMBERS, N_DRAWN, N_DRAWN)
    distribution = np.array([1.0])
    for _ in range(n_draws):
        distribution = np.convolve(distribution, single)
    distribution /= distribution.sum()
    distribution.setflags(write=False)
    return distribution


def top_k(scores: np.ndarray, k: int = N_DRAWN) -> np.ndarray:
    """Indices (0-based) of the k highest scores, in ascending order.

    Ties go to the lower index. A stable sort makes that true on every
    platform; `np.argpartition` leaves the choice among tied scores to a
    CPU-specific kernel, so the same history could produce one ticket on a
    laptop and another on the CI runner.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return np.sort(order[:k])


def matches(scores: np.ndarray, actual_balls: np.ndarray) -> int:
    """How many of the model's top-6 picks were actually drawn."""
    picks = set((top_k(scores) + 1).tolist())
    return len(picks & set(actual_balls.tolist()))


def log_loss(scores: np.ndarray, actual_balls: np.ndarray) -> float:
    """Mean negative log-probability assigned to the six drawn numbers.

    `scores` is normalised to a distribution over the 37 numbers first, so this
    is a *proper* scoring rule: it punishes confident wrong answers, which
    top-k hit counting does not. Uniform scores give exactly log(37) = 3.611.
    """
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, None)
    p = p / p.sum()
    return float(-np.mean(np.log(p[actual_balls - 1])))


def coverage(scores: np.ndarray, actual_balls: np.ndarray, k: int) -> float:
    """Fraction of drawn numbers falling inside the model's top-k shortlist.

    Baseline is k/37. Useful because a model can be informative about *which
    numbers are plausible* long before it can pick six.
    """
    short = set((top_k(scores, k) + 1).tolist())
    return len(short & set(actual_balls.tolist())) / N_DRAWN


def prize_tier(n_matched: int, strong_hit: bool) -> str:
    """Israeli Lotto prize tier for a ticket, used for expected-value framing."""
    if n_matched == 6:
        return "1st (6+strong)" if strong_hit else "2nd (6)"
    if n_matched == 5:
        return "3rd (5+strong)" if strong_hit else "4th (5)"
    if n_matched == 4:
        return "5th (4+strong)" if strong_hit else "6th (4)"
    if n_matched == 3:
        return "7th (3+strong)" if strong_hit else "8th (3)"
    return "no prize"


def summarise(
    match_counts: np.ndarray, losses: np.ndarray | None, strong_hits: np.ndarray
) -> dict:
    """Aggregate one predictor's walk-forward results with significance."""
    match_counts = np.asarray(match_counts)
    strong_hits = np.asarray(strong_hits)
    if (match_counts.ndim != 1 or not match_counts.size or match_counts.dtype.kind not in "iu"
            or np.any((match_counts < 0) | (match_counts > N_DRAWN))):
        raise ValueError("match_counts must be a nonempty integer vector in 0..6")
    if strong_hits.shape != match_counts.shape or not np.isin(strong_hits, [0, 1]).all():
        raise ValueError("strong_hits must contain one binary outcome per draw")
    if losses is not None:
        losses = np.asarray(losses, dtype=float)
        if losses.shape != match_counts.shape or not np.isfinite(losses).all() or np.any(losses < 0):
            raise ValueError("losses must contain one finite nonnegative value per draw")
    n = len(match_counts)
    mean_matches = float(np.mean(match_counts))

    # Keep z as an effect-size diagnostic; test with the finite-sample null.
    se = float(np.sqrt(BASELINE_MATCH_VAR / n))
    z = (mean_matches - BASELINE_MATCHES) / se if se > 0 else 0.0
    distribution = match_sum_distribution(n)
    total = int(np.sum(match_counts))
    p_greater = min(1.0, float(distribution[total:].sum()))
    p_two_sided = min(1.0, 2 * min(float(distribution[:total + 1].sum()), p_greater))

    return {
        "n_draws": n,
        "mean_matches": mean_matches,
        "baseline_matches": BASELINE_MATCHES,
        "lift_pct": 100 * (mean_matches / BASELINE_MATCHES - 1),
        "match_se": se,
        "z_score": float(z),
        "p_value": p_two_sided,
        "p_value_greater": p_greater,
        "p_value_method": "Doubled inclusive tail of convolved hypergeometric null",
        # None for rank-only predictors: see `Predictor.emits_probabilities`.
        "mean_log_loss": None if losses is None else float(np.mean(losses)),
        "baseline_log_loss": BASELINE_LOGLOSS,
        "best_draw_matches": int(np.max(match_counts)),
        "hit_3plus_rate": float(np.mean(match_counts >= 3)),
        "strong_accuracy": float(np.mean(strong_hits)),
        "baseline_strong_accuracy": BASELINE_STRONG_ACC,
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni correction.

    Backtesting a dozen strategies and reporting the best one is how people
    convince themselves they beat a lottery. Every headline claim in the report
    goes through this.
    """
    if not 0 < alpha < 1 or any(not np.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("Require 0 < alpha < 1 and finite p-values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    verdict: dict[str, bool] = {}
    rejected_so_far = True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        rejected_so_far = rejected_so_far and p <= threshold
        verdict[name] = rejected_so_far
    return verdict
