"""Scoring rules for set-valued lottery prediction.

The original evaluation measured per-position top-k accuracy against balls
that are *sorted ascending*. That is not a prediction task: `Ball_1` is the
minimum of six draws from 1-37 and is small almost by definition, so a model
that memorises the marginal distribution of each order statistic scores well
while knowing nothing. Every metric here is invariant to ordering and is
compared against the exact combinatorial baseline.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .data import N_DRAWN, N_NUMBERS, N_STRONG

#: E[matches] for a uniformly random ticket: 6 picks x P(any pick is drawn).
BASELINE_MATCHES = N_DRAWN * N_DRAWN / N_NUMBERS  # 0.9730
#: Var[matches] for a random ticket (hypergeometric with N=37, K=6, n=6).
BASELINE_MATCH_VAR = stats.hypergeom(N_NUMBERS, N_DRAWN, N_DRAWN).var()
#: E[log p] for a uniform predictive distribution over the 37 numbers.
BASELINE_LOGLOSS = float(np.log(N_NUMBERS))
BASELINE_STRONG_ACC = 1.0 / N_STRONG


def top_k(scores: np.ndarray, k: int = N_DRAWN) -> np.ndarray:
    """Indices (0-based) of the k highest scores, ties broken deterministically."""
    return np.sort(np.argpartition(-scores, k - 1)[:k])


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
    if n_matched == 3 and strong_hit:
        return "7th (3+strong)"
    return "no prize"


def summarise(match_counts: np.ndarray, losses: np.ndarray, strong_hits: np.ndarray) -> dict:
    """Aggregate one predictor's walk-forward results with significance."""
    n = len(match_counts)
    mean_matches = float(np.mean(match_counts))

    # Under H0 (no skill) each draw's match count is hypergeometric and draws
    # are independent, so the mean is asymptotically normal with known variance.
    # This is an exact-null z-test; no need to bootstrap the baseline.
    se = float(np.sqrt(BASELINE_MATCH_VAR / n))
    z = (mean_matches - BASELINE_MATCHES) / se if se > 0 else 0.0
    p_two_sided = float(2 * stats.norm.sf(abs(z)))

    return {
        "n_draws": n,
        "mean_matches": mean_matches,
        "baseline_matches": BASELINE_MATCHES,
        "lift_pct": 100 * (mean_matches / BASELINE_MATCHES - 1),
        "match_se": se,
        "z_score": float(z),
        "p_value": p_two_sided,
        "mean_log_loss": float(np.mean(losses)),
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
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    verdict: dict[str, bool] = {}
    rejected_so_far = True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        rejected_so_far = rejected_so_far and p <= threshold
        verdict[name] = rejected_so_far
    return verdict
