"""Leak-free walk-forward evaluation.

For each of the last `n_test` draws we hand a predictor *only* the draws that
preceded it, ask for a ticket, and score it. This is the part the original
pipeline got wrong in three separate ways:

1. Draws were fed newest-first, so the model was trained to predict the past
   from the future.
2. The training split was the oldest block and the test split the newest, but
   with the reversed ordering the sequence windows straddled the boundary.
3. Nothing was ever compared against a random ticket, so there was no way to
   tell whether any number meant anything.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .data import Draws
from .predictors import Predictor

logger = logging.getLogger(__name__)


@dataclass
class Result:
    name: str
    description: str
    summary: dict
    match_counts: np.ndarray = field(repr=False)
    seconds: float = 0.0


def walk_forward(
    predictor: Predictor,
    draws: Draws,
    n_test: int = 300,
    min_history: int = 200,
) -> Result:
    """Score `predictor` over the final `n_test` draws, one draw at a time."""
    start_idx = max(len(draws) - n_test, min_history)
    if start_idx >= len(draws):
        raise ValueError(f"Not enough draws ({len(draws)}) for min_history={min_history}")

    match_counts, losses, strong_hits = [], [], []
    t0 = time.perf_counter()

    for i in range(start_idx, len(draws)):
        history = draws.slice(0, i)
        if predictor.refit_every and (i - start_idx) % predictor.refit_every == 0:
            predictor.fit(history)

        ball_scores, strong_scores = predictor.scores(history)
        actual = draws.balls[i]

        match_counts.append(metrics.matches(ball_scores, actual))
        if predictor.emits_probabilities:
            losses.append(metrics.log_loss(ball_scores, actual))
        strong_hits.append(int(np.argmax(strong_scores) + 1 == draws.strong[i]))

    match_counts = np.asarray(match_counts)
    summary = metrics.summarise(
        match_counts,
        np.asarray(losses) if predictor.emits_probabilities else None,
        np.asarray(strong_hits),
    )

    return Result(
        name=predictor.name,
        description=predictor.description,
        summary=summary,
        match_counts=match_counts,
        seconds=time.perf_counter() - t0,
    )


def random_ticket_null(draws: Draws, n_test: int, n_trials: int = 5000, seed: int = 7) -> dict:
    """Empirical distribution of mean-matches for random tickets.

    The z-test in `metrics.summarise` assumes asymptotic normality; this
    Monte-Carlo null confirms it and gives the report a concrete "how good would
    the luckiest of 5000 random strategies have looked?" number, which is the
    right yardstick when a dozen strategies are being compared.
    """
    rng = np.random.default_rng(seed)
    actual = draws.balls[-n_test:]
    hot = np.zeros((n_test, metrics.N_NUMBERS), dtype=bool)
    hot[np.repeat(np.arange(n_test), 6), actual.ravel() - 1] = True

    means = np.empty(n_trials)
    for t in range(n_trials):
        picks = rng.random((n_test, metrics.N_NUMBERS)).argsort(axis=1)[:, :6]
        means[t] = np.take_along_axis(hot, picks, axis=1).sum(axis=1).mean()

    return {
        "n_trials": n_trials,
        "mean": float(means.mean()),
        "std": float(means.std()),
        "p95": float(np.percentile(means, 95)),
        "max": float(means.max()),
        "samples": means,
    }


def run_suite(predictors: list[Predictor], draws: Draws, n_test: int = 300) -> list[Result]:
    results = []
    for p in predictors:
        logger.info("Backtesting %s", p.name)
        results.append(walk_forward(p, draws, n_test=n_test))
    results.sort(key=lambda r: r.summary["mean_matches"], reverse=True)

    verdicts = metrics.holm_bonferroni({r.name: r.summary["p_value"] for r in results})
    for r in results:
        r.summary["significant_after_correction"] = verdicts[r.name]
    return results
