"""Turn one model's forecast into a form you can fill in.

Every model scores the 37 numbers (and the 7 strong numbers) for the next
draw. Its single best ticket is simply its top six numbers and its top strong
number. A whole form is built differently, so that it keeps the exact prize
odds of a coverage design: the design's rows are kept, and its numbers are
relabelled so that the model's favourite numbers take the places used most
often in the design. Relabelling every row the same way keeps every overlap
between rows, so every model's form has the same exact chance of a prize; what
changes is which numbers fill it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from lotto.core import metrics
from lotto.core.data import N_NUMBERS, N_STRONG
from lotto.tickets.forecast_portfolios import ranked_portfolios


def _validate_scores(ball_scores: np.ndarray, strong_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    balls = np.asarray(ball_scores, dtype=float)
    strong = np.asarray(strong_scores, dtype=float)
    if balls.shape != (N_NUMBERS,) or strong.shape != (N_STRONG,):
        raise ValueError(f"Expected ({N_NUMBERS},) ball scores and ({N_STRONG},) strong scores")
    if np.isnan(balls).any() or np.isposinf(balls).any() or not np.isfinite(strong).all():
        raise ValueError("Ball scores must not contain NaN or +inf; strong scores must be finite")
    return balls, strong


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Scores as nonnegative ranks: same order, ties share a rank, -inf ranks last."""
    arr = np.asarray(scores, dtype=float)
    if np.isnan(arr).any() or np.isposinf(arr).any():
        raise ValueError("Scores must not contain NaN or +inf")
    return rankdata(arr)


def best_ticket(ball_scores: np.ndarray, strong_scores: np.ndarray) -> list[int]:
    """The model's top six numbers, ascending, followed by its top strong number.

    Ties go to the lower number, as in the backtest, so this is exactly the
    ticket the scoreboard scored the model on.
    """
    balls, strong = _validate_scores(ball_scores, strong_scores)
    return [*(metrics.top_k(balls) + 1).tolist(), int(np.argmax(strong)) + 1]


def model_form(ball_scores: np.ndarray, strong_scores: np.ndarray, design) -> list[list[int]]:
    """The coverage `design`, relabelled by this model's forecast.

    Each row is six ascending main numbers followed by a strong number.
    """
    balls, strong = _validate_scores(ball_scores, strong_scores)
    rows = ranked_portfolios(design, _ranks(balls)[None], _ranks(strong)[None])[0]
    return rows.tolist()
