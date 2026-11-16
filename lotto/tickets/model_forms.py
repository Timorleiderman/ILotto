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
from lotto.tickets.forecast_portfolios import ranked_portfolios


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Scores as nonnegative ranks: same order, ties share a rank, -inf ranks last."""
    finite = np.nan_to_num(np.asarray(scores, dtype=float), neginf=-1e300, posinf=1e300)
    return rankdata(finite)


def best_ticket(ball_scores: np.ndarray, strong_scores: np.ndarray) -> list[int]:
    """The model's top six numbers, ascending, followed by its top strong number.

    Ties go to the lower number, as in the backtest, so this is exactly the
    ticket the scoreboard scored the model on.
    """
    return [*(metrics.top_k(np.nan_to_num(ball_scores, neginf=-1e300)) + 1).tolist(),
            int(np.argmax(strong_scores)) + 1]


def model_form(ball_scores: np.ndarray, strong_scores: np.ndarray, design) -> list[list[int]]:
    """The coverage `design`, relabelled by this model's forecast.

    Each row is six ascending main numbers followed by a strong number.
    """
    rows = ranked_portfolios(design, _ranks(ball_scores)[None], _ranks(strong_scores)[None])[0]
    return rows.tolist()
