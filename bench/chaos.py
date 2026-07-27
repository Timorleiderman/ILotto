"""Chaos theory applied to the draw sequence.

This is the best-motivated approach in the repository, because its physical
premise is simply *correct*: the lottery machine is a deterministic chaotic
system. Balls obey Newton; their collisions diverge exponentially; nothing
quantum is involved. If you knew the exact state of the chamber, the draw
would be computable. Chaos theory is the mathematics built for exactly this
situation, and its tools are standard:

* **Takens delay embedding** — reconstruct a system's attractor from a single
  observed time series.
* **Grassberger–Procaccia correlation dimension** — chaos concentrates on a
  low-dimensional attractor (the logistic map saturates near 1, Lorenz near
  2.05); noise fills every embedding dimension you give it.
* **Lorenz's method of analogues** — find past states similar to the current
  one and predict from what followed them. The original weather-forecasting
  idea, and a genuine short-horizon predictor for real chaotic systems.
* **Surrogate data** — the canonical chaos-vs-noise hypothesis test: shuffle
  the sequence (destroying dynamics, preserving composition) and ask whether
  the real ordering predicts better than its shuffles.

Why it still fails here is a precise, physical statement, not a technicality.
Takens' theorem requires the observations to be successive measurements of
**one evolving trajectory**. The draw sequence is not a trajectory: the machine
is emptied, reloaded and reset between draws, so no state survives from one
draw to the next. Each draw is a fresh orbit observed once. The chaos is real
but it lives *inside* a draw, on millisecond timescales, and its defining
property — exponential sensitivity — is exactly what guarantees that the reset
erases everything: an immeasurably small difference in how the balls are loaded
grows past macroscopic within a few collisions.

The contrast that proves this is not defeatism: chaos exploitation genuinely
works where the current run's initial conditions are observable before the
outcome closes — the Eudaemons' and Small & Tse's roulette work measured the
wheel and ball *of the spin being bet on*. A lottery reveals nothing about the
current draw before it happens. Cross-draw tools are all that remain, and the
measurements in this module show the cross-draw sequence is indistinguishable
from noise on every instrument chaos theory owns.
"""

from __future__ import annotations

import logging

import numpy as np

from .data import N_DRAWN, N_NUMBERS, N_STRONG, Draws
from .predictors import Predictor

logger = logging.getLogger(__name__)


def delay_embed(x: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
    """Takens delay-coordinate embedding: rows are (x_t, x_{t+tau}, ..., x_{t+(m-1)tau})."""
    n = len(x) - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"series of length {len(x)} too short for m={m}, tau={tau}")
    return np.stack([x[i * tau : i * tau + n] for i in range(m)], axis=1)


def correlation_dimension(
    x: np.ndarray, m: int, theiler: int = 10, n_ref: int = 400, seed: int = 0
) -> float:
    """Grassberger–Procaccia correlation-dimension estimate at embedding dim m.

    The reading that matters is the *trend across m*: a chaotic series saturates
    near its attractor's dimension as m grows; a noise series keeps climbing
    with m because noise fills whatever space it is embedded in. The Theiler
    window excludes temporally adjacent points so autocorrelation cannot pose
    as geometry.
    """
    z = (x - x.mean()) / x.std()
    X = delay_embed(z, m)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_ref, len(X)), replace=False)
    dist = np.linalg.norm(X[idx, None, :] - X[None, :, :], axis=2)
    t_gap = np.abs(idx[:, None] - np.arange(len(X))[None, :])
    dist = dist[t_gap > theiler]

    r_lo, r_hi = np.quantile(dist, 0.05), np.quantile(dist, 0.5)
    rs = np.geomspace(r_lo, r_hi, 12)
    corr = np.array([np.mean(dist < r) for r in rs])
    good = corr > 0
    return float(np.polyfit(np.log(rs[good]), np.log(corr[good]), 1)[0])


def analogue_skill(
    multi_hot: np.ndarray,
    n_eval: int = 120,
    m: int = 3,
    k: int = 20,
) -> float:
    """Mean matched numbers of the method of analogues over the last n_eval draws.

    Used both as the observed statistic of the surrogate test and to score the
    shuffles, so it must be cheap: windows are precomputed once and each step is
    one distance computation against a prefix.
    """
    n = len(multi_hot)
    flat = np.lib.stride_tricks.sliding_window_view(multi_hot, (m, multi_hot.shape[1]))
    windows = flat.reshape(len(flat), -1)  # window t covers draws t .. t+m-1

    hits = np.empty(n_eval)
    for j, i in enumerate(range(n - n_eval, n)):
        target = multi_hot[i - m : i].reshape(-1)
        past = windows[: i - m]  # successors of these windows are all < i
        dist = np.linalg.norm(past - target, axis=1)
        nbrs = np.argpartition(dist, k)[:k]
        pred = multi_hot[nbrs + m].mean(axis=0)
        top6 = np.argpartition(-pred, N_DRAWN)[:N_DRAWN]
        hits[j] = multi_hot[i][top6].sum()
    return float(hits.mean())


class MethodOfAnalogues(Predictor):
    """Lorenz's method of analogues as a walk-forward predictor.

    Embeds the history as overlapping m-draw windows, finds the k windows most
    similar to the last m draws, and predicts the next draw as the smoothed
    average of what followed each analogue. On a genuinely deterministic
    sequence this is devastating — it scores a perfect 6/6 on a periodic
    control, because similar states there really do have similar futures. On
    the real archive the nearest "analogues" are nearest by coincidence, and
    their successors are just k random draws.
    """

    name = "Method of analogues (chaos)"
    description = (
        "Lorenz's nearest-neighbour forecaster from chaotic dynamics: find the k past "
        "3-draw windows most similar to the current one and average their successors. "
        "The standard short-horizon predictor for real chaotic systems."
    )
    # The smoothed successor frequencies are a genuine k-NN estimate of the
    # inclusion probabilities, and they sum to exactly 6 by construction.
    emits_probabilities = True

    def __init__(self, m: int = 3, k: int = 20, prior_strength: float = 5.0):
        self.m = m
        self.k = k
        self.prior = prior_strength

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        # float64 up front: these are returned as probabilities, and float32
        # accumulation puts the sum visibly off 6.
        M = history.multi_hot.astype(np.float64)
        n = len(M)
        target = M[n - self.m :].reshape(-1)

        flat = np.lib.stride_tricks.sliding_window_view(M, (self.m, N_NUMBERS))
        windows = flat.reshape(len(flat), -1)
        # A window's successor is the draw at its end index + 1; the last usable
        # window is the one whose successor is the final known draw.
        usable = windows[:-1]
        dist = np.linalg.norm(usable - target, axis=1)
        nbrs = np.argpartition(dist, self.k)[: self.k]
        successors = M[nbrs + self.m]

        p0 = N_DRAWN / N_NUMBERS
        ball_scores = (successors.sum(axis=0) + self.prior * p0) / (self.k + self.prior)

        strong_counts = np.bincount(
            history.strong[nbrs + self.m] - 1, minlength=N_STRONG
        ).astype(float)
        strong_scores = (strong_counts + self.prior / N_STRONG) / (self.k + self.prior)
        return ball_scores, strong_scores


def chaos_suite() -> list[Predictor]:
    return [MethodOfAnalogues()]
