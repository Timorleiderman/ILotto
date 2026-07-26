"""Candidate strategies, from folklore to neural networks.

Every predictor implements the same contract:

    scores(history) -> (ball_scores[37], strong_scores[7])

`history` contains only draws strictly before the one being predicted, which is
what makes the walk-forward backtest leak-free. Scores need not be normalised;
metrics normalise where a distribution is required.
"""

from __future__ import annotations

import logging

import numpy as np

from .data import N_DRAWN, N_NUMBERS, N_STRONG, Draws

logger = logging.getLogger(__name__)


class Predictor:
    """Base class. Stateless by default; override `fit` for trained models."""

    name = "base"
    description = ""
    #: Refit cadence in draws for models with a `fit` step (0 = never refit).
    refit_every = 0
    #: True only when `scores` returns values proportional to a genuine
    #: inclusion probability per number. Log loss is a proper scoring rule, so
    #: it is meaningless for predictors that emit ranks, gaps or signed values:
    #: normalising those into a distribution is an arbitrary transform, and for
    #: all-negative scores it collapses to the uniform prior and reports exactly
    #: the baseline no matter how the numbers were ranked. Those predictors get
    #: no log loss rather than a flattering one.
    emits_probabilities = False
    #: True for models conditioned on *when* the draw happens rather than on the
    #: preceding draws. The harness then passes the target draw's date. This is
    #: not lookahead: the date of the next draw is public in advance. What it
    #: must never receive is the draw itself, which `LeakageError` enforces.
    wants_target_date = False

    def fit(self, history: Draws) -> None:  # noqa: D102
        pass

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _uniform_strong(self) -> np.ndarray:
        return np.full(N_STRONG, 1.0 / N_STRONG)


class UniformRandom(Predictor):
    name = "Random ticket"
    description = "Six numbers drawn uniformly at random. The honest null hypothesis."

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        return self._rng.random(N_NUMBERS), self._rng.random(N_STRONG)


class Frequency(Predictor):
    """Pick the numbers drawn most often ("hot") or least often ("cold")."""

    def __init__(self, window: int | None = None, cold: bool = False):
        self.window = window
        self.cold = cold
        # Counts normalise to a real categorical distribution; negated counts
        # do not, so the cold variant is scored on matches only.
        self.emits_probabilities = not cold
        span = "all-time" if window is None else f"last {window} draws"
        self.name = f"{'Cold' if cold else 'Hot'} numbers ({span})"
        self.description = (
            f"Ranks numbers by how {'rarely' if cold else 'often'} they appeared over the {span}. "
            f"The classic {'due-number' if cold else 'hot-hand'} heuristic."
        )

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        recent = history if self.window is None else history.slice(-self.window, None)
        counts = np.bincount(recent.balls.ravel(), minlength=N_NUMBERS + 1)[1:].astype(float)
        strong = np.bincount(recent.strong, minlength=N_STRONG + 1)[1:].astype(float)
        if self.cold:
            counts, strong = -counts, -strong
        return counts, strong


class ExponentialRecency(Predictor):
    """Frequency with exponential forgetting: recent draws count for more."""

    emits_probabilities = True

    def __init__(self, half_life: int = 50):
        self.half_life = half_life
        self.name = f"EWMA frequency (half-life {half_life})"
        self.description = (
            f"Frequency count where a draw {half_life} draws ago carries half the weight of the "
            "latest one. Smoothly interpolates between 'hot recently' and 'hot all-time'."
        )

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        n = len(history)
        age = np.arange(n - 1, -1, -1)
        w = 0.5 ** (age / self.half_life)
        ball_scores = (history.multi_hot * w[:, None]).sum(axis=0)
        strong_scores = np.zeros(N_STRONG)
        np.add.at(strong_scores, history.strong - 1, w)
        return ball_scores, strong_scores


class Overdue(Predictor):
    """Rank by how many draws have passed since a number last appeared."""

    name = "Overdue (longest absence)"
    description = (
        "Ranks numbers by draws elapsed since their last appearance. The gambler's-fallacy "
        "strategy, included because it is the most common thing players actually do."
    )

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        m = history.multi_hot
        n = len(history)
        ball_scores = np.full(N_NUMBERS, float(n))
        for num in range(N_NUMBERS):
            idx = np.flatnonzero(m[:, num])
            if len(idx):
                ball_scores[num] = n - idx[-1]
        strong_scores = np.full(N_STRONG, float(n))
        for s in range(N_STRONG):
            idx = np.flatnonzero(history.strong == s + 1)
            if len(idx):
                strong_scores[s] = n - idx[-1]
        return ball_scores, strong_scores


class PairwiseMarkov(Predictor):
    """Score numbers by co-occurrence with the numbers in the previous draw.

    If the machine had any mechanical memory this is where it would show up:
    a ball that tends to follow another would get a persistently high score.
    """

    name = "Pairwise co-occurrence (Markov)"
    description = (
        "Builds a 37x37 co-occurrence matrix over history, then scores each number by how often "
        "it has appeared alongside the numbers in the most recent draw. Laplace-smoothed."
    )

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        m = history.multi_hot
        co = m.T @ m + self.alpha
        np.fill_diagonal(co, 0.0)
        last = history.balls[-1] - 1
        ball_scores = co[last].sum(axis=0)
        ball_scores[last] = -np.inf  # a repeat of the whole last draw is not the hypothesis
        return ball_scores, self._uniform_strong()


class SequentialMarkov(Predictor):
    """P(number appears at t | number appeared at t-1), estimated per number.

    Distinct from `PairwiseMarkov`: that one is about co-occurrence *within* a
    draw, this one is about persistence *across* draws.
    """

    name = "Lag-1 persistence"
    # Returns P(number drawn | its state last draw) per number: calibrated.
    emits_probabilities = True
    description = (
        "Estimates, for each number independently, the probability it is drawn given whether it "
        "was drawn last time, then scores by that conditional probability."
    )

    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        m = history.multi_hot.astype(bool)
        prev, nxt = m[:-1], m[1:]
        was_in = prev.sum(axis=0)
        stayed = (prev & nxt).sum(axis=0)
        left = (~prev & nxt).sum(axis=0)
        was_out = (~prev).sum(axis=0)

        p_given_in = (stayed + self.alpha) / (was_in + 2 * self.alpha)
        p_given_out = (left + self.alpha) / (was_out + 2 * self.alpha)

        last = m[-1]
        return np.where(last, p_given_in, p_given_out), self._uniform_strong()


class SumConstrained(Predictor):
    """Pick the combination whose sum sits at the mode of the sum distribution.

    Does not change the probability of winning, but it does change *who else*
    holds the same ticket. Included to make the point precisely: this is the
    only lever that has ever mattered in a pari-mutuel game.
    """

    name = "Sum-targeted combination"
    description = (
        "Selects six numbers whose total lands near the centre of the historical sum "
        "distribution (~114). Does not improve win probability; included as a control."
    )

    def __init__(self, target: int = 114, seed: int = 0):
        self.target = target
        self._rng = np.random.default_rng(seed)

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        best, best_gap = None, np.inf
        for _ in range(2000):
            pick = self._rng.choice(N_NUMBERS, size=N_DRAWN, replace=False) + 1
            gap = abs(pick.sum() - self.target)
            if gap < best_gap:
                best, best_gap = pick, gap
                if gap == 0:
                    break
        s = np.zeros(N_NUMBERS)
        s[best - 1] = 1.0
        return s, self._uniform_strong()


class UnpopularNumbers(Predictor):
    """Avoid numbers humans over-pick, to reduce the chance of splitting a jackpot.

    This is the one strategy in the file with a real (if small) positive
    expected-value story, and it works on *player* behaviour, not on the
    machine. Calendar-driven picks mean 1-12 and 1-31 are heavily over-played,
    so tickets weighted toward 32-37 are worth more conditional on winning.
    """

    name = "Unpopular numbers (EV-aware)"
    description = (
        "Prefers 32-37 and other numbers rarely chosen by players. Win probability is identical "
        "to a random ticket, but a winning ticket is less likely to be shared. This is the only "
        "strategy here that can raise expected value, and it does so via player behaviour."
    )

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        # Popularity proxy: day-of-month picks dominate, so 1-31 are crowded and
        # 1-12 (months) doubly so. Weights are a documented prior, not fitted.
        s = np.ones(N_NUMBERS)
        s[:31] -= 0.5
        s[:12] -= 0.3
        s[6] -= 0.2  # 7 is the single most over-picked number in most lotteries
        return s, self._uniform_strong()


class Ensemble(Predictor):
    """Rank-average of several predictors.

    Ensembling genuinely helps when components carry independent signal. Here it
    mostly serves as a check: if the parts are noise, so is the whole.
    """

    name = "Rank ensemble"
    description = "Averages the normalised ranks of the statistical predictors."

    def __init__(self, members: list[Predictor]):
        self.members = members

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        ball_acc = np.zeros(N_NUMBERS)
        strong_acc = np.zeros(N_STRONG)
        for m in self.members:
            b, s = m.scores(history)
            ball_acc += np.argsort(np.argsort(np.nan_to_num(b, neginf=-1e18))) / N_NUMBERS
            strong_acc += np.argsort(np.argsort(s)) / N_STRONG
        return ball_acc / len(self.members), strong_acc / len(self.members)


def generative_suite() -> list[Predictor]:
    """The date-conditioned generative model. Imported lazily: it needs SciPy's
    optimiser, and `predictors` is imported by tests that must stay fast."""
    from .generative import DateConditionedCB

    return [DateConditionedCB()]


def statistical_suite() -> list[Predictor]:
    """The non-neural strategies, in report order."""
    members = [
        Frequency(window=None),
        Frequency(window=100),
        Frequency(window=None, cold=True),
        ExponentialRecency(half_life=50),
        Overdue(),
        PairwiseMarkov(),
        SequentialMarkov(),
    ]
    return [
        UniformRandom(seed=12345),
        *members,
        SumConstrained(),
        UnpopularNumbers(),
        Ensemble(members),
    ]
