"""A date-conditioned generative model of the draw.

Type in a date — past or future — and this samples that round's ticket. It is the
only model here that does not read the draw history as a sequence; it reads a
calendar.

Which makes it the most dangerous model in the repository, for a reason that has
nothing to do with lotteries. There is exactly one draw per calendar date, so the
date is a **unique key** into the training set. Any sufficiently expressive
conditional model can store a date -> draw lookup table, and it will then
reproduce past draws essentially perfectly. That looks like prophecy and is
merely compression.

The guard is the *function class*, not the size of the date encoding. Our date
basis includes a linear time trend, which makes it injective over the 1,629
training dates — a unique key in a smooth costume. Feed an injective map into a
neural head and it memorises everything, however few "features" you named. Feed
it into a **linear** head and it provably cannot: moving a coefficient to fit one
date moves all 1,628 others. So ball logits here are linear in the date basis,
and `test_head_is_linear` fails the build if anyone inserts a hidden layer.

Model. Conditional-Bernoulli (a.k.a. conditional-Poisson) over the 37 balls,
normalised to size exactly 6:

    P(S | t) = prod_{i in S} exp(theta_i(t))  /  e_6(exp(theta(t))),   |S| = 6

with `e_6` the elementary symmetric polynomial, computed exactly in log space by
an O(37 x 6) dynamic program. That gives an exact normaliser, exact marginal
inclusion probabilities, exact gradients and an exact sampler — no MCMC, no
variational bound, no seed hidden in a hyperparameter. The strong number is an
independent 7-way softmax on the same basis.

The fair-lottery null is a *nested point* of this model (W = 0, alpha constant),
so "no date effect" is a parameter value it can land on exactly rather than
approach.

See `docs/approaches/date-conditioned.md` for the measured result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import comb, log

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .data import N_DRAWN, N_NUMBERS, N_STRONG, Draws
from .predictors import Predictor

logger = logging.getLogger(__name__)

#: Exact constants, in one place, asserted by `test_null_is_the_default`.
#: An early design draft had C(37,6) = 3,262,623 and ln C = 14.998, which is
#: wrong by 0.339 nats/draw — enough to manufacture a ~1.5% "improvement" out of
#: noise. Nothing here is allowed to be a remembered number.
C_37_6 = comb(N_NUMBERS, N_DRAWN)  # 2_324_784
LOG_C_37_6 = log(C_37_6)  # 14.659137689462016
LOG_C_PER_POSITION = LOG_C_37_6 / N_DRAWN  # 2.443189614910336
LOG_37 = log(N_NUMBERS)  # 3.6109179126442243
LOG_7 = log(N_STRONG)  # 1.9459101490553132
UNIFORM_MARGINAL = N_DRAWN / N_NUMBERS  # 0.162162...

N_FEATURES = 11
EPOCH = np.datetime64("2012-01-03", "D")


class LeakageError(RuntimeError):
    """Raised when a model is asked to score a draw it could already have seen."""


@dataclass(frozen=True)
class FeatureRef:
    """Standardisation constants for the trend, fitted on a training prefix only.

    Normalising the trend with statistics from the *whole* dataset would encode
    the archive's end date into every early refit — a leak that is invisible in
    the metrics and impossible to spot by reading `scores`.
    """

    mu: float
    sigma: float

    @classmethod
    def fit(cls, dates: np.ndarray) -> "FeatureRef":
        years = _years_since_epoch(dates)
        sigma = float(years.std())
        return cls(mu=float(years.mean()), sigma=sigma if sigma > 1e-9 else 1.0)


def _years_since_epoch(dates: np.ndarray) -> np.ndarray:
    days = (dates.astype("datetime64[D]") - EPOCH).astype(np.float64)
    return days / 365.25


def date_features(dates: np.ndarray, ref: FeatureRef) -> np.ndarray:
    """The 11-dimensional date basis, phi(t).

    Deliberately excluded, each traceable to a way of cheating:

    * **Inter-draw gap** — not computable for a date a user types without
      consulting the real schedule, and provably null anyway (all five gap tests
      came back at Holm p = 1.000).
    * **Row index / draw ordinal** — that is the position in the archive, not a
      property of the date, and it also reveals where the target sits relative to
      the data end.
    * **Time of day** — there is none; every timestamp in the archive is midnight.
    """
    dates = np.atleast_1d(np.asarray(dates, dtype="datetime64[D]"))
    n = len(dates)
    phi = np.zeros((n, N_FEATURES), dtype=np.float64)

    years = dates.astype("datetime64[Y]")
    doy = (dates - years).astype(int) + 1  # 1..366
    # Weekday: 1970-01-01 was a Thursday, so shift to make Monday == 0.
    weekday = (dates.astype(int) + 3) % 7
    dom = (dates - dates.astype("datetime64[M]")).astype(int) + 1  # 1..31

    for k in range(1, 4):  # three annual harmonics
        ang = 2 * np.pi * k * doy / 365.25
        phi[:, 2 * (k - 1)] = np.sin(ang)
        phi[:, 2 * (k - 1) + 1] = np.cos(ang)

    # Three weekday levels only. Sun/Mon/Wed/Fri total 58 draws between them;
    # a full one-hot would just fit noise.
    phi[:, 6] = (weekday == 1).astype(float)  # Tuesday
    phi[:, 7] = (weekday == 3).astype(float)  # Thursday
    # Saturday and strays are the reference level.

    ang = 2 * np.pi * (dom - 1) / 30.4375
    phi[:, 8] = np.sin(ang)
    phi[:, 9] = np.cos(ang)

    phi[:, 10] = (_years_since_epoch(dates) - ref.mu) / ref.sigma
    return phi


# --------------------------------------------------------------------------
# Exact conditional-Bernoulli machinery.
#
# Everything below is closed form. The elementary symmetric polynomials are
# accumulated in log space so that extreme logits cannot overflow the way a
# direct product would.
# --------------------------------------------------------------------------

NEG_INF = -np.inf


def _forward_esp(theta: np.ndarray, k: int = N_DRAWN) -> np.ndarray:
    """F[t, m, j] = log e_j(x_1..x_m) for x = exp(theta). Shape (n, m+1, k+1).

    `k` and the board size are parameters rather than constants so the DP can be
    checked against brute-force enumeration on a board small enough to enumerate.
    """
    n, size = theta.shape
    f = np.full((n, size + 1, k + 1), NEG_INF)
    f[:, 0, 0] = 0.0
    for m in range(1, size + 1):
        f[:, m, 0] = 0.0
        for j in range(1, k + 1):
            f[:, m, j] = np.logaddexp(f[:, m - 1, j], theta[:, m - 1] + f[:, m - 1, j - 1])
    return f


def _backward_esp(theta: np.ndarray, k: int = N_DRAWN) -> np.ndarray:
    """B[t, m, j] = log e_j(x_m..x_size), 1-indexed in m. Shape (n, size+2, k+1)."""
    n, size = theta.shape
    b = np.full((n, size + 2, k + 1), NEG_INF)
    b[:, size + 1, 0] = 0.0
    for m in range(size, 0, -1):
        b[:, m, 0] = 0.0
        for j in range(1, k + 1):
            b[:, m, j] = np.logaddexp(b[:, m + 1, j], theta[:, m - 1] + b[:, m + 1, j - 1])
    return b


def log_partition(theta: np.ndarray, k: int = N_DRAWN) -> np.ndarray:
    """log e_k(exp(theta)) — the exact normaliser, per row."""
    theta = np.atleast_2d(theta)
    return _forward_esp(theta, k)[:, theta.shape[1], k]


def marginals(theta: np.ndarray, k: int = N_DRAWN) -> np.ndarray:
    """Exact inclusion probabilities pi_i = P(ball i is drawn | t).

    By construction these sum to exactly 6, which is what makes it legitimate
    for the predictor to declare `emits_probabilities`: dividing by 6 gives a
    genuine categorical distribution over the numbers rather than an arbitrary
    rescaling of some score.
    """
    theta = np.atleast_2d(theta)
    n, size = theta.shape
    f, b = _forward_esp(theta, k), _backward_esp(theta, k)
    log_z = f[:, size, k]

    out = np.empty((n, size))
    for i in range(1, size + 1):
        # e_{k-1} over every ball except i, split either side of i.
        parts = [f[:, i - 1, a] + b[:, i + 1, k - 1 - a] for a in range(k)]
        out[:, i - 1] = theta[:, i - 1] + logsumexp(np.stack(parts, axis=1), axis=1) - log_z
    return np.exp(out)


def sample_set(theta: np.ndarray, rng: np.random.Generator, k: int = N_DRAWN) -> np.ndarray:
    """Draw one exact sample of k distinct balls from P(S | theta)."""
    theta = np.atleast_2d(theta)
    size = theta.shape[1]
    b = _backward_esp(theta, k)[0]
    picks, need = [], k
    for i in range(1, size + 1):
        if need == 0:
            break
        if size - i + 1 == need:  # must take all that remain
            picks.extend(range(i, size + 1))
            break
        p = np.exp(theta[0, i - 1] + b[i + 1, need - 1] - b[i, need])
        if rng.random() < min(max(p, 0.0), 1.0):
            picks.append(i)
            need -= 1
    return np.array(sorted(picks), dtype=np.int64)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _unpack(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = 0
    alpha = w[o : o + N_NUMBERS]
    o += N_NUMBERS
    W = w[o : o + N_NUMBERS * N_FEATURES].reshape(N_NUMBERS, N_FEATURES)
    o += N_NUMBERS * N_FEATURES
    beta = w[o : o + N_STRONG]
    o += N_STRONG
    V = w[o : o + N_STRONG * N_FEATURES].reshape(N_STRONG, N_FEATURES)
    return alpha, W, beta, V


N_PARAMS = N_NUMBERS + N_NUMBERS * N_FEATURES + N_STRONG + N_STRONG * N_FEATURES  # 528


def _objective(w, phi, y_balls, y_strong, tau):
    """Penalised negative log-likelihood and its exact gradient.

    The ball gradient is the textbook exponential-family residual,
    d(-log p)/d theta_i = pi_i - 1[i in S], with pi from the same DP that
    produced the normaliser. `test_gradient_is_exact` checks it numerically.
    """
    alpha, W, beta, V = _unpack(w)
    theta = alpha[None, :] + phi @ W.T
    eta = beta[None, :] + phi @ V.T

    log_z = log_partition(theta)
    nll = float(-(theta * y_balls).sum() + log_z.sum())

    log_soft = eta - logsumexp(eta, axis=1, keepdims=True)
    nll += float(-log_soft[np.arange(len(eta)), y_strong].sum())

    resid_b = marginals(theta) - y_balls
    resid_s = np.exp(log_soft)
    resid_s[np.arange(len(eta)), y_strong] -= 1.0

    g_alpha = resid_b.sum(axis=0) + tau * alpha
    g_W = resid_b.T @ phi + tau * W
    g_beta = resid_s.sum(axis=0) + tau * beta
    g_V = resid_s.T @ phi + tau * V

    nll += 0.5 * tau * float((alpha**2).sum() + (W**2).sum() + (beta**2).sum() + (V**2).sum())
    return nll, np.concatenate([g_alpha, g_W.ravel(), g_beta, g_V.ravel()])


@dataclass
class CBFit:
    alpha: np.ndarray
    W: np.ndarray
    beta: np.ndarray
    V: np.ndarray
    ref: FeatureRef
    tau: float
    nll: float
    #: Latest draw this fit actually saw. The leakage guard compares against
    #: THIS, not against whatever history the caller passes to `scores`: a
    #: caller can fit on everything and then hand over a truncated history,
    #: which would sail past a check that only looked at the argument.
    train_max_date: np.datetime64
    #: Every training date, for labelling reconstructions as such.
    train_dates: frozenset

    def logits(self, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = date_features(dates, self.ref)
        return self.alpha[None, :] + phi @ self.W.T, self.beta[None, :] + phi @ self.V.T


def fit_cb(draws: Draws, tau: float = 10.0, maxiter: int = 500) -> CBFit:
    """Fit by L-BFGS on the exact penalised likelihood.

    The objective is convex, so this is deterministic: one optimum, no epochs, no
    early stopping, no RNG. That removes a whole class of leak (an early-stopping
    criterion peeking at the target) and makes CI runs bit-reproducible.
    """
    ref = FeatureRef.fit(draws.dates)
    phi = date_features(draws.dates, ref)
    y_balls = draws.multi_hot.astype(np.float64)
    y_strong = (draws.strong - 1).astype(np.int64)

    res = minimize(
        _objective,
        np.zeros(N_PARAMS),
        args=(phi, y_balls, y_strong, tau),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    alpha, W, beta, V = _unpack(res.x)
    days = draws.dates.astype("datetime64[D]")
    return CBFit(
        alpha, W, beta, V, ref, tau, float(res.fun),
        train_max_date=days.max(),
        train_dates=frozenset(str(d) for d in days),
    )


class DateConditionedCB(Predictor):
    """Generative model of the draw given its calendar date."""

    name = "Date-conditioned generative (CB-6/37)"
    description = (
        "A conditional-Bernoulli log-linear model over the 37 balls with the set size fixed at 6, "
        "whose logits are linear in an 11-dimensional smooth basis of the draw's date (annual "
        "harmonics, weekday, day-of-month phase, linear trend). Exact normaliser, exact marginals, "
        "exact sampler. Generates a ticket for any date, past or future."
    )
    refit_every = 50
    # Earned honestly: the marginals sum to exactly 6, so dividing by 6 yields a
    # real categorical distribution rather than a rescaling of arbitrary scores.
    emits_probabilities = True
    wants_target_date = True

    def __init__(self, tau: float = 10.0):
        self.tau = tau
        self.fit_: CBFit | None = None

    def fit(self, history: Draws) -> None:
        self.fit_ = fit_cb(history, tau=self.tau)
        logger.info("Fitted %s on %d draws; nll %.1f", self.name, len(history), self.fit_.nll)

    def scores(self, history: Draws, target_date=None):
        if self.fit_ is None:
            self.fit(history)
        if target_date is None:
            raise ValueError(f"{self.name} requires the target draw's date")
        # The whole point of this model is that it takes a date, so the one
        # mistake that would flatter it is scoring a draw it was fitted on.
        # Check the FIT, not the passed history — otherwise fitting on
        # everything and then passing a truncated history slips straight
        # through, which is exactly how an in-sample number becomes a headline.
        target = np.datetime64(target_date, "D")
        if self.fit_.train_max_date >= target:
            raise LeakageError(
                f"model was fitted on draws up to {self.fit_.train_max_date} but the target "
                f"is {target}; the target must be strictly after every training draw"
            )
        if len(history) and np.datetime64(history.dates.max(), "D") >= target:
            raise LeakageError(
                f"history ends {history.dates.max()} but target is {target}"
            )
        theta, eta = self.fit_.logits(np.asarray([target_date], dtype="datetime64[D]"))
        strong = np.exp(eta[0] - logsumexp(eta[0]))
        return marginals(theta)[0], strong

    # -- generative extras. Not part of the scoring contract, never scored. ----

    def sample(self, target_date, rng: np.random.Generator | None = None) -> dict:
        """Generate a ticket for any date.

        The returned `in_sample` flag is the safeguard, not decoration: for a
        date the model was fitted on this is a *reconstruction*, and without the
        label a screenshot of it is indistinguishable from a forecast.
        """
        if self.fit_ is None:
            raise RuntimeError("fit() first")
        rng = rng or np.random.default_rng()
        day = np.datetime64(target_date, "D")
        theta, eta = self.fit_.logits(np.asarray([day]))
        p_strong = np.exp(eta[0] - logsumexp(eta[0]))
        in_sample = str(day) in self.fit_.train_dates
        return {
            "date": str(day),
            "numbers": sample_set(theta, rng).tolist(),
            "strong": int(rng.choice(N_STRONG, p=p_strong) + 1),
            "in_sample": in_sample,
            "label": (
                "in-sample reconstruction, not a prediction"
                if in_sample
                else "out-of-sample generation"
            ),
        }

    def set_log_prob(self, balls: np.ndarray, target_date) -> float:
        """Exact log P(this ticket | this date)."""
        theta, _ = self.fit_.logits(np.asarray([target_date], dtype="datetime64[D]"))
        return float(theta[0, np.asarray(balls) - 1].sum() - log_partition(theta)[0])


class MemorisingDateLookup(Predictor):
    """The same idea with the safety removed — a deliberate adversarial arm.

    Each date gets its own free 37-vector of logits, so the model can store the
    training set outright: 1,629 x 37 = 60,273 parameters against an answer key
    of 1,629 x ln C(37,6) = 23,880 nats = 4.3 KB. It exists to be *reported*,
    not suppressed: it reconstructs in-sample draws perfectly and is worth
    exactly nothing out of sample, which is the clearest way to show that
    "it predicted a past draw" is a statement about storage, not about the future.

    Never headline-eligible. `emits_probabilities` stays False: on an unseen date
    it falls back to the uniform prior, so its log loss is not a measure of
    anything.
    """

    name = "Date lookup table (memorisation control)"
    description = (
        "Gives every date its own free parameters, so it can memorise the archive outright. "
        "Included to measure how good memorisation looks, not because it predicts anything."
    )
    refit_every = 50
    wants_target_date = True

    def __init__(self, seed: int = 0):
        self.table: dict[str, np.ndarray] = {}
        # On an unseen date it knows nothing, so it must behave like a random
        # ticket. Returning a flat vector instead would make `top_k` break ties
        # deterministically and quietly turn the control into "always play
        # 1-6" — a fixed ticket, which is a different null.
        self._rng = np.random.default_rng(seed)

    def fit(self, history: Draws) -> None:
        self.table = {
            str(np.datetime64(d, "D")): mh for d, mh in zip(history.dates, history.multi_hot)
        }

    def scores(self, history: Draws, target_date=None):
        key = str(np.datetime64(target_date, "D"))
        if key in self.table:  # a date it was trained on: perfect recall
            return self.table[key] * 100.0 + UNIFORM_MARGINAL, np.full(N_STRONG, 1 / N_STRONG)
        return self._rng.random(N_NUMBERS), np.full(N_STRONG, 1 / N_STRONG)


def reconstruction_rate(predictor: Predictor, draws: Draws) -> dict:
    """DKRR — Date-Keyed Reconstruction Rate, on dates the model was trained on.

    This is the seductive number: "give it a past date and it gets N of 6 right".
    It is a *compression* statistic, not a prediction statistic — it measures how
    much of the training targets was absorbed into the parameters. It is returned
    alongside `memorisation_gap` and must never be reported without it.
    """
    from . import metrics

    predictor.fit(draws)
    matched, losses = [], []
    for i in range(len(draws)):
        # Deliberately passing the *full* history: that is what reconstruction
        # means, and it is why this number is not a forecast.
        if isinstance(predictor, DateConditionedCB):
            theta, _ = predictor.fit_.logits(draws.dates[i : i + 1])
            ball_scores = marginals(theta)[0]
        else:
            ball_scores, _ = predictor.scores(draws, draws.dates[i])
        matched.append(metrics.matches(ball_scores, draws.balls[i]))
        losses.append(metrics.log_loss(ball_scores, draws.balls[i]))
    return {
        "dkrr_matches": float(np.mean(matched)),
        "dkrr_log_loss": float(np.mean(losses)),
        "n": len(draws),
        "baseline_matches": metrics.BASELINE_MATCHES,
    }
