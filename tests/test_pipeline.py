"""Guards on the parts that were previously wrong.

Each test here corresponds to a specific bug the rewrite fixed, so a regression
shows up as a named failure rather than as a quietly wrong number in a report.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import backtest, data, metrics, predictors, randomness


@pytest.fixture(scope="module")
def draws():
    return data.load(clean_csv=None)


def fake_draws(n=400, seed=0):
    """A synthetic fair 6/37 history, for tests that must not depend on the archive."""
    rng = np.random.default_rng(seed)
    balls = np.sort(rng.random((n, 37)).argsort(axis=1)[:, :6] + 1, axis=1)
    strong = rng.integers(1, 8, size=n)
    dates = np.datetime64("2012-01-01", "D") + np.arange(n) * 3
    return data.Draws(dates.astype("datetime64[ns]"), balls, strong)


def test_draws_are_chronological(draws):
    # The archive ships newest-first; loading it as-is trains models backwards.
    assert (np.diff(draws.dates.astype("datetime64[D]").astype(int)) > 0).all()


def test_only_current_game_format(draws):
    assert draws.balls.min() >= 1 and draws.balls.max() <= data.N_NUMBERS
    assert draws.strong.min() >= 1 and draws.strong.max() <= data.N_STRONG
    assert len(draws) > 1000


def test_no_outcome_based_filtering(draws):
    """Every number must be reachable; a value filter would truncate the top."""
    counts = np.bincount(draws.balls.ravel(), minlength=data.N_NUMBERS + 1)[1:]
    assert (counts > 0).all()
    # A value filter shows up as the highest numbers being rare. Under fair
    # draws no number should sit anywhere near half the expected count.
    assert counts.min() > 0.5 * counts.mean()


def test_multi_hot_round_trip():
    d = fake_draws(50)
    m = d.multi_hot
    assert m.shape == (50, 37)
    assert (m.sum(axis=1) == 6).all()
    assert set(np.flatnonzero(m[0]) + 1) == set(d.balls[0].tolist())


def test_baseline_constant_matches_theory():
    # E[matches] for a random ticket is the hypergeometric mean, 6*6/37.
    from scipy import stats

    assert metrics.BASELINE_MATCHES == pytest.approx(stats.hypergeom(37, 6, 6).mean())


def test_uniform_scores_hit_baseline_log_loss():
    uniform = np.full(37, 1 / 37)
    assert metrics.log_loss(uniform, np.array([1, 2, 3, 4, 5, 6])) == pytest.approx(
        metrics.BASELINE_LOGLOSS
    )


def test_metrics_are_order_invariant():
    """The old per-position metric rewarded knowing the sort order, not the numbers."""
    rng = np.random.default_rng(1)
    scores = rng.random(37)
    balls = np.array([3, 9, 14, 22, 30, 31])
    shuffled = rng.permutation(balls)
    assert metrics.matches(scores, balls) == metrics.matches(scores, shuffled)
    assert metrics.log_loss(scores, balls) == pytest.approx(metrics.log_loss(scores, shuffled))


def test_random_predictor_lands_on_baseline():
    d = fake_draws(900, seed=5)
    r = backtest.walk_forward(predictors.UniformRandom(seed=3), d, n_test=400, min_history=100)
    assert abs(r.summary["z_score"]) < 3.5


def test_walk_forward_never_sees_the_future():
    """A predictor that peeks would score far above chance; assert it cannot."""
    seen: list[int] = []

    class Spy(predictors.Predictor):
        name = "spy"

        def scores(self, history):
            seen.append(len(history))
            return np.zeros(37), np.zeros(7)

    d = fake_draws(300)
    backtest.walk_forward(Spy(), d, n_test=50, min_history=100)
    # Histories are strictly increasing and always stop before the target draw.
    assert seen == list(range(250, 300))


def test_holm_bonferroni_is_monotone():
    verdict = metrics.holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.4})
    assert verdict["a"] is True
    assert verdict["c"] is False
    # Once a hypothesis fails, every less-significant one must fail too.
    assert not (verdict["b"] is False and verdict["c"] is True)


def test_randomness_tests_pass_on_fair_synthetic_data():
    """The test suite must not fire on data that is random by construction."""
    d = fake_draws(1500, seed=11)
    results = randomness.run_all(d)
    assert len(results) == len(randomness.ALL_TESTS)
    corrected = metrics.holm_bonferroni({t["test"]: t["p_value"] for t in results})
    assert not any(corrected.values()), [t for t in results if corrected[t["test"]]]


@pytest.mark.parametrize("predictor", predictors.statistical_suite(), ids=lambda p: p.name)
def test_predictor_contract(predictor):
    d = fake_draws(300)
    ball_scores, strong_scores = predictor.scores(d)
    assert ball_scores.shape == (37,)
    assert strong_scores.shape == (7,)
    assert np.isfinite(strong_scores).all()
    picks = metrics.top_k(np.nan_to_num(ball_scores, neginf=-1e18))
    assert len(set(picks.tolist())) == 6


def test_neural_predictor_end_to_end():
    from bench.nn import NeuralPredictor

    d = fake_draws(300, seed=2)
    p = NeuralPredictor(arch="mlp", epochs=3, refit_every=100, window=10)
    r = backtest.walk_forward(p, d, n_test=20, min_history=100)
    assert r.summary["n_draws"] == 20
    assert 0 <= r.summary["mean_matches"] <= 6


def test_nn_windows_are_causal():
    from bench.nn import make_windows

    d = fake_draws(60)
    X, y, s = make_windows(d, window=10)
    assert len(X) == 50
    # y[i] must be the draw immediately after the window that produced X[i],
    # and must not appear inside it.
    assert np.array_equal(y[0], d.multi_hot[10])
    assert not np.array_equal(X[0][-1][:37], y[0])
