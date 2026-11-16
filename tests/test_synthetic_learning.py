"""Check generator calibration, genuine synthetic learning, and target isolation."""

import numpy as np
import pytest

from lotto.core.data import Draws
from lotto.forecasting.joint_mixture import LOG_FULL_OUTCOMES
from lotto.forecasting.synthetic_learning import components, fit_mixture, frozen_forecast, simulate_episode


def test_fair_generator_has_correct_marginals_and_exact_oracle():
    d, oracle, _ = simulate_episode(60000, np.random.default_rng(1881), 'fair')
    assert np.all(np.diff(d.balls, axis=1) > 0)
    np.testing.assert_allclose(d.multi_hot.mean(axis=0), 6 / 37, atol=.006)
    np.testing.assert_allclose(np.bincount(d.strong, minlength=8)[1:] / len(d), 1 / 7, atol=.006)
    np.testing.assert_array_equal(oracle, np.full(len(d), -LOG_FULL_OUTCOMES))


def test_planted_bias_and_oracle_follow_declared_probabilities():
    d, oracle, blocks = simulate_episode(40000, np.random.default_rng(1882), 'persistent_bias',
                                         multipliers=(2.,))
    spec = blocks[0]
    ball_hit = (d.balls == spec['ball']).any(axis=1)
    strong_hit = d.strong == spec['strong']
    assert ball_hit.mean() == pytest.approx(12 / 37, abs=.008)
    assert strong_hit.mean() == pytest.approx(2 / 7, abs=.008)
    # Independently evaluate the generator's full-outcome probability ratio.
    ratio = np.where(ball_hit, 2., (1 - 12 / 37) / (1 - 6 / 37))
    ratio *= np.where(strong_hit, 2., (1 - 2 / 7) / (1 - 1 / 7))
    np.testing.assert_allclose(oracle, np.log(ratio) - LOG_FULL_OUTCOMES)
    other = np.delete(d.multi_hot.mean(axis=0), spec['ball'] - 1)
    np.testing.assert_allclose(other, (6 - 12 / 37) / 36, atol=.008)


def test_simulation_reproducible_with_independent_split_seeds():
    a, _, blocks = simulate_episode(620, np.random.default_rng([8, 0, 1]), 'changing_bias')
    b, _, again = simulate_episode(620, np.random.default_rng([8, 0, 1]), 'changing_bias')
    c, _, _ = simulate_episode(620, np.random.default_rng([8, 1, 1]), 'changing_bias')
    np.testing.assert_array_equal(a.balls, b.balls)
    assert blocks == again
    assert [(r['start'], r['stop']) for r in blocks] == [(0, 250), (250, 500), (500, 620)]
    assert not np.array_equal(a.balls, c.balls)


def test_current_and_future_targets_cannot_change_forecast():
    d, _, _ = simulate_episode(65, np.random.default_rng(1883), 'fair')
    b, s = d.balls.copy(), d.strong.copy()
    b[40:] = np.arange(1, 7)
    s[40:] = 7
    changed = Draws(d.dates, b, s)
    kwargs = {'windows': (10, None), 'strengths': (100,), 'include_marginals': True}
    first, second = components(d, **kwargs), components(changed, **kwargs)
    for key in ['ball_p', 'strong_p']:
        np.testing.assert_array_equal(first[key][:, :41], second[key][:, :41])
    np.testing.assert_array_equal(first['log_prob'][:, :40], second['log_prob'][:, :40])
    weights = [.2, .3, .5]
    a, b = frozen_forecast(first, weights), frozen_forecast(second, weights)
    np.testing.assert_array_equal(a['ball_p'][:41], b['ball_p'][:41])
    np.testing.assert_allclose(a['ball_p'].sum(axis=1), 6, atol=1e-12)
    np.testing.assert_allclose(a['strong_p'].sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(first['ball_p'][:, 0], 6 / 37)
    np.testing.assert_array_equal(first['log_prob'][:, 0], np.full(3, -LOG_FULL_OUTCOMES))


def test_training_recovers_known_mixture_and_beats_equal_weights():
    # Components predict P(yes)=.8 and .2; the sample frequency is .7.
    # The analytic optimum is (.7-.2)/(.8-.2)=5/6 on the first component.
    lp = np.log(np.array([[.8] * 14 + [.2] * 6, [.2] * 14 + [.8] * 6]))
    fitted = fit_mixture(lp)
    np.testing.assert_allclose(fitted['weights'], [5 / 6, 1 / 6], atol=2e-5)
    assert fitted['mean_training_log_loss'] < fitted['initial_equal_weight_log_loss']


def test_synthetic_training_transfers_to_independent_planted_bias():
    train, _, _ = simulate_episode(4000, np.random.default_rng(1884), 'persistent_bias', multipliers=(2.,))
    test, _, _ = simulate_episode(4000, np.random.default_rng(1885), 'persistent_bias', multipliers=(2.,))
    kwargs = {'windows': (None,), 'strengths': (100, 500)}
    fitted = fit_mixture(components(train, **kwargs)['log_prob'])
    actual = frozen_forecast(components(test, **kwargs), fitted['weights'])
    assert -actual['log_prob'].mean() < LOG_FULL_OUTCOMES - .01


@pytest.mark.parametrize('scenario', ['unknown', None])
def test_bad_scenarios_rejected(scenario):
    with pytest.raises(ValueError):
        simulate_episode(10, np.random.default_rng(1), scenario)


def test_invalid_weights_and_training_values_rejected():
    with pytest.raises(ValueError):
        fit_mixture([[np.nan]])
    with pytest.raises(ValueError):
        fit_mixture([[.1]])
    d, _, _ = simulate_episode(10, np.random.default_rng(8), 'fair')
    c = components(d, windows=(None,), strengths=(100,))
    with pytest.raises(ValueError):
        frozen_forecast(c, [.4, .4])
    with pytest.raises(ValueError):
        components(d, strengths=(0,))
