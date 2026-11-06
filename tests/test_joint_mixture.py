from itertools import combinations
from math import comb

import numpy as np
import pytest
from scipy.special import logsumexp

from bench.data import Draws
from bench.joint_mixture import LOG_FULL_OUTCOMES, build_experts, event_log_ratio, mix_experts


def fixture_draws(n=60):
    rng = np.random.default_rng(218)
    balls = np.array([np.sort(rng.choice(37, 6, replace=False) + 1) for _ in range(n)])
    dates = np.datetime64("2020-01-01") + np.arange(n)
    return Draws(dates, balls, rng.integers(1, 8, n))


def test_sparse_model_normalizes_on_exhaustive_small_board():
    board = list(combinations(range(7), 3))
    hits = np.array([0 in row for row in board])
    for q in [0.01, 0.3, 0.99]:
        p = np.exp(event_log_ratio(q, hits, 3 / 7)) / comb(7, 3)
        assert p.sum() == pytest.approx(1)
        assert p[hits].sum() == pytest.approx(q)
        for other in range(1, 7):
            assert sum(v for row, v in zip(board, p) if other in row) == pytest.approx((3 - q) / 6)


def test_joint_experts_are_coherent_and_uniform_is_exact():
    e = build_experts(fixture_draws(), min_history=20, windows=(10, None), strengths=(100,))
    assert len(e.names) == 47
    assert e.prior.sum() == pytest.approx(1)
    assert e.prior[0] == 0.5
    np.testing.assert_allclose(e.log_prob[0], -LOG_FULL_OUTCOMES)
    np.testing.assert_allclose(e.ball_p.sum(axis=2), 6, atol=1e-12)
    np.testing.assert_allclose(e.strong_p.sum(axis=2), 1, atol=1e-12)
    assert np.all((e.ball_p > 0) & (e.ball_p < 1))
    assert np.all(np.isfinite(e.log_prob))


def test_future_and_current_outcomes_cannot_change_current_forecast():
    d = fixture_draws()
    changed_b, changed_s = d.balls.copy(), d.strong.copy()
    changed_b[40:] = np.arange(1, 7)
    changed_s[40:] = 7
    changed = Draws(d.dates, changed_b, changed_s)
    a = build_experts(d, min_history=20, windows=(10,), strengths=(100,))
    b = build_experts(changed, min_history=20, windows=(10,), strengths=(100,))
    np.testing.assert_allclose(a.ball_p[:, :21], b.ball_p[:, :21])
    np.testing.assert_allclose(a.strong_p[:, :21], b.strong_p[:, :21])
    np.testing.assert_allclose(a.log_prob[:, :20], b.log_prob[:, :20])
    for share in [0, 0.01]:
        ma, mb = mix_experts(a.log_prob, a.prior, share=share), mix_experts(b.log_prob, b.prior, share=share)
        np.testing.assert_allclose(ma['weights_before_draw'][:21], mb['weights_before_draw'][:21])


def test_static_mixture_matches_sequence_probability_and_uniform_loss_bound():
    e = build_experts(fixture_draws(), min_history=20, windows=(10,), strengths=(100,))
    m = mix_experts(e.log_prob, e.prior)
    expected = logsumexp(np.log(e.prior)[:, None] + np.cumsum(e.log_prob, axis=1), axis=0)
    np.testing.assert_allclose(np.cumsum(m['log_prob']), expected, atol=1e-10)
    assert np.all(m['log_e_path'] >= np.log(0.5) - 1e-10)
    np.testing.assert_allclose(m['weights_before_draw'].sum(axis=1), 1, atol=1e-12)


def test_sharing_update_and_frozen_prior_endpoint():
    lp = np.log([[0.2, 0.8], [0.8, 0.2]])
    prior = np.array([0.5, 0.5])
    m = mix_experts(lp, prior, share=0.1)
    np.testing.assert_allclose(m['weights_before_draw'][1], [0.23, 0.77])
    fixed = mix_experts(lp, prior, share=1)
    np.testing.assert_allclose(np.exp(fixed['log_prob']), [0.5, 0.5])


def test_observed_probability_agrees_with_sparse_marginal_event():
    d = fixture_draws()
    e = build_experts(d, min_history=20, windows=(10,), strengths=(100,))
    for i in [0, 18, 36]:
        present = (d.balls[e.targets] == i + 1).any(axis=1)
        expected = event_log_ratio(e.ball_p[1+i, :, i], present, 6/37) - LOG_FULL_OUTCOMES
        np.testing.assert_allclose(e.log_prob[1+i], expected)


@pytest.mark.parametrize('share', [-1, 2, np.nan, True])
def test_bad_mixture_settings_rejected(share):
    with pytest.raises(ValueError):
        mix_experts(np.log([[0.5]]), [1], share=share)


def test_bad_inputs_rejected():
    d = fixture_draws()
    b = d.balls.copy()
    b[0, 1] = b[0, 0]
    with pytest.raises(ValueError):
        build_experts(Draws(d.dates, b, d.strong), min_history=20)
    with pytest.raises(ValueError):
        build_experts(d, min_history=20, strengths=(0,))
    with pytest.raises(ValueError):
        mix_experts(np.array([[np.nan]]), [1])
    with pytest.raises(ValueError):
        mix_experts(np.log([[0.5], [0.5]]), [0.6, 0.6])
