import numpy as np
import pytest

from bench.historical_strikes import longest_run, ranked_portfolios, score_portfolios
from bench.portfolio import portfolio_odds
from bench.strong_assignment import strong_prize_odds


TEMPLATE = [[1, 2, 3, 4, 5, 6, 1], [2, 7, 8, 9, 10, 11, 2], [3, 7, 12, 13, 14, 15, 1]]


def test_uniform_forecast_returns_original_template():
    out = ranked_portfolios(TEMPLATE, np.ones((2, 37)), np.ones((2, 7)))
    np.testing.assert_array_equal(out, [TEMPLATE, TEMPLATE])


def test_relabeling_preserves_exact_main_and_strong_odds():
    rng = np.random.default_rng(71107)
    out = ranked_portfolios(TEMPLATE, rng.random((1, 37)), rng.random((1, 7)))[0]
    assert portfolio_odds(out) == portfolio_odds(TEMPLATE)
    before, after = strong_prize_odds(TEMPLATE), strong_prize_odds(out)
    assert before['covered_full_outcomes'] == after['covered_full_outcomes']
    assert not np.array_equal(out, TEMPLATE)


def test_forecast_mapping_does_not_use_other_times():
    rng = np.random.default_rng(71108)
    balls, strong = rng.random((4, 37)), rng.random((4, 7))
    original = ranked_portfolios(TEMPLATE, balls, strong)
    balls[2:], strong[2:] = 1, 1
    changed = ranked_portfolios(TEMPLATE, balls, strong)
    np.testing.assert_array_equal(original[:2], changed[:2])


def test_score_matches_independent_set_intersections_and_strong_events():
    tickets = np.array([TEMPLATE, TEMPLATE])
    actual = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    strong = np.array([1, 2])
    counts, hits = score_portfolios(tickets, actual, strong)
    for t in range(2):
        for row in range(len(TEMPLATE)):
            assert counts[t, row] == len(set(tickets[t, row, :6]) & set(actual[t]))
            assert hits[t, row] == (tickets[t, row, -1] == strong[t])
    assert longest_run([False, True, True, False, True, True, True]) == 3
    assert longest_run([]) == 0


@pytest.mark.parametrize('balls,strong', [(np.ones((1, 36)), np.ones((1, 7))),
                                         (np.ones((1, 37)), np.ones((2, 7))),
                                         (np.full((1, 37), np.nan), np.ones((1, 7))),
                                         (-np.ones((1, 37)), np.ones((1, 7)))])
def test_bad_forecasts_rejected(balls, strong):
    with pytest.raises(ValueError):
        ranked_portfolios(TEMPLATE, balls, strong)


def test_duplicate_actual_balls_and_fractional_tickets_rejected():
    with pytest.raises(ValueError):
        score_portfolios(np.array([TEMPLATE]), [[1, 1, 3, 4, 5, 6]], [1])
    with pytest.raises(ValueError):
        score_portfolios(np.array([TEMPLATE], dtype=float), [[1, 2, 3, 4, 5, 6]], [1])
