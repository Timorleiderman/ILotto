"""Independent probability oracles and strict past-only forecast checks."""

from itertools import combinations, product
from math import comb

import numpy as np
import pytest
from scipy import stats

from bench import backtest, data, metrics, randomness
from bench.calibration import frequency_forecasts, past_loss_choices
from bench.portfolio import (
    best_match_counts, greedy_coverage_tickets, portfolio_odds, refine_coverage_tickets, single_ticket_prize_odds,
)


def history(n=260):
    rng = np.random.default_rng(123)
    balls = np.sort(rng.random((n, 37)).argsort(axis=1)[:, :6] + 1, axis=1)
    return data.Draws(np.datetime64("2020-01-01") + np.arange(n), balls, rng.integers(1, 8, n))


def test_prize_tiers_include_three_without_strong():
    assert metrics.prize_tier(3, False) == "8th (3)"
    odds = single_ticket_prize_odds()
    assert len(odds) == 9
    assert sum(row["outcomes"] for row in odds.values()) == comb(37, 6) * 7
    assert odds["1st (6+strong)"]["outcomes"] == 1
    assert odds["8th (3)"]["outcomes"] == comb(6, 3) * comb(31, 3) * 6
    assert sum(row["probability"] for tier, row in odds.items() if tier != "no prize") == pytest.approx(
        stats.hypergeom.sf(2, 37, 6, 6))


def test_exact_portfolio_matches_independent_exhaustive_oracle():
    tickets = [(1, 2, 3), (3, 4, 5), (6, 7, 8)]
    expected = np.zeros(4, dtype=int)
    for draw in combinations(range(1, 9), 3):
        best = max(len(set(draw) & set(ticket)) for ticket in tickets)
        expected[best] += 1
    for chunk in (1, 7, 100):
        actual = best_match_counts(tickets, n_numbers=8, n_drawn=3, chunk_size=chunk)
        np.testing.assert_array_equal(actual, expected)
    duplicate = best_match_counts([*tickets, tickets[0]], n_numbers=8, n_drawn=3)
    np.testing.assert_array_equal(duplicate, expected)


def test_single_ticket_coverage_equals_combinatorial_counts():
    counts = best_match_counts([(1, 2, 3)], n_numbers=9, n_drawn=3)
    np.testing.assert_array_equal(counts, [comb(3, k) * comb(6, 3 - k) for k in range(4)])


def test_portfolio_odds_distinguish_main_and_full_duplicates():
    # Reversing the main numbers changes nothing; changing the strong number
    # covers a second jackpot outcome but no new >=3-main outcomes.
    odds = portfolio_odds([[1, 2, 3, 4, 5, 6, 1], [6, 5, 4, 3, 2, 1, 1], [1, 2, 3, 4, 5, 6, 2]])
    assert odds["tickets"] == 3
    assert odds["distinct_full_tickets"] == 2
    assert odds["distinct_main_tickets"] == 1
    assert odds["p_jackpot"] == pytest.approx(2 / (comb(37, 6) * 7))
    assert odds["p_any_prize"] == pytest.approx(stats.hypergeom.sf(2, 37, 6, 6))
    assert sum(odds["best_match_counts"]) == comb(37, 6)


@pytest.mark.parametrize("strong", [0, 8, 1.0, True])
def test_portfolio_rejects_invalid_strong_numbers(strong):
    with pytest.raises(ValueError, match="Strong numbers"):
        portfolio_odds([[1, 2, 3, 4, 5, 6, strong]])


@pytest.mark.parametrize("tickets", [[], [(1, 1, 2)], [(0, 2, 3)], [(1.0, 2, 3)], [(1, 2)]])
def test_bad_portfolios_are_rejected(tickets):
    with pytest.raises(ValueError):
        best_match_counts(tickets, n_numbers=8, n_drawn=3)


def test_greedy_portfolio_is_reproducible_and_distinct():
    first = greedy_coverage_tickets(10, candidates=32, simulations=200, seed=42)
    assert first == greedy_coverage_tickets(10, candidates=32, simulations=200, seed=42)
    assert len({tuple(row[:6]) for row in first}) == 10
    for row in first:
        assert len(set(row[:6])) == 6 and min(row[:6]) >= 1 and max(row[:6]) <= 37
        assert 1 <= row[6] <= 7


def test_finite_sample_match_null_matches_direct_two_draw_enumeration():
    pmf = stats.hypergeom.pmf(np.arange(7), 37, 6, 6)
    expected = np.zeros(13)
    for a, b in product(range(7), repeat=2):
        expected[a + b] += pmf[a] * pmf[b]
    np.testing.assert_allclose(metrics.match_sum_distribution(2), expected)
    summary = metrics.summarise(np.array([6]), None, np.array([0]))
    assert summary["p_value"] == pytest.approx(2 / comb(37, 6))
    assert summary["p_value_greater"] == pytest.approx(1 / comb(37, 6))


def test_direct_null_sampler_has_hypergeometric_moments():
    samples = backtest.random_ticket_null(history(), n_test=100, n_trials=20_000)["samples"]
    assert abs(samples.mean() - metrics.BASELINE_MATCHES) < 0.003
    assert samples.var() == pytest.approx(metrics.BASELINE_MATCH_VAR / 100, rel=0.04)


def test_uniformity_adjustment_matches_indicator_covariance_wald_test():
    draws = history()
    # Independent construction: covariance of the fixed-size binary draw vector.
    p = 6 / 37
    covariance = np.full((37, 37), -p * (1 - p) / 36)
    np.fill_diagonal(covariance, p * (1 - p))
    deviation = draws.multi_hot.sum(axis=0) - len(draws) * p
    expected = deviation @ np.linalg.pinv(len(draws) * covariance) @ deviation
    result = randomness.uniformity(draws)
    assert result["statistic"] == pytest.approx(expected, rel=1e-5)
    assert result["p_value"] == pytest.approx(stats.chi2.sf(expected, 36), rel=1e-5)


def test_frequency_forecasts_do_not_see_target_or_future():
    draws = history()
    _, targets, balls, strong = frequency_forecasts(draws)
    altered = data.Draws(draws.dates, draws.balls.copy(), draws.strong.copy())
    altered.balls[230:] = np.arange(1, 7)
    altered.strong[230:] = 7
    _, _, changed_balls, changed_strong = frequency_forecasts(altered)
    np.testing.assert_array_equal(balls[:, targets <= 230], changed_balls[:, targets <= 230])
    np.testing.assert_array_equal(strong[:, targets <= 230], changed_strong[:, targets <= 230])
    np.testing.assert_allclose(balls.sum(axis=2), 6)
    np.testing.assert_allclose(strong.sum(axis=2), 1)
    np.testing.assert_allclose(balls[-1], 6 / 37)
    assert np.all((balls >= 0) & (balls <= 1))


def test_model_selection_cannot_see_current_or_future_losses():
    losses = np.array([[0.1] * 12, [1.0] * 12])
    chosen = past_loss_choices(losses, baseline=1, warmup=3, lookback=5, refit_every=2)
    np.testing.assert_array_equal(chosen[:3], 1)
    np.testing.assert_array_equal(chosen[3:], 0)
    changed = losses.copy()
    changed[:, 7:] = np.array([[10.0], [0.0]])
    other = past_loss_choices(changed, baseline=1, warmup=3, lookback=5, refit_every=2)
    np.testing.assert_array_equal(chosen[:8], other[:8])
    assert other[-1] == 1


@pytest.mark.parametrize("tickets", [
    [[1, 2, 3, 4, 5, 6], [32, 33, 34, 35, 36, 37]],
    [[1, 9, 12, 20, 29, 37], [9, 13, 17, 24, 30, 37], [3, 8, 15, 23, 31, 36]],
])
def test_compiled_coverage_matches_independent_reference(tickets):
    np.testing.assert_array_equal(best_match_counts(tickets), best_match_counts(tickets, engine="reference"))


def test_exchange_gain_matches_every_swap_on_an_independent_small_universe():
    from bench._portfolio_exact import best_exchange, update_coverage

    # Include the high bit (number 37); use sets for the independent oracle.
    pool = [0, 1, 2, 3, 4, 5, 6, 10, 20, 30, 36]
    draws = np.asarray(list(combinations(pool, 6)), dtype=np.uint8)
    sets = [set(row) for row in draws]
    ticket = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    other = {0, 1, 2, 3, 4, 36}
    old_set = set(ticket)
    masks = np.asarray([sum(1 << int(x) for x in draw) for draw in draws], dtype=np.uint64)
    ticket_mask = np.uint64(sum(1 << int(x) for x in ticket))
    counts = np.array([int(len(s & old_set) >= 3) + int(len(s & other) >= 3) for s in sets], dtype=np.int32)
    before = np.count_nonzero(counts)
    expected_gain = 0
    for removed in ticket:
        for added in set(range(37)) - old_set:
            candidate = old_set - {removed} | {added}
            covered = sum(len(s & candidate) >= 3 or len(s & other) >= 3 for s in sets)
            expected_gain = max(expected_gain, covered - before)
    gain, position, ball = best_exchange(draws, masks, counts, ticket, ticket_mask)
    assert gain == expected_gain and gain > 0
    new_set = old_set - {ticket[position]} | {ball}
    new_mask = np.uint64(sum(1 << int(x) for x in new_set))
    update_coverage(masks, counts, ticket_mask, new_mask)
    expected_counts = [int(len(s & new_set) >= 3) + int(len(s & other) >= 3) for s in sets]
    np.testing.assert_array_equal(counts, expected_counts)
    assert np.count_nonzero(counts) == before + gain


def test_refinement_preserves_inputs_and_never_reduces_exact_coverage():
    original = greedy_coverage_tickets(4, candidates=32, simulations=200, seed=8)
    snapshot = [row[:] for row in original]
    improved, audit = refine_coverage_tickets(original, max_passes=2)
    assert original == snapshot
    assert [row[-1] for row in improved] == [row[-1] for row in original]
    assert len({tuple(row) for row in improved}) == len(original)
    before, after = portfolio_odds(original), portfolio_odds(improved)
    assert after["p_any_prize"] >= before["p_any_prize"]
    assert after["p_jackpot"] == before["p_jackpot"]
    assert audit["covered_outcomes_before"] == sum(before["best_match_counts"][3:])
    assert audit["covered_outcomes_after"] == sum(after["best_match_counts"][3:])
    assert all(s["newly_covered_outcomes"] > 0 for s in audit["swaps"])
    unchanged, disabled = refine_coverage_tickets(original, max_passes=0)
    assert unchanged == original and not disabled["swaps"]


@pytest.mark.parametrize("kwargs", [
    {"count": True}, {"min_matches": 2.5}, {"min_matches": True},
    {"candidates": 1_000_000}, {"simulations": 1_000_000_000}, {"max_working_bytes": 1},
    {"simulations": np.int64(2**63 - 1)},
])
def test_invalid_or_oversized_search_is_rejected_before_allocation(kwargs):
    with pytest.raises(ValueError):
        greedy_coverage_tickets(**kwargs)


@pytest.mark.parametrize("counts", [[], [1.5], [-1], [7], [[1, 2]], [np.nan]])
def test_invalid_match_counts_cannot_produce_significance(counts):
    with pytest.raises(ValueError, match="match_counts"):
        metrics.summarise(np.asarray(counts), None, np.array([0]))


@pytest.mark.parametrize("losses,strong", [([np.nan], [0]), ([1, 2], [0]), ([1], [2]), ([1], [])])
def test_summary_rejects_misaligned_or_invalid_scores(losses, strong):
    with pytest.raises(ValueError):
        metrics.summarise(np.array([1]), losses, strong)


def test_calibration_rejects_truncated_baselines_and_duplicate_models():
    with pytest.raises(ValueError):
        past_loss_choices(np.zeros((2, 5)), baseline=0.5)
    with pytest.raises(ValueError, match="Duplicate"):
        frequency_forecasts(history(), windows=(100, 100))
    with pytest.raises(ValueError, match="Duplicate"):
        frequency_forecasts(history(), strengths=(0, 0))
    with pytest.raises(ValueError):
        metrics.holm_bonferroni({"bad": np.nan})


def test_pair_test_uses_exact_tails_and_dependency_safe_family_correction():
    draws = history(12)
    draws.balls[:8] = np.arange(1, 7)
    p = comb(35, 4) / comb(37, 6)
    marginal_p = []
    for a, b in combinations(range(1, 38), 2):
        hits = sum(a in row and b in row for row in draws.balls)
        lower = stats.binomtest(hits, len(draws), p, alternative="less").pvalue
        upper = stats.binomtest(hits, len(draws), p, alternative="greater").pvalue
        marginal_p.append(min(1, 2 * min(lower, upper)))
    result = randomness.pair_independence(draws)
    assert result["p_value"] == pytest.approx(min(1, 666 * min(marginal_p)), rel=1e-12, abs=0)


def test_drift_matches_full_without_replacement_covariance():
    draws = history(261)
    half = len(draws) // 2
    p = 6 / 37
    covariance = np.full((37, 37), -p * (1 - p) / 36)
    np.fill_diagonal(covariance, p * (1 - p))
    covariance *= 1 / half + 1 / (len(draws) - half)
    deviation = draws.multi_hot[:half].mean(axis=0) - draws.multi_hot[half:].mean(axis=0)
    expected = deviation @ np.linalg.pinv(covariance) @ deviation
    result = randomness.drift(draws)
    assert result["statistic"] == pytest.approx(expected, rel=1e-5)
    assert result["p_value"] == pytest.approx(stats.chi2.sf(expected, 36), rel=1e-5)


def test_published_tied_ticket_matches_backtest_and_rank_scores_are_not_probabilities():
    from bench import predictors, report

    class Tied(predictors.Predictor):
        name = "tied ranks"

        def scores(self, history):
            return np.full(37, -1.0), np.zeros(7)

    prediction = report.predict_next(Tied(), history())
    assert prediction["numbers"] == (metrics.top_k(np.full(37, -1.0)) + 1).tolist()
    assert set(n for n, _ in prediction["shortlist"][:6]) == set(prediction["numbers"])
    assert prediction["shortlist_kind"] == "ranking_score"
    assert all(score == -1 for _, score in prediction["shortlist"])
