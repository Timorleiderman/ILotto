from itertools import combinations, product
from math import comb

import numpy as np
import pytest

from lotto.tickets._kernels import best_strong_partition, coverage_signatures
from lotto.tickets.coverage import portfolio_odds
from lotto.tickets.strong_assignment import optimize_strong_numbers, strong_prize_odds, subset_union_counts


def test_signature_counts_subset_transform_and_partition_match_exhaustive_oracles():
    draws = [set(row) for row in combinations(range(9), 6)]
    tickets = [{0, 1, 2, 3, 4, 5}, {3, 4, 5, 6, 7, 8}, {0, 2, 4, 5, 6, 8}, {0, 1, 3, 5, 7, 8}]
    def masks(rows):
        return np.array([sum(1 << x for x in row) for row in rows], dtype=np.uint64)

    for threshold in (3, 4, 6):
        expected = np.zeros(1 << len(tickets), dtype=np.int64)
        for draw in draws:
            expected[sum(1 << i for i, row in enumerate(tickets) if len(draw & row) >= threshold)] += 1
        signatures = coverage_signatures(masks(draws), masks(tickets), threshold)
        np.testing.assert_array_equal(signatures, expected)
        unions = subset_union_counts(signatures)
        for subset in range(len(unions)):
            assert unions[subset] == sum(any(subset & (1 << i) and len(draw & row) >= threshold
                                             for i, row in enumerate(tickets)) for draw in draws)
        for groups in (1, 2, 3, 4):
            exhaustive = max(sum(any(label == strong and len(draw & row) >= threshold
                                     for label, row in zip(labels, tickets))
                                 for draw in draws for strong in range(groups))
                             for labels in product(range(groups), repeat=len(tickets)))
            optimum, _ = best_strong_partition(unions, groups)
            assert optimum == exhaustive


def test_optimization_preserves_mains_and_matches_independent_full_universe():
    tickets = [[1, 2, 3, 4, 5, 6, 1], [7, 8, 9, 10, 11, 12, 1], [1, 2, 3, 13, 14, 15, 1]]
    snapshot = [row.copy() for row in tickets]
    optimized, audit = optimize_strong_numbers(tickets)
    assert tickets == snapshot
    assert [row[:6] for row in optimized] == [row[:6] for row in tickets]
    assert len({row[-1] for row in optimized}) == 3
    before = strong_prize_odds(tickets, engine='reference')
    after = strong_prize_odds(optimized, engine='reference')
    assert audit['covered_full_outcomes_before'] == before['covered_full_outcomes']
    assert audit['covered_full_outcomes_after'] == after['covered_full_outcomes']
    assert audit['probability_after'] > audit['probability_before']
    assert portfolio_odds(optimized) == portfolio_odds(tickets)
    assert optimize_strong_numbers(tickets) == (optimized, audit)


def test_identical_main_rows_can_be_evaluated_but_not_optimized():
    tickets = [[1, 2, 3, 4, 5, 6, strong] for strong in (1, 1, 2)]
    result = strong_prize_odds(tickets)
    single = sum(comb(6, k) * comb(31, 6-k) for k in range(3, 7))
    assert result['covered_full_outcomes'] == 2 * single
    with pytest.raises(ValueError, match='distinct main'):
        optimize_strong_numbers(tickets)


@pytest.mark.parametrize('threshold', [0, 7, True, 3.0])
def test_invalid_threshold_is_rejected(threshold):
    for operation in (optimize_strong_numbers, strong_prize_odds):
        with pytest.raises(ValueError, match='min_matches'):
            operation([[1, 2, 3, 4, 5, 6, 1]], min_matches=threshold)


@pytest.mark.parametrize('counts', [[], [1], [1, 2, 3], [1.5, 2.5], [-1, 3],
                                   np.array([2**63, 0], dtype=np.uint64)])
def test_invalid_signature_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        subset_union_counts(counts)


def test_excessive_partition_size_is_rejected():
    tickets = [[1, 2, 3, 4, 5, sixth, 1] for sixth in range(6, 21)]
    with pytest.raises(ValueError, match='1..14'):
        optimize_strong_numbers(tickets)
