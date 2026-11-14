import copy

import numpy as np
import pytest

from lotto.tickets.coverage import best_match_counts, search_coverage_tickets


def test_search_retains_best_preserves_input_and_reproduces_exact_counts():
    tickets = [[1, 2, 3, 4, 5, 6, 1], [1, 2, 3, 4, 5, 7, 2]]
    original = copy.deepcopy(tickets)
    result, audit = search_coverage_tickets(tickets, steps=3, seed=70106)
    assert tickets == original
    assert len(result) == len(tickets)
    assert [r[-1] for r in result] == [r[-1] for r in tickets]
    assert all(len(set(r[:6])) == 6 for r in result)
    exact = best_match_counts([r[:6] for r in result], engine='reference')
    assert exact[3:].sum() == audit['covered_outcomes_after']
    assert audit['covered_outcomes_after'] >= audit['covered_outcomes_before']
    assert np.all(np.diff([p['best_covered_outcomes'] for p in audit['proposals']]) >= 0)
    repeated, repeated_audit = search_coverage_tickets(tickets, steps=3, seed=70106)
    assert repeated == result
    assert repeated_audit == audit


@pytest.mark.parametrize('kwargs', [{'steps': 0}, {'steps': True}, {'seed': -1},
                                   {'perturbations': 0}, {'perturbations': True}, {'local_passes': -1},
                                   {'start_temperature': np.nan}, {'end_temperature': 0},
                                   {'start_temperature': 1, 'end_temperature': 2}])
def test_search_rejects_invalid_budget_and_temperature(kwargs):
    with pytest.raises(ValueError):
        search_coverage_tickets([[1, 2, 3, 4, 5, 6, 1]], **kwargs)


def test_search_honors_perturbation_and_local_pass_budgets():
    tickets = [[1, 2, 3, 4, 5, 6, 1], [1, 2, 3, 4, 5, 7, 2]]
    result, audit = search_coverage_tickets(tickets, steps=2, seed=70106, perturbations=4, local_passes=0)
    assert audit['perturbations'] == 4
    assert audit['local_passes'] == 0
    for proposal in audit['proposals']:
        assert len(proposal['perturbations']) == 4
        assert proposal['local_refinement']['passes'] == 0
        assert proposal['local_refinement']['swaps'] == []
    assert audit['covered_outcomes_after'] == best_match_counts([r[:6] for r in result])[3:].sum()
