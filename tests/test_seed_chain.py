"""The chain learns a planted correction while each prediction precedes its target."""

import numpy as np

from lotto.core.data import Draws
from lotto.experiments.seed_chain import replay
from lotto.seedsearch.generator import OUTCOMES, index_ticket, world_outcome


def draws_from_outcomes(days, outcomes):
    tickets = [index_ticket(x) for x in outcomes]
    return Draws(days, np.array([t["balls"] for t in tickets]), np.array([t["strong"] for t in tickets]))


def test_planted_offset_is_learned_after_the_first_forecast_and_frozen():
    days = np.arange("2020-01-01", "2020-01-07", dtype="datetime64[D]")
    outcomes = [123]
    # A different rule in the last three draws would move the estimated offset
    # if the holdout were ever allowed to influence it.
    for day, offset in zip(days[1:], [17, 17, 25, 25, 25]):
        outcomes.append((world_outcome(outcomes[-1], str(day)) + offset) % OUTCOMES)
    report = replay(draws_from_outcomes(days, outcomes), learn_draws=2)
    assert [r["offset_used"] for r in report["predictions"]] == [0, 17, 17, 17, 17]
    assert report["frozen_offset"] == 17
    assert report["draws_supporting_offset"] == 2
    assert report["online_learning"]["corrected"]["exact_tickets"] == 1
    assert report["retrospective_holdout"]["corrected"]["exact_tickets"] == 0


def test_changing_an_outcome_only_affects_predictions_for_later_draws():
    days = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")
    a = replay(draws_from_outcomes(days, [10, 20, 30, 40, 50]), 2)["predictions"]
    b = replay(draws_from_outcomes(days, [10, 20, 30, 41, 50]), 2)["predictions"]
    assert len(a) == len(b) == len(days) - 1
    for key in ("free_running", "previous_actual", "corrected"):
        assert [r[key] for r in a[:3]] == [r[key] for r in b[:3]]
    assert a[-1]["free_running"] == b[-1]["free_running"]
    assert a[-1]["previous_actual"] == world_outcome(40, str(days[-1]))
    assert b[-1]["previous_actual"] == world_outcome(41, str(days[-1]))
