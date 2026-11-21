"""Check the algebraic search against exhaustive search and planted 64-bit seeds."""

from datetime import date

import numpy as np
import pytest

from lotto.core.data import Draws
from lotto.experiments.seed_family import evaluate
from lotto.seedsearch import affine
from lotto.seedsearch.generator import MAX_SEED, Observation, index_ticket


def test_every_pair_matches_brute_force_with_partial_final_key(monkeypatch):
    # Three complete keys plus five legal seeds for the final key.
    monkeypatch.setattr(affine, "OUTCOMES", 4)
    monkeypatch.setattr(affine, "SEEDS_PER_KEY", 16)
    monkeypatch.setattr(affine, "MAX_SEED", 53)
    days = ["2020-01-01", "2020-01-04"]
    by_pair = {}
    for seed in range(affine.MAX_SEED):
        pair = tuple(affine.affine_outcome(seed, i, day) for i, day in enumerate(days))
        by_pair.setdefault(pair, []).append(seed)
    assert len(by_pair) == 16
    for pair, expected in by_pair.items():
        first, second = [Observation(day, value) for day, value in zip(days, pair)]
        assert list(affine.matching_pair(first, second)) == expected


def test_recovers_largest_64_bit_seed_and_predicts_later_draws():
    seed = MAX_SEED - 1
    days = ["2020-01-01", "2020-01-04", "2020-01-11", "2020-01-16"]
    outcomes = [affine.affine_outcome(seed, i, day) for i, day in enumerate(days)]
    seeds = affine.matching_pair(Observation(days[0], outcomes[0]), Observation(days[1], outcomes[1]))
    survivors = [s for s in seeds if affine.affine_outcome(s, 2, days[2]) == outcomes[2]]
    assert survivors == [seed]
    assert affine.affine_outcome(survivors[0], 3, days[3]) == outcomes[3]


def test_selection_uses_only_the_fitted_pair():
    days = np.array([date(2020, 1, i) for i in (1, 4, 11)], dtype="datetime64[D]")
    selected = 42  # belongs to the first key, so is the smallest matching seed
    outcomes = [affine.affine_outcome(selected, i, str(day)) for i, day in enumerate(days)]

    def report(last):
        tickets = [index_ticket(x) for x in [*outcomes[:2], last]]
        return evaluate(Draws(days, np.array([t["balls"] for t in tickets]),
                              np.array([t["strong"] for t in tickets])))

    original, changed = report(outcomes[-1]), report((outcomes[-1] + 1) % affine.OUTCOMES)
    assert original["selected_seed"] == changed["selected_seed"] == selected
    assert original["retrospective_holdout"]["affine"]["exact_tickets"] == 1
    assert changed["retrospective_holdout"]["affine"]["exact_tickets"] == 0


@pytest.mark.parametrize(("seed", "index"), [(-1, 0), (MAX_SEED, 0), (0, -1)])
def test_invalid_inputs(seed, index):
    with pytest.raises(ValueError):
        affine.affine_outcome(seed, index, "2020-01-01")
