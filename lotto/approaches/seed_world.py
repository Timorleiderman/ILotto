"""A toy: the ticket an artificial seeded generator draws on a given date.

This is the generator from the seed-search side quest (`lotto.seedsearch`):
a 64-bit seed and a calendar date are hashed to one of the 16,273,488 possible
draws. The real lottery uses physical ball machines and has no seed, so this
predicts nothing; it is here because "pick a seed, get a ticket" is fun to
play with. It is never part of the benchmark suites.
"""

from __future__ import annotations

import numpy as np

from lotto.approaches.statistical import Predictor
from lotto.core.data import N_NUMBERS, N_STRONG, Draws
from lotto.seedsearch.generator import MAX_SEED, index_ticket, world_outcome


class SeedWorld(Predictor):
    """The draw that seed's artificial world produces on the target date."""

    name = "Seed world (toy)"
    description = (
        "An artificial generator: a seed and a date are hashed to one complete draw. "
        "Not a model of the real lottery."
    )
    wants_target_date = True

    def __init__(self, seed: int = 0):
        if not 0 <= seed < MAX_SEED:
            raise ValueError("seed must be in 0..2**64-1")
        self.seed = seed

    def scores(self, history: Draws, target_date=None) -> tuple[np.ndarray, np.ndarray]:
        if target_date is None:
            raise ValueError("Seed world needs the draw's date")
        ticket = index_ticket(world_outcome(self.seed, str(np.datetime64(target_date, "D"))))
        balls = np.zeros(N_NUMBERS)
        balls[np.array(ticket["balls"]) - 1] = 1.0
        strong = np.zeros(N_STRONG)
        strong[ticket["strong"] - 1] = 1.0
        return balls, strong
