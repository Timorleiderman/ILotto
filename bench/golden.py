"""The golden ratio applied to the draw sequence.

People reach for phi with reason: it genuinely organises sunflower heads,
pinecones and quasicrystals, so "nature's constant" feels like it should
organise other numbers too. This module takes the idea seriously enough to
state what phi would have to do here, measure whether it does, and ship the
phi-based generator as a scoreboard participant.

The finding is an inversion, and it is the whole story. phi's actual
mathematical distinction is being the **most irrational number** — by Hurwitz's
theorem it is the constant *worst* approximated by fractions, which is exactly
why the Kronecker sequence frac(n*phi) has the lowest discrepancy of any
irrational rotation (measured here: 0.0010 at n = 1,629, beating e, pi and
sqrt(2), while the near-rational 22/7 clumps at 0.1433). In plants that is the
point: each new seed lands maximally far from every earlier one. phi is the
constant of maximal pattern *avoidance*. Asking it to find patterns in a
lottery inverts its one superpower — and the long-run pick rate of the golden
schedule below is uniform to four decimals, meaning the phi predictor is
provably the fairest possible fixed schedule: a random ticket with extra steps.

Measured against the archive (see `docs/approaches/golden-ratio.md`): Fibonacci
numbers appear at +0.93 sd of chance, within-draw gap ratios cluster near phi
no more than fair simulations do, and the golden-section numbers 23 and 14 are
unremarkable. Nothing golden survives measurement.
"""

from __future__ import annotations

import logging

import numpy as np

from .data import N_DRAWN, N_NUMBERS, N_STRONG, Draws
from .predictors import Predictor

logger = logging.getLogger(__name__)

PHI = (1 + 5**0.5) / 2

#: Fibonacci numbers on the 1..37 board.
FIBONACCI = (1, 2, 3, 5, 8, 13, 21, 34)


def golden_ticket(index: int) -> list[int]:
    """The 6-number golden-angle schedule at a given draw index.

    Start at frac(index * phi), then step by the golden angle (phi - 1) —
    the phyllotaxis rule, mapped onto the 37-number board. Deterministic,
    parameter-free, and by phi's own equidistribution the long-run frequency
    of every number is exactly 6/37.
    """
    picks: list[int] = []
    x = (index * PHI) % 1.0
    while len(picks) < N_DRAWN:
        v = int(x * N_NUMBERS) + 1
        if v not in picks:
            picks.append(v)
        x = (x + PHI - 1.0) % 1.0
    return sorted(picks)


class GoldenRatio(Predictor):
    """Golden-angle (Kronecker) schedule as a predictor.

    The ticket for the next draw is the golden schedule evaluated at the next
    index. This is the strongest honest version of "use the golden ratio": it
    is exactly the construction phi is famous for, and phi's low-discrepancy
    property guarantees it spreads over the board as evenly as any fixed
    schedule can — which is another way of saying it converges to a uniformly
    random ticket. Its walk-forward score is chance, necessarily.
    """

    name = "Golden ratio (Kronecker schedule)"
    description = (
        "Six numbers placed by the golden angle at the next draw's index — the phyllotaxis "
        "construction mapped onto the board. phi's low-discrepancy property makes this the "
        "most uniformly spread fixed schedule possible, i.e. a random ticket with extra steps."
    )

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        ticket = golden_ticket(len(history))
        scores = np.zeros(N_NUMBERS)
        scores[np.array(ticket) - 1] = 1.0
        # Strong number by the same rotation on the 7-board.
        s = int(((len(history) * PHI) % 1.0) * N_STRONG)
        strong = np.full(N_STRONG, 0.5 / N_STRONG)
        strong[s] = 1.0
        return scores, strong / strong.sum()


class FibonacciTicket(Predictor):
    """Always play Fibonacci numbers — the folk version, kept as a control.

    Statistically it is a fixed ticket like any other (Fibonacci appearances
    measure +0.93 sd of chance in the archive). Its real-world property is
    worse than neutral, though: 1-2-3-5-8-13 is a famously *popular* human
    pick, so in a pari-mutuel game it wins no more often and shares more when
    it does — the exact opposite of the unpopular-numbers logic.
    """

    name = "Fibonacci ticket (control)"
    description = (
        "The fixed ticket 2-3-5-8-13-21. Same odds as any six numbers, but Fibonacci picks "
        "are famously popular with humans, so the pari-mutuel share is worse than average."
    )

    TICKET = (2, 3, 5, 8, 13, 21)

    def scores(self, history: Draws) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros(N_NUMBERS)
        scores[np.array(self.TICKET) - 1] = 1.0
        return scores, np.full(N_STRONG, 1.0 / N_STRONG)


def golden_suite() -> list[Predictor]:
    return [GoldenRatio(), FibonacciTicket()]
