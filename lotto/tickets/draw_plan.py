"""Which fixed form to play on the next draw, from the last official result.

This is a ticket-budget rule, not a forecast. A *bonus* draw pays every small
tier 1.5x (15 NIS instead of 10 for three main numbers); the chance of a prize
is unchanged. In 583 official prize tables (draws 3400-3982) every draw that
followed an unclaimed 40M first prize was a bonus draw (24 of 24), and the
first 40M draw itself never was (7 of 7). Promotional bonus series at lower
jackpots are announced by the operator and cannot be inferred from results,
so they enter through ``announced_bonus``.

The two forms are fixed coverage designs from ``lotto.tickets.coverage`` with the
exact strong-number assignment from ``lotto.tickets.strong_assignment``. Their odds
assume independent uniform fair draws; a consistent relabelling of the
numbers leaves them unchanged.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

CAP = 40_000_000
RESET = 5_000_000
SECOND_BASE = 750_000
ROW_PRICE = 3
TIERS = ("6+strong", "6", "5+strong", "5", "4+strong", "4", "3+strong", "3")

LOTTO_URL = "https://www.pais.co.il/lotto/"
RESULT_URL = "https://www.pais.co.il/lotto/currentlotto.aspx?lotteryId={draw_id}"

#: 8.33% of fair draws win a prize; the smallest prize (10 NIS) exceeds the 6 NIS cost.
ORDINARY_FORM = (
    (8, 9, 11, 13, 15, 21, 1),
    (1, 2, 12, 24, 30, 35, 2),
)

#: 51.01% of fair draws win a prize; 8.33% win three or more main numbers with the strong number.
BONUS_FORM = (
    (8, 12, 15, 20, 29, 32, 1),
    (1, 23, 25, 30, 35, 37, 2),
    (8, 9, 13, 14, 16, 27, 3),
    (2, 6, 9, 17, 18, 28, 4),
    (1, 8, 10, 11, 26, 36, 4),
    (5, 21, 23, 31, 32, 36, 5),
    (18, 24, 26, 27, 29, 34, 5),
    (3, 4, 10, 12, 17, 27, 6),
    (10, 19, 20, 21, 24, 30, 3),
    (7, 19, 28, 31, 33, 35, 1),
    (2, 3, 11, 13, 22, 29, 7),
    (5, 14, 20, 26, 33, 37, 6),
    (4, 6, 15, 16, 22, 34, 2),
    (7, 12, 14, 24, 25, 36, 7),
)


class ParseError(ValueError):
    """The page did not contain a complete result in the expected layout."""


@dataclass(frozen=True)
class Result:
    """One published draw: announced prizes and the regular/Double prize tables."""

    draw_id: int
    date: str
    first_prize: int
    second_prize: int
    winners: tuple[int, ...]
    prizes: tuple[int, ...]
    double_winners: tuple[int, ...]
    double_prizes: tuple[int, ...]

    @property
    def jackpot_won(self) -> bool:
        """True if any row, regular or Double, won the first prize."""
        return self.winners[0] + self.double_winners[0] > 0

    @property
    def second_won(self) -> bool:
        """True if any row won the second prize."""
        return self.winners[1] + self.double_winners[1] > 0

    @property
    def was_bonus(self) -> bool:
        """True if three main numbers paid 15 NIS, the mark of a bonus draw."""
        return self.prizes[7] == 15


@dataclass(frozen=True)
class Plan:
    form: str
    rows: tuple[tuple[int, ...], ...]
    cost: int
    reason: str
    next_first_prize: int | None
    next_second_prize: int


def _text(page: str) -> str:
    page = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", page)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", page)))


def _number(value: str) -> int:
    return int(value.replace(",", ""))


def latest_draw_id(page: str) -> int:
    """Highest draw id linked from the Lotto landing page."""
    ids = [int(x) for x in re.findall(r"lotteryId=(\d+)", page)]
    ids += [int(x) for x in re.findall(r"תוצאות הגרלה מס' (\d+)", _text(page))]
    if not ids:
        raise ParseError("No draw id found on the Lotto page")
    return max(ids)


def parse_result(page: str) -> Result:
    """Read one ``currentlotto.aspx`` result page."""
    text = _text(page)
    draw = re.search(r"תוצאות הגרלה מס' (\d+)", text)
    date = re.search(r"לתאריך\s*(\d\d/\d\d/\d{4})", text)
    first = re.search(r"סכום הפרס הראשון בהגרלה זו עמד על ([\d,]+)", text)
    second = re.search(r"סכום הפרס השני בהגרלה זו עמד על ([\d,]+)", text)
    tiers = re.findall(r"מס' ניחושים\s*(\d(?:\s*\+\s*חזק)?)\s*כמות זכיות\s*([\d,]+)\s*גובה הפרס\s*([\d,]+)", text)
    if not (draw and date and first and second) or len(tiers) < 16:
        raise ParseError("Result page is missing the draw id, date, announced prizes or prize tables")
    labels = tuple(re.sub(r"\s*\+\s*חזק", "+strong", label) for label, _, _ in tiers[:16])
    if labels != TIERS + TIERS:
        raise ParseError(f"Unexpected prize-table layout: {labels}")
    counts = tuple(_number(count) for _, count, _ in tiers[:16])
    prizes = tuple(_number(prize) for _, _, prize in tiers[:16])
    if prizes[7] not in (10, 15) or prizes[15] != 2 * prizes[7]:
        raise ParseError(f"Unexpected fixed prize for three main numbers: {prizes[7]} / {prizes[15]}")
    return Result(int(draw.group(1)), date.group(1), _number(first.group(1)), _number(second.group(1)),
                  counts[:8], prizes[:8], counts[8:], prizes[8:])


def plan(last: Result, *, announced_bonus: bool = False) -> Plan:
    """Choose the form for the draw after ``last``."""
    if last.first_prize > CAP or last.first_prize < RESET:
        raise ValueError(f"First prize {last.first_prize} is outside the 5M-40M range of this game")
    rolled_cap = last.first_prize == CAP and not last.jackpot_won
    if rolled_cap:
        reason = f"Draw {last.draw_id} had a 40M first prize and nobody won it, so the next draw pays bonus prizes."
    elif announced_bonus:
        reason = "Bonus prizes were announced for the next draw."
    elif last.jackpot_won:
        reason = f"The first prize was won in draw {last.draw_id}; the next draw is an ordinary one."
    else:
        reason = (f"Draw {last.draw_id} had a {last.first_prize // 1_000_000}M first prize, below the 40M cap; "
                  "the next draw is an ordinary one unless a bonus series was announced.")
    bonus = rolled_cap or announced_bonus
    rows = BONUS_FORM if bonus else ORDINARY_FORM
    next_first = RESET if last.jackpot_won else CAP if last.first_prize == CAP else None
    next_second = SECOND_BASE if last.second_won else last.second_prize + SECOND_BASE
    return Plan("bonus" if bonus else "ordinary", rows, ROW_PRICE * len(rows), reason, next_first, next_second)
