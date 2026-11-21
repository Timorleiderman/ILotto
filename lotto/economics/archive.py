"""The published prize tables, one row per draw, and the next draw's announced prizes.

``input/lotto_prizes.csv`` is committed for the same reason as the draw
archive: pais.co.il is unreachable from CI, and the pages must still build.
pais.co.il serves a result page with prize tables for every draw from 2982
(30 Jan 2018) on; earlier pages answer "page not found".

Each row holds the announced first and second prize and, for the regular and
the Double tables, the number of winning rows and the amount each was paid.
The file only grows: a refresh appends draws after the last one on record and
never rewrites a row.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from lotto.core import paths
from lotto.tickets.draw_plan import (
    CAP,
    HEADERS,
    LOTTO_URL,
    RESET,
    RESULT_URL,
    SECOND_BASE,
    ParseError,
    Result,
    latest_draw_id,
    parse_result,
)

logger = logging.getLogger(__name__)

PRIZES_CSV_IN_REPO = "input/lotto_prizes.csv"
PRIZES_CSV = str(paths.REPO_ROOT / PRIZES_CSV_IN_REPO)
#: The JSON the Lotto page itself reads for its countdown and "now in Lotto" banner.
NEXT_URL = "https://www.pais.co.il/include/getNextLotteryDate.ashx?type=1"
#: The first draw whose result page pais.co.il still serves.
FIRST_PUBLISHED_DRAW = 2982
#: A refresh fetches at most this many result pages, so one call stays polite and short.
MAX_PAGES_PER_REFRESH = 60

TIER_KEYS = ("6s", "6", "5s", "5", "4s", "4", "3s", "3")
COLUMNS = (["draw_id", "date", "first_prize", "second_prize"]
           + [f"winners_{k}" for k in TIER_KEYS] + [f"prize_{k}" for k in TIER_KEYS]
           + [f"double_winners_{k}" for k in TIER_KEYS] + [f"double_prize_{k}" for k in TIER_KEYS])


@dataclass(frozen=True)
class Announcement:
    """What pais.co.il advertises for the next draw."""

    draw_id: int
    sales_close: str  #: ISO local time in Israel, e.g. 2026-11-21T22:45
    first_prize: int
    second_prize: int


def _iso(date_ddmmyyyy: str) -> str:
    return datetime.strptime(date_ddmmyyyy, "%d/%m/%Y").date().isoformat()


def to_row(result: Result) -> dict:
    """One CSV row; dates are stored as ISO ``YYYY-MM-DD``."""
    date = result.date if "-" in result.date else _iso(result.date)
    row = {"draw_id": result.draw_id, "date": date,
           "first_prize": result.first_prize, "second_prize": result.second_prize}
    for prefix, values in (("winners", result.winners), ("prize", result.prizes),
                           ("double_winners", result.double_winners), ("double_prize", result.double_prizes)):
        row.update({f"{prefix}_{k}": int(v) for k, v in zip(TIER_KEYS, values)})
    return row


def from_row(row: dict) -> Result:
    def tier(prefix):
        return tuple(int(row[f"{prefix}_{k}"]) for k in TIER_KEYS)

    return Result(int(row["draw_id"]), row["date"], int(row["first_prize"]), int(row["second_prize"]),
                  tier("winners"), tier("prize"), tier("double_winners"), tier("double_prize"))


def load(path: str = PRIZES_CSV) -> list[Result]:
    """Every prize table on record, oldest first. Missing file: an empty list."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as stream:
        results = [from_row(row) for row in csv.DictReader(stream)]
    return sorted(results, key=lambda r: r.draw_id)


def save(results: list[Result], path: str = PRIZES_CSV) -> None:
    """Write the archive so readers see either the old file or the complete new one."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp",
                                     newline="", encoding="utf-8", delete=False) as stream:
        staging = stream.name
        try:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for result in sorted(results, key=lambda r: r.draw_id):
                writer.writerow(to_row(result))
            stream.flush()
        except BaseException:
            stream.close()
            os.remove(staging)
            raise
    os.chmod(staging, 0o644)  # a temporary file is private; the archive is an ordinary committed file
    os.replace(staging, path)


def parse_announcement(text: str) -> Announcement:
    """Read the ``getNextLotteryDate.ashx`` answer: a one-element JSON list."""
    try:
        item = json.loads(text)[0]
        close = datetime.strptime(item["nextLottoryDate"], "%b %d, %Y %H:%M:%S")
        first, second = int(item["firstPrize"]), int(item["secondPrize"])
        draw_id = int(item["LotteryNumber"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ParseError(f"Unexpected next-draw announcement: {exc}") from exc
    if first <= 0 or second <= 0:
        raise ParseError("The announcement carries no prizes yet")
    return Announcement(draw_id, close.isoformat(timespec="minutes"), first, second)


def inferred_next_prizes(results: list[Result]) -> tuple[int, int, bool]:
    """(first, second, bonus likely) for the draw after the last table, when no announcement was read.

    The first prize resets to 5M after a win and stays at the 40M cap; otherwise
    it grows by the median step seen after unclaimed first prizes. The second
    prize grows by 750,000 per unclaimed draw.
    """
    last = results[-1]
    if last.jackpot_won:
        first = RESET
    elif last.first_prize >= CAP:
        first = CAP
    else:
        steps = [b.first_prize - a.first_prize for a, b in zip(results, results[1:])
                 if b.draw_id == a.draw_id + 1 and not a.jackpot_won and a.first_prize < b.first_prize <= CAP]
        step = sorted(steps[-100:])[len(steps[-100:]) // 2] if steps else 1_000_000
        first = min(CAP, last.first_prize + step)
    second = SECOND_BASE if last.second_won else last.second_prize + SECOND_BASE
    return first, second, last.first_prize >= CAP and not last.jackpot_won


def _get(url: str, timeout) -> str:
    import requests  # imported here so reading the archive never needs the network stack

    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_announcement(timeout=(5, 20), get: Callable[[str, object], str] = _get) -> Announcement | None:
    """The next draw's advertised prizes, or None if pais.co.il cannot be read."""
    try:
        return parse_announcement(get(NEXT_URL, timeout))
    except Exception as exc:  # network down, blocked, or a changed layout: never fatal
        logger.warning("Could not read the next draw's announced prizes (%s)", exc)
        return None


def refresh(path: str = PRIZES_CSV, timeout=(5, 20), get: Callable[[str, object], str] = _get,
            max_pages: int = MAX_PAGES_PER_REFRESH) -> tuple[str, int]:
    """Append the prize tables published since the last one on record.

    Returns ``(state, added)`` with state "updated", "unchanged" or "failed".
    Draws are fetched in order and the archive is saved after the run, so a
    failure part-way keeps every table read before it. Like the draw archive,
    a failure is never fatal: the committed copy stays in use.
    """
    results = load(path)
    known = results[-1].draw_id if results else FIRST_PUBLISHED_DRAW - 1
    try:
        latest = latest_draw_id(get(LOTTO_URL, timeout))
    except Exception as exc:
        logger.warning("Could not reach pais.co.il for prize tables (%s); using committed copy", exc)
        return "failed", 0
    added, state = [], "unchanged"
    for draw_id in range(known + 1, min(latest, known + max_pages) + 1):
        try:
            result = parse_result(get(RESULT_URL.format(draw_id=draw_id), timeout))
        except Exception as exc:
            logger.warning("Stopped at draw %d: its prize table could not be read (%s)", draw_id, exc)
            state = "failed"
            break
        if result.draw_id != draw_id:
            logger.warning("Asked for draw %d, got %d; stopping", draw_id, result.draw_id)
            state = "failed"
            break
        added.append(result)
    if added:
        save(results + added, path)
        logger.info("Added %d prize table%s (through draw %d)", len(added), "" if len(added) == 1 else "s",
                    added[-1].draw_id)
        state = "updated"
    return state, len(added)
