#!/usr/bin/env python3
"""Say which fixed form to play on the next Lotto draw and print its rows.

Reads the last official result from pais.co.il. The rule and both forms are
documented in ``lotto/tickets/draw_plan.py``; this is a budgeting rule under fair
draws, not a prediction, and a prize does not imply a net profit over time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from lotto.tickets.coverage import portfolio_odds
from lotto.tickets.draw_plan import (
    LOTTO_URL,
    RESULT_URL,
    ParseError,
    latest_draw_id,
    parse_result,
    plan,
)
from lotto.tickets.strong_assignment import strong_prize_odds

#: pais.co.il serves the result pages to browsers only.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def _get(url, timeout):
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bonus", action="store_true",
                        help="Pais announced bonus prizes for the next draw (promotional series)")
    parser.add_argument("--draw-id", type=int, help="Plan from this draw's result instead of the latest one")
    parser.add_argument("--result-html", type=Path, help="Read a saved result page instead of fetching")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    try:
        if args.result_html:
            page = args.result_html.read_text(encoding="utf-8", errors="replace")
        else:
            draw_id = args.draw_id or latest_draw_id(_get(LOTTO_URL, args.timeout))
            page = _get(RESULT_URL.format(draw_id=draw_id), args.timeout)
        last = parse_result(page)
    except (requests.RequestException, OSError, ParseError) as exc:
        sys.exit(f"Could not read the last result ({exc}). Check pais.co.il and rerun, "
                 "or pass --result-html with a saved result page.")
    choice = plan(last, announced_bonus=args.bonus)
    rows = [list(row) for row in choice.rows]
    odds, strong = portfolio_odds(rows), strong_prize_odds(rows)

    print(f"Last result: draw {last.draw_id} on {last.date}, first prize {last.first_prize:,} NIS "
          f"({'won' if last.jackpot_won else 'not won'}), second prize {last.second_prize:,} NIS "
          f"({'won' if last.second_won else 'not won'}).")
    print(choice.reason)
    first = f"{choice.next_first_prize:,} NIS" if choice.next_first_prize else "above the last one (see pais.co.il)"
    print(f"Next draw: first prize {first}, second prize {choice.next_second_prize:,} NIS.")
    print(f"\nPlay the {choice.form} form: {len(rows)} regular rows, {choice.cost} NIS. Do not tick Double.")
    for index, row in enumerate(rows, 1):
        print(f"  {index:>2}.  {'  '.join(f'{n:>2}' for n in row[:6])}   strong {row[6]}")
    print(f"\nChance of a prize on this draw: {odds['p_any_prize']:.2%}. "
          f"Three or more main numbers with the strong number: {strong['p_at_least_matches_and_strong']:.2%}. "
          f"First prize: {odds['jackpot_covered_outcomes']} in {odds['total_full_outcomes']:,}.")
    print(f"Smallest prize on this draw: {15 if choice.form == 'bonus' else 10} NIS.")


if __name__ == "__main__":
    main()
