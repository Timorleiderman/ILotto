#!/usr/bin/env python3
"""Is a Lotto row ever worth its price, after prize sharing and Israeli tax?

Reads the published prize tables (``input/lotto_prizes.csv``, refreshed from
pais.co.il unless ``--offline``), values a row in every past draw, and prices
the next draw: the cost of N rows, their expected return, the first prize at
which a row would break even, and the cost of buying every possible row.
The model is documented in ``lotto/economics/value.py``.
"""

from __future__ import annotations

import argparse
import logging

from lotto.core.files import write_json
from lotto.economics import archive
from lotto.economics import value as ticket_value


def _millions(n: float) -> str:
    return f"{n / 1e6:,.1f}M NIS"


def _next_prizes(results, announcement) -> tuple[int, int, str]:
    last = results[-1]
    if announcement is not None and announcement.draw_id > last.draw_id:
        return announcement.first_prize, announcement.second_prize, f"advertised for draw {announcement.draw_id}"
    first, second, _ = archive.inferred_next_prizes(results)
    return first, second, f"inferred from draw {last.draw_id}; pass --first to set it"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lotto run value", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, help="first prize of the next draw, NIS (default: as advertised)")
    parser.add_argument("--second", type=int, help="second prize of the next draw, NIS")
    parser.add_argument("--rows", type=int, default=10, help="rows to price (default 10)")
    parser.add_argument("--kind", choices=ticket_value.KINDS, default="regular")
    parser.add_argument("--bonus", action="store_true", help="the next draw pays bonus prizes")
    parser.add_argument("--no-tax", action="store_true", help="ignore the tax on prizes")
    parser.add_argument("--offline", action="store_true", help="never contact pais.co.il")
    parser.add_argument("--prizes", default=archive.PRIZES_CSV, help="prize-table archive to read")
    parser.add_argument("--out", help="also write the full result as JSON here (e.g. results/value.json)")
    args = parser.parse_args(argv)
    if not 1 <= args.rows <= ticket_value.FULL_COVERAGE_ROWS:
        parser.error(f"--rows must be between 1 and {ticket_value.FULL_COVERAGE_ROWS:,}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    announcement = None
    if not args.offline:
        archive.refresh(args.prizes)
        announcement = archive.fetch_announcement()
    results = archive.load(args.prizes)
    if len(results) < 20:
        parser.error(f"Only {len(results)} prize tables in {args.prizes}; run without --offline first")
    taxed = not args.no_tax
    first, second, source = _next_prizes(results, announcement)
    first, second = args.first or first, args.second or second
    plan = ticket_value.next_draw(results, first, second, bonus=args.bonus, rows=args.rows, kind=args.kind,
                                  taxed=taxed)
    history = ticket_value.history(results, taxed)
    summary = ticket_value.summarise(history)

    row, be, fc = plan["per_row"], plan["break_even"], plan["full_coverage"]
    after = "after tax" if taxed else "before tax"
    print(f"Next draw: first prize {_millions(first)}, second prize {second:,} NIS ({source}).")
    print(f"About {plan['rows_sold'] / 1e6:.2f}M rows expected to share them.\n")
    print(f"One {args.kind} row costs {row['price']} NIS and returns {row['total']:.2f} NIS on average {after}:")
    print(f"  first prize {row['first_prize']:.2f}, second prize {row['second_prize']:.2f}, "
          f"smaller prizes {row['lower_tiers']:.2f}")
    print(f"{args.rows:,} rows: cost {plan['cost']:,} NIS, expected back {plan['expected_return']:,.2f} NIS, "
          f"expected result {plan['expected_net']:+,.2f} NIS ({plan['return_per_shekel']:.2f} NIS per NIS).")
    if be["sales_grow"] is None:
        print("No first prize makes a row worth its price.")
    else:
        cap = " - above the 40M cap, so no ordinary draw reaches it" if be["sales_grow"] > be["cap"] else ""
        print(f"Break-even first prize: {_millions(be['sales_grow'])}{cap}.")
    print(f"\nEvery row ({fc['rows']:,}): cost {_millions(fc['cost'])}, expected back "
          f"{_millions(fc['expected_return'])}, expected result {_millions(fc['expected_net'])}.")
    print(f"\nHistory, {summary['draws']:,} draws {summary['first_date']} to {summary['last_date']} ({after}): "
          f"a row was worth its price in {summary['positive_draws']} draws. Median return "
          f"{summary['median_return']:.2f} NIS per NIS; best {summary['best']['return_per_shekel']:.2f} "
          f"(draw {summary['best']['draw_id']}, {summary['best']['date']}).")
    if args.out:
        write_json(args.out, {"plan": plan, "summary": summary, "history": history})
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
