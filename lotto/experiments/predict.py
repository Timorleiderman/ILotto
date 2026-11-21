"""What does every model say about the next draw, and how would you fill the form?

    uv run lotto predict                 every model, a 10-row form each
    uv run lotto predict --rows 14       the 14-row design (2, 10 or 14)
    uv run lotto predict --model hot     only models whose name contains "hot"
    uv run lotto predict --quick         skip the three trained models (no TensorFlow)

For each model this prints its single best ticket, its record on the
scoreboard, and a complete form built from the coverage design. Everything is
also saved to results/predict/ as Markdown and JSON.

Every form has the same exact chance of a prize (38.60% for ten rows), because
each is the same design with its numbers relabelled. The models disagree about
which numbers to play; the scoreboard shows none of them has beaten a random
ticket, so treat the choice between them as a matter of taste.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lotto.approaches import full_suite, quick_suite
from lotto.core import backtest, data, paths
from lotto.core.files import write_json
from lotto.tickets.coverage import portfolio_odds
from lotto.tickets.draw_plan import FORMS
from lotto.tickets.model_forms import best_ticket, model_form
from lotto.tickets.strong_assignment import strong_prize_odds

logger = logging.getLogger("lotto.predict")


def scoreboard_record() -> dict[str, dict]:
    """Each strategy's latest walk-forward result from the published report, by name."""
    path = paths.REPO_ROOT / "docs/benchmark/results.json"
    if not path.is_file():
        return {}
    return {row["name"]: row for row in json.loads(path.read_text())["strategies"]}


def predict_all(draws: data.Draws, suite: list, rows: int) -> list[dict]:
    """Ask every model about the next draw and build its form."""
    design = [list(row) for row in FORMS[rows]]
    record = scoreboard_record()
    out = []
    for predictor in suite:
        logger.info("Forecasting with %s", predictor.name)
        target_date, balls, strong = backtest.next_draw_scores(predictor, draws)
        past = record.get(predictor.name, {})
        out.append({
            "model": predictor.name,
            "description": predictor.description,
            "for_draw_date": None if target_date is None else str(target_date),
            "best_ticket": best_ticket(balls, strong),
            "form": model_form(balls, strong, design),
            "scoreboard_mean_matches": past.get("mean_matches"),
            "scoreboard_draws": past.get("n_draws"),
        })
    return out


def _numbers(row: list[int]) -> str:
    return " ".join(f"{n:2d}" for n in row[:6])


def to_markdown(results: list[dict], draws: data.Draws, rows: int) -> str:
    """The summary table and one form per model, as Markdown."""
    design = [list(row) for row in FORMS[rows]]
    odds, strong = portfolio_odds(design), strong_prize_odds(design)
    last = str(draws.dates[-1].astype("datetime64[D]"))
    lines = [
        f"# Next-draw forms, {rows} rows per model",
        "",
        f"Built from {len(draws):,} draws through {last}. Next scheduled draw: "
        f"{data.next_draw_date(draws)}.",
        "",
        f"Every form below has the same exact odds: a prize in **{odds['p_any_prize']:.2%}** of fair "
        f"draws, three or more numbers with the strong number in "
        f"**{strong['p_at_least_matches_and_strong']:.2%}**, and the first prize "
        f"{odds['jackpot_covered_outcomes']} in {odds['total_full_outcomes']:,}. "
        "The models only change which numbers fill it.",
        "",
        "## Every model at a glance",
        "",
        "Scoreboard: mean numbers matched per draw in the latest walk-forward test (a random ticket "
        "averages 0.973).",
        "",
        "| Model | Best single ticket | Strong | Scoreboard |",
        "|---|---|---:|---:|",
    ]
    for r in results:
        score = "—" if r["scoreboard_mean_matches"] is None else f"{r['scoreboard_mean_matches']:.3f}"
        lines.append(f"| {r['model']} | `{_numbers(r['best_ticket'])}` | {r['best_ticket'][6]} | {score} |")
    for r in results:
        lines += ["", f"## {r['model']}", "", "| Row | Numbers | Strong |", "|---:|---|---:|"]
        lines += [f"| {i} | `{_numbers(row)}` | {row[6]} |" for i, row in enumerate(r["form"], 1)]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Forecast the next draw with every model and write one form per model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(prog="lotto predict", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, choices=sorted(FORMS), default=10,
                        help="rows per form: one of the exact coverage designs")
    parser.add_argument("--model", help="only models whose name contains this text (case-insensitive)")
    parser.add_argument("--quick", action="store_true", help="skip the three trained models")
    parser.add_argument("--offline", action="store_true",
                        help="use the committed archive without checking pais.co.il for new draws")
    parser.add_argument("--out-dir", type=Path, default=paths.REPO_ROOT / "results/predict")
    args = parser.parse_args()

    draws = data.load(refresh=not args.offline)
    suite = quick_suite() if args.quick else full_suite()
    if args.model:
        suite = [p for p in suite if args.model.lower() in p.name.lower()]
        if not suite:
            parser.error(f"no model name contains {args.model!r}")
    results = predict_all(draws, suite, args.rows)

    markdown = to_markdown(results, draws, args.rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"next_draw_{args.rows}_rows.md").write_text(markdown, encoding="utf-8")
    write_json(
        args.out_dir / f"next_draw_{args.rows}_rows.json",
        {"draws": len(draws), "last_draw": str(draws.dates[-1].astype("datetime64[D]")),
         "next_draw": str(data.next_draw_date(draws)), "rows": args.rows, "models": results},
    )
    print("\n" + markdown)
    print(f"Saved {args.out_dir / f'next_draw_{args.rows}_rows.md'} and the .json beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
