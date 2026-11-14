#!/usr/bin/env python3
"""Run the full analysis and write the benchmark report.

    uv run lotto report [--offline] [--n-test 300] [--quick] [--out-dir DIR]

Every run first checks pais.co.il for new draws and adds any it finds to the
archive (validated; a failure falls back to the committed copy). `--offline`
skips that check. `--quick` skips the neural models.
Pages go to `results/report/` by default, so trying the command never touches
the published site; pass `--out-dir docs` to rebuild the pages the site serves.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lotto.approaches import full_suite, quick_suite
from lotto.core import backtest, data, randomness
from lotto.report import html as report

logger = logging.getLogger("lotto.report")


def main() -> int:
    """Run the benchmark and write the report pages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="use the committed archive without checking pais.co.il for new draws")
    # Checking for new draws is now the default; the old flag is still accepted.
    ap.add_argument("--refresh", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--n-test", type=int, default=300, help="walk-forward window length")
    ap.add_argument("--quick", action="store_true", help="skip neural models")
    ap.add_argument("--out-dir", default=str(data.REPO_ROOT / "results/report"),
                    help="where to write the pages; use docs to update the published site")
    ap.add_argument("--null-trials", type=int, default=5000)
    args = ap.parse_args()

    draws = data.load(refresh=not args.offline)

    logger.info("Running randomness tests")
    tests = randomness.run_all(draws)
    for t in tests:
        logger.info("  p=%.4f  %s", t["p_value"], t["test"])

    suite = quick_suite() if args.quick else full_suite()

    # One window, resolved once: the backtest, the Monte-Carlo null and the
    # report must all describe the same draws.
    window = backtest.evaluation_window(len(draws), args.n_test)
    if len(window) < args.n_test:
        logger.warning("Only %d draws remain after the warm-up; scoring those instead of %d",
                       len(window), args.n_test)
    logger.info("Backtesting %d strategies over %d draws", len(suite), len(window))
    results = backtest.run_suite(suite, draws, n_test=args.n_test)
    for r in results:
        logger.info(
            "  %-40s matches=%.4f  p=%.3f  (%.1fs)",
            r.name,
            r.summary["mean_matches"],
            r.summary["p_value"],
            r.seconds,
        )

    # The date-conditioned model cannot be judged by the strategy table alone:
    # its in-sample reconstruction is the number people will quote, so it is
    # measured explicitly and shown next to its walk-forward score.
    memorisation = None
    if not args.quick:
        from lotto.approaches import date_conditioned as gen
        from lotto.approaches.date_conditioned import DateConditionedCB, MemorisingDateLookup

        memorisation = []
        for p in (DateConditionedCB(), MemorisingDateLookup()):
            rec = gen.reconstruction_rate(p, draws)
            # The suite already backtested the date-conditioned model; only the
            # lookup control, which is not a scoreboard strategy, needs a run here.
            scored = next((r.summary for r in results if r.name == p.name), None)
            if scored is None:
                scored = backtest.walk_forward(p, draws, n_test=args.n_test).summary
            memorisation.append({
                "name": p.name,
                "dkrr_matches": rec["dkrr_matches"],
                "dkrr_log_loss": rec["dkrr_log_loss"],
                "wf_matches": scored["mean_matches"],
                "wf_p": scored["p_value"],
            })
            logger.info("  %-40s DKRR=%.3f  walk-forward=%.4f",
                        p.name, rec["dkrr_matches"], scored["mean_matches"])

    logger.info("Building Monte-Carlo null (%d trials)", args.null_trials)
    null_dist = backtest.random_ticket_null(draws, len(window), n_trials=args.null_trials)

    # Show the random ticket, the top-ranked strategy, one neural model and the
    # date-conditioned model, so the page shows that agreement is coincidental.
    to_predict = [suite[0]]
    best_name = results[0].name
    to_predict += [p for p in suite if p.name == best_name and p is not suite[0]]
    to_predict += [p for p in suite if p.name.startswith("Neural (gru")]
    # Always show the date-conditioned model's pick: generating a ticket for a
    # named future date is the whole point of it, and this is the one call where
    # it is forecasting rather than reconstructing.
    to_predict += [p for p in suite if p.wants_target_date]

    predictions = []
    seen = set()
    for p in to_predict:
        if p.name in seen:
            continue
        seen.add(p.name)
        logger.info("Predicting next draw with %s", p.name)
        predictions.append(report.predict_next(p, draws))

    # The standalone report lives under docs/benchmark/ so MkDocs copies it
    # verbatim into the built site; docs/index.md is the site landing page.
    out_dir = Path(args.out_dir)
    bench_dir = out_dir / "benchmark"
    report.build(
        draws,
        tests,
        results,
        null_dist,
        predictions,
        data.era_table(),
        out_path=str(bench_dir / "index.html"),
    )
    report.dump_json(
        str(bench_dir / "results.json"), draws, tests, results, null_dist, predictions
    )
    report.dump_markdown(
        str(out_dir / "results.md"), draws, tests, results, null_dist, predictions,
        memorisation=memorisation,
    )

    logger.info("Wrote %s and %s", bench_dir / "index.html", out_dir / "results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
