#!/usr/bin/env python3
"""Run the full analysis and write the GitHub Pages report.

    python scripts/build_report.py [--refresh] [--n-test 300] [--quick]

`--refresh` re-downloads the draw archive; CI uses it. `--quick` skips the
neural models, which is handy when iterating on the report layout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import backtest, data, predictors, randomness, report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("build_report")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the draw archive")
    ap.add_argument("--n-test", type=int, default=300, help="walk-forward window length")
    ap.add_argument("--quick", action="store_true", help="skip neural models")
    ap.add_argument("--out-dir", default="docs", help="MkDocs source tree")
    ap.add_argument("--null-trials", type=int, default=5000)
    args = ap.parse_args()

    draws = data.load(refresh=args.refresh)

    logger.info("Running randomness tests")
    tests = randomness.run_all(draws)
    for t in tests:
        logger.info("  p=%.4f  %s", t["p_value"], t["test"])

    suite = predictors.statistical_suite()
    if not args.quick:
        from bench.nn import NeuralPredictor

        suite += [
            NeuralPredictor(arch="gru", epochs=60, refit_every=50),
            NeuralPredictor(arch="transformer", epochs=60, refit_every=50),
        ]

    logger.info("Backtesting %d strategies over %d draws", len(suite), args.n_test)
    results = backtest.run_suite(suite, draws, n_test=args.n_test)
    for r in results:
        logger.info(
            "  %-40s matches=%.4f  p=%.3f  (%.1fs)",
            r.name,
            r.summary["mean_matches"],
            r.summary["p_value"],
            r.seconds,
        )

    logger.info("Building Monte-Carlo null (%d trials)", args.null_trials)
    null_dist = backtest.random_ticket_null(draws, args.n_test, n_trials=args.null_trials)

    # Predict with the top statistical strategy, a neural model, and the
    # EV-aware ticket, so the page shows agreement is coincidental.
    to_predict = [suite[0]]
    best_name = results[0].name
    to_predict += [p for p in suite if p.name == best_name and p is not suite[0]]
    to_predict += [p for p in suite if p.name.startswith("Neural (gru")]
    to_predict += [p for p in suite if p.name.startswith("Unpopular")]

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
        str(out_dir / "results.md"), draws, tests, results, null_dist, predictions
    )

    logger.info("Wrote %s and %s", bench_dir / "index.html", out_dir / "results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
