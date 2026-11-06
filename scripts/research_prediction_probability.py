#!/usr/bin/env python3
"""Reproduce the prediction-calibration, power, and exact-coverage experiments.

Historical comparisons are exploratory, not a fresh prospective holdout.
No remote machines, purchases, archive refreshes, or seed campaigns are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import backtest, data, metrics  # noqa: E402
from bench.calibration import frequency_forecasts, past_loss_choices  # noqa: E402
from bench.portfolio import greedy_coverage_tickets, portfolio_odds, single_ticket_prize_odds  # noqa: E402


def prediction_experiment(draws):
    names, targets, ball_p, strong_p = frequency_forecasts(draws)
    actual = draws.multi_hot[targets]
    ball_loss = -(actual[None] * np.log(np.clip(ball_p / 6, 1e-12, 1))).sum(axis=2) / 6
    strong_loss = -np.log(np.clip(strong_p[:, np.arange(len(targets)), draws.strong[targets] - 1], 1e-12, 1))
    ball_choices = past_loss_choices(ball_loss, baseline=len(names) - 1)
    strong_choices = past_loss_choices(strong_loss, baseline=len(names) - 1)
    positions = np.arange(len(targets))
    ball_p = np.concatenate([ball_p, ball_p[ball_choices, positions][None]])
    strong_p = np.concatenate([strong_p, strong_p[strong_choices, positions][None]])
    ball_loss = np.concatenate([ball_loss, ball_loss[ball_choices, positions][None]])
    strong_loss = np.concatenate([strong_loss, strong_loss[strong_choices, positions][None]])
    names.append("adaptive past-loss selection")
    # Common block-bootstrap indices keep paired comparisons reproducible.
    rng = np.random.default_rng(20261106)
    starts = rng.integers(0, 300, size=(4000, 15))
    bootstrap_indices = ((starts[:, :, None] + np.arange(20)) % 300).reshape(4000, 300)
    rows = []
    for i, name in enumerate(names):
        picks = np.argpartition(-ball_p[i], 5, axis=1)[:, :6]
        hits = np.take_along_axis(actual, picks, axis=1).sum(axis=1).astype(int)
        strong_hits = np.argmax(strong_p[i], axis=1) + 1 == draws.strong[targets]
        summary = metrics.summarise(hits[-300:], ball_loss[i, -300:], strong_hits[-300:])
        delta = ball_loss[i, -300:] - np.log(37)
        interval = np.quantile(delta[bootstrap_indices].mean(axis=1), [0.025, 0.975])
        folds = []
        for begin in range(len(targets) - 1200, len(targets), 300):
            sl = slice(begin, begin + 300)
            folds.append({"start": str(draws.dates[targets[begin]].astype("datetime64[D]")),
                          "end": str(draws.dates[targets[begin + 299]].astype("datetime64[D]")),
                          "mean_matches": float(hits[sl].mean()),
                          "mean_ball_log_loss": float(ball_loss[i, sl].mean()),
                          "mean_strong_log_loss": float(strong_loss[i, sl].mean())})
        rows.append({"name": name, **summary,
                     "mean_ball_brier": float(((ball_p[i, -300:] - actual[-300:]) ** 2).mean()),
                     "mean_strong_log_loss": float(strong_loss[i, -300:].mean()),
                     "log_loss_difference_vs_uniform": float(delta.mean()),
                     "paired_block_bootstrap_95_interval": interval.tolist(),
                     "folds": folds})
    verdicts = metrics.holm_bonferroni({r["name"]: r["p_value_greater"] for r in rows})
    for row in rows:
        row["accuracy_significant_after_holm"] = verdicts[row["name"]]
    selected, counts = np.unique(ball_choices[-300:], return_counts=True)
    return {"scope": "Retrospective; this archive was already used in earlier project experiments",
            "primary_window": "Last 300 draws; four disjoint 300-draw blocks shown for stability",
            "selection": "Every 50 draws, minimize prior 300 losses after a 200-prediction warmup",
            "bootstrap": "4000 circular moving-block replicates, block length 20; exploratory intervals",
            "adaptive_ball_choices_last300": {names[i]: int(n) for i, n in zip(selected, counts)},
            "baseline_ball_log_loss": float(np.log(37)), "baseline_strong_log_loss": float(np.log(7)),
            "rows": rows}


def uniformity_power(n_draws, simulations=512):
    """Estimate size and power against an explicit single-ball inclusion bias."""
    rng = np.random.default_rng(20261107)
    results = []
    for multiplier in (1.0, 1.2, 1.5):
        raw_rejections = corrected_rejections = 0
        for start in range(0, simulations, 16):
            batch = min(16, simulations - start)
            # Ball 1 has inclusion probability q; conditional on that event,
            # choose the other five/six uniformly from the remaining 36 balls.
            chosen = np.argpartition(rng.random((batch, n_draws, 36)), 5, axis=2)[:, :, :6] + 1
            include_first = rng.random((batch, n_draws)) < multiplier * 6 / 37
            chosen[:, :, 5] = np.where(include_first, 0, chosen[:, :, 5])
            for history in chosen:
                counts = np.bincount(history.ravel(), minlength=37)
                q = ((counts - n_draws * 6 / 37) ** 2 / (n_draws * 6 / 37)).sum()
                raw_rejections += int(stats.chi2.sf(q, 36) < 0.05)
                corrected_rejections += int(stats.chi2.sf(q * 36 / 31, 36) < 0.05)
        interval = stats.binomtest(corrected_rejections, simulations).proportion_ci(method="wilson")
        results.append({"ball1_inclusion_multiplier": multiplier, "simulations": simulations,
                        "uncorrected_rejection_rate": raw_rejections / simulations,
                        "corrected_rejection_rate": corrected_rejections / simulations,
                        "corrected_wilson_95_interval": [float(interval.low), float(interval.high)]})
    return {"alpha": 0.05, "n_draws": n_draws,
            "alternative": "Only ball 1 biased; conditional remaining balls uniform without replacement",
            "note": "Individual-test power, before correction for the larger test family", "rows": results}


def coverage_experiment(seeds):
    from smart_generator import SmartTicketGenerator

    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        random_tickets = [sorted(rng.choice(37, 6, replace=False) + 1) + [int(rng.integers(1, 8))]
                          for _ in range(10)]
        random_tickets = [[int(x) for x in row] for row in random_tickets]
        previous_rng = np.random.get_state()
        try:
            np.random.seed(seed)
            generator = SmartTicketGenerator()
            smart = [[int(x) for x in row] for row, _ in generator.generate_optimized_tickets(n_tickets=10)]
            legacy_coverage = [[int(x) for x in row] for row, _ in generator.generate_coverage_tickets()]
        finally:
            np.random.set_state(previous_rng)
        tick = time.perf_counter()
        greedy = greedy_coverage_tickets(seed=seed)
        optimize_seconds = time.perf_counter() - tick
        for name, tickets in (("random", random_tickets), ("legacy_smart", smart),
                              ("legacy_coverage", legacy_coverage), ("greedy_3plus", greedy)):
            tick = time.perf_counter()
            odds = portfolio_odds(tickets)
            results.append({"method": name, "seed": seed, "portfolio": tickets, **odds,
                            "evaluation_seconds": time.perf_counter() - tick,
                            "optimization_seconds": optimize_seconds if name == "greedy_3plus" else None})
        print(f"Finished exact coverage for seed {seed}", flush=True)
    return {"objective": "At least one prize (>=3 main matches), same ten-ticket budget",
            "verification": "Exhaustive over all 2,324,784 main draws, not the optimization sample",
            "optimizer": "512 candidates; 30,000 synthetic fair draws; greedy incremental coverage",
            "rows": results}


def null_benchmark(draws):
    def previous():
        rng = np.random.default_rng(7)
        hot = draws.multi_hot[-300:].astype(bool)
        means = []
        for _ in range(5000):
            picks = rng.random((300, 37)).argsort(axis=1)[:, :6]
            means.append(np.take_along_axis(hot, picks, axis=1).sum(axis=1).mean())
        return np.asarray(means)

    measured = {}
    for name, run in (("previous_generate_and_sort", previous),
                      ("direct_hypergeometric", lambda: backtest.random_ticket_null(draws, 300)["samples"])):
        timings = []
        for _ in range(3):
            tick = time.perf_counter()
            samples = run()
            timings.append(time.perf_counter() - tick)
        measured[name] = {"median_seconds": float(np.median(timings)),
                          "mean": float(samples.mean()), "std": float(samples.std())}
    measured["speedup"] = (measured["previous_generate_and_sort"]["median_seconds"]
                           / measured["direct_hypergeometric"]["median_seconds"])
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--portfolio-seeds", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.portfolio_seeds <= 20:
        parser.error("portfolio-seeds must be in 1..20")
    draws = data.load(clean_csv=None)
    if len(draws) < 1400:
        parser.error("This protocol needs at least 1400 draws")
    result = {"schema": 1, "archive_sha256": hashlib.sha256(Path(data.RAW_CSV).read_bytes()).hexdigest(),
              "n_draws": len(draws), "last_draw": str(draws.dates[-1].astype("datetime64[D]")),
              "prediction": prediction_experiment(draws), "uniformity_power": uniformity_power(len(draws)),
              "single_ticket_odds": single_ticket_prize_odds(),
              "coverage": coverage_experiment(range(args.portfolio_seeds)),
              "null_benchmark": null_benchmark(draws)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
