"""Fit the first two archive draws exactly and score one fixed seed on every later draw."""

import argparse
import json
from pathlib import Path

import numpy as np

from lotto.core import data, metrics, paths
from lotto.core.files import sha256_file, write_json
from lotto.seedsearch.affine import GENERATOR, affine_outcome, matching_pair
from lotto.seedsearch.generator import MAX_SEED, Observation, index_ticket, ticket_index, world_outcome
from lotto.tickets.forecast_portfolios import score_portfolios


def score(outcomes, held_out):
    rows = []
    for outcome in outcomes:
        ticket = index_ticket(outcome)
        rows.append([*ticket["balls"], ticket["strong"]])
    matches, strong_hits = score_portfolios(np.asarray(rows)[:, None, :], held_out.balls, held_out.strong)
    summary = metrics.summarise(matches[:, 0], None, strong_hits[:, 0])
    return {
        "mean_main_matches": summary["mean_matches"],
        "p_greater_vs_fair_main_matches": summary["p_value_greater"],
        "strong_hits": int(strong_hits.sum()),
        "exact_tickets": int(((matches == data.N_DRAWN) & strong_hits).sum()),
    }


def evaluate(draws: data.Draws) -> dict:
    days = draws.dates.astype("datetime64[D]").astype(str).tolist()
    if len(days) < 3 or any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("Need at least three draws with distinct dates in chronological order")
    first, second = [
        Observation(days[i], ticket_index(draws.balls[i].tolist(), int(draws.strong[i])))
        for i in range(2)
    ]
    candidates = list(matching_pair(first, second))
    selected = candidates[0]  # fixed tie-break before any held-out score or third-draw check
    held_out = draws.slice(2, None)
    predictions = [affine_outcome(selected, i, days[i]) for i in range(2, len(days))]
    baseline = [world_outcome(0, day) for day in days[2:]]
    third = ticket_index(held_out.balls[0].tolist(), int(held_out.strong[0]))
    return {
        "generator": GENERATOR,
        "seed_space_size": MAX_SEED,
        "fit_dates": days[:2],
        "seeds_matching_first_two": len(candidates),
        "seeds_matching_first_three": sum(affine_outcome(s, 2, days[2]) == third for s in candidates),
        "selection_rule": "Smallest seed matching the first two draws",
        "selected_seed": selected,
        "retrospective_holdout": {
            "draws": len(held_out), "first_date": days[2], "last_date": days[-1],
            "affine": score(predictions, held_out), "calendar_sha256_seed_zero": score(baseline, held_out),
            "fair_expected_mean_main_matches": metrics.BASELINE_MATCHES,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=Path(data.RAW_CSV))
    parser.add_argument("--out", type=Path, default=paths.REPO_ROOT / "results/seed-family.json")
    args = parser.parse_args()
    draws = data.load(raw_csv=str(args.raw_csv), clean_csv=None, refresh=False)
    report = evaluate(draws)
    report["archive_sha256"] = sha256_file(args.raw_csv)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
