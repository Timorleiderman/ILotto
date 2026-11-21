"""Seed once from the first complete ticket and test consecutive exact draws."""

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np

from lotto.core import data, paths
from lotto.core.files import sha256_file, write_json
from lotto.seedsearch.generator import OUTCOMES, index_ticket, ticket_index
from lotto.seedsearch.recurrence import first_miss, generate_with_changing_step, search_rules


def describe_miss(miss, days):
    if miss:
        return {
            **miss,
            "date": days[miss["draw_number"] - 1],
            "predicted_ticket": index_ticket(miss["predicted_index"]),
            "actual_ticket": index_ticket(miss["actual_index"]),
        }
    return None


def evaluate(draws: data.Draws) -> dict:
    days = draws.dates.astype("datetime64[D]").astype(str).tolist()
    if len(days) < 3 or any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("Need at least three unique draw dates in chronological order")
    outcomes = [ticket_index(b.tolist(), int(s)) for b, s in zip(draws.balls, draws.strong)]
    rng = np.random.Generator(np.random.PCG64(outcomes[0]))
    pcg_predictions = (int(rng.integers(OUTCOMES)) for _ in range(len(days) - 1))
    miss = first_miss(pcg_predictions, outcomes[1:], 2)

    first, second, third = outcomes[:3]
    step = (second - first) % OUTCOMES
    step_change = (third - second - step) % OUTCOMES
    generated = generate_with_changing_step(first, step, step_change)
    held_out = outcomes[3:]
    predictions = list(islice(generated, 2, len(outcomes) - 1))
    step_miss = first_miss(iter(predictions), held_out, 4)

    # The closed-form geometric sum supplies expected tickets independently of
    # the iterative code used to recover and replay the planted rule.
    synthetic = [((5**i - 1) // 4) % OUTCOMES for i in range(1003)]
    control = search_rules(synthetic)
    return {
        "archive_draws": len(days), "archive_last_date": days[-1], "outcome_count": OUTCOMES,
        "first_three_draws": [
            {"date": day, "index": value, **index_ticket(value)} for day, value in zip(days[:3], outcomes[:3])
        ],
        "pcg64_seeded_from_first_draw": {"numpy_version": np.__version__, "first_miss": describe_miss(miss, days)},
        "linear_congruential_family": search_rules(outcomes),
        "constant_change_in_step": {
            "fit_dates": days[:3], "initial_step": step, "step_change": step_change,
            "held_out_draws": len(held_out),
            "exact_tickets_after_fit": sum(a == b for a, b in zip(predictions, held_out, strict=True)),
            "first_miss_after_fit": describe_miss(step_miss, days),
        },
        "synthetic_control": {"planted_multiplier": 5, "planted_increment": 1,
                              "draws_after_fit": len(synthetic) - 3, "search": control},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=Path(data.RAW_CSV))
    parser.add_argument("--out", type=Path, default=paths.REPO_ROOT / "results/seed-recurrence.json")
    args = parser.parse_args()
    draws = data.load(raw_csv=str(args.raw_csv), clean_csv=None, refresh=False)
    try:
        report = evaluate(draws)
    except ValueError as exc:
        parser.error(str(exc))
    report["archive_sha256"] = sha256_file(args.raw_csv)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
