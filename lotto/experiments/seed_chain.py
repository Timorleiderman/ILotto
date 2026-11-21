"""Start with the first archived draw, then predict, reveal and update in order."""

import argparse
import json
from collections import Counter
from pathlib import Path

from lotto.core import data, metrics, paths
from lotto.core.files import sha256_file, write_json
from lotto.experiments.seed_family import score
from lotto.seedsearch.generator import OUTCOMES, ticket_index, world_outcome


def replay(draws: data.Draws, learn_draws: int = 1000) -> dict:
    days = draws.dates.astype("datetime64[D]").astype(str).tolist()
    if not 1 <= learn_draws < len(days) - 1:
        raise ValueError("Need an initial draw, the learning period and at least one later draw")
    if any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("Draw dates must be unique and chronological")

    initial = ticket_index(draws.balls[0].tolist(), int(draws.strong[0]))
    previous = free_state = initial
    offset_counts = Counter()
    offset = 0
    log = []
    for i, day in enumerate(days[1:], start=1):
        free_state = world_outcome(free_state, day)
        from_previous = world_outcome(previous, day)
        log.append({
            "date": day, "free_running": free_state, "previous_actual": from_previous,
            "offset_used": offset, "corrected": (from_previous + offset) % OUTCOMES,
        })
        previous = ticket_index(draws.balls[i].tolist(), int(draws.strong[i]))
        log[-1]["actual"] = previous
        if i <= learn_draws:
            # Which offset would have made this prediction exact? Keep the most
            # frequent offset learned so far, retaining the incumbent on ties.
            observed_offset = (previous - from_previous) % OUTCOMES
            offset_counts[observed_offset] += 1
            if offset_counts[observed_offset] > offset_counts[offset]:
                offset = observed_offset

    variants = ("free_running", "previous_actual", "corrected")

    def period(start, stop):
        rows = log[start:stop]
        actual = draws.slice(start + 1, stop + 1)
        return {
            "draws": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
            **{name: score((r[name] for r in rows), actual) for name in variants},
        }

    holdout = period(learn_draws, len(log))
    holdout["main_matches_significant_after_holm"] = metrics.holm_bonferroni(
        {name: holdout[name]["p_greater_vs_fair_main_matches"] for name in variants}
    )
    return {
        "generator": "calendar-sha256-chain-v1", "initial_date": days[0],
        "initial_state": initial,
        "fair_expected_mean_main_matches": metrics.BASELINE_MATCHES,
        "online_learning": period(0, learn_draws), "frozen_offset": offset,
        "distinct_observed_offsets": len(offset_counts), "draws_supporting_offset": offset_counts[offset],
        "retrospective_holdout": holdout, "predictions": log,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=Path(data.RAW_CSV))
    parser.add_argument("--learn-draws", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=paths.REPO_ROOT / "results/seed-chain.json")
    args = parser.parse_args()
    draws = data.load(raw_csv=str(args.raw_csv), clean_csv=None, refresh=False)
    try:
        report = replay(draws, args.learn_draws)
    except ValueError as exc:
        parser.error(str(exc))
    report["archive_sha256"] = sha256_file(args.raw_csv)
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
