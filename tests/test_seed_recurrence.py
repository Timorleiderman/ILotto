"""Exhaustively check the congruence solver and the first-draw-only replay."""

import csv
from itertools import combinations, islice, product

import numpy as np

from lotto.core.data import Draws, load
from lotto.experiments.seed_recurrence import evaluate
from lotto.seedsearch.generator import OUTCOMES, index_ticket, ticket_index
from lotto.seedsearch.recurrence import generate, search_rules
from tests import SNAPSHOT


def test_all_rules_and_first_miss_against_brute_force():
    modulus = 4
    for history in product(range(modulus), repeat=5):
        survivors = list(product(range(modulus), repeat=2))
        counts = []
        for previous, actual in zip(history, history[1:]):
            survivors = [(a, b) for a, b in survivors if (a * previous + b) % modulus == actual]
            counts.append(len(survivors))
        result = search_rules(history, modulus)
        assert result["matching_through_draw_two"] == counts[0]
        assert result["matching_through_draw_three"] == counts[1]
        assert result["first_failed_draw"] == (counts.index(0) + 2 if 0 in counts else None)
        rule = result["representative_fitted_rule"]
        if rule:
            end = result["first_failed_draw"] or len(history) + 1
            path = generate(history[0], rule["multiplier"], rule["increment"], modulus)
            assert list(islice(path, end - 2)) == list(history[1:end - 1])


def test_pcg_starts_at_draw_two_and_stops_at_first_difference():
    initial = 123
    rng = np.random.Generator(np.random.PCG64(initial))
    draws = [initial, int(rng.integers(OUTCOMES)), (int(rng.integers(OUTCOMES)) + 1) % OUTCOMES]
    tickets = [index_ticket(value) for value in draws]
    dates = np.array(["2020-01-01", "2020-01-05", "2020-02-03"], dtype="datetime64[D]")
    report = evaluate(Draws(dates, np.array([t["balls"] for t in tickets]), np.array([t["strong"] for t in tickets])))
    pcg = report["pcg64_seeded_from_first_draw"]["first_miss"]
    assert pcg["draw_number"] == 3
    assert pcg["predicted_index"] == (draws[2] - 1) % OUTCOMES
    control = report["synthetic_control"]
    assert control["draws_after_fit"] == 1000
    assert control["search"]["representative_fitted_rule"] == {"multiplier": 5, "increment": 1}
    assert control["search"]["first_failed_draw"] is None


def test_first_archived_failure_against_independent_ranking_and_all_multipliers():
    with open(SNAPSHOT, encoding="latin-1", newline="") as stream:
        rows = {row[0]: row for row in csv.reader(stream) if row and row[0] in {"2321", "2322", "2323"}}
    prefix = [rows[str(draw)] for draw in (2321, 2322, 2323)]
    balls = [tuple(sorted(map(int, row[2:8]))) for row in prefix]
    wanted = set(balls)
    ranks = {b: i for i, b in enumerate(combinations(range(1, 38), 6)) if b in wanted}
    indices = [7 * ranks[b] + int(row[8]) - 1 for b, row in zip(balls, prefix)]
    draws = load(raw_csv=SNAPSHOT, clean_csv=None, refresh=False)
    assert indices == [ticket_index(b.tolist(), int(s)) for b, s in zip(draws.balls[:3], draws.strong[:3])]
    first, second, third = indices
    survivors = 0
    for start in range(0, OUTCOMES, 65_536):
        a = np.arange(start, min(start + 65_536, OUTCOMES), dtype=np.int64)
        b = (second - a * first) % OUTCOMES
        survivors += int(np.count_nonzero((a * second + b) % OUTCOMES == third))
    assert survivors == search_rules(indices)["matching_through_draw_three"] == 0


def test_changing_step_recovers_planted_rule_and_locates_a_corrupted_draw():
    initial, step, change = 23, OUTCOMES - 3, 5
    outcomes = [(initial + i * step + i * (i - 1) // 2 * change) % OUTCOMES for i in range(1003)]
    days = np.datetime64("2020-01-01") + np.arange(len(outcomes))

    def run(values):
        tickets = [index_ticket(x) for x in values]
        draws = Draws(days, np.array([t["balls"] for t in tickets]), np.array([t["strong"] for t in tickets]))
        return evaluate(draws)["constant_change_in_step"]

    good = run(outcomes)
    assert (good["initial_step"], good["step_change"]) == (step, change)
    assert good["exact_tickets_after_fit"] == good["held_out_draws"] == 1000
    assert good["first_miss_after_fit"] is None
    original = outcomes[499]
    outcomes[499] = (original + 1) % OUTCOMES
    broken = run(outcomes)
    assert broken["first_miss_after_fit"]["draw_number"] == 500
    assert broken["first_miss_after_fit"]["predicted_index"] == original
    assert broken["exact_tickets_after_fit"] == 999
