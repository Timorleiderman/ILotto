"""Exact search for fixed linear congruential rules on complete ticket indices."""

from collections.abc import Iterator, Sequence
from itertools import islice
from math import gcd

from lotto.core.validation import is_integer
from lotto.seedsearch.generator import OUTCOMES


def generate(initial: int, multiplier: int, increment: int, modulus: int = OUTCOMES) -> Iterator[int]:
    """Yield draw two onward, using only the initial ticket and the fixed rule."""
    state = initial
    while True:
        state = (multiplier * state + increment) % modulus
        yield state


def generate_with_changing_step(initial: int, step: int, step_change: int) -> Iterator[int]:
    """Advance a ticket index by a step whose change is fixed before replay."""
    state = initial
    while True:
        state = (state + step) % OUTCOMES
        yield state
        step = (step + step_change) % OUTCOMES


def first_miss(predictions: Iterator[int], actual: Sequence[int], start_draw: int) -> dict | None:
    """Compare in order and stop as soon as a complete ticket differs."""
    for number, observed in enumerate(actual, start_draw):
        predicted = next(predictions)
        if predicted != observed:
            return {"draw_number": number, "predicted_index": predicted, "actual_index": observed}
    return None


def search_rules(outcomes: Sequence[int], modulus: int = OUTCOMES) -> dict:
    """Account for all (multiplier, increment) pairs without enumerating them."""
    if (not is_integer(modulus) or modulus < 2 or len(outcomes) < 3
            or any(not is_integer(x) or not 0 <= x < modulus for x in outcomes)):
        raise ValueError("Need a modulus of at least two and at least three valid outcome indices")
    first, second, third = int(outcomes[0]), int(outcomes[1]), int(outcomes[2])
    delta = second - first
    divisor = gcd(delta, modulus)
    change = third - second
    report = {
        "rule_count": modulus**2, "matching_through_draw_two": modulus,
        "matching_through_draw_three": 0, "distinct_possible_third_draws": modulus // divisor,
        "representative_fitted_rule": None, "first_failed_draw": 3, "first_miss_after_fit": None,
    }
    # Draw two fixes b = second - a * first for each a. Draw three then
    # requires a * delta = change (mod modulus).
    if change % divisor:
        return report
    reduced = modulus // divisor
    multiplier = 0
    if reduced > 1:
        inverse = pow(delta // divisor, -1, reduced)
        multiplier = (change // divisor * inverse) % reduced
    increment = (second - multiplier * first) % modulus
    # All solutions differ in a by multiples of 'reduced'. Every later state
    # differs from 'first' by a multiple of 'divisor', so all give the same path.
    generated = generate(first, multiplier, increment, modulus)
    miss = first_miss(islice(generated, 2, None), outcomes[3:], 4)
    return {
        **report, "matching_through_draw_three": divisor,
        "representative_fitted_rule": {"multiplier": multiplier, "increment": increment},
        "first_failed_draw": miss["draw_number"] if miss else None,
        "first_miss_after_fit": miss,
    }
