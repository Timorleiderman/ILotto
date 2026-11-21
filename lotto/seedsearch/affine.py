"""A 64-bit generator whose offset and step can be inferred from two draws."""

from collections.abc import Iterator

from lotto.core.validation import is_integer
from lotto.seedsearch.generator import MAX_SEED, OUTCOMES, Observation, prefix_hash, world_outcome

GENERATOR = "calendar-affine-sha256-v1"
SEEDS_PER_KEY = OUTCOMES**2


def affine_outcome(seed: int, draw_index: int, day: str) -> int:
    """Index draws from zero at the start of the series, including on irregular dates."""
    if not is_integer(seed) or not 0 <= seed < MAX_SEED:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if not is_integer(draw_index) or draw_index < 0:
        raise ValueError("Draw index must be a nonnegative integer")
    key, remainder = divmod(seed, SEEDS_PER_KEY)
    step, offset = divmod(remainder, OUTCOMES)
    return (world_outcome(key, day) + offset + draw_index * step) % OUTCOMES


def matching_pair(first: Observation, second: Observation) -> Iterator[int]:
    """Yield every seed matching draws zero and one in increasing seed order."""
    prefix_hash([first, second])  # validates dates, order and outcome indices
    for key in range((MAX_SEED - 1) // SEEDS_PER_KEY + 1):
        offset = (first.outcome - world_outcome(key, first.date)) % OUTCOMES
        step = (second.outcome - world_outcome(key, second.date) - offset) % OUTCOMES
        seed = key * SEEDS_PER_KEY + step * OUTCOMES + offset
        if seed < MAX_SEED:  # the last key spans only part of the offset/step space
            yield seed
