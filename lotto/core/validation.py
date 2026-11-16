"""Small input checks shared across the package."""

from __future__ import annotations

import numpy as np


def is_integer(value) -> bool:
    """True for Python and NumPy integers, False for booleans.

    `True` is an `int` in Python, so a plain `isinstance(value, int)` would
    accept a flag where a count was meant.
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
