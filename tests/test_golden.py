"""Characterization tests: numbers that must not move while files do.

These pin behaviour, not correctness. If one fails during a mechanical move,
the move changed a result. If one fails after an intended fix, regenerate with
`uv run python -m tests.golden.regenerate` and put the before/after numbers in
the commit message.
"""

import json
from pathlib import Path

import pytest

from lotto.core import backtest

from lotto.core import data

from lotto.approaches import statistical as predictors
from lotto.approaches.chaos import chaos_suite
from lotto.approaches.golden_ratio import golden_suite
from lotto.tickets.coverage import portfolio_odds
from lotto.approaches.perfect_fit import spacetime_suite
from lotto.tickets.strong_assignment import strong_prize_odds
from tests import SNAPSHOT

GOLDEN = Path(__file__).parent / "golden"
ROOT = Path(__file__).parent.parent


def quick_suite_numbers() -> dict:
    draws = data.load(raw_csv=SNAPSHOT, clean_csv=None)
    suite = predictors.statistical_suite() + spacetime_suite() + chaos_suite() + golden_suite()
    results = backtest.run_suite(suite, draws, n_test=100)
    return {
        r.name: {"total_matches": int(r.match_counts.sum()), "p_value": r.summary["p_value"]}
        for r in results
    }


def coverage_numbers() -> dict:
    saved = json.loads((ROOT / "experiments/ticket_chances_2026_11_07.json").read_text())
    rows = saved["selected_portfolio"]
    return {
        "best_match_counts": portfolio_odds(rows)["best_match_counts"],
        "covered_with_strong": strong_prize_odds(rows)["covered_full_outcomes"],
    }


def test_quick_suite_is_unchanged():
    want = json.loads((GOLDEN / "quick_suite_n100.json").read_text())
    got = quick_suite_numbers()
    assert got.keys() == want.keys()
    for name, expected in want.items():
        assert got[name]["total_matches"] == expected["total_matches"], name
        assert got[name]["p_value"] == pytest.approx(expected["p_value"], rel=1e-12), name


def test_selected_portfolio_coverage_is_unchanged():
    assert coverage_numbers() == json.loads((GOLDEN / "coverage.json").read_text())
