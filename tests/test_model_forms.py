"""A model's form keeps the design's exact odds and puts its favourite numbers in the most rows."""

import json
from pathlib import Path

import numpy as np
import pytest

from lotto.tickets.coverage import portfolio_odds
from lotto.tickets.draw_plan import FORMS, TEN_ROW_FORM
from lotto.tickets.model_forms import best_ticket, model_form
from lotto.tickets.strong_assignment import strong_prize_odds

ROOT = Path(__file__).resolve().parent.parent


def test_ten_row_form_is_the_published_design():
    saved = json.loads((ROOT / "experiments/ticket_chances_2026_11_07.json").read_text())
    assert [list(row) for row in TEN_ROW_FORM] == saved["selected_portfolio"]
    assert strong_prize_odds([list(r) for r in TEN_ROW_FORM])["covered_full_outcomes"] == 969420


def test_every_form_keeps_the_design_odds():
    rng = np.random.default_rng(0)
    for design in FORMS.values():
        want = portfolio_odds([list(r) for r in design])["best_match_counts"]
        for _ in range(3):
            form = model_form(rng.normal(size=37), rng.normal(size=7), design)
            assert portfolio_odds(form)["best_match_counts"] == want
            assert all(len(set(row[:6])) == 6 and row[:6] == sorted(row[:6]) for row in form)


def test_favourite_numbers_fill_the_most_rows():
    scores = np.zeros(37)
    scores[[6, 16, 26]] = [3.0, 2.0, 1.0]  # the model likes 7 most, then 17, then 27
    form = model_form(scores, np.zeros(7), TEN_ROW_FORM)
    uses = {n: sum(n in row[:6] for row in form) for n in range(1, 38)}
    assert uses[7] >= uses[17] >= uses[27] >= max(uses[n] for n in uses if n not in (7, 17, 27))


def test_negative_and_infinite_scores_are_handled():
    scores = -np.arange(37, dtype=float)
    scores[0] = -np.inf
    form = model_form(scores, -np.ones(7), TEN_ROW_FORM)
    assert len(form) == 10
    assert best_ticket(scores, -np.ones(7)) == [2, 3, 4, 5, 6, 7, 1]


def test_flat_forecast_returns_the_design():
    assert model_form(np.ones(37), np.ones(7), TEN_ROW_FORM) == [list(r) for r in TEN_ROW_FORM]


def test_nan_and_posinf_scores_are_rejected():
    bad_balls = np.full(37, -3.6)
    bad_balls[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        best_ticket(bad_balls, np.ones(7))
    with pytest.raises(ValueError, match="NaN"):
        model_form(bad_balls, np.ones(7), TEN_ROW_FORM)

    bad_strong = np.array([0.1, 0.2, np.nan, 0.1, 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="finite"):
        best_ticket(np.ones(37), bad_strong)
    with pytest.raises(ValueError, match="finite"):
        model_form(np.ones(37), bad_strong, TEN_ROW_FORM)
