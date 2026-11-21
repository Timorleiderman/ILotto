"""The ticket-value model: exact tier odds, the published tax schedule, prize sharing and the archive."""

import json
from math import comb, exp

import pytest

from lotto.economics import archive
from lotto.economics import value as v
from lotto.tickets.draw_plan import CAP, Result
from tests import PRIZES_SNAPSHOT

RESULTS = archive.load(PRIZES_SNAPSHOT)


# --- exact odds and tax ------------------------------------------------------


def test_tiers_partition_the_winning_rows_exactly():
    assert v.FULL_COVERAGE_ROWS == 16_273_488 == comb(37, 6) * 7
    assert v.TIER_WAYS == (1, 6, 186, 1116, 6975, 41850, 89900, 539400)
    three_plus = sum(comb(6, k) * comb(31, 6 - k) for k in (3, 4, 5, 6)) * 7
    assert sum(v.TIER_WAYS) == three_plus


@pytest.mark.parametrize("prize, tax", [
    (10, 0), (33_840, 0), (50_000, 11_312), (60_000, 18_312), (70_000, 24_500), (500_000, 175_000),
    (1_000_000, 363_922), (1_500_000, 563_922), (5_000_000, 1_963_922), (40_000_000, 15_963_922),
])
def test_tax_matches_the_worked_examples_pais_publishes(prize, tax):
    assert v.tax(prize) == pytest.approx(tax, abs=0.5)


def test_tax_is_continuous_where_the_bands_meet():
    for edge in (v.TAX_EXEMPT, v.TAX_FULL_FROM, v.SURTAX_FROM):
        assert v.tax(edge - 1e-6) == pytest.approx(v.tax(edge + 1e-6), abs=1e-3)


# --- sharing -----------------------------------------------------------------


def test_shared_pool_has_the_closed_form_without_tax():
    p, rows, pool = 1e-6, 2_000_000, 10_000_000
    lam = rows * p
    expected = p * pool * (1 - exp(-lam)) / lam  # E[1/(1+K)] for K ~ Poisson(lam)
    assert v.shared_prize_value(pool, p, rows, taxed=False) == pytest.approx(expected, rel=1e-9)
    assert v.shared_prize_value(pool, p, 0, taxed=False) == pytest.approx(p * pool)
    assert v.shared_prize_value(pool, p, rows, 2, taxed=False) == pytest.approx(2 * expected, rel=1e-9)


def test_the_split_rule_is_what_the_tables_show():
    """Draw 3300: 25M first prize, one regular and two Double winners, paid by headcount, Double twice."""
    draw = next(r for r in RESULTS if r.draw_id == 3300)
    winners = draw.winners[0] + draw.double_winners[0]
    assert draw.prizes[0] == draw.first_prize // winners
    assert draw.double_prizes[0] == pytest.approx(2 * draw.first_prize / winners, abs=1)


def test_double_rows_are_paid_exactly_twice_in_every_lower_tier():
    for r in RESULTS:
        for i in range(2, 8):
            if r.winners[i] and r.double_winners[i]:
                assert r.double_prizes[i] == 2 * r.prizes[i], r.draw_id


def test_rows_sold_recovers_a_synthetic_draw():
    rows = 3_000_000
    winners = tuple(round(rows * p) for p in v.TIER_P)
    result = Result(1, "2026-01-01", 5_000_000, 750_000, winners, (0, 0, 5000, 600, 150, 50, 40, 10),
                    (0,) * 8, (0,) * 8)
    regular, double = v.rows_sold(result)
    assert regular == pytest.approx(rows, rel=1e-4) and double == 0


# --- the archive -------------------------------------------------------------


def test_archive_round_trips_and_only_grows(tmp_path):
    path = tmp_path / "prizes.csv"
    archive.save(RESULTS[:50], str(path))
    assert archive.load(str(path)) == RESULTS[:50]

    pages = {archive.LOTTO_URL: "<a href='CurrentLotto.aspx?lotteryId=3033'>x</a>"}

    def get(url, timeout):
        if url in pages:
            return pages[url]
        draw_id = int(url.rsplit("=", 1)[1])
        if draw_id == 3033:
            raise OSError("timed out")
        return _page(next(r for r in RESULTS if r.draw_id == draw_id))

    state, added = archive.refresh(str(path), get=get)
    assert (state, added) == ("updated", 1)  # 3032 added, stopped at 3033, nothing rewritten
    assert archive.load(str(path)) == RESULTS[:51]
    assert archive.refresh(str(path), get=lambda url, t: (_ for _ in ()).throw(OSError("down")))[0] == "failed"
    assert archive.load(str(path)) == RESULTS[:51]


def _page(r: Result) -> str:
    labels = ("6 + חזק", "6", "5 + חזק", "5", "4 + חזק", "4", "3 + חזק", "3")
    d = r.date.split("-")

    def table(counts, amounts):
        return "".join(f"<div>מס' ניחושים</div><b>{label}</b><div>כמות זכיות</div><b>{c:,}</b>"
                       f"<div>גובה הפרס</div><b>{a:,} ₪</b>" for label, c, a in zip(labels, counts, amounts))
    return (f"<h3>תוצאות הגרלה מס' {r.draw_id}</h3><div>לתאריך</div><strong>{d[2]}/{d[1]}/{d[0]}</strong>"
            f"{table(r.winners, r.prizes)}{table(r.double_winners, r.double_prizes)}"
            f"<div>סכום הפרס הראשון בהגרלה זו עמד על <strong>{r.first_prize:,} ₪</strong></div>"
            f"<div>סכום הפרס השני בהגרלה זו עמד על <strong>{r.second_prize:,} ₪</strong></div>")


def test_announcement_is_read_from_the_lotto_page_json():
    text = ('[{"nextLottoryDate":"Nov 21, 2026 22:45:00","displayDate":"21/11/2026","displayTime":"22:45",'
            '"firstPrize":12000000,"secondPrize":750000,"LotteryNumber":3987}]')
    a = archive.parse_announcement(text)
    assert (a.draw_id, a.sales_close, a.first_prize, a.second_prize) == (3987, "2026-11-21T22:45", 12_000_000, 750_000)
    assert archive.fetch_announcement(get=lambda url, t: "<html>blocked</html>") is None


def test_inferred_prizes_follow_the_game_rules():
    won = next(r for r in RESULTS if r.jackpot_won)
    assert archive.inferred_next_prizes([won])[0] == 5_000_000
    capped = next(r for r in RESULTS if r.first_prize == CAP and not r.jackpot_won)
    assert archive.inferred_next_prizes([capped])[0::2] == (CAP, True)
    # Draw 3986 (10M, not won) was followed by an advertised 12M.
    assert archive.inferred_next_prizes(RESULTS)[:2] == (12_000_000, 750_000)


# --- the verdict -------------------------------------------------------------


def test_no_draw_since_2018_was_worth_its_price_after_tax():
    taxed = v.summarise(v.history(RESULTS))
    assert taxed["draws"] == 1005 and taxed["positive_draws"] == 0
    assert taxed["best"]["return_per_shekel"] == pytest.approx(0.8967, abs=1e-3)
    untaxed = v.summarise(v.history(RESULTS, taxed=False))
    assert untaxed["positive_draws"] == 31  # the tax is what tips them


def test_next_draw_prices_rows_and_full_coverage_consistently():
    plan = v.next_draw(RESULTS, 12_000_000, 750_000, rows=10)
    assert plan["cost"] == 30 and plan["expected_return"] == pytest.approx(10 * plan["per_row"]["total"])
    assert not plan["positive"] and plan["break_even"]["sales_grow"] > CAP
    fc = plan["full_coverage"]
    assert fc["cost"] == 3 * v.FULL_COVERAGE_ROWS and fc["expected_net"] < 0
    json.dumps(plan, allow_nan=False)


def test_break_even_is_where_the_value_meets_the_price():
    plan = v.next_draw(RESULTS, 12_000_000, 750_000)
    lower = tuple(plan["lower_amounts"])
    fixed = plan["break_even"]["sales_fixed"]
    at = v.row_value(fixed, 750_000, lower, plan["rows_sold"])["total"]
    assert at == pytest.approx(3, abs=1e-6)

