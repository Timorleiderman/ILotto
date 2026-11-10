import pytest

from bench.draw_plan import (
    BONUS_FORM, ORDINARY_FORM, ParseError, Result, latest_draw_id, parse_result, plan,
)
from bench.portfolio import portfolio_odds
from bench.strong_assignment import optimize_strong_numbers, strong_prize_odds

LABELS = ("6 + חזק", "6", "5 + חזק", "5", "4 + חזק", "4", "3 + חזק", "3")


def _page(draw_id=3929, first="40,000,000", second="750,000", winners=(0, 1, 62, 345, 2392, 13038, 31644, 172773),
          prizes=(0, 250000, 4843, 693, 135, 48, 34, 10), double_winners=(0, 2, 47, 312, 2138, 12170, 28014, 155349)):
    def table(counts, amounts):
        return "".join(f"<div>מס' ניחושים</div><b>{label}</b><div>כמות זכיות</div><b>{count:,}</b>"
                       f"<div>גובה הפרס</div><b>{amount:,} ₪</b>"
                       for label, count, amount in zip(LABELS, counts, amounts))
    return (f"<html><script>var x = \"מס' ניחושים 9\";</script><body><h1>תוצאות הגרלה מס' {draw_id}</h1>"
            f"<span>לתאריך</span> <span>26/05/2026</span>{table(winners, prizes)}"
            f"{table(double_winners, [2 * p for p in prizes])}"
            f"<p>סכום הפרס הראשון בהגרלה זו עמד על {first} ₪ ובדאבל לוטו עד 80,000,000 ₪</p>"
            f"<p>סכום הפרס השני בהגרלה זו עמד על {second} ₪</p></body></html>")


def _result(first, *, jackpot_winners=0, double_jackpot_winners=0, second=750_000, second_winners=0, tier8=10):
    return Result(3900, "01/01/2026", first, second, (jackpot_winners, second_winners, 0, 0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0, 0, 0, tier8), (double_jackpot_winners, 0, 0, 0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0, 0, 0, 2 * tier8))


def test_parse_result_reads_both_tables_and_ignores_scripts():
    result = parse_result(_page())
    assert (result.draw_id, result.date, result.first_prize, result.second_prize) == (3929, "26/05/2026", 40_000_000,
                                                                                      750_000)
    assert result.winners == (0, 1, 62, 345, 2392, 13038, 31644, 172773)
    assert result.double_winners[7] == 155349 and result.double_prizes[7] == 20
    assert not result.jackpot_won and result.second_won and not result.was_bonus


def test_parse_result_rejects_incomplete_or_unexpected_pages():
    with pytest.raises(ParseError):
        parse_result("<html><body>תוצאות הגרלה מס' 1</body></html>")
    with pytest.raises(ParseError):
        parse_result(_page(prizes=(0, 250000, 4843, 693, 135, 48, 34, 12)))
    with pytest.raises(ParseError):
        parse_result(_page().replace("5 + חזק", "2 + חזק"))


def test_latest_draw_id_takes_the_highest_reference():
    assert latest_draw_id('<a href="currentlotto.aspx?lotteryId=3981">x</a><p>תוצאות הגרלה מס\' 3982</p>') == 3982
    with pytest.raises(ParseError):
        latest_draw_id("<html></html>")


def test_bonus_form_only_after_an_unclaimed_cap_or_an_announcement():
    rolled = plan(_result(40_000_000))
    assert rolled.form == "bonus" and rolled.rows == BONUS_FORM and rolled.cost == 42
    assert rolled.next_first_prize == 40_000_000
    for last in (_result(40_000_000, jackpot_winners=1), _result(40_000_000, double_jackpot_winners=1),
                 _result(35_000_000), _result(5_000_000, tier8=15)):
        assert plan(last).form == "ordinary" and plan(last).rows == ORDINARY_FORM and plan(last).cost == 6
    assert plan(_result(12_000_000), announced_bonus=True).form == "bonus"
    with pytest.raises(ValueError):
        plan(_result(41_000_000))


def test_next_prizes_follow_the_reset_and_rollover_rules():
    assert plan(_result(22_000_000, jackpot_winners=1)).next_first_prize == 5_000_000
    assert plan(_result(22_000_000)).next_first_prize is None
    assert plan(_result(8_000_000, second=2_250_000)).next_second_prize == 3_000_000
    assert plan(_result(8_000_000, second=2_250_000, second_winners=2)).next_second_prize == 750_000


def test_forms_have_the_documented_exact_odds_and_optimal_strong_numbers():
    for form, any_prize, with_strong in ((ORDINARY_FORM, 193_724, 194_124), (BONUS_FORM, 1_185_924, 1_356_068)):
        rows = [list(row) for row in form]
        odds = portfolio_odds(rows)
        assert odds["distinct_main_tickets"] == len(rows) and odds["jackpot_covered_outcomes"] == len(rows)
        assert sum(odds["best_match_counts"][3:]) == any_prize
        assert strong_prize_odds(rows)["covered_full_outcomes"] == with_strong
        assert optimize_strong_numbers(rows)[1]["covered_full_outcomes_after"] == with_strong
