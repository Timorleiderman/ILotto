"""The expected return of one Lotto row, after prize sharing and Israeli tax.

Every number here is a fair-draw expectation. Nothing depends on which numbers
a row holds: the rest of the project shows the draw cannot be predicted, so a
row's value comes only from the prizes on offer, how many other rows would
share them, and the tax on what is left.

How a row is paid (from the prize tables in ``input/lotto_prizes.csv``):

* **First prize** (6 + strong) and **second prize** (6 without strong) are
  announced pools, split equally between the winning rows. A Double row is
  counted once in the split and paid twice its share: in draw 3300 a 25M first
  prize had one regular and two Double winners, who were paid 8.33M and
  16.67M each.
* **Lower tiers** (5+strong down to 3) are paid per winning row, and a Double
  row is paid exactly twice the regular amount.
* A regular row costs 3 NIS and a Double row 6 NIS.

Other winners are modelled as Poisson with mean (rows sold) x (chance of the
tier), the standard approximation for many independent rows. Rows sold are
estimated from the lower-tier winner counts, which follow the exact tier
probabilities whatever numbers were drawn.

Tax follows the schedule Mifal HaPais publishes for draws from 2026
(``TAX_*``), applied to each prize separately; the 5% surtax is applied to
the part of a prize above its threshold, as if the prize were the winner's
only taxable income that year. Past draws are valued with the same 2026
schedule and price.
"""

from __future__ import annotations

from math import comb, exp, isfinite, log, sqrt

import numpy as np

from lotto.tickets.draw_plan import CAP, Result

N_NUMBERS, N_DRAWN, N_STRONG = 37, 6, 7
COMBINATIONS = comb(N_NUMBERS, N_DRAWN)  #: 2,324,784
FULL_COVERAGE_ROWS = COMBINATIONS * N_STRONG  #: 16,273,488: every row that can be filled in
PRICES = {"regular": 3, "double": 6}
MULTIPLIER = {"regular": 1, "double": 2}
KINDS = tuple(PRICES)

#: Tax on prizes, as published by Mifal HaPais for draws from 2026-01-01 (pais.co.il/info/tax.aspx).
TAX_TABLE_YEAR = 2026
TAX_EXEMPT = 33_840
TAX_FULL_FROM = 67_680
TAX_RATE = 0.35
SURTAX_FROM = 721_560
SURTAX_RATE = 0.05

TIER_LABELS = ("6 + strong", "6", "5 + strong", "5", "4 + strong", "4", "3 + strong", "3")


def _ways(matches: int, strong: bool) -> int:
    """Rows, out of all 16,273,488, that land in this tier for any one draw."""
    return comb(N_DRAWN, matches) * comb(N_NUMBERS - N_DRAWN, N_DRAWN - matches) * (1 if strong else N_STRONG - 1)


#: How many of the possible rows fall in each tier, in ``TIER_LABELS`` order.
TIER_WAYS = tuple(_ways(m, s) for m in (6, 5, 4, 3) for s in (True, False))
#: The exact chance that one row lands in each tier.
TIER_P = tuple(w / FULL_COVERAGE_ROWS for w in TIER_WAYS)
#: Tiers used to count rows sold: thousands of winners per draw, so the estimate is tight.
_COUNTING_TIERS = slice(4, 8)


def tax(prize: float) -> float:
    """Tax withheld on one prize under the 2026 schedule.

    Up to 33,840 is exempt. Between 33,840 and 67,680 the exemption shrinks by
    2 NIS per NIS above 33,840, so 35% falls on twice the excess. From 67,680
    35% falls on the whole prize, plus 5% on the part above 721,560.
    """
    if prize <= TAX_EXEMPT:
        return 0.0
    if prize < TAX_FULL_FROM:
        return TAX_RATE * 2 * (prize - TAX_EXEMPT)
    return TAX_RATE * prize + SURTAX_RATE * max(0.0, prize - SURTAX_FROM)


def net(prize: float, taxed: bool = True) -> float:
    """What the winner keeps."""
    return prize - tax(prize) if taxed else float(prize)


def rows_sold(result: Result) -> tuple[float, float]:
    """Estimated (regular, Double) rows in a draw, from its lower-tier winner counts."""
    p = sum(TIER_P[_COUNTING_TIERS])
    return sum(result.winners[_COUNTING_TIERS]) / p, sum(result.double_winners[_COUNTING_TIERS]) / p


def lower_amounts(result: Result) -> tuple[float, ...] | None:
    """The regular-row amount of each lower tier (5+strong to 3) in a draw.

    A tier no regular row won is read from the Double table at half. None if a
    tier had no winner at all, which has not happened since 2018.
    """
    amounts = []
    for i in range(2, 8):
        if result.winners[i] > 0:
            amounts.append(float(result.prizes[i]))
        elif result.double_winners[i] > 0:
            amounts.append(result.double_prizes[i] / 2)
        else:
            return None
    return tuple(amounts)


def _poisson_expectation(fn, mean: float) -> float:
    """E[fn(K)] for K ~ Poisson(mean), summed until the tail is negligible."""
    top = int(mean + 12 * sqrt(mean) + 30)
    pmf, total = exp(-mean), 0.0
    for k in range(top + 1):
        total += pmf * fn(k)
        pmf *= mean / (k + 1)
    return total


def shared_prize_value(pool: float, p: float, rows: float, multiplier: int = 1, taxed: bool = True) -> float:
    """Expected net value, per row, of a pool split equally between its winners.

    ``rows`` is how many other rows are in the draw; each wins with chance ``p``.
    """
    return p * _poisson_expectation(lambda k: net(multiplier * pool / (1 + k), taxed), rows * p)


def row_value(first: float, second: float, lower: tuple[float, ...], rows: float, kind: str = "regular",
              taxed: bool = True) -> dict:
    """Expected net return of one row, by tier group, and that return per shekel spent."""
    m = MULTIPLIER[kind]
    jackpot = shared_prize_value(first, TIER_P[0], rows, m, taxed)
    runner_up = shared_prize_value(second, TIER_P[1], rows, m, taxed)
    small = sum(p * net(m * a, taxed) for p, a in zip(TIER_P[2:], lower))
    total = jackpot + runner_up + small
    return {"kind": kind, "price": PRICES[kind], "first_prize": jackpot, "second_prize": runner_up,
            "lower_tiers": small, "total": total, "return_per_shekel": total / PRICES[kind]}


# ------------------------------------------------------------------ history
def history(results: list[Result], taxed: bool = True) -> list[dict]:
    """Each past draw's value per row, in hindsight.

    "In hindsight" because rows sold and the lower-tier amounts are read from
    that draw's own prize table, published after the draw. Rows sold were a
    fact before the draw (sales close first); the lower-tier amounts are
    pari-mutuel and vary a little with how many rows won.
    """
    out = []
    for r in results:
        lower = lower_amounts(r)
        if lower is None:
            continue
        regular, double = rows_sold(r)
        rows = regular + double
        values = {kind: row_value(r.first_prize, r.second_prize, lower, rows, kind, taxed) for kind in KINDS}
        sales = PRICES["regular"] * regular + PRICES["double"] * double
        paid = sum(w * a for w, a in zip(r.winners, r.prizes)) + sum(w * a for w, a in zip(r.double_winners,
                                                                                          r.double_prizes))
        out.append({
            "draw_id": r.draw_id, "date": r.date, "first_prize": r.first_prize, "second_prize": r.second_prize,
            "bonus": r.was_bonus, "first_won": r.jackpot_won, "rows_sold": rows, "double_share": double / rows,
            "sales": sales, "paid_out": paid, "paid_share": paid / sales,
            "regular": values["regular"], "double": values["double"],
            "positive": values["regular"]["return_per_shekel"] >= 1 or values["double"]["return_per_shekel"] >= 1,
        })
    return out


def summarise(rows: list[dict]) -> dict:
    """The headline numbers of a history: how often, if ever, a row was worth its price."""
    if not rows:
        return {"draws": 0}
    ratio = np.array([r["regular"]["return_per_shekel"] for r in rows])
    best = rows[int(np.argmax(ratio))]
    return {"draws": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
            "positive_draws": sum(r["positive"] for r in rows),
            "mean_return": float(ratio.mean()), "median_return": float(np.median(ratio)),
            "best": {"draw_id": best["draw_id"], "date": best["date"], "first_prize": best["first_prize"],
                     "return_per_shekel": best["regular"]["return_per_shekel"], "bonus": best["bonus"]},
            "mean_paid_share": float(np.mean([r["paid_share"] for r in rows]))}


# ---------------------------------------------------------------- next draw
def sales_model(results: list[Result], window: int = 300) -> dict:
    """Rows sold as a function of the announced first prize: log rows = a + b log prize (+ c bonus).

    Fitted by least squares on the last ``window`` draws. Bigger jackpots sell
    more rows, which means more rows to share the jackpot with.
    """
    recent = results[-window:]
    if len(recent) < 20:
        raise ValueError("At least 20 prize tables are needed to estimate sales")
    y = np.log([sum(rows_sold(r)) for r in recent])
    x = np.log([r.first_prize for r in recent])
    bonus = np.array([r.was_bonus for r in recent], dtype=float)
    columns = [np.ones_like(x), x] + ([bonus] if 0 < bonus.sum() < len(bonus) else [])
    coef, *_ = np.linalg.lstsq(np.column_stack(columns), y, rcond=None)
    resid = y - np.column_stack(columns) @ coef
    return {"intercept": float(coef[0]), "slope": float(coef[1]), "bonus": float(coef[2]) if len(coef) > 2 else 0.0,
            "draws": len(recent), "residual_sd": float(resid.std(ddof=len(coef)))}


def predicted_rows(model: dict, first_prize: float, bonus: bool = False) -> float:
    return float(exp(model["intercept"] + model["slope"] * log(first_prize) + (model["bonus"] if bonus else 0.0)))


def typical_lower_amounts(results: list[Result], bonus: bool = False, window: int = 100) -> tuple[float, ...]:
    """Median lower-tier amounts over recent draws of the same kind (ordinary or bonus)."""
    same = [lower_amounts(r) for r in results if r.was_bonus == bonus]
    same = [a for a in same if a is not None][-window:]
    if len(same) < 5:
        same = [a for a in (lower_amounts(r) for r in results) if a is not None][-window:]
        scale = 1.5 if bonus else 1.0  # a bonus draw pays every small tier 1.5x
        return tuple(float(v) * scale for v in np.median(same, axis=0))
    return tuple(float(v) for v in np.median(same, axis=0))


def break_even_first_prize(second: float, lower: tuple[float, ...], rows_for, kind: str = "regular",
                           taxed: bool = True, high: float = 1e11) -> float | None:
    """The smallest first prize at which a row's expected return equals its price.

    ``rows_for(prize)`` gives the rows sold at that prize, so sales may grow
    with the jackpot. None if no prize up to ``high`` gets there.
    """
    def gap(prize):
        return row_value(prize, second, lower, rows_for(prize), kind, taxed)["total"] - PRICES[kind]

    grid = np.geomspace(1e5, high, 300)
    previous = grid[0]
    if gap(previous) >= 0:
        return float(previous)
    for prize in grid[1:]:
        if gap(prize) >= 0:
            lo, hi = previous, prize
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if gap(mid) >= 0 else (mid, hi)
            return float(hi)
        previous = prize
    return None


def full_coverage(first: float, second: float, lower: tuple[float, ...], rows: float, taxed: bool = True) -> dict:
    """Buy every one of the 16,273,488 rows once: what it costs and what it returns.

    Holding every row wins the first prize for certain (shared with any other
    winners), six second prizes and every lower-tier row. The lower tiers are
    valued at their published amounts; in reality 16 million extra rows would
    swell those pari-mutuel tiers' winner counts and pay each row less.
    """
    jackpot = _poisson_expectation(lambda k: net(first / (1 + k), taxed), rows * TIER_P[0])
    six = TIER_WAYS[1]
    runner_up = _poisson_expectation(lambda k: six * net(second / (six + k), taxed), rows * TIER_P[1])
    small = sum(w * net(a, taxed) for w, a in zip(TIER_WAYS[2:], lower))
    cost = FULL_COVERAGE_ROWS * PRICES["regular"]
    total = jackpot + runner_up + small
    return {"rows": FULL_COVERAGE_ROWS, "cost": cost, "first_prize": jackpot, "second_prize": runner_up,
            "lower_tiers": small, "expected_return": total, "expected_net": total - cost,
            "return_per_shekel": total / cost}


def next_draw(results: list[Result], first: float, second: float, *, bonus: bool = False, rows: int = 1,
              kind: str = "regular", taxed: bool = True) -> dict:
    """The value of ``rows`` rows in a draw with these announced prizes."""
    if kind not in PRICES:
        raise ValueError(f"kind must be one of {KINDS}")
    if not (isfinite(first) and isfinite(second)) or first <= 0 or second <= 0:
        raise ValueError("prizes must be positive")
    model = sales_model(results)
    lower = typical_lower_amounts(results, bonus)
    sold = predicted_rows(model, first, bonus)
    value = row_value(first, second, lower, sold, kind, taxed)
    cost = rows * PRICES[kind]
    expected = rows * value["total"]
    return {
        "first_prize": first, "second_prize": second, "bonus": bonus, "kind": kind, "rows": rows, "taxed": taxed,
        "rows_sold": sold, "sales_model": model, "lower_amounts": list(lower),
        "per_row": value, "cost": cost, "expected_return": expected, "expected_net": expected - cost,
        "return_per_shekel": value["return_per_shekel"], "positive": value["return_per_shekel"] >= 1,
        "p_first_prize": min(rows, FULL_COVERAGE_ROWS) / FULL_COVERAGE_ROWS,
        "break_even": {
            "sales_grow": break_even_first_prize(second, lower, lambda j: predicted_rows(model, j, bonus), kind,
                                                 taxed),
            "sales_fixed": break_even_first_prize(second, lower, lambda j: sold, kind, taxed),
            "cap": CAP,
        },
        "full_coverage": full_coverage(first, second, lower, sold, taxed),
    }
