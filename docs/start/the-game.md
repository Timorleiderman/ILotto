# The game in five minutes

Everything in this project is about one game. This page is all you need to know about it.

## The rules

A draw picks **six different numbers from 1 to 37** and, from a separate machine, **one strong
number from 1 to 7**. A row on your form is the same thing: six numbers and a strong number.

| Fact | Value |
|---|---|
| Price of a row | 3 NIS (a Double row costs 6 NIS and pays every prize twice) |
| Rows per form | 2 to 14 |
| Draw days | Tuesday and Saturday evening, with an occasional Thursday series |
| How balls are drawn | physical ball machines, one for the main numbers and one for the strong number |
| Possible main draws | C(37, 6) = 2,324,784 |
| Possible complete draws | 2,324,784 × 7 = 16,273,488 |

## The prizes

A row wins if it matches at least three main numbers. Matching the strong number as well moves
it one tier up. These probabilities are exact, computed by
`lotto.tickets.coverage.single_ticket_prize_odds()`:

| Tier | Match | Ways | Chance for one row |
|---|---|---:|---:|
| 1st | 6 + strong | 1 | 1 in 16,273,488 |
| 2nd | 6 | 6 | 1 in 2,712,248 |
| 3rd | 5 + strong | 186 | 1 in 87,492 |
| 4th | 5 | 1,116 | 1 in 14,582 |
| 5th | 4 + strong | 6,975 | 1 in 2,333 |
| 6th | 4 | 41,850 | 1 in 389 |
| 7th | 3 + strong | 89,900 | 1 in 181 |
| 8th | 3 | 539,400 | 1 in 30 |
| | any prize | 679,434 | **4.18%** |

The smallest prize is fixed at 10 NIS. The others are shares of a pool, so they vary from draw to
draw.

## Two numbers to remember

**0.973.** A row has six numbers and each is drawn with probability 6/37, so a random row matches
6 × 6/37 = 0.973 numbers on average. Every prediction strategy in this project is measured
against that.

**4.18%.** The chance that one row wins anything. Every ticket design is measured against that.

## Bonus draws

When a 40 million NIS first prize is not won, the next draw is a *bonus draw*: every prize below
the jackpot pays 1.5 times as much (15 NIS instead of 10 for three numbers), while the odds stay
the same. The [draw planner](../levers/draw-planner.md) uses this.

Next: [run it yourself →](run.md)
