# Is a ticket worth its price?

!!! abstract "The concept, in a minute"

    **The question.** The numbers cannot be predicted, so the only edge left would be
    economic: is there a draw where a row returns more than its 3 NIS price on average?

    **The answer.** Not after tax. Across all 1,005 draws with a published prize table
    (January 2018 to November 2026), a regular row returned a median of 0.47 NIS per NIS
    after prize sharing and tax, and never more than 0.90. Before tax, 31 of those draws
    would have been worth playing, 30 of them with the first prize at the 40M cap.
    The tax on the jackpot is what turns every one into a loss.

    **Why it matters.** It is the one honest version of "when should I play": a
    calculation from published prizes, not a forecast.

Code: `lotto/economics/archive.py` (the prize tables), `lotto/economics/value.py` (the model),
`lotto/experiments/ticket_value.py` (the command). Dashboard: the **Value** screen of
`uv run lotto ui`.

```bash
uv run lotto run value                        # reads the advertised prizes, prices 10 rows
uv run lotto run value --offline --first 40000000 --bonus --rows 14
uv run lotto run value --no-tax               # the same, ignoring tax
```

## The data

`input/lotto_prizes.csv` holds one row per draw from 2982 (30 January 2018), the first
result page pais.co.il still serves: the announced first and second prize, and for the
regular and the Double table the number of winning rows and the amount each was paid.
`lotto report`'s draw check does not touch it; the dashboard's **Check for new draws** and
`lotto run value` append new tables, and never rewrite old ones. Like the draw archive it is
committed, because CI cannot reach pais.co.il.

The next draw's prizes come from the JSON the Lotto page itself reads
(`include/getNextLotteryDate.ashx`). Offline, they are inferred from the last result: 5M after a
win, 40M at the cap, otherwise the usual step.

## How a row is paid

Read off the tables themselves:

| Tier | Paid as | Double row |
|---|---|---|
| 6 + strong | the announced first prize, split equally between winning rows | counted once in the split, paid twice its share |
| 6 | the announced second prize, split the same way | the same |
| 5 + strong to 3 | a per-row amount (3 numbers: 10 NIS, 15 in a bonus draw) | exactly twice the regular amount |

Draw 3300 shows the split: a 25M first prize, one regular and two Double winners, paid 8.33M
and 16.67M each. A regular row costs 3 NIS and a Double row 6 NIS.

## The model

For one row in a draw, the expected value is the sum over tiers of the chance of landing in it
times what the winner keeps:

- **Chances** are exact: of the 16,273,488 possible rows, 1 lands in 6 + strong, 6 in 6, 186 in
  5 + strong, and so on.
- **Rows sold** are estimated from the four lowest tiers' winner counts, which follow the exact
  tier chances whatever numbers are drawn: tens of thousands of winners give an estimate
  within about 1%.
- **Sharing.** The other winners of a pooled prize are Poisson with mean (rows sold) x (chance).
  At 3.5 million rows and a 12M jackpot that leaves a winner about 90% of the pool on average.
- **Tax** follows the schedule Mifal HaPais publishes for 2026: exempt up to 33,840 NIS, a
  partial band up to 67,680, then 35% of the whole prize plus 5% on the part above 721,560.
  It applies per prize; the surtax is computed as if the prize were the winner's only income.
- **The next draw** uses a sales model fitted on the last 300 draws (log rows sold against log
  first prize, plus a bonus-draw term: bigger jackpots sell more rows, which then share it) and
  median small-prize amounts from recent draws of the same kind.

The **break-even first prize** is where a row's expected value equals its price, with sales
rising along the fitted curve. For an ordinary draw it is about 72M NIS, far above the 40M cap.

**Full coverage**, every one of the 16,273,488 rows, costs 48.8M NIS and is shown only for
reference. It wins the first prize for certain, but the expected loss is still about 28M NIS
at a 12M jackpot, and larger in reality: 16 million extra rows would multiply the winners in the
shared small tiers and cut each payment.

## What this does not do

- **It does not change the odds.** Nothing here touches which numbers come up.
- **No popular-number logic.** Picking numbers other players avoid could only shrink how many
  people share a jackpot; it never changes the chance of winning. The project removed
  popularity and birthday scoring deliberately and does not model it.
- **Hindsight is labelled.** A past draw is valued with the rows sold and small-prize amounts
  from its own table, which are published after the draw.
- **Today's prices and tax are applied to every past draw.** Older draws had different tax
  thresholds; the conclusion does not depend on the difference.
