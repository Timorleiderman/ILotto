# Algorithms for the ticket

The draw cannot be steered. The ticket can. This part of the project stops asking *which
numbers will come up* and asks a question that has an exact answer: **given a fixed budget,
which rows should I buy?**

| Lever | What it changes | Measured effect |
|---|---|---|
| [Coverage](../approaches/ticket-generators.md) | which rows, so they overlap as little as possible | ten rows win a prize in 38.60% of draws; fourteen in 51.01% |
| [Strong-number assignment](../ticket-chances.md) | which strong number each row carries | 3+ and strong: 5.926% → 5.957% for ten fixed rows, proven optimal |
| [Draw planner](draw-planner.md) | which draw to play | bonus draws pay small prizes 1.5× at the same odds |
| [Ticket value](ticket-value.md) | whether to play at all | after tax, no draw since 2018 returned its price; best 0.90 NIS per NIS |
| [Every model's form](predict.md) | which numbers fill the form | each model's forecast, arranged in the 10-row design; same 38.60% |

These are algorithms problems, not prediction problems, and each uses a different technique:

- **Coverage** is a covering design. Greedy selection finds a good start, exact one-number
  exchanges improve it, and basin hopping escapes local optima. Every candidate is scored by
  counting all 2,324,784 possible draws with bit masks.
- **Strong-number assignment** is an exact dynamic program over subsets of rows.
- **The planner** is a rule read off 583 official prize tables.

None of them changes the jackpot odds of a row, and none makes a portfolio profitable. They
change how often *something* is won for the same money.
