# Glossary

The terms this project uses, in the order you meet them.

**Row.** Six main numbers and one strong number. What you fill in on a form.

**Draw.** One result: six main numbers and one strong number.

**Baseline.** What you get knowing nothing. For matched numbers it is 6 × 6/37 = 0.973 per draw;
for log loss it is ln 37 = 3.611. A strategy is only interesting relative to its baseline.

**Hypergeometric distribution.** The exact distribution of how many of your six numbers are
drawn when six of 37 are picked without replacement. It is why the baseline is a formula, not an
estimate.

**Leakage.** Information about the answer reaching the model before it predicts. The three data
bugs on the [measurement page](../evaluation.md) are all leaks.

**Walk-forward backtest.** Replaying history one draw at a time: for each draw, the strategy
sees only earlier draws, predicts, and is scored. Implemented in `lotto/core/backtest.py`.

**Null distribution.** How a statistic behaves when nothing is going on. A result is surprising
only if it lands far out in its null distribution.

**p-value.** The probability, under the null, of a result at least as extreme as the one
observed. It is not the probability that a strategy works.

**Multiple testing.** Test twenty strategies and the best one looks good by luck alone.

**Holm–Bonferroni correction.** A rule that tightens the p-value threshold according to how many
tests were run, so the chance of even one false alarm stays below 5%.

**Log loss.** The negative log of the probability a forecast gave to what actually happened.
Lower is better. It rewards honest probabilities, which makes it a *proper scoring rule*.

**Calibration.** A forecast is calibrated if things it calls 20% likely happen 20% of the time.

**Set prediction.** Predicting which numbers are drawn, regardless of order. The opposite is
predicting "ball 1, ball 2, …" in sorted order, which leaks.

**Conditional Bernoulli model.** A probability model over sets of exactly six numbers. Used by
the [date-conditioned model](../approaches/date-conditioned.md).

**DKRR (date-keyed reconstruction rate).** How many numbers a date-conditioned model gets right
on dates it was trained on. It measures memory, not skill.

**Coverage.** The probability that at least one row of a portfolio wins a prize.

**Covering design.** A set of rows chosen so that as many draws as possible are covered by at
least one of them.

**Basin hopping.** A search that makes a random jump, improves locally, and keeps the result if
it is better (or occasionally if it is worse), to escape local optima.

**Seed.** The starting value of a pseudo-random generator. The real lottery has none; the
[seed search](../side-quests/seed-search.md) studies an artificial one.
