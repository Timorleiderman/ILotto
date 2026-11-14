# Strong-number assignment

!!! abstract "In one minute"

    **The question.** Ten rows are fixed. Which strong number should each one carry?

    **The answer.** Not one each, spread evenly. An exact dynamic program over every way of
    grouping the rows raises the chance of three-or-more plus the strong number from 5.926%
    to 5.957%, and proves no assignment does better for those rows.

    **Why it matters.** It is a small gain with a proof, found by exact search, not by
    simulation.

## Run it

```bash
uv run lotto run strong-numbers
```

About three minutes. It repeats the 15 coverage searches and the exact assignment, and writes
`results/strong-numbers/`.

## The dated report

Experiment date: 2026-11-07. Ten distinct tickets per draw, using the portfolios from the [previous algorithm research](algorithm-research.md). All probabilities assume independent uniform fair draws.

**The selected portfolio increases the chance of at least one ticket matching three or more main numbers and the strong number from 5.92632630% to 5.95705113%. Its overall chance of any prize remains 38.59820955%.** Broader main-number searches improved weaker starting portfolios but did not beat the previous best.

[Download all experiment data and checksums](data/ticket-chances-2026-11-07.zip). The bundle contains the input portfolios, protocol, all 15 runs and 900 proposal audits, strong-assignment results, and source snapshots.

## Outcomes for the selected portfolio

The game uses six distinct main numbers from 1–37 and one strong number from 1–7. [Official game guide](https://www.pais.co.il/info/lotto-how-to-play.aspx). The experiment uses the existing [eight prize tiers](start/the-game.md#the-prizes); any prize requires at least three main matches.

| Event on at least one ticket | Previous best portfolio | Selected portfolio | Change in percentage points |
| --- | --- | --- | --- |
| Any prize, at least three main matches | 38.59820955% | 38.59820955% | 0 |
| At least three main matches and the strong number | 5.92632630% | 5.95705113% | +0.03072482 |
| At least four main matches | 3.08071632% | 3.08071632% | 0 |
| Jackpot, six main matches and strong | 0.00006145% | 0.00006145% | 0 |

The secondary gain covers **5,000 additional complete draw outcomes**, from 964,420 to 969,420 out of 16,273,488. It is a **0.5184% relative increase** in this event, or **0.03072482 percentage points**. It does not establish increased expected payout or net profit. Each individual ticket retains the same fair-draw odds; rearranging strong numbers reduces overlap between tickets that win together.

## Main number search comparison

Each of the five saved portfolios was tested with 2, 4, and 8 random number changes per proposal. Every run used 60 proposals, two exact local-refinement passes per proposal, temperatures decreasing from 1,000 to 10 covered outcomes, and seed 70107 plus the original portfolio seed. Every run starts from its saved portfolio, retains its best result, and uses the same ten-ticket budget. Settings were fixed before this experiment ran.

| Changes per proposal | Starting seed | Any prize before | Any prize after | Search seconds |
| --- | --- | --- | --- | --- |
| 2 | 0 | 38.59820955% | 38.59820955% | 9.40 |
| 2 | 1 | 38.59820955% | 38.59820955% | 9.05 |
| 2 | 2 | 38.58762793% | 38.59820955% | 9.11 |
| 2 | 3 | 38.58762793% | 38.59820955% | 9.13 |
| 2 | 4 | 38.58762793% | 38.59820955% | 9.13 |
| 4 | 0 | 38.59820955% | 38.59820955% | 9.19 |
| 4 | 1 | 38.59820955% | 38.59820955% | 9.14 |
| 4 | 2 | 38.58762793% | 38.59820955% | 9.15 |
| 4 | 3 | 38.58762793% | 38.59820955% | 9.22 |
| 4 | 4 | 38.58762793% | 38.59820955% | 9.17 |
| 8 | 0 | 38.59820955% | 38.59820955% | 9.45 |
| 8 | 1 | 38.59820955% | 38.59820955% | 9.45 |
| 8 | 2 | 38.58762793% | 38.58762793% | 9.48 |
| 8 | 3 | 38.58762793% | 38.59820955% | 9.45 |
| 8 | 4 | 38.58762793% | 38.59820955% | 9.43 |

For both the two-change and four-change variants, mean coverage increased from **38.59186058% to 38.59820955%**, a gain of **0.00634898 percentage points**. Each improved three of five starting portfolios. The eight-change variant improved two of five, ending at a mean of **38.59609323%**. None exceeded the existing best coverage of 897,325 main outcomes. These are repeated tests of the same five starting portfolios, not fifteen independent starting designs.

The comparison fixes proposal counts, not wall time. Reported seconds include the main search and exclude independent verification and strong-number optimization. Tests overlapped part of the experiment, so timings are descriptive rather than isolated performance measurements.

## Exact strong number assignment

For each fixed main-number portfolio, an exhaustive evaluator counts which rows win on every possible main draw. A subset transform gives the union coverage for every group of rows. A dynamic program then partitions the rows among seven strong-number labels to maximize the sum of these seven coverages. The strong outcomes are mutually exclusive, so their sum divided by 16,273,488 is the exact secondary-event probability.

The partition optimizer considers every grouping, including unbalanced groups. Splitting a group cannot reduce this objective, so it is sufficient to use seven nonempty groups for ten rows. This proves optimality of the strong assignment **for the fixed main tickets**; it does not prove globally optimal main-number selection. The implementation supports up to fourteen distinct main tickets.

| Starting seed | Three or more main matches and strong before | After changing only strong assignments | Additional full outcomes |
| --- | --- | --- | --- |
| 0 | 5.92632630% | 5.95705113% | 5,000 |
| 1 | 5.92632630% | 5.95705113% | 5,000 |
| 2 | 5.94168871% | 5.95705113% | 2,500 |
| 3 | 5.92632630% | 5.95705113% | 5,000 |
| 4 | 5.94168871% | 5.95705113% | 2,500 |

In the selected portfolio, each reused strong number is assigned to two rows with disjoint main numbers. This reduces their overlapping winning outcomes. Main-number rows, their count, and jackpot coverage are unchanged.

## Why four main matches already reach the limit

One ticket covers C(6,4)×C(31,2) + C(6,5)×C(31,1) + 1 = 7,162 main draws with at least four matches. The union of ten such events cannot exceed 71,620 outcomes, or **3.08071632%**. Our selected and baseline portfolios attain this bound.

Every pair of their main tickets shares at most one number. A six-number draw therefore cannot match four numbers on two of these tickets at once: that would require at least seven distinct drawn numbers. Their four-match events are disjoint. Improving this particular probability beyond the bound would require changing the ticket budget or the draw-probability assumptions.

For any-prize coverage, the looser ten-ticket union bound is 41.75097557%. The selected portfolio does not attain it; this experiment establishes no global optimum for that objective.

## Selected ten row portfolio

This is a coverage design, not a prediction of a particular draw. A consistent permutation of number labels leaves its fair-draw probabilities unchanged.

| Row | Main numbers | Strong |
| --- | --- | --- |
| 1 | 11 · 12 · 15 · 22 · 31 · 35 | 1 |
| 2 | 4 · 6 · 7 · 8 · 23 · 30 | 2 |
| 3 | 1 · 18 · 26 · 29 · 34 · 35 | 3 |
| 4 | 1 · 2 · 3 · 11 · 19 · 25 | 4 |
| 5 | 9 · 12 · 18 · 25 · 27 · 28 | 5 |
| 6 | 2 · 14 · 15 · 26 · 27 · 36 | 6 |
| 7 | 5 · 7 · 13 · 16 · 17 · 20 | 3 |
| 8 | 3 · 9 · 14 · 22 · 29 · 33 | 7 |
| 9 | 10 · 17 · 21 · 24 · 32 · 37 | 1 |
| 10 | 19 · 28 · 31 · 33 · 34 · 36 | 2 |

## Reproduction and validation

Run from the repository root:

```bash
uv run lotto run strong-numbers
uv run pytest -q
uv run ruff check .
uv run mkdocs build --strict
```

The protocol is `experiments/ticket_chances_protocol_2026_11_07.json`; complete results are `experiments/ticket_chances_2026_11_07.json`. The result records input, protocol, and implementation checksums, every proposed main-number change, every retained portfolio, all subset coverage counts used in strong assignment, and environment details. Both experiment inputs and older results are preserved.

The independent reference evaluator reproduced every final main-number histogram, every baseline strong-event count before and after optimization, and both selected-portfolio event counts. Small-universe tests exhaustively compared every subset union and every strong-label assignment against the compiled kernels and dynamic program. Additional tests cover duplicate tickets, invalid inputs, determinism, and preservation of main-number and jackpot odds. **208 tests passed, two skipped** because live CUDA and the host C++ probe were unavailable. Ruff and strict documentation validation passed.
