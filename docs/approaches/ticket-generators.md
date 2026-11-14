# Coverage at a fixed budget

!!! abstract "The concept, in a minute"

    **The question.** If you buy ten rows, which ten?

    **The answer.** Rows that overlap as little as possible. One row wins a prize in
    4.18% of fair draws; ten designed rows win in 38.60%, counted exactly over all
    2,324,784 possible main draws. No forecast is involved.

    **Why it matters.** It is the one lever in this project that measurably works,
    and it is a clean algorithms problem: a covering design, solved by greedy
    selection, exact local search and basin hopping.

Code: `lotto/tickets/coverage.py` (the algorithms), `lotto/tickets/_kernels.py` (the
compiled exact counters), `lotto/experiments/coverage.py` (the command).

## Run it

```bash
uv run lotto run coverage --count 10 --seed 0 --out results/coverage.json
```

About a second after the first run, which also compiles the counters. The output lists
the ten rows and their exact odds.

<div class="role-key">
  <span class="k-data"><i></i>data</span>
  <span class="k-learn"><i></i>learned / scored</span>
  <span class="k-op"><i></i>fixed operation</span>
  <span class="k-out"><i></i>output</span>
</div>

## Exact coverage at a fixed ticket count

The eighth prize tier requires three main matches without the strong number, so
the event “at least one prize” is exactly “at least one ticket matches three or
more main numbers.” See [the official prize table, section 21](https://www.pais.co.il/download/files/Lotto-8306.pdf).

The optimizer greedily covers synthetic fair draws, then enumerates all
2,324,784 possible main draws to evaluate the selected portfolio. Exhaustive
evaluation handles overlap between tickets without assuming independent wins.
It now follows that search with up to two passes of exact one-number exchanges.
Each accepted exchange must strictly increase three-or-more-main-match coverage.
Other prize categories are not the optimization objective and can move differently.

In the 2026-11-06 experiment, five portfolios per generator, each with ten rows,
gave these averages:

| Generator | P(at least one prize) |
|---|---:|
| Random | 34.56% |
| Greedy coverage | 38.40% |
| Greedy coverage + exact refinement | 38.58% |

These are exact probabilities for the tested portfolios, averaged over five
seeds. The search is a heuristic, not a proof of optimality. Every portfolio had
ten distinct full combinations and the same jackpot probability:
10 / 16,273,488, approximately one in 1,627,349. A prize may not cover all ticket costs.

```bash
uv run lotto run coverage --count 10 --seed 0 --out results/coverage.json
```

The second iteration compared the initial and refined portfolios for 20 fixed
seeds (0–19), with identical ten-row budgets and initial search settings:

| Method | Mean any-prize probability | Range across 20 portfolios |
|---|---:|---:|
| Initial greedy search | 38.354% | 38.196–38.542% |
| After two exact refinement passes | **38.578%** | **38.563–38.585%** |

All 20 improved. This paired gain is **0.224 percentage points**, not a
prediction-accuracy gain or a proof of a globally optimal portfolio. These
20-seed averages should not be confused with the earlier five-seed comparison.

The JSON includes tickets, configuration, exact match counts, odds, and a swap
audit. `--refine-passes 0` disables refinement. The compiled evaluator reuses
about 31 MiB of immutable outcome arrays; its first call includes JIT cache
loading or compilation. Oversized simulated searches are rejected before
allocation using a conservative working-memory estimate.

The dated [research report](../reference/research-log/prediction-probability.md) and
the [second review](../reference/research-log/coverage-review.md) document the methods
and measurements in full.

## Limits

!!! note "What coverage does not change"

    Every complete ticket has the same jackpot probability, 1 / 16,273,488, and ten
    distinct tickets have exactly ten times that. Coverage raises the chance of *at
    least one* prize, mostly the smallest one; it does not make a portfolio profitable.

---

Next: [which strong number each row should carry →](../ticket-chances.md)
