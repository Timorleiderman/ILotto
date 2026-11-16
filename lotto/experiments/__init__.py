"""The catalog: every experiment in the project, in the order a reader meets them.

The order tells the story, basics first: the game and its data, whether there
is anything to predict, simple strategies, neural networks, probability
forecasts, then the algorithms that work on the ticket instead of the draw.

Adding an experiment means adding one entry here and one module in this
package that exposes ``main()``. The ``lotto`` command, the generated index
page and the completeness test all read this list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    id: str  #: what you type: ``lotto run coverage``
    chapter: str  #: where it sits in the story
    question: str  #: one sentence ending in a question mark
    module: str  #: module in this package exposing ``main()``
    page: str  #: write-up, relative to docs/
    quick: str = ""  #: arguments for a fast first run, shown by ``lotto list``
    needs: str = ""  #: optional requirement, in plain words


CATALOG: tuple[Experiment, ...] = (
    Experiment("report", "1. The benchmark", "Does any strategy beat a random ticket, and is the draw random?",
               "benchmark_report", "results.md", quick="--quick --n-test 100 --null-trials 500"),
    Experiment("calibration", "2. Forecasting", "Do smoothed frequency forecasts beat the uniform forecast?",
               "calibration", "reference/research-log/prediction-probability.md",
               quick="--out results/calibration.json --portfolio-seeds 2"),
    Experiment("joint-mixture", "2. Forecasting", "Can a learner that starts uncertain find a bias in the draws?",
               "joint_mixture", "algorithm-research.md"),
    Experiment("thousand-draws", "2. Forecasting", "Over 1,000 draws, do forecast-ranked tickets win more often?",
               "thousand_draws", "historical-strikes.md"),
    Experiment("synthetic-training", "2. Forecasting",
               "Does training on a million simulated draws transfer to the real ones?",
               "synthetic_training", "synthetic-learning.md"),
    Experiment("predict", "3. Algorithms", "What does every model say about the next draw, as a form to fill?",
               "predict", "levers/predict.md", quick="--quick --offline"),
    Experiment("coverage", "3. Algorithms", "How much prize chance can a fixed number of rows buy?",
               "coverage", "approaches/ticket-generators.md", quick="--count 10 --seed 0 --out results/coverage.json"),
    Experiment("coverage-benchmark", "3. Algorithms", "How much does exact refinement add, and how fast is it?",
               "coverage_benchmark", "approaches/ticket-generators.md", quick="--seeds 3 --out results/coverage-benchmark.json"),
    Experiment("coverage-search", "3. Algorithms", "Can basin hopping escape a local optimum of the coverage design?",
               "coverage_search", "algorithm-research.md"),
    Experiment("strong-numbers", "3. Algorithms", "Which strong number should each row carry?",
               "strong_numbers", "ticket-chances.md"),
    Experiment("plan", "3. Algorithms", "Which form should I play on the next draw?",
               "draw_planner", "levers/draw-planner.md", needs="network access to pais.co.il"),
    Experiment("seed-search", "4. Side quest", "If draws came from a seeded generator, could the seed be found?",
               "seed_search", "side-quests/seed-search.md", quick="--mode toy"),
    Experiment("seed-search-job", "4. Side quest", "How many seeds can one machine test in a fixed time?",
               "seed_search_job", "side-quests/seed-search.md", needs="the Rust engine (cargo build --release)"),
    Experiment("seed-search-remote", "4. Side quest", "How is a search shared with a remote GPU without overlap?",
               "seed_search_remote", "side-quests/seed-search.md", needs="an NVIDIA machine reachable over SSH"),
    Experiment("seed-search-cuda-benchmark", "4. Side quest", "How fast is the CUDA engine, and is it correct?",
               "seed_search_cuda_benchmark", "side-quests/seed-search.md", needs="a built CUDA worker"),
)


def find(experiment_id: str) -> Experiment:
    """The catalog entry with this id; ValueError lists the valid ids."""
    for experiment in CATALOG:
        if experiment.id == experiment_id:
            return experiment
    raise ValueError(f"Unknown experiment {experiment_id!r}; choose from: {', '.join(e.id for e in CATALOG)}")
