# ILotto - Israeli Lottery Analysis

A machine learning project for analyzing Israeli Lotto patterns. Built as a learning exercise to explore different neural network architectures, with proper statistical analysis and evaluation metrics.

> **Note:** Lottery numbers are random by design. No model can predict them. This project demonstrates ML concepts, not a winning strategy.

**📊 [Read the documentation site →](https://timorleiderman.github.io/ILotto/)** — rebuilt
automatically after every draw. Diagrams and explanations for every approach in this repo,
the randomness tests, 13 strategies benchmarked walk-forward against exact nulls, and
next-draw picks.

| | |
|---|---|
| [How it is measured](https://timorleiderman.github.io/ILotto/evaluation/) | the walk-forward harness and the data bugs it exposed |
| [The approaches](https://timorleiderman.github.io/ILotto/approaches/) | diagrammed: statistical rules, neural set prediction, the four legacy models, ticket generators |
| [Results](https://timorleiderman.github.io/ILotto/results/) | the current scoreboard, regenerated each run |
| [Benchmark report](https://timorleiderman.github.io/ILotto/benchmark/) | the full standalone report with charts |

## Features

- **4 Neural Network Architectures** - Seq2Seq, Multi-Output, Transformer, Set Prediction
- **Hybrid Ticket Generator** - Combines model predictions with smart scoring
- **Statistical Analysis** - Chi-square tests, runs test, serial correlation
- **Smart Ticket Generator** - Avoids popular combinations to maximize payout if winning
- **Interactive Visualization** - Netron integration for model architecture exploration
- **Comprehensive Metrics** - Proper evaluation against random baseline
- **Leak-Free Benchmark** - Walk-forward harness with Holm–Bonferroni correction, published to GitHub Pages

---

## How to Run Everything

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/Timorleiderman/ILotto.git
cd ILotto

# Install dependencies (requires uv - https://github.com/astral-sh/uv)
uv sync
```

### 2. Generate Lottery Predictions

#### Option A: Hybrid Generator (Recommended)
Combines neural network predictions with smart scoring for optimal tickets:

```bash
# Using multi-output model (best results, no mode collapse)
uv run python smart_generator.py --model multi_output --count 5

# Using transformer model
uv run python smart_generator.py --model transformer --count 5

# Adjust model weight (0 = pure smart, 1 = pure model)
uv run python smart_generator.py --model multi_output --model-weight 0.7

# Higher temperature = more diverse sampling
uv run python smart_generator.py --model multi_output --temperature 2.0
```

#### Option B: Smart Generator (No Model)
Pure heuristic-based ticket generation:

```bash
# Generate unpopular ticket combinations
uv run python smart_generator.py --count 10
```

#### Option C: All-Model Predictions
Generate predictions from ALL trained models into a single report:

```bash
# Generate predictions from all models
uv run python predict.py

# Select specific models
uv run python predict.py --models multi_output transformer

# Generate more tickets per model
uv run python predict.py --count 10
```

This creates `PREDICTION.md` with:
- Greedy predictions + top probabilities per model
- Hybrid tickets (model + smart scoring)
- Pure smart tickets for comparison

#### Option D: Full Evaluation
Run complete evaluation with model predictions, metrics, and smart tickets:

```bash
uv run python evaluate.py
```

### 3. Train Models

```bash
# Train a specific model
uv run python train.py --model multi_output --epochs 50

# Train with diversity loss (prevents mode collapse)
uv run python train.py --model transformer --epochs 100 --diversity-loss

# Compare all architectures
uv run python train.py --compare-all --epochs 100

# Available models: original, multi_output, transformer, set_prediction
```

### 4. Generate Visualizations

```bash
# Architecture diagrams (saved to model/diagrams/)
uv run python model_viz.py

# Analysis charts (saved to visualizations/)
uv run python visualizations.py

# Interactive model viewer (opens in browser)
uv run netron model/multi_output_functional.keras --host 0.0.0.0 --port 8080
```

---

## Quick Reference

| Task | Command |
|------|---------|
| **All-model predictions** | `uv run python predict.py` |
| **Get predictions** | `uv run python smart_generator.py --model multi_output --count 5` |
| **Full evaluation** | `uv run python evaluate.py` |
| **Train model** | `uv run python train.py --model multi_output --epochs 50` |
| **Compare models** | `uv run python train.py --compare-all` |
| **Visualize architecture** | `uv run netron model/transformer_functional.keras` |
| **Generate charts** | `uv run python visualizations.py` |

---

## Hybrid Generator

The hybrid generator combines:
1. **Model Predictions** - Neural network probability distributions
2. **Smart Scoring** - Avoids popular combinations

```
Ticket #1:
  Numbers:  [ 3 13 18 25 35 37]  Bonus: 2
  Combined Score: 0.517
  ├─ Model confidence: 0.114
  └─ Smart score:      0.920
```

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model to use: `multi_output`, `transformer`, `original` | None (smart only) |
| `--count` | Number of tickets to generate | 5 |
| `--model-weight` | Weight for model vs smart (0-1) | 0.5 |
| `--temperature` | Sampling diversity (higher = more random) | 1.5 |

---

## Model Architectures

| Model | Description | Parameters | Best For |
|-------|-------------|------------|----------|
| **Original** | Seq2Seq with Bahdanau Attention | ~343K | Learning attention mechanisms |
| **Multi-Output** | LSTM + 7 Independent Dense Heads | ~123K | **Recommended** - No mode collapse |
| **Transformer** | Multi-Head Self-Attention | ~244K | Learning transformer architecture |
| **Set Prediction** | Multi-Label Sigmoid Output | ~193K | Treating lottery as unordered set |

### Training Commands

```bash
# Train specific model
uv run python train.py --model multi_output --epochs 100

# With diversity loss (prevents mode collapse)
uv run python train.py --model multi_output --epochs 100 --diversity-loss

# Compare all architectures
uv run python train.py --compare-all --epochs 100

# All options
uv run python train.py --help
```

## Visualization

### Model Architecture Diagrams

```bash
# Generate matplotlib-based architecture diagrams
uv run python model_viz.py

# Interactive model viewer (Netron)
uv run netron model/transformer_functional.keras --host 0.0.0.0 --port 8080
```

Generated diagrams are saved to `model/diagrams/`:
- `original_architecture_detailed.png`
- `multi_output_architecture_detailed.png`
- `transformer_architecture_detailed.png`
- `set_prediction_architecture_detailed.png`
- `all_models_comparison.png`
- `attention_comparison.png`

### Analysis Charts

```bash
uv run python visualizations.py
```

Generates charts in `visualizations/`:
- Number frequency analysis
- Co-occurrence heatmaps
- Model comparison
- Randomness test results

## Evaluation Results

### The data pipeline had to be fixed first

The numbers below changed once `helpers.py`'s data handling was corrected. Three issues,
each now guarded by a test in `tests/`:

| Issue | Effect |
|---|---|
| Rows filtered by **ball value ≤ 37** | A selection on the *outcome*, not a format filter — it deletes every historic draw containing a high number and keeps the low ones, biasing the surviving frequency table downward. It discarded ~2,000 of ~2,100 pre-2004 draws |
| Draws consumed **newest-first** | Sequence windows trained models to predict the past from the future |
| **Mixed game formats** | The archive spans 6/49-ish (to 2004), 6/34 with strong 1–10 (to 2008) and 6/37 (2009+). Only the 1,629 draws from 2012 on are one game |

The era is now selected by **date**, which is unbiased. This matters for the conclusions:
the previously reported chi-square on the main balls (p = 0.015, "minor bias") was an
artifact of that filter. On clean single-era data it is **p = 0.50** — no bias at all.

### Randomness tests (2012+, 1,629 draws)

| Test | P-Value | Result |
|------|---------|--------|
| Ball frequency uniformity (chi-square) | 0.504 | Uniform |
| Strong-number uniformity (chi-square) | 0.845 | Uniform |
| Lag-1 serial independence (hot/due) | 0.315 | Independent |
| Appearance-gap distribution vs geometric | 0.432 | As expected |
| Pairwise co-occurrence (666 pairs, Šidák) | 0.122 | No structure |
| Draw-sum distribution (KS) | 0.710 | As expected |
| First-half vs second-half drift | 0.863 | Stationary |

Nothing fires, before or after Holm–Bonferroni correction.

### Strategy benchmark

Thirteen strategies evaluated walk-forward — for each draw a strategy sees only the draws
before it. A randomly filled ticket scores 6×6/37 = **0.973** matches per draw; that is the
bar. **No strategy beats it after correcting for the number of strategies tried, and none
beats uniform log-loss (ln 37 = 3.611)** — meaning none holds information even in principle.

The instability is the finding: refreshing the archive by 93 draws completely reshuffled
the leaderboard ("Hot last 100" fell from p = 0.007 to p = 0.084). For scale, the luckiest
of 5,000 *purely random* strategies outscored the best real one.

Full detail, charts and next-draw picks: **[the published report](https://timorleiderman.github.io/ILotto/)**.

### Benchmark commands

```bash
uv run pytest -q                                   # 27 tests, ~4s
uv run python scripts/build_report.py --quick      # skip the neural models
uv run python scripts/build_report.py --refresh    # full run against fresh data (~80s)
open docs/index.html
```

## Project Structure

```
ILotto/
├── predict.py            # All-model prediction generator
├── train.py              # Training script with CLI
├── evaluate.py           # Evaluation and metrics
├── models.py             # Neural network architectures
├── ilotto.py             # Original Seq2Seq model
├── metrics.py            # Statistical evaluation
├── smart_generator.py    # Hybrid + smart ticket generator
├── visualizations.py     # Chart generation
├── model_viz.py          # Architecture diagrams
├── helpers.py            # Data loading utilities
├── bench/                # Leak-free benchmark package
│   ├── data.py           # Era-aware loading; chronological; no outcome filtering
│   ├── randomness.py     # 7 tests for exploitable structure
│   ├── predictors.py     # Frequency/Markov/ensemble strategies
│   ├── nn.py             # GRU + Transformer set prediction
│   ├── metrics.py        # Order-invariant scoring, exact nulls, Holm–Bonferroni
│   ├── backtest.py       # Walk-forward harness + Monte-Carlo null
│   └── report.py         # Builds docs/index.html
├── scripts/build_report.py
├── tests/                # Regression guards for the bugs above
├── docs/                 # Published GitHub Pages report
├── ARCHITECTURES.md      # Detailed architecture docs
├── GUIDE.md              # User guide
├── model/                # Saved models and diagrams
│   ├── *.keras           # Trained models
│   ├── *_functional.keras # Functional API models (for Netron)
│   └── diagrams/         # Architecture visualizations
├── visualizations/       # Generated charts
└── input/                # Lottery data
```

## Documentation

- **[GUIDE.md](GUIDE.md)** - Comprehensive user guide
- **[ARCHITECTURES.md](ARCHITECTURES.md)** - Detailed architecture explanations with diagrams

## Key Concepts Demonstrated

1. **Seq2Seq with Attention** - Encoder-decoder architecture with Bahdanau attention
2. **Multi-Head Self-Attention** - Transformer architecture
3. **Multi-Output Learning** - Independent prediction heads
4. **Multi-Label Classification** - Set prediction with sigmoid
5. **Mode Collapse Prevention** - Diversity loss and architecture design
6. **Statistical Hypothesis Testing** - Chi-square, runs test, correlation

## Smart Tickets

Instead of trying to predict numbers (impossible), the smart generator creates tickets that maximize expected payout IF you win:

- **Favor high numbers (32-37)** - Less popular with players who pick birthdays
- **Avoid birthday numbers (1-31)** - Most commonly picked
- **Skip sequential patterns** - 1,2,3,4,5,6 is very popular
- **Ensure good spread** - Avoid clustering

```bash
# Pure smart mode (no model)
uv run python smart_generator.py --count 10

# Hybrid mode (model + smart)
uv run python smart_generator.py --model multi_output --count 10
```

## Requirements

- Python 3.12+
- TensorFlow 2.x
- See `pyproject.toml` for full dependencies

## Publishing

`.github/workflows/report.yml` rebuilds everything on Sunday, Wednesday and Friday mornings
UTC — the mornings after the Saturday, Tuesday and occasional Thursday draws — commits the
refreshed draw archive, and deploys to GitHub Pages. `ci.yml` runs ruff, the test suite, a
quick report build and a strict docs build on every push and PR.

The site is MkDocs Material. `scripts/build_report.py` writes the generated pages into the
`docs/` tree (`docs/benchmark/index.html` and `docs/results.md`), then `mkdocs build`
renders the whole tree into `site/`, which is the published artifact.

```bash
uv sync --group docs
uv run mkdocs serve      # live preview on http://localhost:8000
```

## Disclaimer

**This is an educational project.** Lottery numbers are designed to be random - no prediction method can reliably predict future draws. The models demonstrate ML concepts, not a winning strategy. Please gamble responsibly.
