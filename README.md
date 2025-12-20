# ILotto - Israeli Lottery Analysis

A machine learning project for analyzing Israeli Lotto patterns. Built as a learning exercise to explore different neural network architectures, with proper statistical analysis and evaluation metrics.

> **Note:** Lottery numbers are random by design. No model can predict them. This project demonstrates ML concepts, not a winning strategy.

## Features

- **4 Neural Network Architectures** - Seq2Seq, Multi-Output, Transformer, Set Prediction
- **Hybrid Ticket Generator** - Combines model predictions with smart scoring
- **Statistical Analysis** - Chi-square tests, runs test, serial correlation
- **Smart Ticket Generator** - Avoids popular combinations to maximize payout if winning
- **Interactive Visualization** - Netron integration for model architecture exploration
- **Comprehensive Metrics** - Proper evaluation against random baseline

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

#### Option C: Full Evaluation
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

### Model Performance (vs Random Baseline)

The models are evaluated against a random baseline (expected ~0.97 matches per draw):

| Metric | Value |
|--------|-------|
| Samples Evaluated | 50 |
| Model Avg Matches | 0.24 |
| Baseline Avg Matches | 0.97 |
| Improvement | -75% (mode collapse) |

**Key Finding:** The original model suffers from mode collapse, predicting the same numbers repeatedly. The Multi-Output and Transformer architectures perform near random baseline, which is optimal for truly random data.

### Lottery Randomness Tests

| Test | P-Value | Result |
|------|---------|--------|
| Chi-square (main balls) | 0.015 | Minor bias |
| Chi-square (bonus ball) | 0.995 | Random |
| Runs test | 0.888 | Independent |
| Serial correlation | 0.861 | No correlation |

## Project Structure

```
ILotto/
├── train.py              # Training script with CLI
├── evaluate.py           # Evaluation and metrics
├── models.py             # Neural network architectures
├── ilotto.py             # Original Seq2Seq model
├── metrics.py            # Statistical evaluation
├── smart_generator.py    # Unpopular ticket generator
├── visualizations.py     # Chart generation
├── model_viz.py          # Architecture diagrams
├── helpers.py            # Data loading utilities
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

## Disclaimer

**This is an educational project.** Lottery numbers are designed to be random - no prediction method can reliably predict future draws. The models demonstrate ML concepts, not a winning strategy. Please gamble responsibly.
