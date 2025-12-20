# ILotto - Israeli Lottery Analysis

A machine learning project for analyzing Israeli Lotto patterns. Built as a learning exercise to explore different neural network architectures, with proper statistical analysis and evaluation metrics.

> **Note:** Lottery numbers are random by design. No model can predict them. This project demonstrates ML concepts, not a winning strategy.

## Features

- **4 Neural Network Architectures** - Seq2Seq, Multi-Output, Transformer, Set Prediction
- **Statistical Analysis** - Chi-square tests, runs test, serial correlation
- **Smart Ticket Generator** - Avoids popular combinations to maximize payout if winning
- **Interactive Visualization** - Netron integration for model architecture exploration
- **Comprehensive Metrics** - Proper evaluation against random baseline

## Quick Start

```bash
# Install dependencies
uv sync

# Train a model
uv run python train.py --model multi_output --epochs 50

# Evaluate
uv run python evaluate.py

# Generate smart tickets
uv run python smart_generator.py

# View model architecture (opens in browser)
uv run netron model/multi_output_functional.keras --host 0.0.0.0 --port 8080
```

## Model Architectures

| Model | Description | Parameters | Mode Collapse Risk |
|-------|-------------|------------|-------------------|
| **Original** | Seq2Seq with Bahdanau Attention | ~343K | High |
| **Multi-Output** | LSTM + 7 Independent Dense Heads | ~123K | Low |
| **Transformer** | Multi-Head Self-Attention | ~244K | Low |
| **Set Prediction** | Multi-Label Sigmoid Output | ~193K | Very Low |

### Training All Models

```bash
# Compare all architectures
uv run python train.py --compare-all --epochs 100

# Train specific model with diversity loss (prevents mode collapse)
uv run python train.py --model multi_output --epochs 100 --diversity-loss
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

Instead of trying to predict numbers (impossible), the smart generator creates tickets that:
- Favor high numbers (32-37) - less popular with players
- Avoid birthday numbers (1-31)
- Skip sequential patterns
- Ensure good spread

If you win, you're less likely to share the jackpot.

```bash
uv run python smart_generator.py --count 10
```

## Requirements

- Python 3.12+
- TensorFlow 2.x
- See `pyproject.toml` for full dependencies

## Disclaimer

**This is an educational project.** Lottery numbers are designed to be random - no prediction method can reliably predict future draws. The models demonstrate ML concepts, not a winning strategy. Please gamble responsibly.
