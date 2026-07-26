# ILotto - Complete User Guide

A machine learning project for analyzing the Israeli Lottery (Lotto). This project demonstrates ML concepts while providing practical tools for smart ticket generation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding the Modules](#understanding-the-modules)
5. [Training Models](#training-models)
6. [Evaluating Models](#evaluating-models)
7. [Smart Ticket Generation](#smart-ticket-generation)
8. [Visualizations](#visualizations)
9. [Model Architectures Explained](#model-architectures-explained)
10. [Metrics and Statistical Tests](#metrics-and-statistical-tests)
11. [The Truth About Lottery Prediction](#the-truth-about-lottery-prediction)
12. [API Reference](#api-reference)

---

## Project Overview

### What This Project Does

1. **Trains neural networks** on historical Israeli Lotto data
2. **Evaluates predictions** with proper statistical metrics
3. **Compares model performance** against random baseline
4. **Generates smart tickets** that avoid popular combinations
5. **Visualizes patterns** in lottery data
6. **Tests for randomness** using statistical methods

### Project Structure

```
ILotto/
├── train.py            # Training script for all model architectures
├── evaluate.py         # Comprehensive evaluation with metrics
├── ilotto.py           # Original LSTM Seq2Seq model
├── models.py           # Alternative architectures (Transformer, MultiOutput)
├── metrics.py          # Evaluation metrics and statistical tests
├── smart_generator.py  # Smart ticket generator
├── visualizations.py   # Charts and visual analytics
├── helpers.py          # Data loading and utilities
├── logger.py           # Logging configuration
├── model/              # Saved models and training outputs
├── visualizations/     # Generated charts
└── input/              # Lottery data CSV files
```

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (optional, for faster training)

### Using UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ILotto

# Install dependencies with uv
uv sync

# Verify installation
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

### Using pip

```bash
uv sync            # dependencies come from pyproject.toml + uv.lock
```

### Using Docker

```bash
# Build the Docker image
docker build -t ilotto .

# Run evaluation
docker run ilotto

# Run with GPU support
docker run --gpus all ilotto
```

---

## Quick Start

### 1. Run Full Evaluation (Recommended First Step)

```bash
uv run python evaluate.py
```

This will:
- Download latest lottery data from pais.co.il
- Load the trained model
- Evaluate predictions with proper metrics
- Compare against random baseline
- Generate smart ticket recommendations
- Create README.md with all results

### 2. Generate Smart Tickets

```bash
uv run python smart_generator.py
```

### 3. Train a New Model

```bash
uv run python train.py --model multi_output --epochs 50 --diversity-loss
```

### 4. Generate Visualizations

```bash
uv run python visualizations.py
```

---

## Understanding the Modules

### `train.py` - Model Training

The main training script supporting multiple architectures.

```bash
# See all options
uv run python train.py --help
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Architecture to train | `original` |
| `--epochs` | Number of training epochs | `50` |
| `--batch-size` | Training batch size | `32` |
| `--learning-rate` | Optimizer learning rate | `0.001` |
| `--diversity-loss` | Use loss that prevents mode collapse | `False` |
| `--compare-all` | Train all architectures and compare | `False` |

### `evaluate.py` - Model Evaluation

Comprehensive evaluation with metrics and reporting.

```bash
uv run python evaluate.py
```

**What it produces:**
- Console output with detailed metrics
- Updated `README.md` with results
- Comparison against random baseline
- Smart ticket recommendations

### `smart_generator.py` - Ticket Generation

Generates lottery tickets optimized to avoid popular combinations.

```bash
uv run python smart_generator.py
```

**Strategy:**
- Favors high numbers (32-37) that people rarely pick
- Avoids arithmetic sequences (1,2,3,4,5,6)
- Ensures good number spread across range
- Analyzes historical pair frequencies

### `visualizations.py` - Charts and Plots

Generates visual analytics for the lottery data.

```bash
uv run python visualizations.py
```

**Generated charts:**
- `number_frequency.png` - How often each number appears
- `bonus_frequency.png` - Bonus ball distribution
- `cooccurrence_heatmap.png` - Which numbers appear together
- `match_distribution.png` - How many matches per prediction
- `model_comparison.png` - Model vs random baseline
- `randomness_tests.png` - Statistical test results
- `ticket_scores.png` - Smart ticket scoring breakdown

### `metrics.py` - Evaluation Metrics

Provides proper evaluation framework.

```python
from metrics import LotteryMetrics, RandomnessTests

# Evaluate predictions
metrics = LotteryMetrics()
comparison = metrics.compare_with_baseline(predictions, ground_truth)

# Test for randomness
tests = RandomnessTests(historical_data)
results = tests.run_all_tests()
```

---

## Training Models

### Basic Training

```bash
# Train original LSTM model
uv run python train.py --model original --epochs 50

# Train with diversity loss (recommended to prevent mode collapse)
uv run python train.py --model multi_output --epochs 100 --diversity-loss
```

### Training All Architectures

```bash
# Compare all architectures
uv run python train.py --compare-all --epochs 50
```

### Training Output

After training, you'll find:

```
model/
├── <architecture>.keras           # Saved model
├── <architecture>_best.weights.h5 # Best checkpoint
├── <architecture>_history.csv     # Training metrics
├── <architecture>_training.png    # Loss/accuracy curves
└── logs/<architecture>/           # TensorBoard logs
```

### View Training Progress with TensorBoard

```bash
uv run tensorboard --logdir model/logs/
# Open http://localhost:6006 in browser
```

### Training Tips

1. **Start with `multi_output`** - Less prone to mode collapse
2. **Use `--diversity-loss`** - Prevents predicting same numbers
3. **Early stopping** is enabled by default (patience=10)
4. **Learning rate** reduces automatically when plateauing

---

## Evaluating Models

### Running Evaluation

```bash
uv run python evaluate.py
```

### Understanding the Output

```
--- Model Performance ---
  Samples evaluated:     50
  Average matches:       0.120    # How many numbers match on average
  Expected (random):     0.973    # What random guessing achieves
  Bonus match rate:      0.0%     # Percentage of correct bonus balls

--- Match Distribution ---
  0 matches:   44 ( 88.0%)        # Most predictions match nothing
  1 matches:    6 ( 12.0%)        # Some get 1 match
  2 matches:    0 (  0.0%)
  ...

--- Comparison with Random Baseline ---
  Baseline avg matches:  0.969 ± 0.113
  Z-score:               -75.090   # Negative = worse than random
  P-value:               1.0000
  Significantly better:  No        # Model doesn't beat random
  Improvement:           -87.62%   # Actually worse than random!
```

### What Good Results Look Like

For a lottery (which is random), the **best possible model** will perform **equal to random baseline**:
- Average matches: ~0.97
- Improvement over random: ~0%
- Not significantly better (p > 0.05)

If your model performs **worse than random** (like -87%), it has mode collapse - it keeps predicting the same numbers.

---

## Smart Ticket Generation

### Concept

Since we can't predict winning numbers, we optimize for **expected payout**:
- Avoid numbers that many people pick
- If you win, you share with fewer people
- Same probability of winning, higher expected prize

### Running the Generator

```bash
uv run python smart_generator.py
```

### Understanding Ticket Scores

```
Ticket #1:
  Numbers:  [ 1 14 29 34 35 37]  Bonus: 5
  Score:    0.926
  Details:  high=1.00 spread=0.99 seq=1.00 pattern=1.00
```

**Score Components:**

| Component | What it measures | Good score |
|-----------|------------------|------------|
| `high` | Numbers > 31 included | 1.0 (at least 2) |
| `spread` | Even distribution across range | > 0.8 |
| `seq` | Avoids arithmetic sequences | 1.0 (no sequences) |
| `pattern` | Avoids common patterns | 1.0 |
| `pair_rarity` | Uses uncommon number pairs | > 0.5 |

**Total Score:**
- 0.9+ = Excellent (very unpopular combination)
- 0.7-0.9 = Good
- 0.5-0.7 = Average
- < 0.5 = Popular (many people pick similar)

### Using in Python

```python
from smart_generator import SmartTicketGenerator

generator = SmartTicketGenerator()

# Generate 5 optimized tickets
tickets = generator.generate_optimized_tickets(n_tickets=5, n_candidates=5000)

for ticket, scores in tickets:
    print(f"Numbers: {ticket[:6]}, Bonus: {ticket[6]}, Score: {scores['total']:.3f}")
```

### Popular vs Unpopular Example

**Popular ticket (low score ~0.45):**
```
Numbers: [1, 2, 3, 7, 14, 21]  # Sequence + all low numbers
- Many people pick sequences
- All birthday-range numbers (1-31)
- If this wins, you share with many
```

**Smart ticket (high score ~0.92):**
```
Numbers: [1, 14, 29, 34, 35, 37]
- Includes high numbers (34, 35, 37)
- Good spread across range
- Uncommon pair combinations
- Fewer people pick this
```

---

## Visualizations

### Generating All Charts

```bash
uv run python visualizations.py
```

### Available Visualizations

#### 1. Number Frequency (`number_frequency.png`)
Shows how often each main ball (1-37) appears in historical draws.
- Green bars: Above expected frequency
- Blue bars: Near expected
- Red bars: Below expected

#### 2. Bonus Frequency (`bonus_frequency.png`)
Distribution of bonus ball (1-7) appearances.

#### 3. Co-occurrence Heatmap (`cooccurrence_heatmap.png`)
Shows which number pairs frequently appear together.
- Darker = more frequent
- Useful for understanding common combinations

#### 4. Match Distribution (`match_distribution.png`)
How many numbers the model correctly predicts per draw.

#### 5. Model Comparison (`model_comparison.png`)
Side-by-side comparison of model vs random baseline.

#### 6. Randomness Tests (`randomness_tests.png`)
Visual representation of statistical test results.

#### 7. Ticket Scores (`ticket_scores.png`)
Breakdown of smart ticket scoring components.

### Using Visualizations in Python

```python
from visualizations import (
    plot_number_frequency,
    plot_model_comparison,
    generate_all_visualizations,
)

# Generate single chart
plot_number_frequency(data, save_path="my_chart.png", show=True)

# Generate all charts
generate_all_visualizations(
    data, metrics_comparison, randomness_results, smart_tickets,
    output_dir="visualizations/",
    show=False  # Don't display, just save
)
```

---

## Model Architectures Explained

### 1. Original (`ILotto` in `ilotto.py`)

**Architecture:** Seq2Seq LSTM with Attention

```
Input (10 draws) → Embedding → Bidirectional LSTM → Attention → Dense → Output (7 numbers)
```

**Characteristics:**
- Uses Bahdanau (additive) attention
- Treats output as a sequence (position matters)
- Prone to mode collapse (predicts same number repeatedly)

**Use case:** Educational - demonstrates seq2seq with attention

### 2. Multi-Output (`MultiOutputLotto` in `models.py`)

**Architecture:** LSTM with separate output heads

```
Input → Embedding → LSTM → 7 separate Dense heads → 7 outputs
```

**Characteristics:**
- Each ball position has its own classifier
- Less prone to mode collapse
- Each position can predict different numbers

**Use case:** Better for training, avoids mode collapse

### 3. Transformer (`TransformerLotto` in `models.py`)

**Architecture:** Transformer encoder

```
Input → Embedding + Position → Multi-Head Attention → FFN → Pool → Output heads
```

**Characteristics:**
- Self-attention captures relationships
- Parallel processing (faster)
- Modern architecture

**Use case:** Learning about transformers

### 4. Set Prediction (`SetPredictionLotto` in `models.py`)

**Architecture:** Multi-label classification

```
Input → Embedding → LSTM → Sigmoid output (37 classes)
```

**Characteristics:**
- Predicts which numbers will appear (not positions)
- Uses binary cross-entropy
- More aligned with lottery nature (order doesn't matter)

**Use case:** Conceptually closest to the actual problem

### Diversity Loss

All architectures can use `DiversityLoss` to prevent mode collapse:

```python
# Standard loss: just cross-entropy
loss = cross_entropy(y_true, y_pred)

# Diversity loss: penalizes similar predictions
loss = cross_entropy(y_true, y_pred) + diversity_weight * similarity_penalty
```

---

## Metrics and Statistical Tests

### Evaluation Metrics

#### Average Matches
- How many of 6 predicted numbers match actual draw
- Expected for random: 6 × (6/37) ≈ 0.97

#### Match Distribution
- Count of 0, 1, 2, 3, 4, 5, 6 matches
- Random baseline: mostly 0-1 matches

#### Win Rates
- Percentage achieving 3+, 4+, 5+, 6 matches
- For random: 3+ ≈ 4%, 6 ≈ 0%

### Statistical Tests

#### Chi-Square Test
- Tests if number frequencies are uniform
- p > 0.05 → Cannot reject randomness
- p < 0.05 → Possible bias detected

#### Runs Test
- Tests independence of consecutive draws
- Checks if there are patterns in sequences

#### Serial Correlation
- Tests correlation between consecutive draws
- Checks if draw N affects draw N+1

### Interpreting Results

```
Chi-square uniformity (main balls):
  P-value: 0.0150
  Result:  Non-random bias detected
```

This doesn't mean lottery is rigged! It means:
- With ~2300 draws, some numbers appear more than others
- This is **expected statistical variation**
- With infinite draws, it would be perfectly uniform

---

## The Truth About Lottery Prediction

### Why ML Can't Predict Lottery

1. **Lottery is designed to be random**
   - Uses certified random number generators
   - Each draw is independent
   - No pattern exists to learn

2. **Past doesn't predict future**
   - Ball 7 appearing 10 times doesn't make it more or less likely
   - This is the "Gambler's Fallacy"

3. **No hidden signal**
   - Unlike stock prices or weather, there's no underlying pattern
   - The best any model can do is match random guessing

### What This Project Actually Teaches

1. **ML concepts:**
   - Seq2Seq models, Attention mechanisms, Transformers
   - Training, validation, evaluation
   - Handling multi-output classification

2. **Proper evaluation:**
   - Comparing against baseline
   - Statistical significance testing
   - Understanding mode collapse

3. **Practical strategy:**
   - Smart ticket generation
   - Avoiding popular combinations
   - Expected value optimization

### The Only Valid Strategy

1. **Don't play** - Expected value is negative (house always wins)
2. **If you play**, use smart tickets:
   - Same odds of winning
   - Higher expected prize (fewer people share)
3. **Play for fun, not profit**

---

## API Reference

### `metrics.py`

```python
from metrics import LotteryMetrics, RandomnessTests

# Initialize
metrics = LotteryMetrics(n_main=37, n_bonus=7)

# Count matches for single prediction
main_matches, bonus_match = metrics.count_matches(prediction, ground_truth)

# Evaluate batch of predictions
results = metrics.evaluate_predictions(predictions, ground_truths)
# Returns: avg_matches, match_distribution, win_rates, etc.

# Compare with random baseline
comparison = metrics.compare_with_baseline(
    model_predictions, 
    ground_truths,
    n_baseline_runs=100
)
# Returns: model metrics, baseline stats, z-score, p-value

# Randomness tests
tests = RandomnessTests(historical_data)
results = tests.run_all_tests()
# Returns: chi_square_main, chi_square_bonus, runs_test, serial_correlation
```

### `smart_generator.py`

```python
from smart_generator import SmartTicketGenerator, TicketScorer

# Initialize with historical data
generator = SmartTicketGenerator(historical_data)

# Generate single random ticket
ticket = generator.generate_random_ticket()

# Generate smart ticket with constraints
ticket = generator.generate_smart_ticket(min_high_numbers=2)

# Generate optimized tickets (best of many candidates)
tickets = generator.generate_optimized_tickets(
    n_tickets=5,      # How many tickets to return
    n_candidates=1000 # How many to generate and score
)

# Generate diverse tickets (minimize overlap)
tickets = generator.generate_coverage_tickets(
    n_tickets=10,
    overlap_threshold=3  # Max shared numbers between tickets
)

# Score any ticket
scorer = TicketScorer(historical_data)
scores = scorer.score_ticket([1, 14, 29, 34, 35, 37, 5])
# Returns: high_numbers, spread, sequence_avoidance, pattern_avoidance, pair_rarity, total
```

### `models.py`

```python
from models import MultiOutputLotto, TransformerLotto, SetPredictionLotto, create_model

# Create model by name
model = create_model("multi_output")

# Or directly
model = MultiOutputLotto(
    n_main=37,
    n_bonus=7,
    embed_dim=32,
    lstm_units=64,
    dropout_rate=0.3
)

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Predict
predictions = model.predict(X_test)
```

### `train.py`

```python
from train import get_compiled_model, train_model, evaluate_model

# Get compiled model
model = get_compiled_model(
    architecture="multi_output",  # or "original", "transformer", "set_prediction"
    learning_rate=0.001,
    use_diversity_loss=True
)

# Train
history = train_model(
    model, X_train, y_train, X_test, y_test,
    epochs=50,
    batch_size=32,
    model_name="my_model",
    early_stopping=True
)

# Evaluate
results = evaluate_model(model, X_test, y_test, architecture="multi_output")
```

---

## Troubleshooting

### Common Issues

**1. Mode Collapse (model predicts same numbers)**
```bash
# Use diversity loss
uv run python train.py --model multi_output --diversity-loss
```

**2. GPU not detected**
```bash
# Check TensorFlow sees GPU
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**3. Out of memory**
```bash
# Reduce batch size
uv run python train.py --batch-size 16
```

**4. Data not downloading**
```python
# Manually download from
# https://pais.co.il/Lotto/lotto_resultsDownload.aspx
```

### Getting Help

1. Check TensorFlow logs for errors
2. Use `--help` flag on any script
3. Review this guide for usage examples

---

## License

This project is for educational purposes. Use responsibly. Gambling can be addictive - please play responsibly.

---

*Happy learning (and good luck if you play)!*
