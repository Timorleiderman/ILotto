# Model Architectures - Deep Dive

This document provides detailed explanations of each neural network architecture used in ILotto, including visual diagrams, layer-by-layer breakdowns, and the intuition behind each design choice.

## Table of Contents

1. [The Problem We're Solving](#the-problem-were-solving)
2. [Original Model (Seq2Seq with Attention)](#1-original-model-seq2seq-with-attention)
3. [Multi-Output Model](#2-multi-output-model)
4. [Transformer Model](#3-transformer-model)
5. [Set Prediction Model](#4-set-prediction-model)
6. [Comparison Summary](#comparison-summary)
7. [Generating Architecture Diagrams](#generating-architecture-diagrams)

---

## The Problem We're Solving

### Input
- **10 previous lottery draws** (sequence of 10 timesteps)
- **Each draw has 7 numbers**: 6 main balls (0-36) + 1 bonus ball (0-6)
- **Input shape**: `(batch_size, 10, 7)`

### Output
- **Next lottery draw**: 7 numbers
- **Output shape**: `(batch_size, 7, 37)` - probability distribution over 37 numbers for each position

### The Challenge
We're treating this as a **sequence-to-sequence** problem:
- Input: sequence of 10 draws
- Output: sequence of 7 numbers

---

## 1. Original Model (Seq2Seq with Attention)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENCODER                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (batch, 10, 7)                                                       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  EMBEDDING LAYER (for each of 7 positions)                  │            │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │            │
│  │  │Emb 1│ │Emb 2│ │Emb 3│ │Emb 4│ │Emb 5│ │Emb 6│ │Emb 7│   │            │
│  │  │30dim│ │30dim│ │30dim│ │30dim│ │30dim│ │30dim│ │30dim│   │            │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │            │
│  │     │       │       │       │       │       │       │       │            │
│  │     ▼       ▼       ▼       ▼       ▼       ▼       ▼       │            │
│  │  SpatialDropout1D (0.5)                                     │            │
│  │     │       │       │       │       │       │       │       │            │
│  │     └───────┴───────┴───────┼───────┴───────┴───────┘       │            │
│  │                             │                                │            │
│  │                    CONCATENATE                               │            │
│  │                             │                                │            │
│  │                             ▼                                │            │
│  │                   (batch, 10, 210)                           │            │
│  └─────────────────────────────┼───────────────────────────────┘            │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │              BIDIRECTIONAL LSTM LAYER 1                      │            │
│  │  ┌──────────────────┐    ┌──────────────────┐               │            │
│  │  │   Forward LSTM   │    │  Backward LSTM   │               │            │
│  │  │    64 units      │    │    64 units      │               │            │
│  │  └────────┬─────────┘    └────────┬─────────┘               │            │
│  │           │                       │                          │            │
│  │           └───────────┬───────────┘                          │            │
│  │                       │                                      │            │
│  │              CONCATENATE (128 units)                         │            │
│  │                       │                                      │            │
│  └───────────────────────┼─────────────────────────────────────┘            │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │              BIDIRECTIONAL LSTM LAYER 2                      │            │
│  │  ┌──────────────────┐    ┌──────────────────┐               │            │
│  │  │   Forward LSTM   │    │  Backward LSTM   │               │            │
│  │  │    32 units      │    │    32 units      │               │            │
│  │  └────────┬─────────┘    └────────┬─────────┘               │            │
│  │           │                       │                          │            │
│  │           └───────────┬───────────┘                          │            │
│  │                       │                                      │            │
│  │              CONCATENATE (64 units)                          │            │
│  │                       │                                      │            │
│  │         Returns: sequence_output, h, c                       │            │
│  └───────────────────────┼─────────────────────────────────────┘            │
│                          │                                                   │
└──────────────────────────┼──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DECODER                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                          │                                                   │
│              ┌───────────┴───────────┐                                       │
│              │     RepeatVector(7)   │                                       │
│              │   Repeat h 7 times    │                                       │
│              └───────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│              (batch, 7, 64)                                                  │
│                          │                                                   │
│              ┌───────────┴───────────┐                                       │
│              │    LSTM Decoder 1     │                                       │
│              │      64 units         │                                       │
│              │  (init with h1, c1)   │                                       │
│              └───────────┬───────────┘                                       │
│                          │                                                   │
│              ┌───────────┴───────────┐                                       │
│              │    LSTM Decoder 2     │                                       │
│              │      64 units         │                                       │
│              │  (init with h2, c2)   │                                       │
│              └───────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                  BAHDANAU ATTENTION                          │            │
│  │                                                              │            │
│  │   Query: decoder output (batch, 7, 64)                       │            │
│  │   Key/Value: encoder sequence (batch, 10, 128)               │            │
│  │                                                              │            │
│  │   Attention weights = softmax(score(query, key))             │            │
│  │   Context = weighted sum of values                           │            │
│  │                                                              │            │
│  │   score(q, k) = V · tanh(W1·q + W2·k)  [Additive/Bahdanau]  │            │
│  │                                                              │            │
│  └───────────────────────┬─────────────────────────────────────┘            │
│                          │                                                   │
│              ┌───────────┴───────────┐                                       │
│              │     CONCATENATE       │                                       │
│              │  [context, decoder]   │                                       │
│              └───────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│              ┌───────────────────────┐                                       │
│              │    Dense (37 units)   │                                       │
│              │   Softmax activation  │                                       │
│              └───────────┬───────────┘                                       │
│                          │                                                   │
│                          ▼                                                   │
│              Output: (batch, 7, 37)                                          │
│              Probability for each of 37 numbers at each of 7 positions       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Explanation

#### 1. Embedding Layer
```python
Embedding(input_dim=37, output_dim=30)
```
- **Purpose**: Convert discrete numbers (0-36) into dense vectors
- **Why 7 separate embeddings?** Each position might have different patterns
- **Output**: Each number becomes a 30-dimensional vector

#### 2. SpatialDropout1D
```python
SpatialDropout1D(rate=0.5)
```
- **Purpose**: Regularization - drops entire feature channels
- **Why spatial?** Better for sequential data than regular dropout

#### 3. Bidirectional LSTM Encoder
```python
Bidirectional(LSTM(64, return_sequences=True, return_state=True))
```
- **Bidirectional**: Reads sequence forward AND backward
- **Why?** Captures context from both directions
- **Returns**: Sequence output + hidden states (h, c)

#### 4. RepeatVector
```python
RepeatVector(7)
```
- **Purpose**: Copy final encoder state 7 times (one for each output)
- **Output**: (batch, 7, hidden_dim)

#### 5. LSTM Decoder
- Takes repeated vector as input
- Initialized with encoder's final states
- Generates output sequence

#### 6. Bahdanau Attention
```python
AdditiveAttention(dropout=0.5)
```
- **Purpose**: Focus on relevant parts of input sequence
- **How it works**:
  - Compute alignment scores between decoder state and encoder outputs
  - Softmax to get attention weights
  - Weighted sum of encoder outputs = context vector

#### 7. Dense Output
```python
Dense(37, activation='softmax')
```
- **Output**: Probability distribution over 37 numbers
- **Applied to each of 7 positions**

### Why This Architecture Has Problems

**Mode Collapse Issue**: The model tends to predict the same number for all positions because:
1. Shared decoder state → similar outputs
2. Softmax encourages one dominant class
3. No mechanism to enforce diversity

---

## 2. Multi-Output Model

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Input: (batch, 10, 7)                                                       │
│         │                                                                    │
│         ├────────────────────────────────┐                                   │
│         │                                │                                   │
│         ▼                                ▼                                   │
│  ┌──────────────┐                ┌──────────────┐                           │
│  │ Main Balls   │                │ Bonus Ball   │                           │
│  │ Embedding    │                │ Embedding    │                           │
│  │ (37 → 32)    │                │ (7 → 32)     │                           │
│  └──────┬───────┘                └──────┬───────┘                           │
│         │                                │                                   │
│         └────────────────┬───────────────┘                                   │
│                          │                                                   │
│                    CONCATENATE                                               │
│                          │                                                   │
│                          ▼                                                   │
│                  (batch, 10, 224)                                            │
│                    [6×32 + 32]                                               │
│                          │                                                   │
│         ┌────────────────┴────────────────┐                                  │
│         │                                 │                                  │
│         ▼                                 │                                  │
│  ┌──────────────┐                         │                                  │
│  │   LSTM 1     │                         │                                  │
│  │  64 units    │                         │                                  │
│  │ return_seq   │                         │                                  │
│  │ dropout=0.3  │                         │                                  │
│  └──────┬───────┘                         │                                  │
│         │                                 │                                  │
│         ▼                                 │                                  │
│  ┌──────────────┐                         │                                  │
│  │   LSTM 2     │                         │                                  │
│  │  64 units    │                         │                                  │
│  │ dropout=0.3  │                         │                                  │
│  └──────┬───────┘                         │                                  │
│         │                                 │                                  │
│         ▼                                 │                                  │
│  ┌──────────────┐                         │                                  │
│  │   Dropout    │                         │                                  │
│  │    (0.3)     │                         │                                  │
│  └──────┬───────┘                         │                                  │
│         │                                 │                                  │
│         │    Shared representation        │                                  │
│         │    (batch, 64)                  │                                  │
│         │                                 │                                  │
│         ├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┐  │
│         │          │          │          │          │          │         │  │
│         ▼          ▼          ▼          ▼          ▼          ▼         ▼  │
│  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│  │ Dense 1  ││ Dense 2  ││ Dense 3  ││ Dense 4  ││ Dense 5  ││ Dense 6  ││Dense Bonus│
│  │ 37 units ││ 37 units ││ 37 units ││ 37 units ││ 37 units ││ 37 units ││ 7 units  │
│  │ softmax  ││ softmax  ││ softmax  ││ softmax  ││ softmax  ││ softmax  ││ softmax  │
│  └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
│       │          │          │          │          │          │         │  │
│       ▼          ▼          ▼          ▼          ▼          ▼         ▼  │
│   Ball 1     Ball 2     Ball 3     Ball 4     Ball 5     Ball 6    Bonus  │
│                                                                              │
│                          Stack outputs                                       │
│                               │                                              │
│                               ▼                                              │
│                    Output: (batch, 7, 37)                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Original

1. **Separate Output Heads**: Each ball position has its own Dense layer
2. **Independent Predictions**: Position 1 prediction doesn't directly affect position 2
3. **Shared Encoder**: All positions share the same LSTM encoding

### Why This Helps with Mode Collapse

```python
# Original: Single Dense layer applied to all positions
output = Dense(37)(decoder_output)  # Same weights for all positions

# Multi-Output: Separate Dense layers
ball_1 = Dense(37, name='ball_1')(shared_repr)  # Unique weights
ball_2 = Dense(37, name='ball_2')(shared_repr)  # Unique weights
# ... each learns independently
```

### Advantages
- ✅ Less prone to mode collapse
- ✅ Each position can specialize
- ✅ Simpler architecture

### Disadvantages
- ❌ Positions can predict same number (no coordination)
- ❌ More parameters (7 separate heads)

---

## 3. Transformer Model

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Input: (batch, 10, 7)                                                       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                       FLATTEN                                 │           │
│  │                  (batch, 10×7) = (batch, 70)                  │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│         ┌───────────────────────┼───────────────────────┐                   │
│         │                       │                       │                   │
│         ▼                       ▼                       │                   │
│  ┌──────────────┐       ┌──────────────┐               │                   │
│  │   Number     │       │   Position   │               │                   │
│  │  Embedding   │       │  Embedding   │               │                   │
│  │  (37 → 64)   │       │  (70 → 64)   │               │                   │
│  └──────┬───────┘       └──────┬───────┘               │                   │
│         │                       │                       │                   │
│         └───────────┬───────────┘                       │                   │
│                     │                                   │                   │
│                   ADD                                   │                   │
│                     │                                   │                   │
│                     ▼                                   │                   │
│             (batch, 70, 64)                             │                   │
│                     │                                   │                   │
│  ┌──────────────────┴──────────────────────────────────┐│                   │
│  │           TRANSFORMER BLOCK 1                        ││                   │
│  │  ┌─────────────────────────────────────────────┐    ││                   │
│  │  │          MULTI-HEAD ATTENTION               │    ││                   │
│  │  │                                             │    ││                   │
│  │  │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   │    ││                   │
│  │  │   │Head1│   │Head2│   │Head3│   │Head4│   │    ││                   │
│  │  │   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   │    ││                   │
│  │  │      │         │         │         │       │    ││                   │
│  │  │      └─────────┴────┬────┴─────────┘       │    ││                   │
│  │  │                     │                       │    ││                   │
│  │  │               CONCATENATE                   │    ││                   │
│  │  │                     │                       │    ││                   │
│  │  │            Linear projection                │    ││                   │
│  │  │                     │                       │    ││                   │
│  │  └─────────────────────┼───────────────────────┘    ││                   │
│  │                        │                             ││                   │
│  │           ┌────────────┘                             ││                   │
│  │           │                                          ││                   │
│  │           ▼                                          ││                   │
│  │    ┌──────────────┐                                  ││                   │
│  │    │   Dropout    │                                  ││                   │
│  │    └──────┬───────┘                                  ││                   │
│  │           │                                          ││                   │
│  │           ▼                                          ││                   │
│  │    ┌──────────────┐                                  ││                   │
│  │    │ ADD & NORM   │◄─────── Residual connection      ││                   │
│  │    └──────┬───────┘                                  ││                   │
│  │           │                                          ││                   │
│  │           ▼                                          ││                   │
│  │    ┌──────────────┐                                  ││                   │
│  │    │ Feed Forward │                                  ││                   │
│  │    │ Dense(128)   │─── GELU activation               ││                   │
│  │    │ Dropout      │                                  ││                   │
│  │    │ Dense(64)    │                                  ││                   │
│  │    └──────┬───────┘                                  ││                   │
│  │           │                                          ││                   │
│  │           ▼                                          ││                   │
│  │    ┌──────────────┐                                  ││                   │
│  │    │ ADD & NORM   │◄─────── Residual connection      ││                   │
│  │    └──────┬───────┘                                  ││                   │
│  │           │                                          ││                   │
│  └───────────┼──────────────────────────────────────────┘│                   │
│              │                                           │                   │
│              ▼                                           │                   │
│  ┌───────────────────────────────────────────────────────┐                   │
│  │           TRANSFORMER BLOCK 2                         │                   │
│  │              (Same structure)                         │                   │
│  └───────────────────────┬───────────────────────────────┘                   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────┐                   │
│  │         GLOBAL AVERAGE POOLING                        │                   │
│  │    Average across sequence dimension                  │                   │
│  │    (batch, 70, 64) → (batch, 64)                      │                   │
│  └───────────────────────┬───────────────────────────────┘                   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────┐                   │
│  │              Dense(256) + GELU                        │                   │
│  └───────────────────────┬───────────────────────────────┘                   │
│                          │                                                   │
│         ┌────────────────┼────────────────┬───────────────┐                  │
│         │                │                │               │                  │
│         ▼                ▼                ▼               ▼                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │ Ball 1   │    │ Ball 2   │    │ ...      │    │ Bonus    │              │
│   │ Dense(37)│    │ Dense(37)│    │          │    │ Dense(7) │              │
│   │ Softmax  │    │ Softmax  │    │          │    │ Softmax  │              │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│        │               │               │               │                     │
│        └───────────────┴───────────────┴───────────────┘                     │
│                                │                                             │
│                          Stack outputs                                       │
│                                │                                             │
│                                ▼                                             │
│                     Output: (batch, 7, 37)                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Self-Attention Explained

```
Self-Attention computes relationships between ALL positions:

Input: [x1, x2, x3, ..., x70]  (flattened sequence)

For each position, compute:
  - Query (Q): What am I looking for?
  - Key (K): What do I contain?
  - Value (V): What information do I provide?

Attention(Q, K, V) = softmax(QK^T / √d) · V

        Q·K^T gives us attention matrix:
        
             k1   k2   k3   k4   ...  k70
        q1  [0.1  0.3  0.2  0.1  ...  0.1]
        q2  [0.2  0.1  0.4  0.1  ...  0.1]
        q3  [0.1  0.2  0.1  0.3  ...  0.1]
        ...
        q70 [0.1  0.1  0.1  0.1  ...  0.4]
        
        Each row shows how much each position attends to others
```

### Multi-Head Attention

```
Instead of one attention, use multiple "heads":

Head 1: Might focus on "which numbers appeared recently"
Head 2: Might focus on "number magnitude patterns"
Head 3: Might focus on "position patterns"
Head 4: Might focus on "bonus ball patterns"

Final output = Concat(Head1, Head2, Head3, Head4) × W_O
```

### Advantages
- ✅ Captures long-range dependencies
- ✅ Parallel processing (faster than LSTM)
- ✅ Modern, well-understood architecture
- ✅ No recurrence = no vanishing gradients

### Disadvantages
- ❌ More parameters
- ❌ Needs more data to train well
- ❌ Position information via embedding only

---

## 4. Set Prediction Model

### The Key Insight

**Lottery is a SET, not a SEQUENCE!**

The numbers [3, 7, 15, 22, 31, 35] win regardless of order.
Previous models treat it as sequence prediction, which is wrong.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Input: (batch, 10, 7)                                                       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                      EMBEDDING                                │           │
│  │                     (37 → 32)                                 │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│                        (batch, 10, 7, 32)                                    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                      RESHAPE                                  │           │
│  │               (batch, 10, 7×32) = (batch, 10, 224)            │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                        LSTM                                   │           │
│  │                     128 units                                 │           │
│  │                   dropout=0.3                                 │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│                         (batch, 128)                                         │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                     Dense(64) + ReLU                          │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                      Dropout(0.3)                             │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│              ┌──────────────────┴──────────────────┐                        │
│              │                                     │                        │
│              ▼                                     ▼                        │
│  ┌─────────────────────────┐         ┌─────────────────────────┐           │
│  │   MAIN BALLS OUTPUT     │         │   BONUS BALL OUTPUT     │           │
│  │                         │         │                         │           │
│  │   Dense(37)             │         │   Dense(7)              │           │
│  │   SIGMOID activation    │         │   SOFTMAX activation    │           │
│  │                         │         │                         │           │
│  │   Output: (batch, 37)   │         │   Output: (batch, 7)    │           │
│  │                         │         │                         │           │
│  │   Each output is        │         │   Single number         │           │
│  │   probability that      │         │   classification        │           │
│  │   number appears        │         │                         │           │
│  └─────────────────────────┘         └─────────────────────────┘           │
│                                                                              │
│                                                                              │
│  INTERPRETATION:                                                             │
│                                                                              │
│  Main output [0.1, 0.05, 0.8, 0.02, ..., 0.7, 0.3]                          │
│               │     │     │    │         │     │                             │
│               ▼     ▼     ▼    ▼         ▼     ▼                             │
│              #1    #2    #3   #4   ... #36   #37                             │
│                                                                              │
│  Take TOP 6 with highest probability as prediction                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Label vs Multi-Class

```
MULTI-CLASS (other models):
- Output: probability distribution over 37 classes
- Sum of probabilities = 1
- Exactly ONE class is correct
- Loss: Categorical Cross-Entropy

MULTI-LABEL (this model):
- Output: 37 independent probabilities
- Each can be 0-1 independently
- MULTIPLE classes can be correct (exactly 6)
- Loss: Binary Cross-Entropy (for each number)

Example:
  Multi-class: [0.7, 0.1, 0.1, 0.05, 0.05, 0, 0, ...]  # sums to 1
  Multi-label: [0.9, 0.1, 0.8, 0.05, 0.95, 0.7, 0.2, ...]  # independent
```

### Advantages
- ✅ Conceptually correct for lottery
- ✅ Order doesn't matter
- ✅ Can predict which 6 numbers appear
- ✅ Natural handling of duplicates (they can't happen)

### Disadvantages
- ❌ Need to pick top-6 from probabilities
- ❌ Doesn't predict bonus ball position correlation
- ❌ Binary loss doesn't know exactly 6 should be picked

---

## Comparison Summary

| Feature | Original | Multi-Output | Transformer | Set Prediction |
|---------|----------|--------------|-------------|----------------|
| **Architecture** | Seq2Seq + Attention | LSTM + 7 heads | Self-Attention | LSTM + Sigmoid |
| **Output Type** | Sequence | 7 classifications | 7 classifications | Multi-label |
| **Mode Collapse** | High risk | Low risk | Low risk | Very low risk |
| **Parameters** | ~500K | ~300K | ~400K | ~200K |
| **Training Speed** | Medium | Fast | Fast | Fast |
| **Conceptual Fit** | Poor | Medium | Medium | Best |
| **Complexity** | High | Low | Medium | Low |

### Recommendation

1. **For learning**: Try all architectures to understand different approaches
2. **For best results**: Use `multi_output` with `--diversity-loss`
3. **Conceptually correct**: `set_prediction` (but harder to train)

---

## Generating Architecture Diagrams

You can generate visual diagrams of the models:

```bash
# Generate all model diagrams
uv run python -c "from model_viz import generate_all_diagrams; generate_all_diagrams()"
```

Or use TensorFlow's built-in visualization:

```python
from tensorflow.keras.utils import plot_model
from models import MultiOutputLotto

model = MultiOutputLotto()
model.build((None, 10, 7))

plot_model(
    model, 
    to_file='model_diagram.png',
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True
)
```

---

## Further Reading

1. **Seq2Seq with Attention**: [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473)
2. **Transformers**: [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. **Multi-Label Classification**: [Zhang & Zhou, 2014](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde13rev.pdf)
