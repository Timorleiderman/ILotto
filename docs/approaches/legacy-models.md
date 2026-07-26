# The four legacy models

In `models.py` and `ilotto.py`, trained via `train.py --model <name>`. All four take the
same input — 10 previous draws as `(batch, 10, 7)` integer ball IDs — and, except for the
last, emit `(batch, 7, 37)`: a probability distribution per output position.

The [architecture deep dive](../reference/architectures.md) covers each layer in prose. This
page is the visual summary, plus the one structural property that decides how to read their
scores.

<div class="role-key">
  <span class="k-data"><i></i>tensors</span>
  <span class="k-learn"><i></i>learned weights</span>
  <span class="k-op"><i></i>fixed operation</span>
  <span class="k-out"><i></i>output</span>
  <span class="k-flaw"><i></i>leaks sort order</span>
</div>

!!! warning "Three of the four optimise a target that leaks"

    `original`, `multi_output` and `transformer` all supervise softmaxes against balls in
    **ascending order**. Their reported top-k accuracy is inflated by the order statistics
    of sorting, as explained in [set prediction](set-prediction.md#why-sigmoid-and-not-softmax).
    Only `set_prediction` avoids this. It is the one whose numbers can be read at face value.

---

## 1. Original — Seq2Seq with attention

The starting point: an encoder–decoder over per-position embeddings, with attention between
decoder and encoder states.

```mermaid
flowchart TB
  I["input (B, 10, 7)"] --> E["7 separate Embeddings<br/>one per ball position"]
  E --> SD["SpatialDropout1D ×7"]
  SD --> CC["concat → (B, 10, 7·64)"]
  CC --> EN["Bidirectional LSTM stack<br/>128 → 64 → 32"]
  EN --> ST["final states h, c"]
  ST --> RV["RepeatVector(7)"]
  RV --> DE["LSTM decoder stack<br/>(seeded with encoder states)"]
  EN --> AT["MultiHeadAttention<br/>8 heads"]
  DE --> AT
  AT --> CAT["concat(attended, decoder)"]
  CAT --> DN["Dense 128 → 64"]
  DN --> O["Dense 37, softmax<br/>(B, 7, 37)"]
  O --> L["sparse categorical CE<br/>vs <b>sorted</b> balls"]

  class I,CC,ST data
  class E,EN,DE,AT,DN,O learn
  class SD,RV,CAT op
  class L flaw
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

**Known failure mode: mode collapse.** With one shared output layer applied across all
positions, the cheapest way to reduce loss is to emit nearly the same distribution
everywhere. The measured result in `ARCHITECTURES.md` was 0.24 matches against a 0.97
baseline — *worse* than random, which is the signature of collapse rather than of a weak
signal.

## 2. Multi-output — one head per position

The direct fix for collapse: give each of the seven positions its own Dense head so they
cannot share a single degenerate solution.

```mermaid
flowchart TB
  I["input (B, 10, 7)"] --> SP["split: 6 main | 1 bonus"]
  SP --> EM["Embedding(37, 32)<br/>shared across main balls"]
  SP --> EB["Embedding(7, 32)<br/>bonus"]
  EM --> FL["reshape + concat<br/>(B, 10, 224)"]
  EB --> FL
  FL --> L1["LSTM 64, return_sequences"]
  L1 --> L2["LSTM 64"]
  L2 --> DO["Dropout 0.3"]
  DO --> H1["Dense 37 softmax — ball 1"]
  DO --> H2["Dense 37 softmax — ball 2"]
  DO --> H6["… ball 6"]
  DO --> HB["Dense 7 softmax — strong"]
  H1 --> S["stack → (B, 7, 37)"]
  H2 --> S
  H6 --> S
  HB --> PAD["zero-pad 7 → 37"]
  PAD --> S
  S --> L["CE vs <b>sorted</b> balls"]

  class I,FL data
  class EM,EB,L1,L2,H1,H2,H6,HB learn
  class SP,DO,PAD,S op
  class L flaw
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

Independent heads do prevent collapse — but each head now learns its own order statistic
very well, which is exactly the artefact. Solving collapse made the leak *cleaner*, not
smaller. This is the architecture the old README recommended.

## 3. Transformer — self-attention over flattened draws

Flattens the window into 70 tokens (10 draws × 7 balls) and runs standard pre-LN
transformer blocks over them.

```mermaid
flowchart TB
  I["input (B, 10, 7)"] --> R["reshape → (B, 70)"]
  R --> NE["number Embedding(37, 64)"]
  R --> PE["position Embedding(100, 64)"]
  NE --> ADD["token + position"]
  PE --> ADD
  ADD --> TB["2 × transformer block"]
  TB --> MHA["MultiHeadAttention<br/>4 heads, key_dim 64"]
  MHA --> N1["LayerNorm(x + attn)"]
  N1 --> FF["Dense 128 gelu → Dense 64"]
  FF --> N2["LayerNorm(x + ffn)"]
  N2 --> GP["GlobalAveragePooling1D"]
  GP --> DN["Dense 256 gelu → Dropout"]
  DN --> HH["6 × Dense 37 softmax<br/>+ Dense 7 softmax"]
  HH --> O["stack → (B, 7, 37)"]
  O --> L["CE vs <b>sorted</b> balls"]

  class I,R,ADD data
  class NE,PE,MHA,FF,DN,HH learn
  class TB,N1,N2,GP,O op
  class L flaw
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

Every ball attends to every other ball across all ten draws — the most expressive of the
four. It also lands at the baseline, which is the informative part: capacity is not the
binding constraint when there is nothing to learn.

## 4. Set prediction — the sound one

The only legacy architecture that treats a draw as a set.

```mermaid
flowchart TB
  I["input (B, 10, 7)"] --> EM["Embedding(37, 32)"]
  EM --> RS["reshape → (B, 10, 224)"]
  RS --> LS["LSTM 128"]
  LS --> DN["Dense 64 relu → Dropout"]
  DN --> MO["Dense 37, <b>sigmoid</b>"]
  DN --> BO["Dense 7, softmax"]
  MO --> L["binary cross-entropy<br/>vs multi-hot set"]
  MO --> TK["top-k → 6 numbers"]

  class I,RS data
  class EM,LS,DN,MO,BO learn
  class TK op
  class L,TK out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

Same insight as [`bench/nn.py`](set-prediction.md), reached independently. The differences
are that `bench` adds recency features, base-rate bias initialisation, and — crucially — a
walk-forward harness that scores it against an exact null.

## DiversityLoss

A custom loss used to fight mode collapse in the position-softmax models:

\[
\mathcal{L} = \text{CE}(y, \hat y) + \lambda \cdot \underbrace{\overline{\sum_b \hat y_b \cdot \bar{y}_b}}_{\text{similarity to the mean prediction}}
\]

with \(\lambda = 0.1\). It penalises every position for resembling the average across
positions, pushing the heads apart.

```mermaid
flowchart LR
  P["predictions (B, 7, 37)"] --> AVG["mean over positions"]
  P --> DOT["Σ ŷ · mean"]
  AVG --> DOT
  DOT --> PEN["penalty × 0.1"]
  P --> CE["cross-entropy"]
  CE --> SUM["total loss"]
  PEN --> SUM

  class P data
  class AVG,DOT,CE op
  class PEN,SUM out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

It treats the symptom. Collapse in the original model comes from a shared output layer over
positions whose *true* distributions genuinely differ — which is itself a consequence of
supervising on sorted targets. Predicting the set removes the cause, so the set-prediction
model needs no diversity term at all.

## Side by side

| Model | Encoder | Output | Target | Order-invariant |
|---|---|---|---|---|
| Original | BiLSTM 128/64/32 + MHA | `(7, 37)` softmax | sorted balls | **no** |
| Multi-output | LSTM 64 ×2 | 7 independent softmaxes | sorted balls | **no** |
| Transformer | 2 × MHA blocks over 70 tokens | 7 independent softmaxes | sorted balls | **no** |
| Set prediction | LSTM 128 | 37 sigmoids | multi-hot set | **yes** |
| `bench` GRU / Transformer | GRU 96/64 or MHA | 37 sigmoids + 7 softmax | multi-hot set | **yes** |

---

Full layer-by-layer prose: [Architecture deep dive](../reference/architectures.md).
Measured results: [Results](../results.md).
