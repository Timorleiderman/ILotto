# The approaches

Every approach in this repository is one of four families. They differ in what they
*believe* about the draw, and that belief is what the diagrams make visible.

```mermaid
flowchart TB
  H["Draw history"] --> S["<b>Statistical</b><br/>counting rules over past draws"]
  H --> N["<b>Neural set prediction</b><br/>learn P(number ∈ next draw)"]
  H --> L["<b>Legacy sequence models</b><br/>learn P(ball at position k)"]
  H --> G["<b>Ticket generators</b><br/>optimise the payout, not the odds"]
  CAL["Calendar date"] --> DC["<b>Date-conditioned generative</b><br/>p(draw | date)"]

  S --> B["the same walk-forward harness"]
  DC --> B
  N --> B
  L --> B
  G --> B
  B --> V["matches vs 0.973 · log loss vs ln 37"]

  class H,CAL data
  class S,N,L,G,DC learn
  class B op
  class V out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

| Family | Believes | Can it beat chance? |
|---|---|---|
| [Statistical](statistical.md) | past frequencies carry information about the next draw | No — and the tests say there is nothing to carry |
| [Neural set prediction](set-prediction.md) | a network can find structure counting misses | No — it converges on the uniform prior, which is correct |
| [Legacy sequence models](legacy-models.md) | the draw is a sequence to be decoded position by position | No — and three of the four optimise a target that leaks sort order |
| [Ticket generators](ticket-generators.md) | *which* numbers you pick changes the payout, not the odds | **Yes, for payout** — the only real edge here |
| [Date-conditioned generative](date-conditioned.md) | the draw is a function of *when* it happened | No — but it reconstructs past draws perfectly, which is an instructive failure |
| [The perfect-fit function](perfect-fit.md) | one function through all past draws, extrapolated | No — three exact fits disagree about the future, and the archive contains 0 of the 23.96 bits/draw that choosing between them requires |
| [Chaos theory](chaos.md) | the machine is deterministic, so dynamics tools can recover structure | No — the premise is *true*, but the reset between draws erases the state Takens' theorem needs; no attractor, analogues at chance, surrogates indistinguishable |

## The one honest edge

Nothing predicts the draw. But the lottery is pari-mutuel: prizes are split among winners.
Players over-pick calendar dates, so numbers above 31 are systematically under-played.

```mermaid
flowchart LR
  T["Any ticket"] --> W["P(win)<br/>identical for<br/>every combination"]
  T --> U["choose under-played<br/>numbers (32–37)"]
  U --> SH["fewer co-winners<br/>if it hits"]
  SH --> EV["higher expected<br/><b>payout</b> given a win"]
  W -.->|"unchanged"| EV

  class T data
  class W op
  class U learn
  class SH op
  class EV out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

This changes expected value without predicting anything. It is a fact about *players*, not
about the machine — and it is the only lever in this repository that moves the number.

---

Start with [how everything is measured](../evaluation.md), or jump to a family above.
