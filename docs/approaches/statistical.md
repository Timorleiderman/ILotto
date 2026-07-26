# Statistical strategies

Eleven counting rules, in `bench/predictors.py`. None of them trains anything — each is a
deterministic function from history to a 37-vector of scores. That makes them the cleanest
possible test of lottery folklore: if "hot numbers" meant anything, this is where it would
show up.

<div class="role-key">
  <span class="k-data"><i></i>data</span>
  <span class="k-op"><i></i>fixed operation</span>
  <span class="k-out"><i></i>output</span>
</div>

## The shared contract

Every strategy plugs into the same socket, which is why they are directly comparable.

```mermaid
flowchart LR
  H["history<br/>(n draws)"] --> M["multi_hot<br/>(n, 37)"]
  M --> R["scoring rule<br/><i>this is what differs</i>"]
  R --> S["ball_scores<br/>(37,)"]
  S --> T["argsort → top 6"]
  T --> O["ticket"]

  class H,M,S data
  class R,T op
  class O out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

---

## Hot numbers

**Believes:** numbers drawn often will keep being drawn often.

Counts appearances over a window (all-time, or the last 100 draws) and picks the six
highest.

\[
\text{score}(b) = \sum_{t \in \text{window}} \mathbb{1}[b \in \text{draw}_t]
\]

```mermaid
flowchart LR
  W["last W draws"] --> C["bincount over<br/>37 numbers"]
  C --> S["counts<br/>(37,)"]
  S --> P["top 6"]
  class W data
  class C,P op
  class S out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

Counts normalise to a genuine categorical distribution, so this strategy **does** get a log
loss. It scores worse than uniform.

## Cold numbers

**Believes:** numbers that are behind must catch up.

Identical machinery, negated. That negation is why it gets no log loss — see
[the scoring rules](../evaluation.md#the-two-scoring-rules) — and it is a good illustration of
why a metric has to be earned rather than computed.

## EWMA frequency

**Believes:** recent draws matter more than old ones.

A draw \(k\) draws ago carries weight \(0.5^{k/h}\) with half-life \(h = 50\). It smoothly
interpolates between "hot recently" and "hot all-time", which means it also spans the whole
family of window choices a person might have tuned to.

```mermaid
flowchart LR
  H["history"] --> A["age of each draw"]
  A --> W["w = 0.5^(age / 50)"]
  H --> M["multi_hot (n, 37)"]
  M --> X["weighted sum<br/>Σ w · multi_hot"]
  W --> X
  X --> S["scores (37,)"]
  class H,M,W,A data
  class X op
  class S out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

## Overdue

**Believes:** the longer a number has been absent, the more it is "due" — the gambler's
fallacy, stated precisely so it can be measured.

Scores each number by draws elapsed since its last appearance. Included because it is the
most common thing real players actually do.

## Pairwise co-occurrence (Markov)

**Believes:** the machine has mechanical memory — some balls travel together.

Builds a 37×37 co-occurrence matrix over all history, then scores each number by how often
it appeared *alongside* the numbers in the most recent draw.

```mermaid
flowchart LR
  M["multi_hot<br/>(n, 37)"] --> CO["Mᵀ M + α<br/>(37, 37)"]
  CO --> Z["zero the diagonal"]
  L["last draw<br/>6 numbers"] --> SEL["select those 6 rows"]
  Z --> SEL
  SEL --> SUM["column sum"]
  SUM --> S["scores (37,)"]
  class M,L,CO data
  class Z,SEL,SUM op
  class S out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

The corresponding hypothesis test — every one of the 666 pairs, Šidák-corrected — is in the
[randomness suite](../evaluation.md), and it does not fire.

## Lag-1 persistence

**Believes:** whether a number appeared last time changes its odds this time.

Distinct from the pairwise version: that one is co-occurrence *within* a draw, this is
persistence *across* draws. For each number independently it estimates two smoothed
conditional probabilities and picks whichever applies.

\[
\hat p_{\text{in}}(b) = \frac{\#\{b \in t-1 \wedge b \in t\} + \alpha}{\#\{b \in t-1\} + 2\alpha}
\qquad
\hat p_{\text{out}}(b) = \frac{\#\{b \notin t-1 \wedge b \in t\} + \alpha}{\#\{b \notin t-1\} + 2\alpha}
\]

These are real inclusion probabilities, so this strategy is scored on log loss too. Both
estimates land on \(6/37\) to within noise, which is the whole story.

## Sum-targeted combination

**Believes:** nothing, deliberately. It is a control.

Draw sums cluster near 114, so this picks a combination summing near the mode. That changes
*which* ticket you hold, never the probability it wins — which is exactly the point of
including it next to strategies that claim more.

## Unpopular numbers

**Believes:** other players are predictable, even though the machine is not.

The only strategy here whose logic is sound. See
[ticket generators](ticket-generators.md) for the full treatment.

## Rank ensemble

**Believes:** averaging independent signals helps.

Averages the normalised ranks of the seven statistical members. Ensembling genuinely helps
when the components carry independent information — so this is a useful check. If the parts
are noise, the average is noise, and that is what the numbers show.

```mermaid
flowchart LR
  A["Hot"] --> R["normalised rank"]
  B["EWMA"] --> R
  C["Overdue"] --> R
  D["Markov"] --> R
  E["… 7 members"] --> R
  R --> M["mean rank"]
  M --> S["scores (37,)"]
  class A,B,C,D,E data
  class R,M op
  class S out
  classDef data fill:none,stroke:#8a8a85,stroke-width:1.5px
  classDef learn fill:#2a78d626,stroke:#2a78d6,stroke-width:2px
  classDef op fill:none,stroke:#8a8a85,stroke-width:1.5px,stroke-dasharray:4 3
  classDef out fill:#199e7026,stroke:#199e70,stroke-width:2.5px
  classDef flaw fill:#e3494826,stroke:#e34948,stroke-width:3px
```

## Random ticket

**Believes:** correctly.

Six uniform picks. It is in the table as a full participant, and it has finished *above*
several of the strategies above. That is not a bug in those strategies — it is the size of
the noise, made visible.

---

Measured results for all of these: [Results](../results.md).
