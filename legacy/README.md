# The first attempt (2023–2025)

This folder is an exhibit. It holds the original Keras models and their training script, kept
because their mistakes are the best teaching material in the project. Nothing they print is
evidence about the lottery.

## What is here

| File | What it is |
|---|---|
| `ilotto.py` | The original Seq2Seq model with Bahdanau attention |
| `models.py` | Three later architectures: multi-output heads, a Transformer, a set predictor |
| `train.py` | Training command for all four |
| `helpers.py` | The original data loading and train/test split |
| `metrics.py` | The original evaluation metrics |
| `ARCHITECTURES.md` | Layer-by-layer explanation of the four models |

## What is wrong with it

Each defect is real, is still in the code on purpose, and is fixed in the main package.

| Defect | Where | Why it matters | Done properly in |
|---|---|---|---|
| Rows are filtered by ball value | `helpers.py:55` | Dropping every draw that contains a number above 37 deletes exactly the draws with high numbers, which biases the frequency table downward | `lotto/core/data.py` filters by date |
| Time runs backwards | `helpers.py:102-116` | The archive is newest-first, so each training window predicts an older draw from newer ones | `lotto/core/data.py` sorts chronologically |
| Test windows do not belong to their targets | `helpers.py:110,119-124` | Test inputs come from the end of the file, test targets from the start | `lotto/core/backtest.py` |
| The test set selects the model | `train.py:230` | Checkpointing and early stopping watch the "test" loss, so no untouched data remains | walk-forward refits on history only |
| Targets are sorted balls | `train.py` | Ball 1 is the smallest of six, so per-position accuracy measures order statistics | `lotto/approaches/neural.py` predicts the set |
| Attention context is concatenated twice | `ilotto.py:143,148` | Harmless, but not what the diagram says | not carried over |

`tests/test_legacy_exhibit.py` pins the split defect, so the exhibit cannot be repaired silently
while the pages that describe it stay unchanged.

## Running it

```bash
uv sync --extra neural --extra legacy
uv run python legacy/train.py --model multi_output --epochs 5
```

Run from the repository root; five epochs take a few seconds. Models and plots are written to
`legacy/model/` and a log to `evaluate.log`; neither is tracked. The exhibit reads the committed
archive in `input/` and never downloads; it does still write its own value-filtered copy,
`input/lotto_IL_filtered.csv`, which is git-ignored.
