# The dashboard

A page in your browser for playing with every model in the project: change its settings,
get a ticket and a form for the next draw, see the exact odds, test a setting on past draws,
and keep a record of what you predicted.

```bash
uv run lotto ui
```

It opens `http://localhost:6037` (6 of 37) and runs until you press Ctrl-C. Everything happens
on your machine. The only thing it ever contacts is pais.co.il, and only when you press
**Check for new draws**.

| Option | What it does |
|---|---|
| `--port 7000` | use another port |
| `--no-browser` | start the server without opening a browser |
| `--offline` | never contact pais.co.il |

## The four screens

### Predict

Pick a model, change its settings and seed, choose how many rows to fill in, and press
**Generate**.

- **The ticket**: the model's six numbers and strong number.
- **Odds tiles**: the chance of any prize, of three or more with the strong number, of the
  first prize, and the cost. They are exact, and the same for every model: they come from how
  the rows are arranged, not from the numbers in them.
- **The form**: one row per line, with a button to copy them.
- **Why these numbers**: the numbers 1–37 shaded by the model's score, with the chosen ones
  ringed. Point at a number to see its value.
- **Record**: how these exact settings did on the published scoreboard. Change a setting and
  the record goes away, because nothing has measured the new one yet; the Lab can.
- **Save to log** and **Open report**, a page you can print.

Two models take any date, not only the next draw: the date-conditioned model and the seed-world
toy. A past date is a *look back*: the model is given only the draws before that date, the real
result is shown beside its ticket, and it cannot be saved as a prediction.

### Compare

Runs all 19 scoreboard models on the next draw: each one's ticket and record in one table, and
a board showing how many models picked each number. **Open** loads a model into Predict.

### Lab

Replays the settings from Predict over the last 100 to 1,000 draws. Each forecast sees only
earlier draws. You get the mean number of matches against chance (0.973), a p-value, and a chart
of how often it matched 0, 1, 2… numbers beside what chance predicts.

Every setting you try goes into a **notebook** that is kept between visits, and the verdict is
corrected for all of them. Try five windows and the best one has to clear a higher bar. That is
the project's central lesson, and here you can feel it.

### Log

Predictions you saved before a draw. Once the result is in the archive, each one is scored: the
matches in every row, any prize tier hit, and a running total against what chance would give.
Entries cannot be edited or deleted in the dashboard, because a record you can prune is not a
record. A prediction saved on the draw day itself is kept but left out of the totals: the
dashboard cannot tell whether it was saved before the draw.

## Using it from another machine

The dashboard listens only on the machine it runs on. To run it on one machine and view it on
another, forward the port over SSH, then open `http://localhost:6037` as usual:

```bash
ssh -L 6037:localhost:6037 user@other-machine
```

## Where things are kept

Results, the log and the notebook are plain JSON under `results/ui/`, which git ignores. The
dashboard never writes anywhere else, except when a check for new draws adds them to `input/`.

## How it is built

No extra dependencies: Python's built-in web server and one page of plain JavaScript.

| File | What it holds |
|---|---|
| `lotto/ui/registry.py` | every model, its settings and their allowed ranges |
| `lotto/ui/service.py` | predict, compare, backtest and log scoring, all by calling existing code |
| `lotto/ui/jobs.py` | one background worker, so the page stays responsive while a model trains |
| `lotto/ui/storage.py` | the files under `results/ui/` |
| `lotto/ui/server.py` | the web server and its two safety checks |
| `lotto/ui/static/` | the page: `index.html`, `app.css`, `app.js` |

Next: [how results are measured →](../evaluation.md)
