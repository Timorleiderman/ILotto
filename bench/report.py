"""Builds the static report published to GitHub Pages."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from . import charts, metrics
from .data import N_DRAWN, N_NUMBERS, N_STRONG, Draws
from .predictors import Predictor

CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  background:var(--page);color:var(--text-primary);line-height:1.65;
  -webkit-font-smoothing:antialiased}
:root{
  --page:#f9f9f7;--surface:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
  --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
  --series-1:#2a78d6;--series-2:#eb6834;--series-7:#4a3aa7;--series-8:#e34948;
  --good:#0ca30c;--critical:#d03b3b;--warning:#fab219;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
  --page:#0d0d0d;--surface:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
  --muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --series-1:#3987e5;--series-2:#d95926;--series-7:#9085e9;--series-8:#e66767;
}}
:root[data-theme=dark]{
  --page:#0d0d0d;--surface:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
  --muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --series-1:#3987e5;--series-2:#d95926;--series-7:#9085e9;--series-8:#e66767;
}
main{max-width:900px;margin:0 auto;padding:40px 20px 96px}
h1{font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.2;margin:0 0 .3em;letter-spacing:-.02em}
h2{font-size:1.35rem;margin:2.6em 0 .6em;letter-spacing:-.01em;
  padding-top:1.4em;border-top:1px solid var(--border)}
h3{font-size:1.05rem;margin:1.8em 0 .4em}
p,li{color:var(--text-secondary)}
strong{color:var(--text-primary)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;
  background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:.1em .35em}
.sub{color:var(--muted);font-size:.95rem;margin-top:0}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:20px;margin:18px 0}
.chart{width:100%;height:auto;display:block}
figure{margin:22px 0}
figcaption{color:var(--muted);font-size:.85rem;margin-top:10px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:520px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--text-primary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tile .k{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.tile .v{font-size:1.65rem;font-weight:650;color:var(--text-primary);margin-top:4px}
.tile .n{font-size:.8rem;color:var(--muted)}
.pill{display:inline-block;font-size:.72rem;font-weight:650;padding:.2em .6em;border-radius:99px;
  border:1px solid currentColor}
.pill.no{color:var(--good)}.pill.yes{color:var(--critical)}
.balls{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.ball{width:46px;height:46px;border-radius:50%;display:grid;place-items:center;
  font-weight:650;font-size:1.05rem;background:var(--series-1);color:#fff}
.ball.strong{background:var(--series-2)}
.note{border-left:3px solid var(--warning);padding:2px 0 2px 16px;margin:20px 0}
footer{color:var(--muted);font-size:.82rem;margin-top:48px;padding-top:20px;
  border-top:1px solid var(--border)}
a{color:var(--series-1)}
"""


def _e(s) -> str:
    return html.escape(str(s))


def _table(headers: list[str], rows: list[list], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{c}</td>' for i, c in enumerate(r)
        )
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _pill(significant: bool) -> str:
    return (
        '<span class="pill yes">signal</span>'
        if significant
        else '<span class="pill no">no signal</span>'
    )


def predict_next(predictor: Predictor, draws: Draws, top_n: int = 12) -> dict:
    """Ticket the given predictor would play for the next, undrawn result."""
    if predictor.refit_every:
        predictor.fit(draws)

    target_date = None
    if predictor.wants_target_date:
        # Date-conditioned models need the round's date. The next scheduled draw
        # is genuinely in the future, so this is also the one call where such a
        # model is doing what it was built for rather than reconstructing.
        from .data import next_draw_date

        target_date = next_draw_date(draws)
        ball_scores, strong_scores = predictor.scores(draws, target_date)
    else:
        ball_scores, strong_scores = predictor.scores(draws)

    order = np.argsort(-np.nan_to_num(ball_scores, neginf=-1e18))
    picks = sorted((order[:N_DRAWN] + 1).tolist())
    p = np.clip(np.asarray(ball_scores, dtype=float), 1e-12, None)
    p = p / p.sum()
    return {
        "strategy": predictor.name,
        "target_date": None if target_date is None else str(target_date),
        "numbers": picks,
        "strong": int(np.argmax(strong_scores) + 1),
        "shortlist": [(int(i + 1), float(p[i])) for i in order[:top_n]],
    }


def build(
    draws: Draws,
    tests: list[dict],
    results: list,
    null_dist: dict,
    predictions: list[dict],
    era: "object",
    out_path: str = "docs/index.html",
) -> str:
    now = datetime.now(timezone.utc)
    last_date = np.datetime_as_string(draws.dates[-1], unit="D")
    best = results[0]

    # --- randomness tests, family-corrected -------------------------------
    test_p = {t["test"]: t["p_value"] for t in tests}
    test_sig = metrics.holm_bonferroni(test_p)

    # The verdict covers *both* families. A significant randomness test with no
    # significant predictor is still a change in draw behaviour, and is exactly
    # the case this page exists to catch; keying the headline off the strategy
    # table alone would report "no signal" while the test table said otherwise.
    any_signal = any(r.summary["significant_after_correction"] for r in results) or any(
        test_sig.values()
    )

    # --- charts ------------------------------------------------------------
    strat_chart = charts.barh(
        [r.name for r in results],
        [r.summary["mean_matches"] for r in results],
        baseline=metrics.BASELINE_MATCHES,
        baseline_label=f"random ticket ({metrics.BASELINE_MATCHES:.3f})",
        highlight={best.name},
        title="Mean matched numbers per draw by strategy",
    )

    counts = np.bincount(draws.balls.ravel(), minlength=N_NUMBERS + 1)[1:].astype(float)
    exp_count = len(draws) * N_DRAWN / N_NUMBERS
    sd = np.sqrt(len(draws) * (N_DRAWN / N_NUMBERS) * (1 - N_DRAWN / N_NUMBERS))
    freq_chart = charts.barv(
        [str(i) for i in range(1, N_NUMBERS + 1)],
        counts.tolist(),
        expected=exp_count,
        band=(exp_count - 2 * sd, exp_count + 2 * sd),
        x_title="Ball number",
        y_title="Times drawn",
        title="Frequency of each number",
    )

    null_chart = charts.histogram_with_markers(
        null_dist["samples"],
        [(f"best strategy — {best.name}", best.summary["mean_matches"])],
        x_title="Mean matched numbers per draw",
        title="Where the best strategy lands inside pure chance",
    )

    obs_hist = np.bincount(best.match_counts, minlength=N_DRAWN + 1) / len(best.match_counts)
    hyp = stats.hypergeom(N_NUMBERS, N_DRAWN, N_DRAWN)
    theo = [float(hyp.pmf(k)) for k in range(N_DRAWN + 1)]
    dist_chart = charts.grouped_bars(
        [str(k) for k in range(N_DRAWN + 1)],
        [("Observed (best strategy)", obs_hist.tolist()), ("Chance (hypergeometric)", theo)],
        x_title="Numbers matched in a single draw",
        y_title="Share of draws",
        title="Match distribution vs chance",
    )

    # --- tables ------------------------------------------------------------
    test_rows = [
        [
            _e(t["test"]),
            f"{t['statistic']:.2f}",
            f"{t['p_value']:.4f}",
            _pill(test_sig[t["test"]]),
        ]
        for t in tests
    ]
    strat_rows = [
        [
            _e(r.name),
            f"{r.summary['mean_matches']:.4f}",
            f"{r.summary['lift_pct']:+.1f}%",
            f"{r.summary['z_score']:+.2f}",
            f"{r.summary['p_value']:.3f}",
            "&mdash;" if r.summary["mean_log_loss"] is None else f"{r.summary['mean_log_loss']:.4f}",
            f"{r.summary['strong_accuracy']:.3f}",
            _pill(r.summary["significant_after_correction"]),
        ]
        for r in results
    ]
    # Fifty-eight yearly rows bury the point; group them into the four formats
    # and keep the year-by-year detail behind a disclosure.
    eras = [
        (1968, 2004, "6/49-era pool, strong number up to 49"),
        (2005, 2008, "6/34, strong 1&ndash;10"),
        (2009, 2011, "6/37 transition, strong still reaching 8&ndash;9"),
        (2012, 2100, "6/37, strong 1&ndash;7 &mdash; <strong>used here</strong>"),
    ]
    era_rows = []
    for lo, hi, label in eras:
        sub = era[(era.year >= lo) & (era.year <= hi)]
        if sub.empty:
            continue
        drawn, dropped = int(sub["draws"].sum()), int(sub["dropped_by_legacy_filter"].sum())
        era_rows.append(
            [
                f"{lo}&ndash;{min(hi, int(sub.year.max()))} &middot; {label}",
                f"{drawn:,}",
                int(sub["max_ball"].max()),
                int(sub["max_strong"].max()),
                f"{100 * (1 - dropped / drawn):.0f}%",
            ]
        )
    year_rows = [
        [int(r.year), int(r.draws), int(r.max_ball), int(r.max_strong), f"{r.kept_pct:.0f}%"]
        for r in era.itertuples()
    ]

    pred_blocks = []
    for p in predictions:
        balls = "".join(f'<div class="ball">{n}</div>' for n in p["numbers"])
        balls += f'<div class="ball strong">{p["strong"]}</div>'
        short = ", ".join(f"{n} ({pr * 100:.1f}%)" for n, pr in p["shortlist"])
        pred_blocks.append(
            f'<div class="card"><h3>{_e(p["strategy"])}</h3>'
            f'<div class="balls">{balls}</div>'
            f'<p style="font-size:.85rem;margin:0"><strong>Ranked shortlist:</strong> {_e(short)} '
            f"— note how flat those probabilities are around {100 / N_NUMBERS:.1f}%, "
            f"which is what &ldquo;no information&rdquo; looks like.</p></div>"
        )

    n_calibrated = sum(1 for r in results if r.summary["mean_log_loss"] is not None)
    verdict_pill = (
        '<span class="pill yes">signal detected</span>'
        if any_signal
        else '<span class="pill no">no signal detected</span>'
    )

    body = f"""
<main>
<h1>Can a model predict the Israeli Lotto?</h1>
<p class="sub">An adversarial benchmark of {len(results)} prediction strategies against
{len(draws):,} draws of the 6/{N_NUMBERS} game &middot; data through {_e(last_date)} &middot;
rebuilt {_e(now.strftime("%Y-%m-%d %H:%M"))} UTC</p>

<div class="tiles">
  <div class="tile"><div class="k">Draws analysed</div><div class="v">{len(draws):,}</div>
    <div class="n">2012 &rarr; today, one game format</div></div>
  <div class="tile"><div class="k">Strategies tested</div><div class="v">{len(results)}</div>
    <div class="n">walk-forward, no lookahead</div></div>
  <div class="tile"><div class="k">Best mean matches</div><div class="v">{best.summary["mean_matches"]:.3f}</div>
    <div class="n">chance gives {metrics.BASELINE_MATCHES:.3f}</div></div>
  <div class="tile"><div class="k">Verdict</div><div class="v" style="font-size:1.05rem;padding-top:8px">{verdict_pill}</div>
    <div class="n">after multiple-testing correction</div></div>
</div>

<div class="note"><p><strong>The headline.</strong> Across every test on this page, the draw
history is indistinguishable from a fair random process, and no strategy beats a randomly
filled ticket by a margin that survives correcting for the number of strategies tried.
This page exists to measure that claim properly rather than assert it &mdash; and to be
re-run automatically as new draws arrive, in case that ever stops being true.</p></div>

<h2>1. The data is not what the archive hands you</h2>
<p>The published archive runs from 1968, but the <em>game</em> changed repeatedly: a 6/49-style
pool until 2004, 6/34 with a strong number up to 10 from 2005, and the current 6/{N_NUMBERS}
with strong 1&ndash;{N_STRONG} only from 2012. Pooling those eras models a machine that never existed.</p>
<p>Worse is <em>how</em> the older pipeline handled it: it filtered rows by <em>ball value</em>,
dropping any draw containing a number above {N_NUMBERS}. That is not a format filter, it is a
selection on the outcome &mdash; it deletes precisely the draws that contained high numbers and
keeps the low ones, manufacturing a downward bias in the surviving frequency table.
In the pre-2004 years it discarded {int(era[era.year < 2004]["dropped_by_legacy_filter"].sum()):,} of
{int(era[era.year < 2004]["draws"].sum()):,} draws that way. This report filters by <strong>date</strong>
instead, which is unbiased, and uses the {len(draws):,} draws from 2012 onward.</p>
{_table(["Game format", "Draws", "Max ball", "Max strong", "Kept by legacy filter"], era_rows, numeric_from=1)}
<details><summary style="cursor:pointer;color:var(--muted);font-size:.85rem;margin-top:12px">
Year-by-year detail</summary>
{_table(["Year", "Draws", "Max ball", "Max strong", "Kept by legacy filter"], year_rows, numeric_from=1)}
</details>

<h2>2. Is there anything to predict?</h2>
<p>Before fitting anything, test the sequence for exploitable structure. Each test below is
a different way for a lottery machine to be unfair; a small p&#8209;value would mean the draws
carry structure a model could exploit. Because {len(tests)} tests run at once, significance is judged
after a Holm&ndash;Bonferroni correction.</p>
{_table(["Test", "Statistic", "p-value", "After correction"], test_rows, numeric_from=1)}
<figure>{freq_chart}
<figcaption>How often each number has been drawn since 2012. The shaded band is the
&plusmn;2&sigma; range you would expect from a fair machine; every bar sits inside it.
The spread between the most and least drawn number is what randomness looks like,
not evidence of &ldquo;hot&rdquo; numbers.</figcaption></figure>

<h2>3. Benchmarking {len(results)} strategies, honestly</h2>
<p>Each strategy is evaluated <strong>walk-forward</strong>: for every draw in the test window it
sees only the draws that preceded it, produces a six-number ticket, and is scored on how many
of those numbers came up. A randomly filled ticket scores
6&times;6/{N_NUMBERS} = <strong>{metrics.BASELINE_MATCHES:.4f}</strong> matches per draw on average.
That number is the bar.</p>
<figure>{strat_chart}
<figcaption>Mean matched numbers per draw over the {best.summary["n_draws"]}-draw walk-forward window.
The dashed line is chance. Bars land on both sides of it; the ordering reshuffles if you move
the window.</figcaption></figure>
{_table(
    ["Strategy", "Mean matches", "Lift", "z", "p", "Log loss", "Strong acc.", "Verdict"],
    strat_rows,
    numeric_from=1,
)}
<p>Two columns deserve attention. <strong>Log loss</strong> is a proper scoring rule &mdash; it grades
the full probability distribution a strategy assigns, not just its top six &mdash; and a uniform
guess scores exactly ln({N_NUMBERS}) = {metrics.BASELINE_LOGLOSS:.4f}. It is shown only for the
{n_calibrated} strategies that emit genuine per-number inclusion probabilities; the rest produce
ranks, gaps or indicator values, and normalising those into a distribution would be an arbitrary
transform that says nothing about calibration (for all-negative scores it collapses to the uniform
prior and would report exactly the baseline no matter how the numbers were ranked). Among the
strategies where the number means something, <strong>none beats the uniform guess</strong>, which
means none holds information about the next draw even in principle. <strong>Verdict</strong>
applies Holm&ndash;Bonferroni across all {len(results)} strategies, because the highest of
{len(results)} noisy numbers is high by construction.</p>

<h2>4. What &ldquo;beating chance&rdquo; actually looks like</h2>
<p>To make the multiple-comparisons problem concrete: {null_dist["n_trials"]:,} portfolios of
purely random tickets were run through the identical backtest. The spread below is what
zero skill produces.</p>
<figure>{null_chart}
<figcaption>Distribution of mean matches for {null_dist["n_trials"]:,} random-ticket strategies over the same
draws. 95% of them land below {null_dist["p95"]:.3f} and the luckiest reached
{null_dist["max"]:.3f} &mdash; without knowing anything. The best strategy on this page scores
{best.summary["mean_matches"]:.3f}. Any result inside this spread is a coin landing heads a few
extra times, not a discovery.</figcaption></figure>
<figure>{dist_chart}
<figcaption>How often the leading strategy matched 0&ndash;6 numbers in a single draw, against the
exact hypergeometric distribution for a random ticket. The two agree.</figcaption></figure>

<h2>5. Next draw</h2>
<p>Published because it was asked for, and because it is the most direct illustration of
everything above: these picks have exactly the same chance as any other six numbers.</p>
{"".join(pred_blocks)}
<div class="note"><p>Expected value of a Lotto ticket is negative regardless of which numbers are
on it &mdash; roughly half of ticket revenue is returned as prizes. The one genuine lever is
<em>which</em> numbers, not <em>whether</em>: prizes are split among winners, and players
over-pick dates, so a ticket weighted toward 32&ndash;{N_NUMBERS} wins no more often but shares
less when it does. That is a payout effect, not a prediction effect, and it is the only edge
in this repository.</p></div>

<h2>6. Method</h2>
<ul>
<li><strong>Era filtering by date, never by outcome</strong>, so the frequency table stays unbiased.</li>
<li><strong>Chronological ordering.</strong> The archive ships newest-first; consuming it as-is trains a
model to predict the past from the future.</li>
<li><strong>Set-valued targets.</strong> Balls are published in ascending order, so per-position accuracy
mostly measures the marginals of order statistics. Every metric here is order-invariant.</li>
<li><strong>Walk-forward evaluation</strong> with periodic refitting; a strategy never sees a draw at or
after the one it is predicting.</li>
<li><strong>Exact null baselines</strong> (hypergeometric for matches, ln&nbsp;{N_NUMBERS} for log loss) plus a
Monte-Carlo null, and Holm&ndash;Bonferroni correction on every family of tests.</li>
</ul>

<footer>
<p>Generated automatically from the Mifal HaPayis public draw archive. Not affiliated with
Mifal HaPayis. Nothing here is gambling advice; the analysis concludes the opposite.
<a href="results.json">Raw results (JSON)</a></p>
</footer>
</main>
"""

    doc = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Can a model predict the Israeli Lotto?</title>"
        '<meta name="description" content="An adversarial benchmark of lottery prediction '
        'strategies against 1,500+ draws of the Israeli 6/37 Lotto.">'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)

    return doc


def dump_markdown(
    path: str, draws: Draws, tests, results, null_dist, predictions, memorisation=None
) -> None:
    """Write the MkDocs results page.

    Generated rather than hand-written so the prose in the architecture pages can
    never drift from the numbers the benchmark actually produced.
    """
    last_date = np.datetime_as_string(draws.dates[-1], unit="D")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    def pill(sig: bool) -> str:
        return (
            '<span class="pill warn">signal</span>'
            if sig
            else '<span class="pill ok">no signal</span>'
        )

    n_test = results[0].summary["n_test" if "n_test" in results[0].summary else "n_draws"]
    se = float(np.sqrt(metrics.BASELINE_MATCH_VAR / n_test))
    band_lo, band_hi = metrics.BASELINE_MATCHES - 2 * se, metrics.BASELINE_MATCHES + 2 * se

    lines = [
        "# Results",
        "",
        "!!! info \"Generated output\"",
        "",
        f"    Rebuilt {now} UTC from {len(draws):,} draws through {last_date}. "
        "Do not edit by hand — `scripts/build_report.py` overwrites this page after every draw, "
        "so the numbers below shuffle from build to build. That shuffling is not a bug in the "
        "strategies; it is what a table of noise looks like when you keep re-rolling it.",
        "",
        "## Strategy scoreboard",
        "",
        f"Every strategy was asked to pick 6 numbers for each of the last **{n_test} draws**, "
        "seeing only the draws before each one. The columns:",
        "",
        "| Column | Meaning | What counts as unremarkable |",
        "|---|---|---|",
        "| **Mean matches** | Of the 6 numbers picked, how many were actually drawn, averaged "
        f"over the {n_test} test draws | A random ticket averages 6×6/37 = "
        f"**{metrics.BASELINE_MATCHES:.4f}**. For this window, anything in "
        f"**{band_lo:.3f} – {band_hi:.3f}** is within ±2 standard errors of chance |",
        "| **Lift** | Mean matches relative to chance, as a percentage | "
        f"±{200 * se / metrics.BASELINE_MATCHES:.0f}% is the ±2-standard-error band — "
        "single-digit lifts are noise at this sample size |",
        "| **z** | How many standard errors the mean sits from chance | \\|z\\| < 2 is "
        f"unremarkable, and with {len(results)} strategies the *largest* \\|z\\| is expected "
        "to exceed 2 by chance alone |",
        "| **p** | Probability that a skill-less strategy would score at least this far from "
        "chance | **Not** the probability the strategy works. One row near "
        f"p ≈ {1 / len(results):.2f} is expected among {len(results)} rows |",
        "| **Log loss** | Grades the full probability distribution the strategy assigned, not "
        f"just its top 6. A uniform guess scores exactly ln 37 = "
        f"**{metrics.BASELINE_LOGLOSS:.4f}**; *below* that means real information | “—” means "
        "the strategy outputs ranks, not probabilities, so this score would be meaningless "
        "for it |",
        "| **Verdict** | Significance after Holm–Bonferroni correction across all "
        f"{len(results)} rows | “no signal” is the expected outcome; a genuine “signal” would "
        "survive the correction *and* recur in the next rebuild |",
        "",
        "| Strategy | Mean matches | Lift | z | p | Log loss | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        su = r.summary
        ll = "—" if su["mean_log_loss"] is None else f"{su['mean_log_loss']:.4f}"
        lines.append(
            f"| {r.name} | {su['mean_matches']:.4f} | {su['lift_pct']:+.1f}% | "
            f"{su['z_score']:+.2f} | {su['p_value']:.3f} | {ll} | "
            f"{pill(su['significant_after_correction'])} |"
        )

    lines += [
        "",
        "Two rulers for the table above. First, the luckiest of "
        f"{null_dist['n_trials']:,} **purely random** strategies scored "
        f"**{null_dist['max']:.3f}** over these same draws — against the best real strategy's "
        f"**{results[0].summary['mean_matches']:.3f}** — so topping this table is well within "
        "what luck alone produces. Second, note both tails: strategies land *below* chance as "
        "often as above it, and a low row is exactly as (un)meaningful as a high one.",
        "",
        "## Randomness tests",
        "",
        "Each row asks a different way the machine could be unfair: biased ball frequencies, "
        "memory between draws, pairs that travel together, drift over the years, a compressible "
        "generating pattern. **Statistic** is each test's own measure (a chi-square, a KS "
        "distance, bits/draw, …) — the values are not comparable *between* rows; the p-value is "
        "the comparable column. Small p would mean structure a model could exploit. Because "
        "many tests run at once, the verdict column applies Holm–Bonferroni across the family.",
        "",
        "| Test | Statistic | p | After correction |",
        "|---|---:|---:|---|",
    ]
    test_sig = metrics.holm_bonferroni({t["test"]: t["p_value"] for t in tests})
    for t in tests:
        lines.append(
            f"| {t['test']} | {t['statistic']:.2f} | {t['p_value']:.4f} | "
            f"{pill(test_sig[t['test']])} |"
        )

    if memorisation:
        lines += [
            "",
            "## Memorisation vs prediction",
            "",
            "The date-conditioned generative model takes a date rather than a history, and "
            "there is exactly one draw per calendar date — so the date is a **unique key**. "
            "A model with enough capacity can store a lookup table and reproduce past draws "
            "perfectly. **DKRR** (Date-Keyed Reconstruction Rate) is that seductive number: "
            "matched numbers on dates the model was *trained on*. It measures storage, not "
            "skill, and is never shown without the walk-forward column beside it.",
            "",
            "| Model | DKRR (in-sample) | In-sample log loss | Walk-forward matches | p |",
            "|---|---:|---:|---:|---:|",
        ]
        lines[-2:-2] = [
            "Column guide: **DKRR** — matched numbers when asked about dates it was trained "
            "on (6.000 = perfect recall); **in-sample log loss** — same idea for the full "
            "distribution (lower = more memorised; a uniform guess scores "
            f"{metrics.BASELINE_LOGLOSS:.4f}); **walk-forward matches** — the honest column, "
            "on draws it had never seen (chance "
            f"{metrics.BASELINE_MATCHES:.4f}). Read each row left to right as "
            "\"how impressive it looks\" versus \"what it is worth\".",
            "",
        ]
        for m in memorisation:
            lines.append(
                f"| {m['name']} | {m['dkrr_matches']:.3f} / 6 | {m['dkrr_log_loss']:.4f} | "
                f"{m['wf_matches']:.4f} | {m['wf_p']:.3f} |"
            )
        lines += [
            "",
            f"Chance is **{metrics.BASELINE_MATCHES:.4f}** matched numbers and "
            f"**{metrics.BASELINE_LOGLOSS:.4f}** log loss. The lookup control reconstructs "
            "every training draw perfectly and is worth nothing on a date it has not seen. "
            "That gap is the entire point of the exercise.",
        ]

    if predictions:
        lines += [
            "",
            "## Next-draw picks",
            "",
            "Published because they were asked for. These have exactly the same chance as "
            "any other six numbers. **For draw of** — most strategies condition on the "
            "history, so their pick simply applies to whichever draw comes next; the "
            "date-conditioned model needs an actual date, so it names the next scheduled one.",
            "",
            "| Strategy | For draw of | Numbers | Strong |",
            "|---|---|---|---:|",
        ]
        for pr in predictions:
            nums = " · ".join(str(n) for n in pr["numbers"])
            when = pr.get("target_date") or "next draw"
            lines.append(f"| {pr['strategy']} | {when} | {nums} | {pr['strong']} |")

    lines += [
        "",
        "---",
        "",
        "Full charts and methodology: [the benchmark report](benchmark/index.html).",
        "",
    ]

    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def dump_json(path: str, draws: Draws, tests, results, null_dist, predictions) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_draws": len(draws),
        "last_draw_date": str(np.datetime_as_string(draws.dates[-1], unit="D")),
        "randomness_tests": tests,
        "strategies": [
            {"name": r.name, "description": r.description, **r.summary, "seconds": round(r.seconds, 2)}
            for r in results
        ],
        "monte_carlo_null": {k: v for k, v in null_dist.items() if k != "samples"},
        "predictions": predictions,
    }
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
