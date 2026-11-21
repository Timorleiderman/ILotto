"""A completed result as a page to print or as Markdown. Rendering only."""

from __future__ import annotations

import html

from lotto.tickets.draw_plan import ROW_PRICE

CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b;background:#fff;line-height:1.5;
  max-width:820px;margin:32px auto;padding:0 20px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.05rem;margin:26px 0 8px;border-top:1px solid #ddd;padding-top:16px}
p,li{color:#333}.sub{color:#666;font-size:.92rem;margin:0}
table{border-collapse:collapse;width:100%;font-size:.92rem}thead{display:table-header-group}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #ddd}th{color:#666;font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.ball{display:inline-grid;place-items:center;width:34px;height:34px;border-radius:50%;background:#2162ae;color:#fff;
  font-weight:650;margin-right:5px}.ball.s{background:#ad431b}
.note{border-left:3px solid #fab219;padding:4px 0 4px 12px;color:#444}
.fine{color:#555;font-size:.8rem;margin-top:28px;overflow-wrap:anywhere}
@media print{body{margin:0;max-width:none}.ball{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
tr{break-inside:avoid}
"""


def _e(value) -> str:
    return html.escape(str(value))


def _balls(row: list[int]) -> str:
    return ("".join(f'<span class="ball">{n}</span>' for n in row[:6])
            + f'<span aria-label="Strong number {row[6]}"> strong <span class="ball s">{row[6]}</span></span>')


def _numbers(row: list[int]) -> str:
    return " ".join(f"{n:2d}" for n in row[:6])


def _percent(p: float) -> str:
    return f"{100 * p:.5f}%" if p < 0.001 else f"{100 * p:.2f}%"


def _settings(result: dict) -> list[tuple[str, str]]:
    rows = [(k.replace("_", " "), "all" if v is None else ", ".join(v) if isinstance(v, list) else str(v))
            for k, v in result["params"].items()]
    if result["seed"] is not None:
        rows.append(("seed", result["seed"]))
    return rows


def _mode_note(result: dict) -> str:
    if result["mode"] == "retrospective" and not result["benchmark"]:
        return f"A look back at {result['target_date']}, a date that has already passed. This is not a prediction."
    if result["mode"] == "retrospective":
        return (f"A look back: the model was given only the {result['history_draws']:,} draws before "
                f"{result['target_date']}. This is not a prediction made in advance.")
    return f"A forecast for the draw expected on {result['target_date']}, from draws through {result['trained_through']}."


def _odds_rows(odds: dict) -> list[tuple[str, float]]:
    return [("3 or more numbers (any prize)", odds["p_at_least"]["3"]), ("4 or more", odds["p_at_least"]["4"]),
            ("5 or more", odds["p_at_least"]["5"]), ("all 6", odds["p_at_least"]["6"]),
            ("3 or more with the strong number", odds["p_three_plus_and_strong"]),
            ("all 6 and the strong number (first prize)", odds["p_first_prize"])]


def _reference(result: dict) -> str:
    ref = result["reference"]
    if ref is None:
        return "These exact settings have not been measured on the published scoreboard."
    return (f"Published scoreboard, from draws through {ref['archive_last_date']}: {ref['mean_matches']:.3f} matched "
            f"numbers per draw over {ref['n_draws']} draws (a random ticket averages 0.973); "
            f"{'significant' if ref['significant'] else 'not significant'} after correction for "
            f"{ref['strategies_compared']} strategies. This is a preset reference: the published file has no "
            "complete configuration manifest, so an exact settings match is unverified.")


def _provenance(result: dict) -> str:
    archive = result["archive"]
    text = (f"Result {result['id']} · archive of {archive['draws']:,} draws through {archive['last_date']} "
            f"· archive SHA-256 {archive['sha256']} · schema {result.get('schema', 1)} "
            f"· algorithm SHA-256 {result.get('algorithm', 'not recorded')}")
    if result.get("design_sha256"):
        text += f" · design SHA-256 {result['design_sha256']}"
    if result.get("trained_through"):
        text += f" · training cutoff {result['trained_through']}"
    return text


def prediction_html(result: dict) -> str:
    """One prediction as a printable page."""
    odds = result["odds"]
    settings = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in _settings(result)) \
        or '<tr><td colspan="2">This model has no settings.</td></tr>'
    form = "".join(f'<tr><td class="n">{i}</td><td>{_balls(row)}</td></tr>' for i, row in enumerate(result["form"], 1))
    table = "".join(f'<tr><td>{_e(k)}</td><td class="n">{_percent(p)}</td><td class="n">{round(1 / p):,}</td></tr>'
                    for k, p in _odds_rows(odds))
    actual = f"<p>Actual result that day: {_balls(result['actual'])}</p>" if result.get("actual") else ""
    kind = "Sampled ticket" if result["ticket_kind"] == "sampled" else "Best single ticket"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ILotto prediction, {_e(result['target_date'])}</title><style>{CSS}</style></head><body>
<h1>{_e(result['model_label'])}</h1>
<p class="sub">Draw of {_e(result['target_date'])} · {result['rows']} row{'s' if result['rows'] != 1 else ''} ·
generated {_e(result['generated_utc'])} UTC</p>
<p class="note">{_e(_mode_note(result))}</p>
<h2>{kind}</h2><p>{_balls(result['ticket'])}</p>{actual}
<h2>The form</h2><table><thead><tr><th class="n">Row</th><th>Numbers and strong number</th></tr></thead><tbody>{form}</tbody></table>
<h2>Exact odds of this form under a fair draw</h2>
<table><thead><tr><th>At least one row matches</th><th class="n">Chance</th><th class="n">1 in</th></tr></thead><tbody>{table}</tbody></table>
<p class="sub">Estimated base cost {odds['cost']} NIS at {ROW_PRICE} NIS a row. A prize is not necessarily a profit. The odds come from how
the rows are arranged, so they are the same whichever model fills the form.</p>
<h2>Settings</h2><table><tbody>{settings}</tbody></table>
<h2>Record</h2><p>{_e(_reference(result))}</p>
<p class="fine">{_e(_provenance(result))} · ILotto dashboard</p>
</body></html>"""


def prediction_markdown(result: dict) -> str:
    """One prediction as Markdown, with the same content as the page."""
    odds = result["odds"]
    lines = [f"# {result['model_label']}", "",
             f"Draw of {result['target_date']} · {result['rows']} row{'s' if result['rows'] != 1 else ''} · "
             f"generated {result['generated_utc']} UTC", "",
             _mode_note(result), "",
             f"**{'Sampled ticket' if result['ticket_kind'] == 'sampled' else 'Best single ticket'}:** "
             f"`{_numbers(result['ticket'])}` strong {result['ticket'][6]}", ""]
    if result.get("actual"):
        lines += [f"Actual result that day: `{_numbers(result['actual'])}` strong {result['actual'][6]}", ""]
    lines += ["## The form", "", "| Row | Numbers | Strong |", "|---:|---|---:|"]
    lines += [f"| {i} | `{_numbers(row)}` | {row[6]} |" for i, row in enumerate(result["form"], 1)]
    lines += ["", "## Exact odds of this form under a fair draw", "", "| At least one row matches | Chance | 1 in |",
              "|---|---:|---:|"]
    lines += [f"| {k} | {_percent(p)} | {round(1 / p):,} |" for k, p in _odds_rows(odds)]
    lines += ["", f"Estimated base cost {odds['cost']} NIS at {ROW_PRICE} NIS a row. "
              "A prize is not necessarily a profit.", "", "## Settings", ""]
    lines += [f"- {k}: {v}" for k, v in _settings(result)] or ["- none"]
    lines += ["", "## Record", "", _reference(result), "", _provenance(result), ""]
    return "\n".join(lines)


def comparison_html(result: dict) -> str:
    """Every model's ticket for one draw as a printable page."""
    rows = "".join(
        f"<tr><td>{_e(r['name'])}<br><small>{_e(_comparison_settings(r))}</small></td>"
        f"<td>{_balls(r['ticket']) if r['state'] == 'complete' else _e(r.get('message', r['state']))}</td>"
        f"<td class='n'>{'—' if r['reference_mean'] is None else format(r['reference_mean'], '.3f')}</td></tr>"
        for r in result["rows"])
    agreed = [str(i + 1) for i, c in enumerate(result["consensus"]) if c >= 5]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ILotto comparison, {_e(result['target_date'])}</title><style>{CSS}</style></head><body>
<h1>Every model, draw of {_e(result['target_date'])}</h1>
<p class="sub">Generated {_e(result['generated_utc'])} UTC from {result['archive']['draws']:,} draws through
{_e(result['archive']['last_date'])}.</p>
<table><thead><tr><th>Model</th><th>Best single ticket</th><th class="n">Scoreboard</th></tr></thead><tbody>{rows}</tbody></table>
<p class="sub">Scoreboard: mean matched numbers per draw on the published benchmark; a random ticket averages 0.973.
These are preset references with unverified configuration matches.</p>
<h2>Agreement</h2><p>{len(agreed)} numbers were picked by five or more of the {result['counted']} models:
{_e(', '.join(agreed) or 'none')}. Agreement describes the models, which mostly read the same history; it is not confidence.</p>
<p class="fine">{_e(_provenance(result))} · ILotto dashboard</p></body></html>"""


def _comparison_settings(row: dict) -> str:
    parts = [
        f"{key.replace('_', ' ')}: {'all' if value is None else ', '.join(value) if isinstance(value, list) else value}"
        for key, value in row.get("params", {}).items()
    ]
    if row.get("seed") is not None:
        parts.append(f"seed: {row['seed']}")
    if row.get("reference"):
        parts.append(f"reference through {row['reference']['archive_last_date']}")
    return "; ".join(parts)


def comparison_markdown(result: dict) -> str:
    """Every model's ticket for one draw as Markdown."""
    lines = [f"# Every model, draw of {result['target_date']}", "",
             f"Generated {result['generated_utc']} UTC from {result['archive']['draws']:,} draws through "
             f"{result['archive']['last_date']}.", "", "| Model | Best single ticket | Strong | Scoreboard |",
             "|---|---|---:|---:|"]
    for r in result["rows"]:
        score = "—" if r["reference_mean"] is None else f"{r['reference_mean']:.3f}"
        name = r["name"] + (f" ({_comparison_settings(r)})" if _comparison_settings(r) else "")
        if r["state"] == "complete":
            lines.append(f"| {name} | `{_numbers(r['ticket'])}` | {r['ticket'][6]} | {score} |")
        else:
            lines.append(f"| {name} | {r.get('message', r['state'])} | | {score} |")
    agreed = [str(i + 1) for i, count in enumerate(result["consensus"]) if count >= 5]
    lines += ["", "Scoreboard: mean matches; chance 0.973. Preset references have unverified configuration matches.",
              "", "## Agreement", "",
              f"{len(agreed)} numbers picked by five or more of {result['counted']} models: "
              f"{', '.join(agreed) or 'none'}. Agreement is descriptive, not confidence.", "", _provenance(result)]
    return "\n".join(lines) + "\n"


def render(result: dict, markdown: bool) -> tuple[str, str]:
    """(content type, body) for a stored result."""
    if result["kind"] == "prediction":
        body = prediction_markdown(result) if markdown else prediction_html(result)
    else:
        body = comparison_markdown(result) if markdown else comparison_html(result)
    return ("text/markdown; charset=utf-8" if markdown else "text/html; charset=utf-8"), body
