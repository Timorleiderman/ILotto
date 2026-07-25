"""Hand-rolled inline SVG charts.

GitHub Pages serves these under a strict-ish, dependency-free setup and the
report must stay a single self-contained file, so no plotting library is used.
Colors come in as CSS custom properties (`var(--series-1)` etc.) so the page's
light/dark theming applies to the marks without regenerating anything.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

MUTED = "var(--muted)"
GRID = "var(--grid)"
INK = "var(--text-primary)"
INK2 = "var(--text-secondary)"


def _esc(s) -> str:
    return html.escape(str(s))


@dataclass
class Box:
    w: int = 720
    h: int = 380
    left: int = 190
    right: int = 60
    top: int = 16
    bottom: int = 44

    @property
    def pw(self) -> int:
        return self.w - self.left - self.right

    @property
    def ph(self) -> int:
        return self.h - self.top - self.bottom


def _open(box: Box, label: str) -> list[str]:
    return [
        f'<svg viewBox="0 0 {box.w} {box.h}" role="img" aria-label="{_esc(label)}" '
        f'preserveAspectRatio="xMidYMid meet" class="chart">'
    ]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    step = min([m * mag for m in (1, 2, 2.5, 5, 10)], key=lambda s: abs(s - raw))
    start = (lo // step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


def barh(
    labels: list[str],
    values: list[float],
    *,
    baseline: float | None = None,
    baseline_label: str = "baseline",
    highlight: set[str] | None = None,
    value_fmt: str = "{:.3f}",
    title: str = "",
) -> str:
    """Horizontal bars for magnitude comparison across named strategies.

    Every bar is one series (identity is carried by the axis label, not by
    color), so a legend would be noise; the only color split is the highlight,
    which is explained in the caption.
    """
    highlight = highlight or set()
    # Left gutter sized for the longest strategy name; SVG text is not clipped
    # but it does run off the viewBox, so this has to be generous.
    box = Box(w=780, left=max(190, int(7.0 * max(len(s) for s in labels)) + 16),
              h=max(200, 34 * len(labels) + 70))
    hi = max(max(values), baseline or 0) * 1.12
    lo = 0.0
    out = _open(box, title or "strategy comparison")

    def x(v: float) -> float:
        return box.left + (v - lo) / (hi - lo) * box.pw

    for t in _nice_ticks(lo, hi):
        px = x(t)
        out.append(
            f'<line x1="{px:.1f}" y1="{box.top}" x2="{px:.1f}" y2="{box.top + box.ph}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{px:.1f}" y="{box.top + box.ph + 20}" fill="{MUTED}" font-size="12" '
            f'text-anchor="middle">{t:g}</text>'
        )

    bar_h = 18
    step = box.ph / max(len(labels), 1)
    for i, (lab, val) in enumerate(zip(labels, values)):
        cy = box.top + step * (i + 0.5)
        color = "var(--series-2)" if lab in highlight else "var(--series-1)"
        w = max(x(val) - box.left, 0.5)
        out.append(
            f'<rect x="{box.left}" y="{cy - bar_h / 2:.1f}" width="{w:.1f}" height="{bar_h}" '
            f'rx="4" fill="{color}"><title>{_esc(lab)}: {value_fmt.format(val)}</title></rect>'
        )
        out.append(
            f'<text x="{box.left - 10}" y="{cy + 4:.1f}" fill="{INK2}" font-size="12.5" '
            f'text-anchor="end">{_esc(lab)}</text>'
        )
        out.append(
            f'<text x="{x(val) + 8:.1f}" y="{cy + 4:.1f}" fill="{INK}" font-size="12.5" '
            f'font-variant-numeric="tabular-nums">{value_fmt.format(val)}</text>'
        )

    if baseline is not None:
        px = x(baseline)
        out.append(
            f'<line x1="{px:.1f}" y1="{box.top - 4}" x2="{px:.1f}" y2="{box.top + box.ph + 4}" '
            f'stroke="var(--series-8)" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        out.append(
            f'<text x="{px:.1f}" y="{box.top + box.ph + 38}" fill="var(--series-8)" font-size="12" '
            f'text-anchor="middle">{_esc(baseline_label)}</text>'
        )

    out.append("</svg>")
    return "".join(out)


def barv(
    labels: list[str],
    values: list[float],
    *,
    expected: float | None = None,
    band: tuple[float, float] | None = None,
    x_title: str = "",
    y_title: str = "",
    title: str = "",
) -> str:
    """Vertical bars over an ordered numeric domain (the 37 ball numbers)."""
    box = Box(w=760, h=300, left=54, right=16, top=18, bottom=48)
    hi = max(values) * 1.14
    out = _open(box, title or "distribution")

    def y(v: float) -> float:
        return box.top + box.ph - (v / hi) * box.ph

    if band is not None:
        y0, y1 = y(band[1]), y(band[0])
        out.append(
            f'<rect x="{box.left}" y="{y0:.1f}" width="{box.pw}" height="{max(y1 - y0, 1):.1f}" '
            f'fill="var(--series-1)" opacity="0.10"/>'
        )

    for t in _nice_ticks(0, hi):
        py = y(t)
        out.append(
            f'<line x1="{box.left}" y1="{py:.1f}" x2="{box.left + box.pw}" y2="{py:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{box.left - 8}" y="{py + 4:.1f}" fill="{MUTED}" font-size="11" '
            f'text-anchor="end">{t:g}</text>'
        )

    slot = box.pw / len(values)
    bw = max(slot - 2, 1)
    for i, (lab, v) in enumerate(zip(labels, values)):
        px = box.left + slot * i + 1
        out.append(
            f'<rect x="{px:.1f}" y="{y(v):.1f}" width="{bw:.1f}" height="{box.top + box.ph - y(v):.1f}" '
            f'rx="3" fill="var(--series-1)"><title>{_esc(lab)}: {v:g}</title></rect>'
        )
        if len(values) <= 40 and i % 3 == 0:
            out.append(
                f'<text x="{px + bw / 2:.1f}" y="{box.top + box.ph + 16}" fill="{MUTED}" '
                f'font-size="10" text-anchor="middle">{_esc(lab)}</text>'
            )

    if expected is not None:
        py = y(expected)
        out.append(
            f'<line x1="{box.left}" y1="{py:.1f}" x2="{box.left + box.pw}" y2="{py:.1f}" '
            f'stroke="var(--series-8)" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        out.append(
            f'<text x="{box.left + box.pw}" y="{py - 6:.1f}" fill="var(--series-8)" font-size="11.5" '
            f'text-anchor="end">expected if fair ({expected:.0f})</text>'
        )

    if x_title:
        out.append(
            f'<text x="{box.left + box.pw / 2:.1f}" y="{box.h - 8}" fill="{INK2}" font-size="12" '
            f'text-anchor="middle">{_esc(x_title)}</text>'
        )
    if y_title:
        out.append(
            f'<text transform="translate(14,{box.top + box.ph / 2:.1f}) rotate(-90)" fill="{INK2}" '
            f'font-size="12" text-anchor="middle">{_esc(y_title)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def grouped_bars(
    labels: list[str],
    series: list[tuple[str, list[float]]],
    *,
    x_title: str = "",
    y_title: str = "",
    title: str = "",
) -> str:
    """Two-series grouped bars — observed against theoretical, side by side."""
    box = Box(w=720, h=346, left=62, right=16, top=18, bottom=82)
    hi = max(max(v) for _, v in series) * 1.16
    colors = ["var(--series-1)", "var(--series-2)"]
    out = _open(box, title or "grouped comparison")

    def y(v: float) -> float:
        return box.top + box.ph - (v / hi) * box.ph

    for t in _nice_ticks(0, hi):
        py = y(t)
        out.append(
            f'<line x1="{box.left}" y1="{py:.1f}" x2="{box.left + box.pw}" y2="{py:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{box.left - 8}" y="{py + 4:.1f}" fill="{MUTED}" font-size="11" '
            f'text-anchor="end">{t:g}</text>'
        )

    slot = box.pw / len(labels)
    bw = (slot - 10) / len(series)
    for gi, (sname, vals) in enumerate(series):
        for i, v in enumerate(vals):
            px = box.left + slot * i + 5 + gi * (bw + 2)
            out.append(
                f'<rect x="{px:.1f}" y="{y(v):.1f}" width="{max(bw - 2, 1):.1f}" '
                f'height="{box.top + box.ph - y(v):.1f}" rx="3" fill="{colors[gi]}">'
                f"<title>{_esc(sname)} — {_esc(labels[i])}: {v:.4f}</title></rect>"
            )
    for i, lab in enumerate(labels):
        out.append(
            f'<text x="{box.left + slot * (i + 0.5):.1f}" y="{box.top + box.ph + 18}" fill="{MUTED}" '
            f'font-size="11.5" text-anchor="middle">{_esc(lab)}</text>'
        )

    lx = box.left
    for gi, (sname, _) in enumerate(series):
        out.append(f'<rect x="{lx}" y="{box.h - 20}" width="11" height="11" rx="3" fill="{colors[gi]}"/>')
        out.append(
            f'<text x="{lx + 16}" y="{box.h - 10}" fill="{INK2}" font-size="12">{_esc(sname)}</text>'
        )
        lx += 26 + 7.2 * len(sname)

    if x_title:
        out.append(
            f'<text x="{box.left + box.pw / 2:.1f}" y="{box.top + box.ph + 40}" fill="{INK2}" '
            f'font-size="12" text-anchor="middle">{_esc(x_title)}</text>'
        )
    if y_title:
        out.append(
            f'<text transform="translate(16,{box.top + box.ph / 2:.1f}) rotate(-90)" fill="{INK2}" '
            f'font-size="12" text-anchor="middle">{_esc(y_title)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def histogram_with_markers(
    samples, markers: list[tuple[str, float]], *, x_title: str = "", title: str = ""
) -> str:
    """Monte-Carlo null distribution with the observed strategies marked on it.

    This is the chart that does the most work in the report: it shows where the
    best backtested strategy lands inside the spread of pure chance.
    """
    import numpy as np

    box = Box(w=720, h=344, left=54, right=20, top=22, bottom=94)
    counts, edges = np.histogram(samples, bins=36)
    hi = counts.max() * 1.1
    lo_x, hi_x = float(edges[0]), float(edges[-1])
    for _, mv in markers:
        lo_x, hi_x = min(lo_x, mv), max(hi_x, mv)
    pad = (hi_x - lo_x) * 0.04
    lo_x, hi_x = lo_x - pad, hi_x + pad

    out = _open(box, title or "null distribution")

    def x(v: float) -> float:
        return box.left + (v - lo_x) / (hi_x - lo_x) * box.pw

    def y(v: float) -> float:
        return box.top + box.ph - (v / hi) * box.ph

    for t in _nice_ticks(lo_x, hi_x, 6):
        out.append(
            f'<text x="{x(t):.1f}" y="{box.top + box.ph + 18}" fill="{MUTED}" font-size="11" '
            f'text-anchor="middle">{t:.2f}</text>'
        )

    for i, c in enumerate(counts):
        x0, x1 = x(float(edges[i])), x(float(edges[i + 1]))
        out.append(
            f'<rect x="{x0:.1f}" y="{y(float(c)):.1f}" width="{max(x1 - x0 - 2, 1):.1f}" '
            f'height="{box.top + box.ph - y(float(c)):.1f}" rx="2" fill="var(--series-1)" '
            f'opacity="0.75"><title>{edges[i]:.3f}–{edges[i + 1]:.3f}: {c}</title></rect>'
        )

    out.append(
        f'<line x1="{box.left}" y1="{box.top + box.ph:.1f}" x2="{box.left + box.pw}" '
        f'y2="{box.top + box.ph:.1f}" stroke="var(--axis)" stroke-width="1"/>'
    )

    colors = ["var(--series-2)", "var(--series-7)", "var(--series-8)"]
    ly = box.h - 26
    for mi, (name, mv) in enumerate(markers):
        col = colors[mi % len(colors)]
        px = x(mv)
        out.append(
            f'<line x1="{px:.1f}" y1="{box.top - 6}" x2="{px:.1f}" y2="{box.top + box.ph:.1f}" '
            f'stroke="{col}" stroke-width="2"/>'
        )
        out.append(f'<rect x="{box.left}" y="{ly - 9}" width="11" height="11" rx="3" fill="{col}"/>')
        out.append(
            f'<text x="{box.left + 16}" y="{ly}" fill="{INK2}" font-size="12">'
            f"{_esc(name)} ({mv:.3f})</text>"
        )
        ly += 17

    if x_title:
        out.append(
            f'<text x="{box.left + box.pw / 2:.1f}" y="{box.top + box.ph + 38}" fill="{INK2}" '
            f'font-size="12" text-anchor="middle">{_esc(x_title)}</text>'
        )
    out.append("</svg>")
    return "".join(out)
