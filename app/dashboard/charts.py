"""Tiny dependency-free SVG chart helper for the dashboard.

No JS charting library, no CDN - consistent with the project's existing
policy of keeping the runtime dependency surface small (see
app/signals/indicators.py's docstring on why no numpy/pandas), and it means
the equity curve renders even in a fully offline/air-gapped deployment.
"""
from __future__ import annotations

import datetime as dt

# The rendered height in CSS pixels. Exported because the template's
# overlay of tappable markers has to be exactly the same box as the SVG
# for percentage positions to land on the line.
CURVE_HEIGHT = 160
CURVE_WIDTH = 760
CURVE_PAD = 8


def _projector(values: list[float], width: int, height: int, pad: int):
    """x_of/y_of for the curve's viewBox, shared by every renderer here.

    Kept in one place because the marker overlay positions itself over the
    drawn line: if the two derived the scale separately, a later tweak to
    the padding would move the line and leave the hit targets behind.
    """
    vmin, vmax = min(values), max(values)
    vrange = (vmax - vmin) or 1.0
    n = len(values)

    def x_of(i: int) -> float:
        return pad + (i / (n - 1)) * (width - 2 * pad)

    def y_of(v: float) -> float:
        return height - pad - ((v - vmin) / vrange) * (height - 2 * pad)

    return x_of, y_of


def equity_curve_svg(
    points: list[tuple[dt.datetime, float]],
    *,
    width: int = CURVE_WIDTH,
    height: int = CURVE_HEIGHT,
) -> str:
    """Render an equity curve as an inline <svg> polyline.

    Returns an empty string for fewer than 2 points - a single point has no
    line to draw, and the caller should show an empty state instead.
    """
    if len(points) < 2:
        return ""

    values = [v for _, v in points]
    x_of, y_of = _projector(values, width, height, CURVE_PAD)
    pad = CURVE_PAD

    coords = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
    color = "#3ddc97" if values[-1] >= values[0] else "#ff5c7a"
    baseline_y = y_of(values[0])

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" role="img" aria-label="equity curve">'
        f'<line x1="{pad}" y1="{baseline_y:.1f}" x2="{width - pad}" y2="{baseline_y:.1f}" '
        f'stroke="#232838" stroke-width="1" stroke-dasharray="4,4" />'
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" />'
        f'</svg>'
    )


def curve_positions(
    values: list[float], *, width: int = CURVE_WIDTH, height: int = CURVE_HEIGHT
) -> list[tuple[float, float]]:
    """Each point's position as a percentage of the chart box.

    The markers a reader hovers or taps are plain HTML positioned on top of
    the SVG rather than circles inside it, because the chart is drawn with
    preserveAspectRatio="none": the same stretch that lets the line fill
    the panel width would squash any circle in the viewBox into an ellipse.
    Percentages of the box survive that stretch exactly, so the overlay
    stays on the line at every width.

    Returns [] for fewer than 2 points, matching equity_curve_svg - there
    is no line to attach markers to.
    """
    if len(values) < 2:
        return []
    x_of, y_of = _projector(values, width, height, CURVE_PAD)
    return [
        (x_of(i) / width * 100, y_of(v) / height * 100) for i, v in enumerate(values)
    ]
