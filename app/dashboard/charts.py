"""Tiny dependency-free SVG chart helper for the dashboard.

No JS charting library, no CDN - consistent with the project's existing
policy of keeping the runtime dependency surface small (see
app/signals/indicators.py's docstring on why no numpy/pandas), and it means
the equity curve renders even in a fully offline/air-gapped deployment.
"""
from __future__ import annotations

import datetime as dt


def equity_curve_svg(
    points: list[tuple[dt.datetime, float]], *, width: int = 760, height: int = 160
) -> str:
    """Render an equity curve as an inline <svg> polyline.

    Returns an empty string for fewer than 2 points - a single point has no
    line to draw, and the caller should show an empty state instead.
    """
    if len(points) < 2:
        return ""

    values = [v for _, v in points]
    vmin, vmax = min(values), max(values)
    vrange = (vmax - vmin) or 1.0
    pad = 8
    n = len(points)

    def x_of(i: int) -> float:
        return pad + (i / (n - 1)) * (width - 2 * pad)

    def y_of(v: float) -> float:
        return height - pad - ((v - vmin) / vrange) * (height - 2 * pad)

    coords = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, (_, v) in enumerate(points))
    color = "#3ddc97" if values[-1] >= values[0] else "#ff5c7a"
    baseline_y = y_of(points[0][1])

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" role="img" aria-label="equity curve">'
        f'<line x1="{pad}" y1="{baseline_y:.1f}" x2="{width - pad}" y2="{baseline_y:.1f}" '
        f'stroke="#232838" stroke-width="1" stroke-dasharray="4,4" />'
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" />'
        f'</svg>'
    )
