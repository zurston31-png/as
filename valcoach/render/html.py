"""Standalone HTML report.

One self-contained file: no scripts loaded from anywhere, no fonts fetched, no
network at all. It opens from disk, works at phone width, and follows the
reader's light/dark preference.

The map panels plot every death position, which is the fastest way to see a
habit — clusters jump out of a scatter in a way they never do from a table.
"""

from __future__ import annotations

import html
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..analysis.detectors import Finding
from ..models import ATTACK, DEFENSE
from ..analysis.report import Report

SEVERITY_LABEL = {
    "critical": "Critical", "high": "High", "medium": "Medium", "low": "Minor",
    "strength": "Strength",
}

CSS = """
:root {
  color-scheme: light dark;
  --bg: #f7f4f3;
  --panel: #fffdfc;
  --panel-2: #efeae9;
  --ink: #191415;
  --ink-2: #6b6062;
  --line: #e2dad9;
  --accent: #b4363f;
  --critical: #b4363f;
  --high: #d1662b;
  --medium: #b08a1e;
  --low: #4a6fa5;
  --strength: #2f7a55;
  --attack: #c2603a;
  --defense: #3f7192;
  --radius: 10px;
  --display: "Oswald", "Archivo Narrow", "Roboto Condensed", "Liberation Sans Narrow",
             ui-sans-serif, system-ui, sans-serif;
  --body: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #131011;
    --panel: #1c1819;
    --panel-2: #262021;
    --ink: #efeae9;
    --ink-2: #a59a9c;
    --line: #302a2b;
    --accent: #ff6b74;
    --critical: #ff6b74;
    --high: #ff9a5a;
    --medium: #e8c364;
    --low: #8ab4f8;
    --strength: #6bd6a0;
    --attack: #ff8f66;
    --defense: #74aed6;
  }
}
:root[data-theme="dark"] {
  --bg: #131011;
  --panel: #1c1819;
  --panel-2: #262021;
  --ink: #efeae9;
  --ink-2: #a59a9c;
  --line: #302a2b;
  --accent: #ff6b74;
  --critical: #ff6b74;
  --high: #ff9a5a;
  --medium: #e8c364;
  --low: #8ab4f8;
  --strength: #6bd6a0;
  --attack: #ff8f66;
  --defense: #74aed6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 var(--body);
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1040px; margin: 0 auto; padding-block: 28px; padding-left: 18px; padding-right: 18px; }
header.top { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 26px; }
h1 { font-family: var(--display); font-size: 34px; font-weight: 500; margin: 0 0 4px;
     letter-spacing: 0.005em; text-transform: uppercase; text-wrap: balance; }
h2 { font-family: var(--display); font-size: 15px; text-transform: uppercase;
     letter-spacing: 0.14em; color: var(--ink-2); margin: 38px 0 14px;
     font-weight: 500; display: flex; align-items: center; gap: 12px; }
h2::after { content: ""; flex: 1; height: 1px; background: var(--line); }
h3 { font-size: 17px; margin: 0 0 6px; text-wrap: balance; }
p { margin: 0 0 10px; }
.sub { color: var(--ink-2); font-size: 14px; }
.tiles { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 12px 14px; }
.tile .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.09em;
           color: var(--ink-2); font-weight: 600; }
.tile .v { font-family: var(--display); font-size: 27px; font-weight: 500;
           margin-top: 2px; letter-spacing: 0.01em;
           font-variant-numeric: tabular-nums; }
.tile .n { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
.tile.warn .v { color: var(--high); }
.tile.good .v { color: var(--strength); }
.card { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--line);
        border-radius: var(--radius); padding: 16px 18px; margin-bottom: 12px; }
.card.critical { border-left-color: var(--critical); }
.card.high { border-left-color: var(--high); }
.card.medium { border-left-color: var(--medium); }
.card.low { border-left-color: var(--low); }
.card.strength { border-left-color: var(--strength); }
.badge { display: inline-block; font-family: var(--display); font-size: 12px;
         font-weight: 500; letter-spacing: 0.1em;
         text-transform: uppercase; padding: 2px 7px; border-radius: 5px;
         background: var(--panel-2); color: var(--ink-2); }
.badge.critical { color: #fff; background: var(--critical); }
.badge.high { color: var(--high); }
.badge.medium { color: var(--medium); }
.badge.strength { color: var(--strength); }
.card .meta { font-size: 12px; color: var(--ink-2); margin: 6px 0 10px;
              font-family: var(--mono); font-variant-numeric: tabular-nums; }
.card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em;
               color: var(--ink-2); margin-top: 12px; }
.evidence { list-style: none; padding: 0; margin: 6px 0 0; }
.evidence li { font-family: var(--mono);
               font-size: 12px; color: var(--ink-2); padding: 4px 0 4px 10px;
               border-left: 2px solid var(--line); margin-bottom: 3px; word-break: break-word; }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px;
         font-variant-numeric: tabular-nums; }
th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
     color: var(--ink-2); font-weight: 600; padding: 8px 10px; border-bottom: 1px solid var(--line); }
td { padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
tr:last-child td { border-bottom: none; }
td.wrap-cell { white-space: normal; min-width: 240px; }
.win { color: var(--strength); font-weight: 600; }
.loss { color: var(--critical); font-weight: 600; }
.maps { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.mapcard { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 12px; }
.mapcard h3 { font-size: 14px; margin-bottom: 8px; }
.mapcard svg { width: 100%; height: auto; max-width: 100%; display: block; }
.legend { font-size: 11.5px; color: var(--ink-2); margin-top: 8px; display: flex; gap: 12px; flex-wrap: wrap; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
.bar { height: 8px; background: var(--panel-2); border-radius: 4px; overflow: hidden; margin-top: 5px; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.narrative { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
             padding: 18px 20px; }
.narrative h3 { margin-top: 18px; }
.narrative h3:first-child { margin-top: 0; }
.focus { counter-reset: focus; list-style: none; padding: 0; }
.focus li { background: var(--panel); border: 1px solid var(--line);
            border-radius: var(--radius); padding: 14px 16px 14px 52px;
            margin-bottom: 10px; position: relative; counter-increment: focus; }
.focus li::before { content: counter(focus); position: absolute; left: 16px; top: 13px;
                    font-family: var(--display); font-size: 20px; font-weight: 500;
                    color: var(--accent); }
.focus li b { display: block; margin-bottom: 4px; }
.delta.up { color: var(--strength); }
.delta.down { color: var(--critical); }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line);
         color: var(--ink-2); font-size: 12px; }
.notice { display: flex; gap: 10px; align-items: baseline; background: var(--panel-2);
          border: 1px dashed var(--line); border-radius: var(--radius);
          padding: 11px 14px; margin-bottom: 22px; font-size: 13px;
          color: var(--ink-2); }
.notice b { font-family: var(--display); text-transform: uppercase;
            letter-spacing: 0.1em; color: var(--ink); font-weight: 500;
            white-space: nowrap; }
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
@media (max-width: 560px) {
  h1 { font-size: 26px; }
  .tile .v { font-size: 22px; }
  .wrap { padding-block: 20px; }
}
"""


def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def _tile(key: str, value: Any, note: str = "", tone: str = "") -> str:
    cls = f"tile {tone}".strip()
    return (
        f'<div class="{cls}"><div class="k">{_esc(key)}</div>'
        f'<div class="v">{_esc(value)}</div>'
        + (f'<div class="n">{_esc(note)}</div>' if note else "")
        + "</div>"
    )


def _finding_card(f: Finding) -> str:
    sev = f.severity if f.severity in SEVERITY_LABEL else "low"
    bits = [f'<div class="card {sev}">']
    bits.append(
        f'<span class="badge {sev}">{_esc(SEVERITY_LABEL[sev])}</span>'
        f'<h3 style="margin-top:8px">{_esc(f.title)}</h3>'
    )
    bits.append(f"<p>{_esc(f.summary)}</p>")
    meta = []
    if f.value is not None and f.benchmark is not None:
        unit = f.unit or ""
        meta.append(f"{f.value:g} vs {f.benchmark:g} benchmark ({_esc(unit)})")
    if f.sample:
        meta.append(_esc(f.sample))
    if f.confidence:
        meta.append(f"{_esc(f.confidence)} confidence")
    if meta:
        bits.append(f'<div class="meta">{" · ".join(meta)}</div>')
    if f.why:
        why_label = "Why it matters" if sev == "strength" else "Why it costs rounds"
        bits.append(f'<div class="label">{why_label}</div><p>{_esc(f.why)}</p>')
    if f.fix:
        bits.append(f'<div class="label">What to do</div><p>{_esc(f.fix)}</p>')
    if f.evidence:
        items = "".join(f"<li>{_esc(e)}</li>" for e in f.evidence)
        bits.append(f'<div class="label">Evidence</div><ul class="evidence">{items}</ul>')
    bits.append("</div>")
    return "".join(bits)


def _death_scatter(map_name: str, deaths: Sequence[Any], size: int = 240) -> str:
    """Plot deaths in the map's own coordinate space, normalised to the box."""
    points = [(d.x, d.y, d) for d in deaths if d.x is not None and d.y is not None]
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y) or 1.0
    pad = 18

    def place(x: float, y: float) -> Tuple[float, float]:
        # Riot's y axis runs opposite to SVG's, and the axes are swapped
        # relative to the minimap, so this is a consistent relative view.
        px = pad + (x - min_x) / span * (size - 2 * pad)
        py = size - pad - (y - min_y) / span * (size - 2 * pad)
        return round(px, 1), round(py, 1)

    circles = []
    for x, y, death in points:
        cx, cy = place(x, y)
        color = "var(--attack)" if death.side == ATTACK else (
            "var(--defense)" if death.side == DEFENSE else "var(--ink-2)"
        )
        radius = 6.5 if death.first_death_of_round else 4.5
        title = _esc(death.describe())
        circles.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}" '
            f'fill-opacity="0.62" stroke="{color}" stroke-width="1">'
            f"<title>{title}</title></circle>"
        )
    grid = "".join(
        f'<line x1="{pad}" y1="{pad + i * (size - 2 * pad) / 4}" x2="{size - pad}" '
        f'y2="{pad + i * (size - 2 * pad) / 4}" stroke="var(--line)" stroke-width="1"/>'
        f'<line y1="{pad}" x1="{pad + i * (size - 2 * pad) / 4}" y2="{size - pad}" '
        f'x2="{pad + i * (size - 2 * pad) / 4}" stroke="var(--line)" stroke-width="1"/>'
        for i in range(5)
    )
    attack = sum(1 for _, _, d in points if d.side == ATTACK)
    defense = sum(1 for _, _, d in points if d.side == DEFENSE)
    return (
        f'<div class="mapcard"><h3>{_esc(map_name)} '
        f'<span class="sub">· {len(points)} deaths</span></h3>'
        f'<svg viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="Death positions on {_esc(map_name)}">'
        f"{grid}{''.join(circles)}</svg>"
        f'<div class="legend">'
        f'<span><i class="dot" style="background:var(--attack)"></i>attack {attack}</span>'
        f'<span><i class="dot" style="background:var(--defense)"></i>defense {defense}</span>'
        f"<span>larger = first death of the round</span></div></div>"
    )


def _markdown_lite(text: str) -> str:
    """Render the coach's markdown headings, lists and bold text."""
    out: List[str] = []
    in_list = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        if line.startswith("### "):
            body, tag = line[4:], "h3"
        elif line.startswith("## "):
            body, tag = line[3:], "h3"
        elif line.startswith("# "):
            body, tag = line[2:], "h3"
        else:
            body, tag = line, ""
        stripped = body.strip()
        bullet = stripped.startswith(("- ", "* ", "• "))
        numbered = len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ")."
        content = _esc(stripped[2:].strip() if (bullet or numbered) else stripped)
        # Inline bold / italic, applied after escaping.
        for marker, html_tag in (("**", "strong"), ("__", "strong")):
            while content.count(marker) >= 2:
                content = content.replace(marker, f"<{html_tag}>", 1)
                content = content.replace(marker, f"</{html_tag}>", 1)
        content = content.replace("`", "")
        if tag:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<{tag}>{content}</{tag}>")
        elif bullet or numbered:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{content}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{content}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


FONT_LINK = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Oswald:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&"
    'family=IBM+Plex+Mono:wght@400&display=swap">'
)


def render_html(
    report: Report,
    title: Optional[str] = None,
    notice: str = "",
    webfonts: bool = False,
) -> str:
    """Render a self-contained report page.

    ``notice`` puts a labelled banner at the top (the demo uses it to say the
    data is synthetic). ``webfonts`` links the display faces from Google Fonts;
    off by default so the file stays usable with no network at all.
    """
    m = report.metrics
    who = report.riot_id or report.puuid[:8] or "player"
    page_title = title or f"{who} · VALORANT review"
    generated = time.strftime("%d %b %Y, %H:%M", time.localtime(report.generated_at))

    tiles = [
        _tile("Record", f"{m.match_wins}-{m.matches - m.match_wins}",
              f"{m.round_win_rate}% of rounds"),
        _tile("K/D", f"{m.kd}", f"{m.kills} / {m.deaths} / {m.assists}"),
        _tile("ADR", f"{m.adr:.0f}", f"{m.rounds} rounds"),
        _tile("Headshot %", f"{m.hs_pct}%", f"{m.headshots} of {m.shots} shots",
              "good" if m.hs_pct >= 22 else ("warn" if m.hs_pct < 18 else "")),
        _tile("KAST", f"{m.kast_pct}%", "rounds you affected",
              "warn" if m.kast_pct < 65 else ""),
        _tile("Opening duels", f"{m.opening_win_rate}%",
              f"{m.first_bloods} won / {m.first_deaths} lost",
              "warn" if m.opening_win_rate < 50 else "good"),
        _tile("Deaths untraded", f"{m.untraded_death_rate}%",
              f"{m.untraded_deaths} of {m.deaths}",
              "warn" if m.untraded_death_rate > 60 else ""),
        _tile("Avg death time", f"{m.avg_death_time_ms / 1000:.0f}s",
              f"{m.opening_death_rate}% in first 15s"),
        _tile("Utility", f"{m.util_per_round}/rd", f"{m.no_util_rounds} empty rounds",
              "warn" if m.util_per_round < 1.6 else ""),
        _tile("Clutches", f"{m.clutch_wins}/{m.clutch_attempts}",
              f"{m.clutch_rate}% converted"),
    ]

    # Splits
    split_tables = []
    for label, splits in (("Side", m.by_side), ("Map", m.by_map), ("Agent", m.by_agent)):
        if not splits:
            continue
        rows = "".join(
            f"<tr><td>{_esc(s.label)}</td><td>{s.rounds}</td>"
            f"<td>{s.win_rate:.1f}%</td><td>{s.kd:.2f}</td><td>{s.adr:.0f}</td>"
            f"<td>{s.first_death_rate:.1f}%</td></tr>"
            for s in sorted(splits.values(), key=lambda s: -s.rounds)[:8]
        )
        split_tables.append(
            f'<div class="table-scroll"><table><thead><tr><th>{_esc(label)}</th>'
            f"<th>Rounds</th><th>Win %</th><th>K/D</th><th>ADR</th>"
            f"<th>First death %</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    # Matches
    match_rows = "".join(
        f"<tr><td>{time.strftime('%d %b %H:%M', time.localtime(s.started_at))}</td>"
        f"<td>{_esc(s.map_name)}</td><td>{_esc(s.agent)}</td>"
        f'<td class="{"win" if s.result == "win" else "loss"}">{_esc(s.result)}</td>'
        f"<td>{_esc(s.score)}</td><td>{s.kills}/{s.deaths}/{s.assists}</td>"
        f"<td>{s.adr:.0f}</td><td>{s.first_deaths}</td><td>{s.untraded_deaths}</td></tr>"
        for s in sorted(report.matches, key=lambda s: -s.started_at)
    )

    # Death log
    death_rows = "".join(
        f"<tr><td>{_esc(d.map_name)}</td><td>R{d.round_index + 1}</td>"
        f"<td>{d.time_in_round_ms / 1000:.0f}s</td><td>{_esc(d.side)}</td>"
        f"<td>{_esc(d.killer_name)}</td><td>{_esc(d.weapon)}</td>"
        f"<td>{_esc(d.place)}</td>"
        f'<td>{"yes" if d.traded else "no"}</td>'
        f'<td>{"yes" if d.first_death_of_round else ""}</td>'
        f"<td>{d.loadout_value}</td></tr>"
        for d in sorted(report.deaths, key=lambda d: (-d.started_at, d.round_index))
    )

    # Maps scatter
    by_map: Dict[str, List[Any]] = {}
    for death in report.deaths:
        by_map.setdefault(death.map_name or "unknown", []).append(death)
    scatters = "".join(
        _death_scatter(name, deaths)
        for name, deaths in sorted(by_map.items(), key=lambda kv: -len(kv[1]))
    )

    progress_html = ""
    if report.progress:
        rows = ""
        for line in report.progress:
            cls = "" if line.improved is None else ("up" if line.improved else "down")
            arrow = "=" if line.improved is None else ("▲" if line.improved else "▼")
            rows += (
                f"<tr><td>{_esc(line.label)}</td><td>{line.previous:g}</td>"
                f"<td>{line.current:g}</td>"
                f'<td class="delta {cls}">{arrow} {abs(line.delta):g}</td></tr>'
            )
        progress_html = (
            "<h2>Since your last review</h2>"
            f'<div class="table-scroll"><table><thead><tr><th>Metric</th><th>Then</th>'
            f"<th>Now</th><th>Change</th></tr></thead><tbody>{rows}"
            "</tbody></table></div>"
        )
        if report.repeated_findings:
            still = "; ".join(_esc(t) for t in report.repeated_findings)
            progress_html += f'<p class="sub">Still open from last time: {still}</p>'

    narrative_html = ""
    if report.narrative:
        narrative_html = (
            '<h2>Your coach</h2><div class="narrative">'
            + _markdown_lite(report.narrative)
            + "</div>"
        )

    focus_html = ""
    if report.problems:
        items = "".join(
            f"<li><b>{_esc(f.title)}</b>{_esc(f.fix)}</li>" for f in report.focus(3)
        )
        focus_html = f'<h2>Focus for your next session</h2><ul class="focus">{items}</ul>'

    problems = "".join(_finding_card(f) for f in report.problems)
    strengths = "".join(_finding_card(f) for f in report.strengths)

    subtitle = _esc(report.window)
    if report.queue_filter:
        subtitle += f" · {_esc(report.queue_filter)}"

    notice_html = (
        f'<div class="notice"><b>Note</b><span>{_esc(notice)}</span></div>'
        if notice else ""
    )

    return f"""<title>{_esc(page_title)}</title>
{FONT_LINK if webfonts else ""}
<style>{CSS}</style>
<div class="wrap">
{notice_html}
<header class="top">
  <h1>{_esc(who)} — what to fix</h1>
  <div class="sub">{subtitle} · generated {_esc(generated)}</div>
</header>

<h2>The numbers</h2>
<div class="tiles">{''.join(tiles)}</div>

{progress_html}

{focus_html}

{narrative_html}

<h2>What's going wrong</h2>
{problems or '<p class="sub">Nothing crossed a benchmark in this window.</p>'}

{"<h2>What's working</h2>" + strengths if strengths else ""}

<h2>Where you die</h2>
<div class="maps">{scatters or '<p class="sub">No position data available.</p>'}</div>

<h2>Splits</h2>
{''.join(split_tables)}

<h2>Matches</h2>
<div class="table-scroll"><table><thead><tr><th>When</th><th>Map</th><th>Agent</th>
<th>Result</th><th>Score</th><th>K/D/A</th><th>ADR</th><th>First deaths</th>
<th>Untraded</th></tr></thead><tbody>{match_rows}</tbody></table></div>

<h2>Every death</h2>
<div class="table-scroll"><table><thead><tr><th>Map</th><th>Round</th><th>Time</th>
<th>Side</th><th>Killed by</th><th>Weapon</th><th>Where</th><th>Traded</th>
<th>First death</th><th>Loadout</th></tr></thead><tbody>{death_rows}</tbody></table></div>

<footer>
  Generated by valcoach from {m.matches} matches ({m.rounds} rounds,
  {m.deaths} deaths). Benchmarks are guidelines for ranked play, not absolutes.
</footer>
</div>
"""
