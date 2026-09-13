"""Terminal rendering.

This is the offline coach: with no API key at all, this output alone tells you
what is going wrong, with the deaths that prove it and what to do about it.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Any, List, Optional, Sequence

from ..analysis.detectors import Finding
from ..analysis.report import Report

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    "critical": "\033[97;41m",
    "high": "\033[91m",
    "medium": "\033[93m",
    "low": "\033[94m",
    "strength": "\033[92m",
    "good": "\033[92m",
    "bad": "\033[91m",
    "head": "\033[96m",
}
SEVERITY_LABEL = {
    "critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "MINOR",
    "strength": "STRENGTH",
}


def use_color(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


class Painter:
    def __init__(self, color: bool = True):
        self.color = color

    def __call__(self, text: str, style: str = "") -> str:
        if not self.color or not style:
            return text
        code = COLORS.get(style, style)
        return f"{code}{text}{RESET}"

    def bold(self, text: str) -> str:
        return f"{BOLD}{text}{RESET}" if self.color else text

    def dim(self, text: str) -> str:
        return f"{DIM}{text}{RESET}" if self.color else text


def _width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 110))


def _wrap(text: str, indent: int = 0, width: Optional[int] = None) -> List[str]:
    import textwrap

    body_width = (width or _width()) - indent
    return [
        " " * indent + line
        for line in textwrap.wrap(text, body_width) or [""]
    ]


def _rule(char: str = "─") -> str:
    return char * _width()


def _fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    if unit.startswith("%"):
        return f"{text}%"
    return text


def render_findings(
    findings: Sequence[Finding],
    paint: Painter,
    detail: bool = True,
    limit: Optional[int] = None,
) -> str:
    lines: List[str] = []
    shown = list(findings)[: limit or len(findings)]
    for i, f in enumerate(shown, 1):
        label = SEVERITY_LABEL.get(f.severity, f.severity.upper())
        head = f" {i}. {f.title} "
        lines.append(
            paint(f" {label:8} ", f.severity) + paint(paint.bold(head), "")
        )
        lines.extend(_wrap(f.summary, indent=4))
        if f.value is not None and f.benchmark is not None:
            arrow = "vs"
            lines.append(
                paint.dim(
                    f"    {_fmt(f.value, f.unit)} {arrow} "
                    f"{_fmt(f.benchmark, f.unit)} benchmark"
                    f"  ·  {f.sample}  ·  {f.confidence} confidence"
                )
            )
        elif f.sample:
            lines.append(paint.dim(f"    {f.sample}  ·  {f.confidence} confidence"))
        if detail:
            if f.why:
                lines.append("")
                lines.extend(_wrap(f"Why it costs rounds: {f.why}", indent=4))
            if f.fix:
                lines.append("")
                lines.extend(_wrap(f"What to do: {f.fix}", indent=4))
            if f.evidence:
                lines.append("")
                lines.append(paint.dim("    Evidence:"))
                for item in f.evidence:
                    lines.extend(_wrap(f"· {item}", indent=6))
        lines.append("")
    return "\n".join(lines)


def render_metrics(report: Report, paint: Painter) -> str:
    m = report.metrics
    rows = [
        ("Record", f"{m.match_wins}-{m.matches - m.match_wins}",
         f"{m.match_win_rate}% of matches, {m.round_win_rate}% of rounds"),
        ("K / D / A", f"{m.kills} / {m.deaths} / {m.assists}",
         f"{m.kd} K/D · {m.kda} KDA"),
        ("ADR", f"{m.adr:.0f}", f"{m.damage:,} damage over {m.rounds} rounds"),
        ("Headshot %", f"{m.hs_pct}%", f"{m.headshots} of {m.shots} shots"),
        ("KAST", f"{m.kast_pct}%", f"{m.kast_rounds} of {m.rounds} rounds"),
        ("Opening duels", f"{m.opening_win_rate}%",
         f"{m.first_bloods} won / {m.first_deaths} lost"),
        ("Deaths untraded", f"{m.untraded_death_rate}%",
         f"{m.untraded_deaths} of {m.deaths}"),
        ("Deaths isolated", f"{m.isolated_death_rate}%",
         f"avg {m.avg_nearest_teammate / 100:.0f}m to nearest teammate"),
        ("Avg death time", f"{m.avg_death_time_ms / 1000:.0f}s",
         f"{m.opening_death_rate}% inside the first 15s"),
        ("Utility", f"{m.util_per_round}/round",
         f"{m.no_util_rounds} rounds with none"),
        ("Clutches", f"{m.clutch_wins}/{m.clutch_attempts}",
         f"{m.clutch_rate}% converted"),
        ("Economy", f"{int(m.avg_loadout)}c avg",
         f"{m.saves} saves, {m.gear_lost_credits:,}c of gear lost"),
    ]
    lines = []
    for label, value, note in rows:
        lines.append(
            f"  {label:<16} {paint.bold(str(value)):<22} {paint.dim(note)}"
        )
    return "\n".join(lines)


def render_splits(report: Report, paint: Painter) -> str:
    m = report.metrics
    lines = []
    for title, splits in (
        ("By side", m.by_side), ("By map", m.by_map), ("By agent", m.by_agent),
    ):
        if not splits:
            continue
        items = sorted(splits.values(), key=lambda s: -s.rounds)
        lines.append(paint(f"  {title}", "head"))
        lines.append(
            paint.dim(f"    {'':<12}{'rds':>5}{'win%':>7}{'K/D':>7}{'ADR':>7}{'FD%':>7}")
        )
        for split in items[:6]:
            lines.append(
                f"    {split.label[:12]:<12}{split.rounds:>5}"
                f"{split.win_rate:>7.1f}{split.kd:>7.2f}"
                f"{split.adr:>7.0f}{split.first_death_rate:>7.1f}"
            )
        lines.append("")
    return "\n".join(lines)


def render_deaths(
    deaths: Sequence[Any], paint: Painter, limit: int = 40, group_by_match: bool = True
) -> str:
    lines: List[str] = []
    current = None
    shown = list(deaths)[:limit]
    for death in shown:
        if group_by_match and death.match_id != current:
            current = death.match_id
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(death.started_at))
            lines.append("")
            lines.append(
                paint(f"  {death.map_name} · {when} · {death.match_id[:8]}", "head")
            )
        prefix = "  ✗ "
        style = "bad" if death.first_death_of_round else ""
        lines.append("  " + paint(prefix.strip() + " " + death.describe(), style))
    if len(deaths) > len(shown):
        lines.append(paint.dim(f"\n  … {len(deaths) - len(shown)} more deaths"))
    return "\n".join(lines)


def render_progress(report: Report, paint: Painter) -> str:
    if not report.progress:
        return ""
    lines = [paint("  Since your last review", "head")]
    for line in report.progress:
        if line.delta == 0:
            mark, style = "=", ""
        elif line.improved:
            mark, style = "▲", "good"
        else:
            mark, style = "▼", "bad"
        lines.append(
            f"    {line.label:<22}{line.previous:>8} → {line.current:>8}  "
            + paint(f"{mark} {abs(line.delta):g}", style)
        )
    if report.repeated_findings:
        lines.append("")
        lines.extend(
            _wrap(
                "Still open from last time: " + "; ".join(report.repeated_findings),
                indent=4,
            )
        )
    return "\n".join(lines)


def render_report(
    report: Report,
    color: Optional[bool] = None,
    detail: bool = True,
    death_limit: int = 25,
    show_deaths: bool = True,
) -> str:
    paint = Painter(use_color(color))
    out: List[str] = []
    title = f"VALORANT review · {report.riot_id or report.puuid[:8]}"
    out.append(paint.bold(title))
    subtitle = report.window
    if report.queue_filter:
        subtitle += f" · {report.queue_filter}"
    if report.strict:
        subtitle += " · strict benchmarks"
    out.append(paint.dim(subtitle))
    out.append(_rule())

    if not report.contexts:
        out.append("")
        out.append("  No matches in this window. Run `valcoach sync` first.")
        return "\n".join(out)

    out.append("")
    out.append(paint("THE NUMBERS", "head"))
    out.append(render_metrics(report, paint))
    out.append("")
    out.append(render_splits(report, paint))

    progress = render_progress(report, paint)
    if progress:
        out.append(progress)
        out.append("")

    problems = report.problems
    out.append(_rule())
    out.append("")
    if problems:
        out.append(paint.bold(f"WHAT'S GOING WRONG  ({len(problems)} findings)"))
        out.append("")
        out.append(render_findings(problems, paint, detail=detail))
    else:
        out.append(paint("Nothing crossed a benchmark in this window. "
                         "Play more games or try --strict.", "good"))
        out.append("")

    if report.strengths:
        out.append(paint.bold("WHAT'S WORKING"))
        out.append("")
        out.append(render_findings(report.strengths, paint, detail=False))

    if show_deaths:
        out.append(_rule())
        out.append("")
        out.append(paint.bold(f"HOW YOU DIED  ({len(report.deaths)} deaths)"))
        out.append(render_deaths(report.deaths, paint, limit=death_limit))
        out.append("")

    if report.narrative:
        out.append(_rule())
        out.append("")
        out.append(paint.bold("YOUR COACH"))
        out.append("")
        out.append(report.narrative)
        out.append("")

    if problems:
        out.append(_rule())
        focus = report.focus(3)
        out.append(paint.bold("FOCUS FOR YOUR NEXT SESSION"))
        for i, f in enumerate(focus, 1):
            out.append(f"  {i}. {f.title}")
            out.extend(_wrap(f.fix, indent=5))
        out.append("")
    return "\n".join(out)


def render_watch_line(report: Report, paint: Optional[Painter] = None) -> str:
    """One-line summary printed by ``valcoach watch`` after each new match."""
    paint = paint or Painter(use_color())
    ctx = report.contexts[-1] if report.contexts else None
    if ctx is None:
        return "no new matches"
    focus = report.focus(1)
    verdict = focus[0].title if focus else "nothing flagged"
    return (
        f"{ctx.map_name} {ctx.score[0]}-{ctx.score[1]} "
        f"({'win' if ctx.won else 'loss'}) · "
        f"{len(ctx.kills)}/{len(ctx.deaths)} · "
        f"biggest issue: {verdict}"
    )
