"""Assemble a full review: metrics, findings, death log, and progress over time."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..maps import MapIndex
from ..models import Match
from .context import DeathContext, MatchContext, build_contexts
from .detectors import BENCHMARKS, DetectorInput, Finding, run_detectors
from .metrics import Metrics, add_shot_totals, compute_metrics

# Metrics worth tracking session over session.
TRACKED = (
    ("kd", "K/D", True),
    ("adr", "ADR", True),
    ("hs_pct", "Headshot %", True),
    ("kast_pct", "KAST %", True),
    ("first_death_rate", "First-death rate", False),
    ("untraded_death_rate", "Untraded deaths", False),
    ("isolated_death_rate", "Isolated deaths", False),
    ("opening_win_rate", "Opening duels won", True),
    ("trade_participation", "Trade participation", True),
    ("util_per_round", "Utility per round", True),
    ("round_win_rate", "Round win rate", True),
)


@dataclass
class MatchSummary:
    match_id: str
    map_name: str
    queue: str
    agent: str
    started_at: int
    result: str
    score: str
    kills: int
    deaths: int
    assists: int
    adr: float
    first_deaths: int
    untraded_deaths: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgressLine:
    key: str
    label: str
    current: float
    previous: float
    higher_is_better: bool

    @property
    def delta(self) -> float:
        return round(self.current - self.previous, 2)

    @property
    def improved(self) -> Optional[bool]:
        if self.delta == 0:
            return None
        return self.delta > 0 if self.higher_is_better else self.delta < 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(delta=self.delta, improved=self.improved)
        return data


@dataclass
class Report:
    riot_id: str
    puuid: str
    generated_at: int
    metrics: Metrics
    findings: List[Finding] = field(default_factory=list)
    contexts: List[MatchContext] = field(default_factory=list)
    matches: List[MatchSummary] = field(default_factory=list)
    progress: List[ProgressLine] = field(default_factory=list)
    repeated_findings: List[str] = field(default_factory=list)
    narrative: str = ""
    queue_filter: str = ""
    strict: bool = False

    # ---- views ---------------------------------------------------------
    @property
    def deaths(self) -> List[DeathContext]:
        return [d for c in self.contexts for d in c.deaths]

    @property
    def problems(self) -> List[Finding]:
        return [f for f in self.findings if f.severity != "strength"]

    @property
    def strengths(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "strength"]

    def focus(self, count: int = 3) -> List[Finding]:
        return self.problems[:count]

    @property
    def window(self) -> str:
        if not self.contexts:
            return "no matches"
        first = time.strftime("%Y-%m-%d", time.localtime(self.contexts[0].started_at))
        last = time.strftime("%Y-%m-%d", time.localtime(self.contexts[-1].started_at))
        span = first if first == last else f"{first} → {last}"
        return f"{len(self.contexts)} matches ({span})"

    def to_dict(self, include_deaths: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "riot_id": self.riot_id,
            "puuid": self.puuid,
            "generated_at": self.generated_at,
            "window": self.window,
            "queue_filter": self.queue_filter,
            "strict": self.strict,
            "metrics": self.metrics.summary(),
            "findings": [f.to_dict() for f in self.findings],
            "matches": [m.to_dict() for m in self.matches],
            "progress": [p.to_dict() for p in self.progress],
            "repeated_findings": self.repeated_findings,
            "narrative": self.narrative,
        }
        if include_deaths:
            data["deaths"] = [asdict(d) for d in self.deaths]
        return data


def _match_summaries(contexts: Sequence[MatchContext]) -> List[MatchSummary]:
    out: List[MatchSummary] = []
    for ctx in contexts:
        kills = len(ctx.kills)
        deaths = len(ctx.deaths)
        damage = sum(r.damage for r in ctx.rounds)
        rounds = len(ctx.rounds) or 1
        out.append(
            MatchSummary(
                match_id=ctx.match_id,
                map_name=ctx.map_name,
                queue=ctx.queue,
                agent=ctx.agent,
                started_at=ctx.started_at,
                result=("win" if ctx.won else "loss" if ctx.won is False else "—"),
                score=f"{ctx.score[0]}-{ctx.score[1]}",
                kills=kills,
                deaths=deaths,
                assists=ctx.assists,
                adr=round(damage / rounds, 1),
                first_deaths=sum(1 for d in ctx.deaths if d.first_death_of_round),
                untraded_deaths=sum(1 for d in ctx.deaths if not d.traded),
            )
        )
    return out


def _progress(metrics: Metrics, previous: Optional[Dict[str, Any]]) -> List[ProgressLine]:
    if not previous:
        return []
    old = previous.get("metrics") or {}
    now = metrics.summary()
    lines: List[ProgressLine] = []
    for key, label, higher_better in TRACKED:
        if key not in old or key not in now:
            continue
        try:
            lines.append(
                ProgressLine(
                    key=key, label=label, current=float(now[key]),
                    previous=float(old[key]), higher_is_better=higher_better,
                )
            )
        except (TypeError, ValueError):
            continue
    return lines


def build_report(
    matches: Iterable[Match],
    puuid: str,
    riot_id: str = "",
    trade_window_ms: int = 4000,
    map_index: Optional[MapIndex] = None,
    strict: bool = False,
    benchmarks: Optional[Dict[str, float]] = None,
    previous_reports: Optional[List[Dict[str, Any]]] = None,
    queue_filter: str = "",
) -> Report:
    matches = list(matches)
    contexts = build_contexts(matches, puuid, trade_window_ms, map_index)
    metrics = compute_metrics(contexts, riot_id=riot_id, puuid=puuid)
    add_shot_totals(metrics, matches, puuid)

    findings = run_detectors(
        DetectorInput(
            metrics=metrics,
            contexts=contexts,
            deaths=[d for c in contexts for d in c.deaths],
            rounds=[r for c in contexts for r in c.rounds],
            map_index=map_index,
            benchmarks=dict(benchmarks or BENCHMARKS),
            strict=strict,
        )
    )

    previous = (previous_reports or [None])[0] if previous_reports else None
    repeated: List[str] = []
    if previous:
        previous_ids = {
            f.get("id") for f in (previous.get("findings") or [])
            if f.get("severity") != "strength"
        }
        repeated = [f.title for f in findings if f.id in previous_ids
                    and f.severity != "strength"]

    return Report(
        riot_id=riot_id or metrics.riot_id,
        puuid=puuid,
        generated_at=int(time.time()),
        metrics=metrics,
        findings=findings,
        contexts=contexts,
        matches=_match_summaries(contexts),
        progress=_progress(metrics, previous),
        repeated_findings=repeated,
        queue_filter=queue_filter,
        strict=strict,
    )
