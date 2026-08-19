"""The scanner funnel, built from the append-only pipeline event log.

Answers the question that matters when the bot is quiet:

    "Why did 500 discovered tokens produce only 3 trades?"

The old funnel counted RiskEvent rows, which only exist for rejections that
someone remembered to log, and could not report which individual pre-screen
threshold did the work. This reads app/pipeline.py's stage events instead,
so every stage has both a numerator and a denominator and the conversion
between them is arithmetic rather than inference.

Two rules keep the numbers honest.

CONVERSION IS MEASURED AGAINST WHAT REACHED THE STAGE, not against the top
of the funnel. "2% of discovered tokens got a technical score" and "80% of
the tokens that reached scoring passed it" describe completely different
systems, and only the second tells you whether the score threshold is the
bottleneck.

A STAGE NOBODY REACHED IS NOT A STAGE THAT REJECTED EVERYTHING. Zero
entries means no data, and it is reported as such rather than as a 0%
pass rate - a distinction that decides whether you go and change a
threshold or go and fix the scanner.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.pipeline import STAGE_ORDER, TERMINAL_STAGES


@dataclass
class StageStats:
    stage: str
    entered: int
    passed: int

    @property
    def terminal(self) -> bool:
        """EXIT is an outcome, not a filter.

        Its `passed` flag records whether the trade was PROFITABLE, so
        reading it as a conversion rate would report the loss rate as a
        rejection rate.
        """
        return self.stage in TERMINAL_STAGES

    @property
    def rejected(self) -> int:
        return self.entered - self.passed

    @property
    def pass_rate(self) -> float | None:
        """Share of tokens REACHING this stage that cleared it.

        None when nothing reached it - which is a different fact from 0%
        and must not be rendered as one.

        On a terminal stage this is the WIN RATE, not a conversion rate.
        `meaning` below says which.
        """
        return (self.passed / self.entered) if self.entered else None

    @property
    def meaning(self) -> str:
        return "profitable" if self.terminal else "advanced"

    @property
    def reached(self) -> bool:
        return self.entered > 0

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "entered": self.entered,
            "passed": self.passed,
            "rejected": self.rejected,
            "pass_rate_pct": round(self.pass_rate * 100, 1) if self.pass_rate is not None else None,
            "reached": self.reached,
            "terminal": self.terminal,
            "meaning": self.meaning,
        }


@dataclass
class PrescreenBreakdown:
    """How many tokens each individual pre-screen threshold rejected.

    The single most useful table for tuning the scanner: "480 failed
    liquidity, 15 failed volume, 2 failed age" tells you exactly which
    number to look at, where "497 failed pre-screen" tells you nothing.

    Counts overlap on purpose - a token can fail several checks at once,
    and hiding that would make each threshold look more decisive than it
    is.
    """

    evaluated: int = 0
    passed_by_check: dict[str, int] = field(default_factory=dict)
    failed_by_check: dict[str, int] = field(default_factory=dict)

    def pass_rate(self, check: str) -> float | None:
        if not self.evaluated:
            return None
        return self.passed_by_check.get(check, 0) / self.evaluated

    def as_dict(self) -> dict:
        checks = sorted(set(self.passed_by_check) | set(self.failed_by_check))
        return {
            "evaluated": self.evaluated,
            "checks": [
                {
                    "name": name,
                    "passed": self.passed_by_check.get(name, 0),
                    "failed": self.failed_by_check.get(name, 0),
                    "pass_rate_pct": (
                        round(self.pass_rate(name) * 100, 1) if self.pass_rate(name) is not None else None
                    ),
                }
                for name in checks
            ],
        }


@dataclass
class StageFunnel:
    window_hours: float | None
    stages: list[StageStats] = field(default_factory=list)
    prescreen: PrescreenBreakdown = field(default_factory=PrescreenBreakdown)
    rejection_reasons: list[tuple[str, str, int]] = field(default_factory=list)  # (stage, reason, count)
    strategy_versions: dict[str, int] = field(default_factory=dict)

    def stage(self, name: str) -> StageStats | None:
        return next((s for s in self.stages if s.stage == name), None)

    @property
    def discovered(self) -> int:
        first = self.stage(STAGE_ORDER[0])
        return first.entered if first else 0

    @property
    def positions_opened(self) -> int:
        opened = self.stage("OPEN_POSITION")
        return opened.passed if opened else 0

    @property
    def overall_conversion(self) -> float | None:
        """Discovered -> position, end to end. None when nothing was seen."""
        if not self.discovered:
            return None
        return self.positions_opened / self.discovered

    @property
    def bottleneck(self) -> StageStats | None:
        """The FILTER stage that rejected the largest NUMBER of tokens.

        Deliberately the count and not the rate: a stage that rejects 100%
        of the four tokens that reached it is not the reason 500 became 3.

        Terminal stages are excluded. EXIT's "failures" are losing trades,
        and every position reaching it has already been opened - calling it
        the pipeline's bottleneck would be both wrong and actively
        misleading, since the fix it implies (loosen the filter) has nothing
        to do with the problem it names.
        """
        candidates = [s for s in self.stages if s.reached and s.rejected > 0 and not s.terminal]
        return max(candidates, key=lambda s: s.rejected) if candidates else None

    @property
    def exit_stats(self) -> StageStats | None:
        """The EXIT stage, reported separately from the funnel proper."""
        return next((s for s in self.stages if s.terminal and s.reached), None)

    def explain(self) -> str:
        """One-paragraph answer to 'why so few trades?'."""
        if not self.discovered:
            return (
                "No tokens were discovered in this window. The scanner is not seeing "
                "candidates at all - check SCANNER_ENABLED and the discovery sources' health "
                "before touching any threshold."
            )
        neck = self.bottleneck
        opened = self.positions_opened
        head = (
            f"{self.discovered} tokens discovered, {opened} position(s) opened "
            f"({self.overall_conversion * 100:.2f}% conversion)."
        )
        if neck is None:
            return head + " No filter rejected anything - every discovered token is still in flight."
        text = (
            head
            + f" The largest single loss is at {neck.stage}, which rejected {neck.rejected} of the "
            f"{neck.entered} tokens that reached it"
            + (f" ({(1 - neck.pass_rate) * 100:.0f}%)." if neck.pass_rate is not None else ".")
        )
        exits = self.exit_stats
        if exits:
            text += (
                f" Separately, {exits.passed} of {exits.entered} closed position(s) were "
                "profitable - an outcome, not a funnel stage."
            )
        return text

    def as_dict(self) -> dict:
        return {
            "window_hours": self.window_hours,
            "discovered": self.discovered,
            "positions_opened": self.positions_opened,
            "overall_conversion_pct": (
                round(self.overall_conversion * 100, 3) if self.overall_conversion is not None else None
            ),
            "bottleneck": self.bottleneck.stage if self.bottleneck else None,
            "exit": self.exit_stats.as_dict() if self.exit_stats else None,
            "explain": self.explain(),
            "stages": [s.as_dict() for s in self.stages],
            "prescreen": self.prescreen.as_dict(),
            "rejection_reasons": [
                {"stage": stage, "reason": reason, "count": count}
                for stage, reason, count in self.rejection_reasons
            ],
            "strategy_versions": dict(self.strategy_versions),
        }


def _shorten(reason: str, limit: int = 70) -> str:
    """Collapse a reason to its category.

    Reasons embed symbols, dollar amounts and scores, so grouping on the
    raw string would produce one 'category' per event. Truncating at the
    first number keeps 'liquidity $12,431 below scanner minimum' and
    'liquidity $801 below scanner minimum' in one bucket, which is the
    bucket a human wants to count.
    """
    out = []
    for ch in reason:
        if ch.isdigit():
            break
        out.append(ch)
    collapsed = "".join(out).strip(" ,.:;-")
    collapsed = collapsed or reason
    return collapsed[:limit]


def build_stage_funnel(
    db: Session, *, window_hours: float | None = 24.0, strategy_version: str | None = None
) -> StageFunnel:
    """Count every stage over the window, from the pipeline event log."""
    query = db.query(models.PipelineEvent)
    if window_hours is not None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
        query = query.filter(models.PipelineEvent.occurred_at >= cutoff)
    if strategy_version is not None:
        query = query.filter(models.PipelineEvent.strategy_version == strategy_version)
    events = query.all()

    entered: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    reasons: Counter[tuple[str, str]] = Counter()
    prescreen = PrescreenBreakdown()

    for event in events:
        entered[event.stage] += 1
        if event.passed:
            passed[event.stage] += 1
        else:
            reasons[(event.stage, _shorten(event.reason or "no reason recorded"))] += 1
        versions[event.strategy_version or "unversioned"] += 1

        if event.stage == "PRESCREEN":
            prescreen.evaluated += 1
            for check in (event.detail or {}).get("checks", []):
                name = check.get("name", "unknown")
                bucket = prescreen.passed_by_check if check.get("passed") else prescreen.failed_by_check
                bucket[name] = bucket.get(name, 0) + 1

    stages = [StageStats(name, entered.get(name, 0), passed.get(name, 0)) for name in STAGE_ORDER]

    return StageFunnel(
        window_hours=window_hours,
        stages=stages,
        prescreen=prescreen,
        rejection_reasons=[(stage, reason, count) for (stage, reason), count in reasons.most_common(25)],
        strategy_versions=dict(versions),
    )
