"""Where do setups die?

A strategy that fires three times in four thousand bars is not obviously
broken - it might be a strict conjunction doing exactly what you asked. The way
to tell the difference is to look at each rule's own pass rate and then at the
funnel: how many bars survive each required rule in turn.

If one rule passes 6% of the time and another passes 7%, requiring both leaves
you roughly 0.4% of bars before anything else is even considered. That is a
choice, not a bug - but it should be a choice you made on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Candle, Series, Side
from ..strategy.context import ContextBuilder
from ..strategy.engine import StrategyEngine


@dataclass
class RuleStat:
    rule: str
    mode: str
    passed: int = 0
    evaluated: int = 0

    @property
    def rate(self) -> float:
        return self.passed / self.evaluated if self.evaluated else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "mode": self.mode, "passed": self.passed,
                "evaluated": self.evaluated, "rate": round(self.rate, 5)}


@dataclass
class RuleReport:
    bars: int = 0
    evaluations: int = 0
    qualified: int = 0
    stats: list[RuleStat] = field(default_factory=list)
    funnel: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "bars": self.bars,
            "evaluations": self.evaluations,
            "qualified": self.qualified,
            "rules": [s.to_dict() for s in self.stats],
            "funnel": self.funnel,
        }

    def render(self) -> str:
        lines = [
            f"  {self.strategy} · {self.bars} bars · {self.evaluations} side-evaluations "
            f"· {self.qualified} qualified",
            "",
            f"  {'rule':<24}{'mode':<10}{'passes':>9}{'rate':>8}",
            "  " + "-" * 52,
        ]
        for stat in self.stats:
            lines.append(f"  {stat.rule:<24}{stat.mode:<10}{stat.passed:>9}{stat.rate * 100:>7.1f}%")

        lines += ["", "  Funnel - bars surviving each required rule, in order", "  " + "-" * 52]
        for step in self.funnel:
            lines.append(
                f"  {step['rule']:<24}{step['surviving']:>9}"
                f"{step['share_of_all'] * 100:>7.2f}%   "
                f"(kept {step['kept_from_previous'] * 100:.0f}% of the previous step)"
            )
        if self.qualified == 0 and self.evaluations:
            lines += ["", "  Nothing qualified. The funnel above shows which rule ends the run;",
                      "  relax that one, or move it to advisory, before blaming the data."]
        return "\n".join(lines)


def analyse(config, candles: list[Candle], ignore_session: bool = False) -> RuleReport:
    import copy

    cfg = copy.deepcopy(config)
    if ignore_session:
        cfg.market.trade_session_only = False

    engine = StrategyEngine(cfg)
    builder = ContextBuilder(cfg)
    series = Series(cfg.market.symbol, cfg.market.timeframe, maxlen=600)
    specs = engine.spec.active()
    stats = {s.rule: RuleStat(s.rule, s.mode) for s in specs}
    required = [s.rule for s in specs if s.required]
    survivors = {rule: 0 for rule in required}

    evaluations = qualified = 0
    need = engine.spec.min_bars()

    for candle in candles:
        series.upsert(candle)
        ctx = builder.build(series, structure_window=engine.spec.structure_window)
        if ctx is None or not ctx.warm(need) or not ctx.in_session:
            continue
        for side in (Side.LONG, Side.SHORT):
            verdict = engine.evaluate_side(ctx, side)
            evaluations += 1
            if verdict.passed:
                qualified += 1
            results = {c.name: c.passed for c in verdict.checks}
            for name, ok in results.items():
                stats[name].evaluated += 1
                stats[name].passed += int(ok)
            alive = True
            for rule in required:
                alive = alive and results.get(rule, False)
                if not alive:
                    break
                survivors[rule] += 1

    funnel: list[dict[str, Any]] = []
    previous = evaluations
    for rule in required:
        count = survivors[rule]
        funnel.append({
            "rule": rule,
            "surviving": count,
            "share_of_all": count / evaluations if evaluations else 0.0,
            "kept_from_previous": count / previous if previous else 0.0,
        })
        previous = count or 1

    return RuleReport(
        bars=len(candles),
        evaluations=evaluations,
        qualified=qualified,
        stats=[stats[s.rule] for s in specs],
        funnel=funnel,
        strategy=engine.spec.name,
    )
