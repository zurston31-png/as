"""Which rule is removing the setups - and is removing it an improvement?

`tradebot rules` shows the funnel: where setups die. This answers the next
question, which the funnel cannot: whether the rule that kills the most setups
is *earning* its strictness.

Two modes, both over identical data:

* **incremental** - stack the rules up one at a time, in config order. Each row
  shows what adding that rule did to the numbers.
* **leave-one-out** - the full stack, minus one required rule at a time. Each
  row shows what that single rule contributes on top of everything else. This
  is the one that answers "is `liquidity_sweep` worth the 96% of setups it
  costs me?".

Both run rules-only under permissive risk limits, so the comparison measures
the rules rather than the daily caps.

Three setup columns, and the differences between them matter.

* `qualif` counts side-evaluations that passed, before anything else looked at
  them. Relaxing a rule can only raise this, so it is the clean measure of a
  rule's strictness.
* `ambig` counts bars where *both* directions qualified at once. The engine
  treats those as no-setup, so a rule set loose enough to be ambiguous produces
  fewer usable setups, not more. If dropping a rule spikes this column, the rule
  was supplying the directional decision - `ema_stack` typically is.
* `signals` counts what survived the duplicate and position gates. Those couple
  decisions across time: a looser rule set enters earlier, holds the book
  longer, and can end up with fewer admitted signals than a stricter one.

Judge strictness on `qualif`, direction on `ambig`, and results on `trades`.

Read the output carefully. A variant with more trades and better expectancy is
a genuine improvement. A variant with more trades and *similar* expectancy is
usually just more samples of the same edge - which may still be worth having,
because 8 trades cannot distinguish an edge from luck. A variant with many more
trades and worse expectancy is the rule doing its job.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ..models import Candle
from ..strategy.presets import PRESETS
from ..strategy.spec import StrategySpec
from .confirmers import NullConfirmer
from .runner import Backtester, BacktestResult, permissive_risk

log = logging.getLogger(__name__)

Mode = Literal["incremental", "leave-one-out", "both"]


@dataclass
class Variant:
    label: str
    rules: list[dict[str, Any]]
    removed: Optional[str] = None      # the rule this variant drops, if any
    added: Optional[str] = None        # the rule this variant adds, if any
    result: Optional[BacktestResult] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "removed": self.removed,
            "added": self.added,
            "rules": [r.get("rule") for r in self.rules],
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class RuleAblationReport:
    baseline: Optional[Variant] = None
    incremental: list[Variant] = field(default_factory=list)
    leave_one_out: list[Variant] = field(default_factory=list)
    bars: int = 0
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "bars": self.bars,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "incremental": [v.to_dict() for v in self.incremental],
            "leave_one_out": [v.to_dict() for v in self.leave_one_out],
            "findings": self.findings(),
        }

    # --------------------------------------------------------------- verdict

    def findings(self) -> list[dict[str, Any]]:
        """Per-rule contribution, computed from the leave-one-out arms."""
        if not self.baseline or not self.baseline.result:
            return []
        base = self.baseline.result
        out: list[dict[str, Any]] = []
        for variant in self.leave_one_out:
            if not variant.result:
                continue
            without = variant.result
            expectancy_delta = without.metrics.expectancy_r - base.metrics.expectancy_r
            out.append({
                "rule": variant.removed,
                # Qualifying bars, not admitted signals: the duplicate and
                # position gates couple decisions across time, so a looser rule
                # set can enter earlier, hold the book longer, and end up with
                # *fewer* admitted signals. Only the pre-gate count measures
                # how much the rule itself removes.
                "qualified_without_it": without.qualified,
                "qualified_with_it": base.qualified,
                "setups_it_removes": without.qualified - base.qualified,
                "signals_without_it": len(without.signals),
                "signals_with_it": len(base.signals),
                "expectancy_without_it": round(without.metrics.expectancy_r, 4),
                "expectancy_with_it": round(base.metrics.expectancy_r, 4),
                "expectancy_delta_from_keeping_it": round(-expectancy_delta, 4),
                "drawdown_without_it": round(without.metrics.max_drawdown_pct, 3),
                "drawdown_with_it": round(base.metrics.max_drawdown_pct, 3),
                "ambiguous_without_it": without.ambiguous,
                "reading": _read(base, without),
            })
        out.sort(key=lambda f: -f["setups_it_removes"])
        return out

    def render(self) -> str:
        lines: list[str] = []
        if self.incremental:
            lines += ["  Incremental - rules stacked up in config order", ""]
            lines += _table(self.incremental)
        if self.leave_one_out:
            if lines:
                lines.append("")
            lines += ["  Leave-one-out - the full stack minus one required rule", ""]
            lines += _table([self.baseline, *self.leave_one_out])

        findings = self.findings()
        if findings:
            lines += ["", "  What each rule is buying you", "  " + "-" * 100]
            for f in findings:
                lines.append(
                    f"  {f['rule']:<22} removes {f['setups_it_removes']:>5} qualifying bars   "
                    f"expectancy {f['expectancy_with_it']:+.3f}R with / "
                    f"{f['expectancy_without_it']:+.3f}R without"
                )
                if f["ambiguous_without_it"] > max(10, f["qualified_with_it"] * 0.1):
                    lines.append(
                        f"  {'':<22} without it, {f['ambiguous_without_it']} bars qualify in "
                        f"both directions at once - this rule picks the side"
                    )
                lines.append(f"  {'':<22} {f['reading']}")
        return "\n".join(lines)


def _table(variants) -> list[str]:
    header = (f"  {'variant':<26}{'qualif':>8}{'ambig':>7}{'signals':>8}{'trades':>8}{'win%':>7}"
              f"{'expectancy':>13}{'PF':>7}{'maxDD%':>9}{'TP1%':>7}{'TP2%':>7}")
    rows = [header, "  " + "-" * 100]
    for variant in variants:
        if not variant or not variant.result:
            continue
        m = variant.result.metrics
        pf = f"{m.profit_factor:.2f}" if m.profit_factor else ("inf" if m.trades else "-")
        rows.append(
            f"  {variant.label[:25]:<26}{variant.result.qualified:>8}"
            f"{variant.result.ambiguous:>7}{len(variant.result.signals):>8}{m.trades:>8}"
            f"{(m.win_rate * 100 if m.trades else 0):>6.0f}%"
            f"{m.expectancy_r:>+12.3f}R{pf:>7}{m.max_drawdown_pct:>9.2f}"
            f"{m.tp1_hit_rate * 100:>6.0f}%{m.tp2_hit_rate * 100:>6.0f}%"
        )
    return rows


def _read(base: BacktestResult, without: BacktestResult) -> str:
    """A plain-language reading of one leave-one-out arm."""
    if not base.metrics.trades or not without.metrics.trades:
        return "not enough trades either way to say anything"

    extra = without.metrics.trades - base.metrics.trades
    better = base.metrics.expectancy_r > without.metrics.expectancy_r
    margin = abs(base.metrics.expectancy_r - without.metrics.expectancy_r)
    thin = base.metrics.trades < 30

    if better and margin > 0.05:
        verdict = "keeping it improves expectancy"
    elif not better and margin > 0.05:
        verdict = "dropping it improves expectancy"
    else:
        verdict = "expectancy is effectively unchanged either way"

    if extra > 0:
        verdict += f"; dropping it adds {extra} trades"
    if thin:
        verdict += " (too few trades to trust - treat as a hint, not a result)"
    return verdict


# ---------------------------------------------------------------- building

def expand(config) -> StrategySpec:
    """Resolve the configured strategy - preset or not - into explicit rules."""
    raw = config.strategy if isinstance(config.strategy, dict) else {}
    return StrategySpec.from_config(raw, PRESETS)


def with_rules(config, rules: list[dict[str, Any]], name: str):
    cfg = copy.deepcopy(config)
    base = cfg.strategy if isinstance(cfg.strategy, dict) else {}
    cfg.strategy = {**{k: v for k, v in base.items() if k != "preset"},
                    "name": name, "rules": rules}
    return cfg


def build_variants(spec: StrategySpec, mode: Mode) -> tuple[Variant, list[Variant], list[Variant]]:
    active = [r.to_dict() for r in spec.active()]
    baseline = Variant("full stack", active)

    incremental: list[Variant] = []
    if mode in ("incremental", "both"):
        for i in range(1, len(active) + 1):
            subset = active[:i]
            # A prefix that is all-advisory would be rejected by the spec, so
            # promote the first rule - the point is the incremental stack, not
            # the mode of rule one.
            subset = _ensure_one_required(subset)
            incremental.append(Variant(
                label=f"+ {subset[-1]['rule']}" if i > 1 else subset[0]["rule"],
                rules=subset,
                added=subset[-1]["rule"],
            ))

    leave_one_out: list[Variant] = []
    if mode in ("leave-one-out", "both"):
        required = [r["rule"] for r in active if r["mode"] == "required"]
        for rule in required:
            remaining = [r for r in active if r["rule"] != rule]
            if not remaining:
                continue
            leave_one_out.append(Variant(
                label=f"- {rule}",
                rules=_ensure_one_required(remaining),
                removed=rule,
            ))

    return baseline, incremental, leave_one_out


def _ensure_one_required(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = [dict(r) for r in rules]
    if any(r.get("mode") == "required" for r in rules):
        return rules
    rules[0]["mode"] = "required"
    return rules


async def run_rule_ablation(config, candles: list[Candle], mode: Mode = "both"
                            ) -> RuleAblationReport:
    spec = expand(config)
    loose = permissive_risk(config)
    baseline, incremental, leave_one_out = build_variants(spec, mode)

    async def evaluate(variant: Variant, tag: str) -> None:
        cfg = with_rules(loose, variant.rules, variant.label)
        variant.result = await Backtester(
            cfg, NullConfirmer(), variant.label, tag).run_candles(candles)

    await evaluate(baseline, "ra_full")
    for i, variant in enumerate(incremental):
        await evaluate(variant, f"ra_inc_{i}")
    for i, variant in enumerate(leave_one_out):
        await evaluate(variant, f"ra_loo_{i}")

    return RuleAblationReport(
        baseline=baseline,
        incremental=incremental,
        leave_one_out=leave_one_out,
        bars=len(candles),
        strategy=spec.name,
    )
