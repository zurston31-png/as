"""Does the AI layer actually earn its place?

Four arms over identical data:

1. **rules**            - the rules alone, sizing only, no daily limits.
2. **rules+ai**         - the same, plus the AI veto.
3. **rules+ai+risk**    - the full production stack.
4. **rules+random+risk** - arm 3 with the AI replaced by coin flips vetoing at
   the *same rate* arm 2's model did.

Arm 4 is the one that matters. A veto layer reduces trade count, and reducing
trade count changes drawdown and expectancy on its own - so "adding the AI
improved the numbers" proves nothing until you show that vetoing the same
proportion of trades *at random* does not improve them equally. If arm 2 does
not beat arm 4, the model is contributing no information; it is just trading
less, and a lower `max_trades_per_day` would do the same thing for free.

Verdicts are cached on disk by decision identity, so arms 2 and 3 see exactly
the same model output and re-running is free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..ai.confirm import ConfirmationLayer
from ..models import Candle
from .confirmers import CachedConfirmer, NullConfirmer, RandomConfirmer
from .runner import Backtester, BacktestResult, permissive_risk

log = logging.getLogger(__name__)


@dataclass
class AblationReport:
    arms: list[BacktestResult] = field(default_factory=list)
    veto_rate: float = 0.0
    seeds: list[int] = field(default_factory=list)
    random_arms: list[BacktestResult] = field(default_factory=list)
    ai_verdicts: int = 0          # how many real model verdicts the AI arms saw
    veto_rate_source: str = "observed"

    def by_label(self, label: str) -> Optional[BacktestResult]:
        return next((a for a in self.arms if a.label == label), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms": [a.to_dict() for a in self.arms],
            "veto_rate": round(self.veto_rate, 4),
            "veto_rate_source": self.veto_rate_source,
            "ai_verdicts": self.ai_verdicts,
            "random_control_seeds": self.seeds,
            "random_control_arms": [a.to_dict() for a in self.random_arms],
            "verdict": self.verdict(),
        }

    def verdict(self) -> dict[str, Any]:
        """The comparison the whole exercise exists for."""
        ai = self.by_label("rules+ai")
        rules = self.by_label("rules")
        if not ai or not rules or not ai.metrics.trades or not rules.metrics.trades:
            return {"conclusion": "not enough trades to compare"}
        if self.ai_verdicts == 0:
            # Without verdicts the AI arms are the rules arm under another name,
            # and the random control is calibrated to a veto rate of zero. Saying
            # "the AI does not help" here would be a claim about nothing.
            return {
                "conclusion": (
                    "no AI verdicts were available, so the AI arms are identical to the "
                    "rules arm and this comparison is meaningless. Re-run with --ai (which "
                    "calls the API and costs money) or against a populated verdict cache."
                ),
                "ai_verdicts": 0,
                "inconclusive": True,
            }

        random_expectancies = [a.metrics.expectancy_r for a in self.random_arms
                               if a.metrics.trades]
        random_mean = (sum(random_expectancies) / len(random_expectancies)
                       if random_expectancies else None)
        random_dd = [a.metrics.max_drawdown_pct for a in self.random_arms if a.metrics.trades]
        random_dd_mean = sum(random_dd) / len(random_dd) if random_dd else None

        beats_rules = ai.metrics.expectancy_r > rules.metrics.expectancy_r
        beats_random = random_mean is not None and ai.metrics.expectancy_r > random_mean
        drawdown_ok = ai.metrics.max_drawdown_pct <= rules.metrics.max_drawdown_pct + 1e-9

        if beats_rules and beats_random and drawdown_ok:
            conclusion = "the AI veto adds measurable value on this data"
        elif beats_rules and not beats_random:
            conclusion = ("the AI beats the rules but not random vetoing at the same rate - "
                          "it is trading less, not trading better")
        elif beats_rules and not drawdown_ok:
            conclusion = "the AI improves expectancy but increases drawdown"
        else:
            conclusion = "the AI veto does not improve results on this data"

        return {
            "conclusion": conclusion,
            "beats_rules_alone": beats_rules,
            "beats_random_control": beats_random,
            "drawdown_not_worse": drawdown_ok,
            "expectancy_r": {
                "rules": round(rules.metrics.expectancy_r, 4),
                "rules+ai": round(ai.metrics.expectancy_r, 4),
                "random_control_mean": round(random_mean, 4) if random_mean is not None else None,
            },
            "max_drawdown_pct": {
                "rules": round(rules.metrics.max_drawdown_pct, 3),
                "rules+ai": round(ai.metrics.max_drawdown_pct, 3),
                "random_control_mean": (round(random_dd_mean, 3)
                                        if random_dd_mean is not None else None),
            },
            "trades": {a.label: a.metrics.trades for a in self.arms},
        }

    def render(self) -> str:
        rows = [*self.arms, *self.random_arms]
        lines = [
            f"  {'arm':<22}{'setups':>8}{'taken':>7}{'vetoed':>8}{'trades':>8}"
            f"{'win%':>7}{'expectancy':>13}{'maxDD%':>9}",
            "  " + "-" * 82,
        ]
        for arm in rows:
            m = arm.metrics
            lines.append(
                f"  {arm.label:<22}{len(arm.signals):>8}{arm.taken:>7}{arm.ai_vetoes:>8}"
                f"{m.trades:>8}{(m.win_rate * 100 if m.trades else 0):>6.0f}%"
                f"{m.expectancy_r:>+12.3f}R{m.max_drawdown_pct:>9.2f}"
            )
        v = self.verdict()
        lines += ["  " + "-" * 82,
                  f"  AI veto rate: {self.veto_rate * 100:.1f}% ({self.veto_rate_source}) "
                  f"from {self.ai_verdicts} model verdict(s)",
                  f"  => {v['conclusion']}"]
        return "\n".join(lines)


async def run_ablation(
    config,
    candles: list[Candle],
    use_live_ai: bool = False,
    cache_path: str | Path = "data/ai_verdicts.json",
    control_runs: int = 5,
    seed: int = 1234,
    veto_rate: Optional[float] = None,
) -> AblationReport:
    """Run all four arms over the same candles.

    With `use_live_ai=False` the AI arms use whatever is already in the verdict
    cache and skip the rest. That is useful for wiring the comparison up before
    paying for a full run, but if the cache is empty the AI arms are just the
    rules arm again and the report says so rather than drawing a conclusion.

    `veto_rate` forces the control arms' rate instead of taking it from the
    model - handy for asking "what would vetoing 30% at random do to this
    strategy?" without spending anything.
    """
    loose = permissive_risk(config)

    arm_rules = await Backtester(loose, NullConfirmer(), "rules", "arm_rules").run_candles(candles)

    inner = ConfirmationLayer(config) if use_live_ai else _OfflineLayer()
    confirmer = CachedConfirmer(inner, cache_path)
    arm_ai = await Backtester(loose, confirmer, "rules+ai", "arm_ai").run_candles(candles)
    arm_full = await Backtester(config, confirmer, "rules+ai+risk", "arm_full").run_candles(candles)

    ai_verdicts = confirmer.hits + (confirmer.misses if use_live_ai else 0)

    # Calibrate the control to the veto rate the model actually produced, unless
    # the caller pinned one.
    if veto_rate is None:
        consulted = len(arm_ai.signals)
        observed = (arm_ai.ai_vetoes / consulted) if consulted else 0.0
        source = "observed"
    else:
        observed, source = max(0.0, min(1.0, veto_rate)), "forced"

    seeds = [seed + i for i in range(max(1, control_runs))]
    random_arms = []
    for i, s in enumerate(seeds):
        result = await Backtester(
            config, RandomConfirmer(observed, s),
            f"random#{i + 1}", f"arm_random_{i + 1}",
        ).run_candles(candles)
        random_arms.append(result)

    return AblationReport(
        arms=[arm_rules, arm_ai, arm_full],
        veto_rate=observed,
        seeds=seeds,
        random_arms=random_arms,
        ai_verdicts=ai_verdicts,
        veto_rate_source=source,
    )


class _OfflineLayer:
    """Stands in for the real layer when running from the cache alone."""

    last_error = "offline: no cached verdict for this decision"

    async def confirm(self, *_args, **_kwargs):
        from ..models import AIVerdict, Decision
        return AIVerdict(Decision.UNAVAILABLE, 0.0, self.last_error, source="none")

    def accepts(self, verdict):
        # Offline gaps must not silently become vetoes and skew the arm; they
        # pass through, and the run reports how many were missing.
        from ..models import Decision
        if verdict.decision is Decision.UNAVAILABLE:
            return True, "no cached verdict - passed through"
        if verdict.decision in (Decision.REJECT, Decision.WAIT):
            return False, f"AI {verdict.decision.value}: {verdict.rationale}"
        return True, f"AI confirmed ({verdict.confidence:.2f})"
