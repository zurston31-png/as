"""Historical backtest - stage one of the rollout.

It drives the *live* orchestrator, bar by bar, rather than reimplementing the
pipeline. That is the point: a backtest that runs different code from the live
loop measures something you will never actually trade. Every gate applies -
freshness, the duplicate-signal gate, reconciliation, the risk engine - and
fills go through the same pessimistic paper broker.

No lookahead is structural rather than checked after the fact: the context for
a decision is built from closed bars at or before that bar's timestamp
(`strategy/context.py`), and the broker marks existing positions against a bar
*before* any new entry is opened on it, so a position never fills against the
bar that created it.

The AI layer is off by default here. A backtest that calls a model on every bar
is slow and expensive, and you are trying to measure the rules.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..audit import NullAuditLog
from ..data.replay import read_csv
from ..models import Candle, ClosedTrade, TradeSignal
from ..orchestrator import Orchestrator
from .confirmers import NullConfirmer
from .metrics import Metrics, compute

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    label: str
    metrics: Metrics
    trades: list[ClosedTrade] = field(default_factory=list)
    signals: list[TradeSignal] = field(default_factory=list)
    bars: int = 0
    blocked: dict[str, int] = field(default_factory=dict)
    suppressed: int = 0
    qualified: int = 0        # side-evaluations that passed, before arbitration and the gates
    ambiguous: int = 0        # bars where both directions qualified at once
    period: tuple[Optional[str], Optional[str]] = (None, None)
    ai_vetoes: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def taken(self) -> int:
        return len([s for s in self.signals if s.actionable])

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "period": {"start": self.period[0], "end": self.period[1]},
            "bars": self.bars,
            "signals": len(self.signals),
            "taken": self.taken,
            "suppressed_duplicates": self.suppressed,
            "qualified": self.qualified,
            "ambiguous": self.ambiguous,
            "ai_vetoes": self.ai_vetoes,
            "blocked_by": self.blocked,
            "metrics": self.metrics.to_dict(),
            "notes": self.notes,
            "trades": [t.to_dict() for t in self.trades],
        }


def sandbox(config, tag: str = "backtest"):
    """A copy of the config that can never touch live state or your speakers."""
    cfg = copy.deepcopy(config)
    state_dir = Path(config.risk.state_path).parent
    cfg.risk.state_path = str(state_dir / f"{tag}_risk_state.json")
    cfg.risk.audit_path = str(state_dir / f"{tag}_audit.jsonl")
    cfg.execution.trades_path = str(state_dir / f"{tag}_trades.jsonl")
    cfg.notify.sound = cfg.notify.desktop = cfg.notify.console = False
    cfg.vision.enabled = False
    cfg.feed.source = "replay"
    Path(cfg.risk.state_path).unlink(missing_ok=True)
    Path(cfg.execution.trades_path).unlink(missing_ok=True)
    return cfg


def permissive_risk(config):
    """Sizing and a required stop only - used by arms that isolate the rules.

    Nothing here is a recommendation. It exists so "rules alone" measures the
    rules rather than the risk engine's daily caps.
    """
    cfg = copy.deepcopy(config)
    cfg.risk.max_trades_per_day = 10_000
    cfg.risk.max_daily_loss_pct = 1e6
    cfg.risk.max_consecutive_losses = 10_000
    cfg.risk.cooldown_minutes_after_loss = 0
    cfg.risk.daily_drawdown_kill_pct = 1e6
    cfg.risk.max_risk_per_trade_pct = 100.0
    return cfg


class Backtester:
    def __init__(self, config, confirmer=None, label: str = "backtest",
                 tag: Optional[str] = None) -> None:
        self.cfg = sandbox(config, tag or label.replace(" ", "_").lower())
        self.label = label
        self.confirmer = confirmer

    async def run(self, csv_path: str | Path, limit: Optional[int] = None) -> BacktestResult:
        candles = read_csv(csv_path)
        if limit:
            candles = candles[-limit:]
        return await self.run_candles(candles)

    async def run_candles(self, candles: list[Candle]) -> BacktestResult:
        if not candles:
            return BacktestResult(self.label, Metrics(final_equity=self.cfg.risk.starting_equity))

        bot = Orchestrator(self.cfg, audit=NullAuditLog())
        bot.confirmation = self.confirmer if self.confirmer is not None else NullConfirmer()
        # Freshness is a live-feed concern; replayed bars are "old" by definition.
        bot.monitor.max_stale_seconds = 10 ** 9

        for candle in candles:
            await bot.on_candle(candle)

        if bot.broker.positions:
            bot.broker.close_all(candles[-1].close, candles[-1], "end_of_data")

        blocked: dict[str, int] = {}
        ai_vetoes = 0
        for signal in bot.signals:
            if signal.actionable:
                continue
            key = _blocked_key(signal.blocked_by)
            blocked[key] = blocked.get(key, 0) + 1
            if signal.blocked_by.startswith(("AI ", "randomised")):
                ai_vetoes += 1

        return BacktestResult(
            label=self.label,
            metrics=compute(bot.broker.closed, self.cfg.risk.starting_equity),
            trades=list(bot.broker.closed),
            signals=list(bot.signals),
            bars=len(candles),
            blocked=blocked,
            suppressed=bot.strategy.gate.suppressed,
            qualified=bot.strategy.qualified_sides,
            ambiguous=bot.strategy.ambiguous,
            period=(candles[0].ts.isoformat(), candles[-1].ts.isoformat()),
            ai_vetoes=ai_vetoes,
            notes={"strategy": bot.strategy.spec.name,
                   "confirmer": getattr(bot.confirmation, "name", "live")},
        )


def _blocked_key(reason: str) -> str:
    """Group block reasons so the summary counts causes, not individual messages."""
    if reason.startswith("risk["):
        return "risk: " + reason.split("[", 1)[1].split("]", 1)[0]
    for prefix in ("AI rejected", "AI says wait", "AI unavailable", "AI confidence",
                   "randomised veto", "reconciliation", "screen/feed disagreement",
                   "duplicate"):
        if reason.startswith(prefix):
            return prefix
    return reason.split("(")[0].strip()[:60]


def run_sync(config, csv_path: str | Path, confirmer=None,
             limit: Optional[int] = None) -> BacktestResult:
    return asyncio.run(Backtester(config, confirmer).run(csv_path, limit))
