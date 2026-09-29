"""Historical backtest - stage one of the rollout.

Runs the exact same strategy, sizing and risk code the live loop uses, over a
CSV. The AI layer is off by default: a backtest that calls a model once per bar
is slow and expensive, and more importantly the model would be seeing bars it
has no business seeing when you are trying to measure the *rules*. Turn it on
with --ai to sanity check the confirmation layer over a short window.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..data.replay import read_csv
from ..execution.paper import PaperBroker
from ..models import Candle, ClosedTrade, Series, TradeSignal
from ..risk.engine import RiskEngine
from ..strategy.context import ContextBuilder
from ..strategy.engine import StrategyEngine
from .metrics import Metrics, compute

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    metrics: Metrics
    trades: list[ClosedTrade]
    signals: list[TradeSignal]
    bars: int
    blocked: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "bars": self.bars,
            "signals": len(self.signals),
            "taken": len([s for s in self.signals if s.actionable]),
            "blocked_by": self.blocked,
            "trades": [t.to_dict() for t in self.trades],
        }


class Backtester:
    def __init__(self, config: Config, use_ai: bool = False) -> None:
        self.cfg = config
        self.use_ai = use_ai
        # A backtest must never inherit or mutate live risk state.
        self.cfg.risk.state_path = str(Path(config.risk.state_path).with_name("backtest_state.json"))
        Path(self.cfg.risk.state_path).unlink(missing_ok=True)
        self.cfg.execution.trades_path = str(
            Path(config.execution.trades_path).with_name("backtest_trades.jsonl")
        )
        Path(self.cfg.execution.trades_path).unlink(missing_ok=True)

    async def run(self, csv_path: str | Path, limit: Optional[int] = None) -> BacktestResult:
        candles = read_csv(csv_path)
        if limit:
            candles = candles[-limit:]
        return await self.run_candles(candles)

    async def run_candles(self, candles: list[Candle]) -> BacktestResult:
        cfg = self.cfg
        series = Series(cfg.market.symbol, cfg.market.timeframe, maxlen=600)
        builder = ContextBuilder(cfg)
        strategy = StrategyEngine(cfg)
        risk = RiskEngine(cfg)
        broker = PaperBroker(cfg, on_close=risk.register_close)

        confirmation = None
        if self.use_ai and cfg.ai.enabled:
            from ..ai.confirm import ConfirmationLayer
            confirmation = ConfirmationLayer(cfg)

        signals: list[TradeSignal] = []
        blocked: dict[str, int] = {}
        min_bars = builder.min_bars()

        for candle in candles:
            series.upsert(candle)
            broker.on_candle(candle)
            if len(series) < min_bars:
                continue

            ctx = builder.build(series)
            if ctx is None:
                continue
            risk.tick(candle.ts)

            verdict = strategy.evaluate(ctx)
            if not verdict.passed:
                continue
            signal = strategy.build_signal(ctx, verdict)
            if signal is None:
                continue

            if confirmation is not None:
                signal.ai = await confirmation.confirm(signal, ctx, series.closed_candles())
                allowed, reason = confirmation.accepts(signal.ai)
                if not allowed:
                    signals.append(_blocked(signal, reason, blocked))
                    continue

            decision = risk.evaluate(signal, len(broker.positions), candle.ts)
            signal.risk = decision
            if not decision.allowed:
                signals.append(_blocked(signal, decision.reason, blocked))
                continue

            signal.qty = decision.qty
            signal.risk_pct = decision.risk_pct
            signal.risk_amount = decision.risk_amount
            signal.actionable = True
            signals.append(signal)
            broker.open(signal, decision.qty)
            risk.register_entry(signal)

        if broker.positions and candles:
            broker.close_all(candles[-1].close, candles[-1], "end_of_data")

        metrics = compute(broker.closed, cfg.risk.starting_equity)
        return BacktestResult(metrics, broker.closed, signals, len(candles), blocked)


def _blocked(signal: TradeSignal, reason: str, counter: dict[str, int]) -> TradeSignal:
    signal.actionable = False
    signal.blocked_by = reason
    key = reason.split("(")[0].strip()[:60]
    counter[key] = counter.get(key, 0) + 1
    return signal


def run_sync(config: Config, csv_path: str | Path, use_ai: bool = False,
             limit: Optional[int] = None) -> BacktestResult:
    return asyncio.run(Backtester(config, use_ai).run(csv_path, limit))
