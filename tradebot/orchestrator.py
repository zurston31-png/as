"""The pipeline.

    market data
        -> series + indicator context
        -> strategy engine            (deterministic; can it be a trade at all?)
        -> AI confirmation layer      (can only veto)
        -> risk engine                (can only veto, and does the sizing)
        -> LONG / SHORT / NO TRADE
        -> notification + paper fill

Two things are deliberate. Vetoes are one-directional: no later stage can
resurrect a setup an earlier one killed, and the AI cannot invent a trade the
rules didn't find. And the screen-vision layer sits *beside* this chain rather
than in it - it corroborates or contradicts, it doesn't originate.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from .ai.chat import ChatSession
from .ai.client import availability as ai_availability
from .ai.confirm import ConfirmationLayer
from .ai.vision import ChartVision
from .bus import EventBus
from .config import Config
from .data.poll import PollSource
from .data.replay import ReplaySource, read_csv
from .data.synthetic import SyntheticSource
from .data.webhook import WebhookSource
from .execution.base import LiveBroker
from .execution.paper import PaperBroker
from .models import Candle, ClosedTrade, Series, TradeSignal, utcnow
from .notify import Notifier
from .risk.engine import RiskEngine
from .strategy.context import ContextBuilder, MarketContext
from .strategy.engine import StrategyEngine

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config, bus: Optional[EventBus] = None) -> None:
        self.cfg = config
        self.bus = bus or EventBus()

        self.series = Series(config.market.symbol, config.market.timeframe)
        self.context_builder = ContextBuilder(config)
        self.strategy = StrategyEngine(config)
        self.confirmation = ConfirmationLayer(config)
        self.vision = ChartVision(config)
        self.risk = RiskEngine(config)
        self.notifier = Notifier(config.notify)
        self.broker = self._make_broker()
        self.chat = ChatSession(config, self.state)

        self.context: Optional[MarketContext] = None
        self.signals: list[TradeSignal] = []
        self.last_actionable: Optional[TradeSignal] = None
        self.last_bar_at: Optional[datetime] = None
        self.started_at = utcnow()
        self.bars_seen = 0
        self.source = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._evaluating = False

    # ------------------------------------------------------------- wiring

    def _make_broker(self):
        mode = self.cfg.execution.mode
        if mode == "live":
            return LiveBroker()  # raises with an explanatory message by design
        return PaperBroker(self.cfg, on_close=self._on_trade_closed)

    def build_source(self):
        f = self.cfg.feed
        m = self.cfg.market
        if f.source == "replay":
            return ReplaySource(f.csv_path, f.replay_speed)
        if f.source == "webhook":
            return WebhookSource(m.timeframe)
        if f.source == "poll":
            return PollSource(f.poll_url, m.timeframe, f.poll_interval_seconds)
        # The synthetic feed stamps its bars with the current clock, so a real
        # session filter would silently reject all of them and the demo would
        # look broken. Nothing about this applies to a real feed.
        if m.trade_session_only:
            m.trade_session_only = False
            log.info("synthetic feed: session filter off (its bars aren't real market hours)")
        return SyntheticSource(m.symbol, m.timeframe, bars_per_second=4.0)

    def warmup(self) -> int:
        """Preload history so the indicators aren't cold when live bars start."""
        path = self.cfg.feed.warmup_csv
        if not path:
            return 0
        try:
            candles = read_csv(path)
        except (FileNotFoundError, OSError) as exc:
            log.warning("warmup CSV unusable: %s", exc)
            return 0
        self.series.extend(candles)
        self.context = self.context_builder.build(self.series)
        log.info("warmed up with %d bars from %s", len(candles), path)
        return len(candles)

    # -------------------------------------------------------------- loop

    async def run(self) -> None:
        self._running = True
        self.source = self.build_source()
        self.warmup()
        log.info(
            "starting: %s %s via %s feed, execution=%s, AI=%s",
            self.cfg.market.symbol, self.cfg.market.timeframe,
            getattr(self.source, "name", "?"), self.cfg.execution.mode,
            "on" if self.cfg.ai.enabled else "off",
        )
        self.bus.publish("status", self.status())
        try:
            async for candle in self.source.stream():
                if not self._running:
                    break
                await self.on_candle(candle)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("data loop crashed")
            self.bus.publish("error", {"message": "data feed stopped - see logs"})
        finally:
            self._running = False

    def start(self) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._task = asyncio.create_task(self.run(), name="tradebot-loop")
        return self._task

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ------------------------------------------------------------- per bar

    async def on_candle(self, candle: Candle) -> None:
        is_new = self.series.upsert(candle)
        self.last_bar_at = utcnow()
        self.bars_seen += is_new

        # Positions are marked against every bar, forming or closed, so a stop
        # that trades through intrabar is seen as soon as the feed shows it.
        for trade in self.broker.on_candle(candle):
            log.info("closed %s %s for %+.2f (%.2fR)", trade.side.value, trade.symbol,
                     trade.pnl, trade.r_multiple)

        self.context = self.context_builder.build(self.series)
        if self.context:
            self.bus.publish("tick", {
                "candle": candle.to_dict(),
                "context": self.context.to_dict(),
                "positions": [p.to_dict(candle.close) for p in self.broker.positions],
                "equity": round(self.risk.equity, 2),
            })

        if not candle.closed or not self.context:
            return
        if self._evaluating:
            return  # an AI call is still in flight; skip rather than queue up
        self._evaluating = True
        try:
            await self.evaluate()
        finally:
            self._evaluating = False

    async def evaluate(self) -> Optional[TradeSignal]:
        """Run the full gate chain for the just-closed bar."""
        ctx = self.context
        if ctx is None:
            return None
        self.risk.tick()

        verdict = self.strategy.evaluate(ctx)
        if not verdict.passed:
            self.bus.publish("evaluation", {
                "actionable": False,
                "ts": ctx.candle.ts.isoformat(),
                "verdict": verdict.to_dict(),
                "reason": "; ".join(verdict.notes) or "no setup",
            })
            return None

        signal = self.strategy.build_signal(ctx, verdict)
        if signal is None:
            return None

        # --- backup layer: does the screen contradict the feed? -------------
        vision_note = None
        if self.vision.enabled and self._vision_should_run():
            reading = await self.vision.read()
            vision_note = self.vision.note_for_prompt(reading)
            conflict = self.vision.disagreement(reading, ctx.price)
            if conflict:
                return self._block(signal, f"screen/feed disagreement: {conflict}")

        # --- AI confirmation ------------------------------------------------
        ai_verdict = await self.confirmation.confirm(
            signal, ctx, self.series.closed_candles(), vision_note
        )
        signal.ai = ai_verdict
        allowed, reason = self.confirmation.accepts(ai_verdict)
        if not allowed:
            return self._block(signal, reason)

        # --- risk engine ----------------------------------------------------
        decision = self.risk.evaluate(signal, len(self.broker.positions))
        signal.risk = decision
        if not decision.allowed:
            return self._block(signal, f"risk: {decision.reason}")

        signal.qty = decision.qty
        signal.risk_pct = decision.risk_pct
        signal.risk_amount = decision.risk_amount
        signal.actionable = True
        signal.reasons.append(reason)

        self._record(signal)
        self.last_actionable = signal
        self.notifier.signal(signal)

        if self.cfg.execution.mode == "paper":
            self.broker.open(signal, decision.qty)
            self.risk.register_entry(signal)
        elif self.cfg.execution.mode == "alerts_only":
            self.risk.register_entry(signal)

        log.info(signal.headline())
        return signal

    def _vision_should_run(self) -> bool:
        if self.cfg.vision.mode == "always":
            return True
        # "backup" mode: only when the structured feed has gone quiet.
        if self.last_bar_at is None:
            return True
        stale_after = timedelta(seconds=self.cfg.vision.stale_feed_seconds)
        return utcnow() - self.last_bar_at > stale_after

    def _block(self, signal: TradeSignal, reason: str) -> TradeSignal:
        signal.actionable = False
        signal.blocked_by = reason
        self._record(signal)
        self.notifier.signal(signal)
        log.info("no trade: %s", reason)
        return signal

    def _record(self, signal: TradeSignal) -> None:
        self.signals.append(signal)
        self.signals = self.signals[-200:]
        self.bus.publish("signal", signal.to_dict())

    def _on_trade_closed(self, trade: ClosedTrade) -> None:
        self.risk.register_close(trade)
        self.bus.publish("trade_closed", {
            "trade": trade.to_dict(),
            "risk": self.risk.snapshot(),
        })
        if self.risk.state.kill_switch:
            self.bus.publish("kill_switch", {"active": True, "reason": self.risk.state.kill_reason})
            self.notifier.desktop("Trading halted", self.risk.state.kill_reason)

    # ------------------------------------------------------------- controls

    def kill(self, reason: str = "manual: stopped from the dashboard") -> dict[str, Any]:
        self.risk.trip_kill_switch(reason)
        self.bus.publish("kill_switch", {"active": True, "reason": reason})
        return self.risk.snapshot()

    def resume(self) -> dict[str, Any]:
        self.risk.release_kill_switch()
        self.risk.clear_cooldown()
        self.bus.publish("kill_switch", {"active": False, "reason": ""})
        return self.risk.snapshot()

    def flatten(self) -> list[dict[str, Any]]:
        """Close every paper position at the last price."""
        last = self.series.last
        if not last or not isinstance(self.broker, PaperBroker):
            return []
        closed = self.broker.close_all(last.close, last, "flatten")
        return [t.to_dict() for t in closed]

    # --------------------------------------------------------------- state

    def status(self) -> dict[str, Any]:
        last = self.series.last
        stale_for = (utcnow() - self.last_bar_at).total_seconds() if self.last_bar_at else None
        return {
            "running": self._running,
            "symbol": self.cfg.market.symbol,
            "timeframe": self.cfg.market.timeframe,
            "strategy": self.cfg.strategy.name,
            "execution_mode": self.cfg.execution.mode,
            "feed": getattr(self.source, "name", self.cfg.feed.source),
            "bars_seen": self.bars_seen,
            "series_length": len(self.series),
            "last_price": last.close if last else None,
            "last_bar_at": self.last_bar_at.isoformat() if self.last_bar_at else None,
            "feed_stale_seconds": round(stale_for, 1) if stale_for is not None else None,
            "started_at": self.started_at.isoformat(),
            "ai": {**ai_availability(), "enabled": self.cfg.ai.enabled,
                   "model": self.cfg.ai.model, "last_error": self.confirmation.last_error},
            "vision": self.vision.status(),
        }

    def state(self) -> dict[str, Any]:
        """The full snapshot - served to the dashboard and given to the chat layer."""
        last = self.series.last
        price = last.close if last else None
        return {
            "status": self.status(),
            "context": self.context.to_dict() if self.context else None,
            "risk": self.risk.snapshot(),
            "positions": [p.to_dict(price) for p in self.broker.positions],
            "recent_signals": [s.to_dict() for s in self.signals[-12:]],
            "last_actionable_signal": self.last_actionable.to_dict() if self.last_actionable else None,
            "closed_trades": [t.to_dict() for t in getattr(self.broker, "closed", [])[-20:]],
            "performance": self.performance(),
            "config": {
                "strategy": self.cfg.strategy.__dict__,
                "risk_limits": self.risk.snapshot()["limits"],
            },
        }

    def performance(self) -> dict[str, Any]:
        trades = list(getattr(self.broker, "closed", []))
        if not trades:
            return {"trades": 0, "win_rate": None, "net_pnl": 0.0, "expectancy_r": None}
        wins = [t for t in trades if t.win]
        net = sum(t.pnl for t in trades)
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
            "win_rate": round(len(wins) / len(trades), 3),
            "net_pnl": round(net, 2),
            "expectancy_r": round(sum(t.r_multiple for t in trades) / len(trades), 3),
            "best_r": round(max(t.r_multiple for t in trades), 2),
            "worst_r": round(min(t.r_multiple for t in trades), 2),
        }

    # -------------------------------------------------------------- inbound

    def submit_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(self.source, WebhookSource):
            return {"accepted": False, "reason": "feed.source is not 'webhook'"}
        return self.source.submit(payload)
