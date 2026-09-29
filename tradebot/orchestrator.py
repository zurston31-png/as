"""The pipeline.

    market data
        -> freshness check          (stale feed => no decision at all)
        -> indicator context        (closed bars only, as of decision_ts)
        -> strategy rules           (deterministic; is there a setup?)
        -> duplicate gate           (one setup, one signal)
        -> AI confirmation          (can only veto; failure counts as WAIT)
        -> position reconciliation  (book vs. broker)
        -> risk engine              (limits, then explicit sizing)
        -> LONG / SHORT / NO TRADE
        -> audit record, notification, paper fill

Vetoes are one-directional: no later stage can resurrect a setup an earlier one
killed, and the AI cannot invent a trade the rules didn't find. Every outcome -
taken or blocked - is written to the audit log with the full chain behind it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Optional

from .ai.chat import ChatSession
from .ai.client import availability as ai_availability
from .ai.confirm import ConfirmationLayer
from .ai.vision import ChartVision
from .audit import AuditLog, fingerprint
from .bus import EventBus
from .data.base import timeframe_seconds
from .data.poll import PollSource
from .data.replay import ReplaySource, read_csv
from .data.synthetic import SyntheticSource
from .data.webhook import WebhookSource
from .execution.base import LiveBroker
from .execution.book import PositionBook
from .execution.paper import PaperBroker
from .models import Candle, ClosedTrade, Series, TradeSignal, utcnow
from .monitor import FeedMonitor
from .notify import Notifier
from .risk.engine import RiskEngine
from .strategy.context import ContextBuilder, MarketContext
from .strategy.engine import StrategyEngine

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config, bus: Optional[EventBus] = None,
                 audit: Optional[AuditLog] = None) -> None:
        self.cfg = config
        self.bus = bus or EventBus()

        self.series = Series(config.market.symbol, config.market.timeframe)
        self.context_builder = ContextBuilder(config)
        self.strategy = StrategyEngine(config)
        self.confirmation = ConfirmationLayer(config)
        self.vision = ChartVision(config)
        self.risk = RiskEngine(config)
        self.book = PositionBook()
        self.notifier = Notifier(config.notify)
        self.broker = self._make_broker()
        self.chat = ChatSession(config, self.state)
        self.audit = audit or AuditLog(config.risk.audit_path,
                                       fingerprint(config.to_dict()))
        self.monitor = FeedMonitor(
            bar_seconds=timeframe_seconds(config.market.timeframe),
            max_stale_bars=config.feed.max_stale_bars,
            max_stale_seconds=config.feed.max_stale_seconds,
        )

        self.context: Optional[MarketContext] = None
        self.signals: list[TradeSignal] = []
        self.last_actionable: Optional[TradeSignal] = None
        self.started_at = utcnow()
        self.source = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._evaluating = False
        self._display: list[dict[str, Any]] = []
        self._decision_position = None      # position state as of the current decision

    # ------------------------------------------------------------- wiring

    def _make_broker(self):
        if self.cfg.execution.mode == "live":
            return LiveBroker()  # raises with an explanatory message by design
        return PaperBroker(self.cfg, on_close=self._on_trade_closed)

    def build_source(self):
        f, m = self.cfg.feed, self.cfg.market
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
        path = self.cfg.feed.warmup_csv
        if not path:
            return 0
        try:
            candles = read_csv(path)
        except (FileNotFoundError, OSError) as exc:
            log.warning("warmup CSV unusable: %s", exc)
            return 0
        self.series.extend(candles)
        log.info("warmed up with %d bars from %s", len(candles), path)
        return len(candles)

    # -------------------------------------------------------------- loop

    async def run(self) -> None:
        self._running = True
        self.source = self.build_source()
        self.warmup()
        log.info(
            "starting: %s %s via %s feed, strategy=%s, execution=%s, AI=%s",
            self.cfg.market.symbol, self.cfg.market.timeframe,
            getattr(self.source, "name", "?"), self.strategy.spec.name,
            self.cfg.execution.mode, "on" if self.cfg.ai.enabled else "off",
        )
        self.audit.append("session_start", {
            "config": self.cfg.to_dict(),
            "strategy": self.strategy.spec.to_dict(),
            "feed": getattr(self.source, "name", self.cfg.feed.source),
        })
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
            self.audit.append("error", {"message": "data feed stopped"})
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
        self.monitor.record(candle.ts, is_new)

        # Positions are marked against every bar, forming or closed, so a stop
        # that trades through intrabar is seen as soon as the feed shows it.
        # This runs *before* any new entry, so a position opened at this bar's
        # close is never filled against the same bar.
        for trade in self.broker.on_candle(candle):
            self.book.record_exit(trade.signal_id)
            log.info("closed %s %s for %+.2f (%.2fR)", trade.side.value, trade.symbol,
                     trade.pnl, trade.r_multiple)
        for position in self.broker.positions:
            self.book.record_partial(position.signal.id, position.remaining)

        self.context = self._build_context()
        if self.context:
            self._display = self.strategy.display(self.context)
            self.bus.publish("tick", {
                "candle": candle.to_dict(),
                "context": self.context.to_dict(self._display),
                "positions": [p.to_dict(candle.close) for p in self.broker.positions],
                "position_state": self.position_state().to_dict(),
                "equity": round(self.risk.equity, 2),
                "feed": self.monitor.to_dict(self._in_session()),
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

    def _build_context(self) -> Optional[MarketContext]:
        closed = [c for c in self.series.candles if c.closed]
        if not closed:
            return None
        decision_ts = closed[-1].ts
        health = self.monitor.health(self.context_builder.in_session(decision_ts))
        return self.context_builder.build(
            self.series, decision_ts, health, self.strategy.spec.structure_window)

    def _in_session(self) -> bool:
        return self.context.in_session if self.context else True

    def position_state(self):
        return self.book.state_of(self.broker.positions)

    # ------------------------------------------------------------ evaluate

    async def evaluate(self) -> Optional[TradeSignal]:
        """Run the full gate chain for the just-closed bar."""
        ctx = self.context
        if ctx is None:
            return None
        self.risk.tick(ctx.decision_ts)

        verdict = self.strategy.evaluate(ctx)
        self.strategy.observe(ctx, verdict)

        if not verdict.passed:
            self.bus.publish("evaluation", {
                "actionable": False,
                "decision_ts": ctx.decision_ts.isoformat(),
                "verdict": verdict.to_dict(),
                "reason": "; ".join(verdict.notes) or "no setup",
                "feed": ctx.feed.to_dict(),
                "gate": self.strategy.gate.to_dict(),
            })
            return None

        # Captured once, before anything is opened: this is the state the risk
        # decision was actually made against, and it is what the audit record
        # has to show. Reading it again after an entry would log the position
        # the signal created, not the one it was judged against.
        position = self.position_state()
        self._decision_position = position
        admitted, admit_reason = self.strategy.admit(verdict, position.side)
        if not admitted:
            self.bus.publish("evaluation", {
                "actionable": False,
                "decision_ts": ctx.decision_ts.isoformat(),
                "verdict": verdict.to_dict(),
                "reason": admit_reason,
                "suppressed": True,
                "gate": self.strategy.gate.to_dict(),
            })
            return None

        signal = self.strategy.build_signal(ctx, verdict)
        if signal is None:
            return None
        signal.sequence = self.audit.sequence + 1
        # One qualifying setup produces exactly one signal, taken or not.
        self.strategy.mark_fired(verdict.side)

        # --- backup layer: does the screen contradict the feed? -------------
        vision_note = None
        if self.vision.enabled and self._vision_should_run(ctx):
            reading = await self.vision.read()
            vision_note = self.vision.note_for_prompt(reading)
            conflict = self.vision.disagreement(reading, ctx.price)
            if conflict:
                return self._block(signal, f"screen/feed disagreement: {conflict}")

        # --- AI confirmation ------------------------------------------------
        signal.ai = await self.confirmation.confirm(
            signal, ctx, list(ctx.candles), vision_note)
        allowed, reason = self.confirmation.accepts(signal.ai)
        if not allowed:
            return self._block(signal, reason)

        # --- position reconciliation ---------------------------------------
        report = self.book.reconcile(self.broker.positions, utcnow().isoformat())
        if not report.ok:
            log.error("reconciliation failed: %s", report.message)
            self.risk.trip_kill_switch(f"auto: {report.message}")
            return self._block(signal, f"reconciliation: {report.message}")

        # --- risk engine ----------------------------------------------------
        decision = self.risk.evaluate(signal, position, ctx.decision_ts, True, report.message)
        signal.risk = decision
        if not decision.allowed:
            return self._block(signal, f"risk[{decision.reason_code}]: {decision.reason}")

        signal.qty = decision.qty
        signal.risk_pct = decision.risk_pct
        signal.risk_amount = decision.risk_amount
        signal.actionable = True
        signal.reasons.append(reason)

        if self.cfg.execution.mode == "paper":
            self.broker.open(signal, decision.qty)
            self.book.record_entry(signal, decision.qty)
            self.risk.register_entry(signal)
        elif self.cfg.execution.mode == "alerts_only":
            self.risk.register_entry(signal)

        self._record(signal, ctx)
        self.last_actionable = signal
        self.notifier.signal(signal)
        log.info(signal.headline())
        return signal

    def _vision_should_run(self, ctx: MarketContext) -> bool:
        if self.cfg.vision.mode == "always":
            return True
        # "backup" mode: the feed monitor already flags staleness, so this only
        # needs to catch the softer case of a feed that is lagging but not yet
        # over the hard threshold.
        return ctx.feed.bar_age_seconds > self.cfg.vision.stale_feed_seconds

    def _block(self, signal: TradeSignal, reason: str) -> TradeSignal:
        signal.actionable = False
        signal.blocked_by = reason
        self._record(signal, self.context)
        self.notifier.signal(signal)
        log.info("no trade: %s", reason)
        return signal

    def _record(self, signal: TradeSignal, ctx: Optional[MarketContext]) -> None:
        # `_decision_position` is the state as of the decision, not as of now -
        # by the time a taken signal is recorded its own entry is already open.
        position = self._decision_position or self.position_state()
        record = self.audit.signal(
            signal,
            ctx.to_dict(self._display) if ctx else {},
            {"position_state": position.to_dict(),
             "position_state_now": self.position_state().to_dict(),
             "gate": self.strategy.gate.to_dict()},
        )
        signal.sequence = record["sequence"]
        self.signals.append(signal)
        self.signals = self.signals[-200:]
        self.bus.publish("signal", signal.to_dict())

    def _on_trade_closed(self, trade: ClosedTrade) -> None:
        self.risk.register_close(trade)
        self.book.record_exit(trade.signal_id)
        self.audit.append("trade_closed", {
            "trade": trade.to_dict(), "equity": round(self.risk.equity, 2)})
        self.bus.publish("trade_closed", {
            "trade": trade.to_dict(), "risk": self.risk.snapshot()})
        if self.risk.state.kill_switch:
            self.bus.publish("kill_switch", {"active": True, "reason": self.risk.state.kill_reason})
            self.notifier.desktop("Trading halted", self.risk.state.kill_reason)

    # ------------------------------------------------------------- controls

    def kill(self, reason: str = "manual: stopped from the dashboard") -> dict[str, Any]:
        self.risk.trip_kill_switch(reason)
        self.audit.append("kill_switch", {"active": True, "reason": reason})
        self.bus.publish("kill_switch", {"active": True, "reason": reason})
        return self.risk.snapshot()

    def resume(self) -> dict[str, Any]:
        self.risk.release_kill_switch()
        self.risk.clear_cooldown()
        self.audit.append("kill_switch", {"active": False, "reason": ""})
        self.bus.publish("kill_switch", {"active": False, "reason": ""})
        return self.risk.snapshot()

    def flatten(self) -> list[dict[str, Any]]:
        last = self.series.last
        if not last or not isinstance(self.broker, PaperBroker):
            return []
        closed = self.broker.close_all(last.close, last, "flatten")
        for trade in closed:
            self.book.record_exit(trade.signal_id)
        self.audit.append("flatten", {"closed": [t.signal_id for t in closed]})
        return [t.to_dict() for t in closed]

    def reconcile(self) -> dict[str, Any]:
        report = self.book.reconcile(self.broker.positions, utcnow().isoformat())
        self.audit.append("reconciliation", report.to_dict())
        return report.to_dict()

    # --------------------------------------------------------------- state

    def status(self) -> dict[str, Any]:
        last = self.series.last
        in_session = self._in_session()
        return {
            "running": self._running,
            "symbol": self.cfg.market.symbol,
            "timeframe": self.cfg.market.timeframe,
            "strategy": self.strategy.spec.name,
            "execution_mode": self.cfg.execution.mode,
            "feed_source": getattr(self.source, "name", self.cfg.feed.source),
            "feed": self.monitor.to_dict(in_session),
            "series_length": len(self.series),
            "last_price": last.close if last else None,
            "started_at": self.started_at.isoformat(),
            "overlays": self.strategy.spec.overlays(),
            "gate": self.strategy.gate.to_dict(),
            "reconciliation": self.book.last_reconciliation.to_dict(),
            "audit": {"path": str(self.audit.path), "records": self.audit.sequence},
            "ai": {**ai_availability(), "enabled": self.cfg.ai.enabled,
                   "model": self.cfg.ai.model, "on_failure": self.cfg.ai.on_failure,
                   "last_error": self.confirmation.last_error},
            "vision": self.vision.status(),
        }

    def state(self) -> dict[str, Any]:
        last = self.series.last
        price = last.close if last else None
        return {
            "status": self.status(),
            "context": self.context.to_dict(self._display) if self.context else None,
            "risk": self.risk.snapshot(),
            "positions": [p.to_dict(price) for p in self.broker.positions],
            "position_state": self.position_state().to_dict(),
            "recent_signals": [s.to_dict() for s in self.signals[-12:]],
            "last_actionable_signal": self.last_actionable.to_dict() if self.last_actionable else None,
            "closed_trades": [t.to_dict() for t in getattr(self.broker, "closed", [])[-20:]],
            "performance": self.performance(),
            "strategy": self.strategy.spec.to_dict(),
        }

    def performance(self) -> dict[str, Any]:
        trades = list(getattr(self.broker, "closed", []))
        if not trades:
            return {"trades": 0, "win_rate": None, "net_pnl": 0.0, "expectancy_r": None}
        wins = [t for t in trades if t.win]
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
            "win_rate": round(len(wins) / len(trades), 3),
            "net_pnl": round(sum(t.pnl for t in trades), 2),
            "expectancy_r": round(sum(t.r_multiple for t in trades) / len(trades), 3),
            "best_r": round(max(t.r_multiple for t in trades), 2),
            "worst_r": round(min(t.r_multiple for t in trades), 2),
        }

    # -------------------------------------------------------------- inbound

    def submit_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(self.source, WebhookSource):
            return {"accepted": False, "reason": "feed.source is not 'webhook'"}
        result = self.source.submit(payload)
        if not result.get("accepted"):
            self.audit.append("webhook_rejected", {"reason": result.get("reason")})
        return result
