"""The last gate before anything reaches your screen as actionable.

Order matters. The kill switch is checked first and short-circuits everything;
position-state reconciliation is checked next, because sizing a new trade
against a position book that disagrees with the broker is worse than not
trading at all.

Every refusal carries a stable `reason_code` so rejections can be counted and
compared across runs rather than string-matched.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..execution.book import PositionState
from ..models import ClosedTrade, RiskDecision, TradeSignal, utcnow
from .state import RiskStore


class RejectCode:
    """Stable identifiers for every way a signal can be refused."""

    KILL_SWITCH = "kill_switch_active"
    RECONCILIATION = "position_state_unreconciled"
    NO_STOP = "stop_loss_required"
    ZERO_STOP = "stop_distance_zero"
    MIN_RR = "reward_to_risk_below_minimum"
    SAME_DIRECTION = "already_in_this_direction"
    OPPOSITE_DIRECTION = "opposite_position_open"
    MAX_OPEN = "max_open_positions"
    MAX_TRADES = "max_trades_per_day"
    DAILY_LOSS = "daily_loss_limit"
    COOLDOWN = "cooldown_after_losses"
    SIZE_BELOW_MIN = "position_size_below_minimum"
    RISK_CEILING = "risk_per_trade_above_ceiling"
    OK = "ok"


class RiskEngine:
    def __init__(self, config) -> None:
        self.cfg = config
        self.store = RiskStore(config.risk.state_path, config.risk.starting_equity)
        if config.risk.kill_switch:
            self.store.state.kill_switch = True
            self.store.state.kill_reason = "config: kill_switch enabled"
        try:
            self.tz = ZoneInfo(config.market.session_tz)
        except (ZoneInfoNotFoundError, KeyError):
            self.tz = ZoneInfo("UTC")
        # The day is rolled lazily, from the timestamp of whatever event arrives
        # first. Rolling from the wall clock here would reset a backtest's
        # counters the moment a historical bar was evaluated.

    # ------------------------------------------------------------- queries

    @property
    def state(self):
        return self.store.state

    @property
    def equity(self) -> float:
        return self.store.state.equity

    def tick(self, now: Optional[datetime] = None) -> None:
        """Roll the trading day over. Safe to call as often as you like."""
        now = now or utcnow()
        self.store.roll_day(now.astimezone(self.tz).date())

    # -------------------------------------------------------------- sizing

    def position_size(self, signal: TradeSignal) -> dict[str, Any]:
        """The sizing formula, written out so the audit log can re-derive it.

            risk_dollars           = equity x risk_percent
            stop_risk_per_contract = |entry - stop| x point_value
            contracts              = floor(risk_dollars / stop_risk_per_contract)

        Rounding is always *down*. If that reaches zero the trade is refused -
        risk is never increased to make a position fit.
        """
        m, r = self.cfg.market, self.cfg.risk
        equity = self.store.state.equity or r.starting_equity
        risk_percent = r.risk_per_trade_pct / 100.0
        risk_dollars = equity * risk_percent
        stop_distance = abs(signal.entry - signal.stop)
        stop_risk_per_contract = stop_distance * m.point_value

        record: dict[str, Any] = {
            "equity": round(equity, 2),
            "risk_percent": risk_percent,
            "risk_dollars": round(risk_dollars, 4),
            "entry": signal.entry,
            "stop": signal.stop,
            "stop_distance": round(stop_distance, 10),
            "point_value": m.point_value,
            "stop_risk_per_contract": round(stop_risk_per_contract, 6),
            "qty_step": m.qty_step,
            "min_qty": m.min_qty,
        }
        if stop_risk_per_contract <= 0:
            record.update({"raw_contracts": 0.0, "contracts": 0.0,
                           "risk_amount": 0.0, "risk_pct": 0.0})
            return record

        raw = risk_dollars / stop_risk_per_contract
        step = m.qty_step or 1.0
        contracts = round(math.floor(raw / step) * step, 10)
        if contracts < m.min_qty:
            contracts = 0.0
        risk_amount = contracts * stop_risk_per_contract
        record.update({
            "raw_contracts": round(raw, 6),
            "contracts": contracts,
            "risk_amount": round(risk_amount, 4),
            "risk_pct": risk_amount / equity if equity else 0.0,
        })
        return record

    # -------------------------------------------------------------- gating

    def evaluate(
        self,
        signal: TradeSignal,
        position: Optional[PositionState] = None,
        now: Optional[datetime] = None,
        reconciled: bool = True,
        reconciliation_message: str = "",
    ) -> RiskDecision:
        now = now or utcnow()
        self.tick(now)
        r = self.cfg.risk
        st = self.store.state
        position = position or PositionState()

        def refuse(code: str, reason: str, sizing: Optional[dict] = None) -> RiskDecision:
            return RiskDecision(False, reason, reason_code=code, violations=[code],
                                sizing=sizing or {})

        if st.kill_switch:
            return refuse(RejectCode.KILL_SWITCH, f"kill switch active ({st.kill_reason})")

        if not reconciled:
            return refuse(RejectCode.RECONCILIATION,
                          reconciliation_message or "position state could not be reconciled")

        if r.require_stop and not signal.stop:
            return refuse(RejectCode.NO_STOP, "a stop loss is required on every signal")
        if abs(signal.entry - signal.stop) <= 0:
            return refuse(RejectCode.ZERO_STOP, "entry and stop are the same price")

        rr = abs(signal.tp1 - signal.entry) / abs(signal.entry - signal.stop)
        if rr < r.min_rr:
            return refuse(RejectCode.MIN_RR,
                          f"R:R to TP1 is {rr:.2f}, below the {r.min_rr:g} minimum")

        if position.side is signal.side and not position.flat:
            return refuse(RejectCode.SAME_DIRECTION,
                          f"already {position.side.value} - not adding to the same direction")
        if not position.flat and position.side is not signal.side:
            return refuse(RejectCode.OPPOSITE_DIRECTION,
                          f"holding a {position.state} position; flatten before reversing")
        if position.count >= r.max_open_positions:
            return refuse(RejectCode.MAX_OPEN, f"already holding {position.count} position(s)")

        if st.trades_today >= r.max_trades_per_day:
            return refuse(RejectCode.MAX_TRADES,
                          f"daily trade cap reached ({st.trades_today}/{r.max_trades_per_day})")

        if st.daily_pnl_pct <= -abs(r.max_daily_loss_pct):
            self.trip_kill_switch(f"auto: daily loss limit hit ({st.daily_pnl_pct:.2f}%)")
            return refuse(RejectCode.DAILY_LOSS,
                          f"daily loss limit hit ({st.daily_pnl_pct:.2f}%)")

        if self.store.cooldown_active(now):
            return refuse(RejectCode.COOLDOWN,
                          f"cooling down after {st.consecutive_losses} loss(es) "
                          f"until {st.cooldown_until}")

        sizing = self.position_size(signal)
        if sizing["contracts"] <= 0:
            return refuse(
                RejectCode.SIZE_BELOW_MIN,
                f"position size below minimum: {sizing['risk_dollars']:.2f} of risk budget "
                f"buys {sizing.get('raw_contracts', 0):.4f} contracts at "
                f"{sizing['stop_risk_per_contract']:.2f} each (minimum {sizing['min_qty']:g})",
                sizing,
            )
        if sizing["risk_pct"] > r.max_risk_per_trade_pct / 100.0:
            return refuse(
                RejectCode.RISK_CEILING,
                f"sized risk {sizing['risk_pct'] * 100:.2f}% exceeds the "
                f"{r.max_risk_per_trade_pct:g}% ceiling", sizing)

        return RiskDecision(
            True, "within all limits", reason_code=RejectCode.OK,
            qty=sizing["contracts"], risk_amount=sizing["risk_amount"],
            risk_pct=sizing["risk_pct"], sizing=sizing,
        )

    # ------------------------------------------------------------ mutation

    def register_entry(self, signal: TradeSignal) -> None:
        self.tick(signal.ts)
        self.store.state.trades_today += 1
        self.store.save()

    def register_close(self, trade: ClosedTrade) -> None:
        """Book a realised result and apply the loss-streak / cooldown rules."""
        st = self.store.state
        r = self.cfg.risk
        self.tick(trade.closed_at)
        st.equity += trade.pnl
        st.realized_today += trade.pnl
        if trade.pnl < 0:
            st.consecutive_losses += 1
            if st.consecutive_losses >= r.max_consecutive_losses and r.cooldown_minutes_after_loss > 0:
                # Anchored to the trade's own exit time, not wall clock, so a
                # historical replay cools down for 30 bars' worth of minutes
                # rather than for the whole backtest.
                base = trade.closed_at or utcnow()
                st.cooldown_until = (
                    base + timedelta(minutes=r.cooldown_minutes_after_loss)
                ).isoformat()
        else:
            st.consecutive_losses = 0
            st.cooldown_until = None

        if st.daily_pnl_pct <= -abs(r.daily_drawdown_kill_pct):
            self.trip_kill_switch(f"auto: daily drawdown {st.daily_pnl_pct:.2f}%")
        elif st.daily_pnl_pct <= -abs(r.max_daily_loss_pct):
            self.trip_kill_switch(f"auto: daily loss limit {st.daily_pnl_pct:.2f}%")
        self.store.save()

    def trip_kill_switch(self, reason: str) -> None:
        self.store.state.kill_switch = True
        self.store.state.kill_reason = reason
        self.store.save()

    def release_kill_switch(self) -> None:
        self.store.state.kill_switch = False
        self.store.state.kill_reason = ""
        self.store.save()

    def clear_cooldown(self) -> None:
        self.store.state.cooldown_until = None
        self.store.state.consecutive_losses = 0
        self.store.save()

    def snapshot(self) -> dict[str, Any]:
        r = self.cfg.risk
        st = self.store.state
        return {
            **st.to_dict(),
            "limits": {
                "risk_per_trade_pct": r.risk_per_trade_pct,
                "max_trades_per_day": r.max_trades_per_day,
                "max_daily_loss_pct": r.max_daily_loss_pct,
                "max_open_positions": r.max_open_positions,
                "max_consecutive_losses": r.max_consecutive_losses,
                "cooldown_minutes_after_loss": r.cooldown_minutes_after_loss,
                "min_rr": r.min_rr,
            },
            "trades_remaining": max(0, r.max_trades_per_day - st.trades_today),
            "cooldown_active": self.store.cooldown_active(utcnow()),
        }
