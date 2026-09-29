"""The last gate before anything reaches your screen as actionable.

Order matters: the kill switch is checked first and short-circuits everything,
because that's the whole point of a kill switch.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import Config
from ..models import ClosedTrade, RiskDecision, TradeSignal, utcnow
from .state import RiskStore


class RiskEngine:
    def __init__(self, config: Config) -> None:
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

    # -------------------------------------------------------------- gating

    def evaluate(
        self, signal: TradeSignal, open_positions: int, now: Optional[datetime] = None
    ) -> RiskDecision:
        now = now or utcnow()
        self.tick(now)
        r = self.cfg.risk
        st = self.store.state
        violations: list[str] = []

        if st.kill_switch:
            return RiskDecision(False, f"kill switch active ({st.kill_reason})", violations=["kill_switch"])

        if r.require_stop and not signal.stop:
            violations.append("missing_stop")
        if signal.stop_distance <= 0:
            violations.append("zero_stop_distance")
        if violations:
            return RiskDecision(False, "a stop loss is required on every signal", violations=violations)

        rr = abs(signal.tp1 - signal.entry) / signal.stop_distance
        if rr < r.min_rr:
            violations.append("min_rr")
            return RiskDecision(False, f"R:R to TP1 is {rr:.2f}, below the {r.min_rr:g} minimum",
                                violations=violations)

        if open_positions >= r.max_open_positions:
            violations.append("max_open_positions")
            return RiskDecision(False, f"already holding {open_positions} position(s)",
                                violations=violations)

        if st.trades_today >= r.max_trades_per_day:
            violations.append("max_trades_per_day")
            return RiskDecision(False, f"daily trade cap reached ({st.trades_today}/{r.max_trades_per_day})",
                                violations=violations)

        if st.daily_pnl_pct <= -abs(r.max_daily_loss_pct):
            self.trip_kill_switch(f"auto: daily loss limit hit ({st.daily_pnl_pct:.2f}%)")
            violations.append("max_daily_loss")
            return RiskDecision(False, f"daily loss limit hit ({st.daily_pnl_pct:.2f}%)",
                                violations=violations)

        if self.store.cooldown_active(now):
            violations.append("cooldown")
            return RiskDecision(
                False,
                f"cooling down after {st.consecutive_losses} loss(es) until {st.cooldown_until}",
                violations=violations,
            )

        qty, risk_amount, risk_pct = self.position_size(signal)
        if qty <= 0:
            violations.append("size_zero")
            return RiskDecision(
                False,
                "position size rounds to zero - stop is too wide for the configured risk",
                violations=violations,
            )
        if risk_pct > r.max_risk_per_trade_pct / 100.0:
            violations.append("max_risk_per_trade")
            return RiskDecision(
                False,
                f"sized risk {risk_pct * 100:.2f}% exceeds the {r.max_risk_per_trade_pct:g}% ceiling",
                violations=violations,
            )

        return RiskDecision(True, "within all limits", qty=qty, risk_amount=risk_amount, risk_pct=risk_pct)

    def position_size(self, signal: TradeSignal) -> tuple[float, float, float]:
        """Risk-based sizing: qty = (equity * risk%) / (stop distance * point value)."""
        m, r = self.cfg.market, self.cfg.risk
        equity = self.store.state.equity or r.starting_equity
        budget = equity * (r.risk_per_trade_pct / 100.0)
        per_unit_risk = signal.stop_distance * m.point_value
        if per_unit_risk <= 0:
            return 0.0, 0.0, 0.0
        raw = budget / per_unit_risk
        step = m.qty_step or 1.0
        qty = math.floor(raw / step) * step
        qty = round(qty, 10)
        if qty < m.min_qty:
            qty = 0.0
        risk_amount = qty * per_unit_risk
        risk_pct = risk_amount / equity if equity else 0.0
        return qty, risk_amount, risk_pct

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
                until = base + timedelta(minutes=r.cooldown_minutes_after_loss)
                st.cooldown_until = until.isoformat()
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

    def snapshot(self) -> dict:
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
            },
            "trades_remaining": max(0, r.max_trades_per_day - st.trades_today),
            "cooldown_active": self.store.cooldown_active(utcnow()),
        }
