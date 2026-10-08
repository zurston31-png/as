"""Account-level risk limits: the only stateful object in the risk layer.

ARCHITECTURE.md section 8 fixes both the set of limits and the ORDER they are
checked in, each with its own WAIT reason:

    kill_switch -> daily_loss_limit -> max_consecutive_losses
      -> cooldown_active -> max_open_positions -> max_portfolio_heat

The order is not cosmetic. Checks run cheapest-and-most-terminal first, so the
reason recorded for a blocked bar is the most fundamental one that applies: an
account that has tripped its kill switch should report `kill_switch`, not
`max_open_positions`, even when both are true. Section 12's rejection
analysis counts these, so a blocked bar attributed to the wrong gate
misstates which constraint is actually shaping the results.

Three design decisions that are easy to get wrong:

**The clock is an argument, never a reading.** Nothing here calls
`datetime.now()`. Every method takes the timestamp it should reason about,
because this object is driven by the backtest clock and must produce the same
decisions on a replay. A limit manager that consults the wall clock cannot be
backtested at all.

**"Daily" means the trading session, not the UTC date.** The caller supplies
`session_date` from the trading calendar. Keying a daily loss limit on a UTC
date would reset it mid-session for any instrument whose session spans
midnight UTC -- the limit would silently stop limiting on exactly the
overnight sessions where it matters most. The same class of timezone error
was a real finding in `validation/splits.py`, where a seal boundary moved
with the caller's tzinfo.

**Equity is marked, limits are realized.** The kill switch watches *marked*
equity including open positions, because an account can be destroyed without
closing anything. The daily and weekly loss limits watch *realized* PnL,
because an unrealized excursion that recovers before the exit was never a
loss. Mixing the two makes a limit fire on noise or not at all.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from pydantic import Field

from flow_model.core.enums import WaitReason
from flow_model.core.model import FrozenModel

__all__ = ["LimitBreach", "LimitDecision", "RiskLimitManager"]


class LimitBreach:
    """Namespace of breach codes, one per limit, so rejection analysis works."""

    KILL_SWITCH = "kill_switch"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"
    COOLDOWN_ACTIVE = "cooldown_active"
    MAX_OPEN_POSITIONS = "max_open_positions"
    MAX_POSITIONS_PER_SYMBOL = "max_positions_per_symbol"
    MAX_PORTFOLIO_HEAT = "max_portfolio_heat"


class LimitDecision(FrozenModel):
    """Whether a new position may be opened, and if not, which limit said no."""

    allowed: bool
    breach: str = ""
    wait_reason: WaitReason | None = None
    detail: str = ""
    terminal: bool = False
    """True when the account is permanently halted, not merely paused."""

    headroom: dict[str, float] = Field(default_factory=dict)
    """How much slack remains in each measurable limit, for diagnostics."""


class RiskLimitManager:
    """Stateful gate in front of every new position.

    Usage from the backtest loop:

        decision = limits.check(ts=bar_ts, session_date=d, equity=eq,
                                symbol="NQ", prospective_risk=intent.risk_dollars)
        if not decision.allowed: journal.record_wait(decision)

    then, as trades resolve:

        limits.on_position_opened(symbol, risk_dollars)
        limits.on_trade_closed(ts=exit_ts, session_date=d, symbol=..., pnl=...)
    """

    def __init__(self, config, starting_equity: float | None = None) -> None:
        self.config = config
        if config.risk_per_trade_pct > config.max_risk_per_trade_pct:
            # Section 8: "All limits are hard: the manager cannot be configured
            # to exceed max_risk_per_trade_pct." Sizing also caps this; the
            # duplicate refusal here is deliberate, because a config that
            # states an impossible intention should fail loudly at construction
            # rather than be silently clamped on every bar.
            raise ValueError(
                f"risk_per_trade_pct {config.risk_per_trade_pct} exceeds "
                f"max_risk_per_trade_pct {config.max_risk_per_trade_pct}"
            )
        equity = float(starting_equity if starting_equity is not None else config.starting_equity)
        if equity <= 0:
            raise ValueError(f"starting equity must be positive, got {equity}")

        self.starting_equity = equity
        self.high_water_mark = equity
        self._killed = False
        self._kill_detail = ""
        self._consecutive_losses = 0
        self._cooldown_until: datetime | None = None
        self._realized_by_session: dict[date, float] = {}
        self._realized_by_week: dict[tuple[int, int], float] = {}
        self._open_risk: dict[str, float] = {}
        self._open_count_by_symbol: dict[str, int] = {}
        self._limit_trip_count: dict[str, int] = {}

    # --- state the caller feeds in ------------------------------------

    def mark_equity(self, equity: float) -> None:
        """Update the high-water mark. Called on every mark-to-market.

        The high-water mark only ever rises, so the kill switch measures
        drawdown from the peak rather than from the start -- an account up 40%
        and then down 15% from that peak has lost real money even though it is
        still ahead of where it began.
        """
        if equity > self.high_water_mark:
            self.high_water_mark = equity

    def on_position_opened(self, symbol: str, risk_dollars: float) -> None:
        self._open_risk[symbol] = self._open_risk.get(symbol, 0.0) + float(risk_dollars)
        self._open_count_by_symbol[symbol] = self._open_count_by_symbol.get(symbol, 0) + 1

    def on_trade_closed(
        self,
        *,
        ts: datetime,
        session_date: date,
        symbol: str,
        pnl: float,
        risk_dollars: float = 0.0,
    ) -> None:
        """Record a closed trade: realized PnL, streaks, and cooldowns."""
        self._open_count_by_symbol[symbol] = max(
            0, self._open_count_by_symbol.get(symbol, 0) - 1
        )
        remaining = self._open_risk.get(symbol, 0.0) - float(risk_dollars)
        if self._open_count_by_symbol[symbol] == 0 or remaining <= 0:
            # Clearing on the last exit rather than letting float residue
            # accumulate: heat that never returns to zero slowly blocks all
            # trading, which looks like a strategy that stopped finding setups.
            self._open_risk.pop(symbol, None)
        else:
            self._open_risk[symbol] = remaining

        self._realized_by_session[session_date] = (
            self._realized_by_session.get(session_date, 0.0) + float(pnl)
        )
        week = session_date.isocalendar()[:2]
        self._realized_by_week[week] = self._realized_by_week.get(week, 0.0) + float(pnl)

        if pnl < 0:
            self._consecutive_losses += 1
            minutes = float(self.config.cooldown_minutes_after_loss)
            if minutes > 0:
                self._arm_cooldown(ts, minutes)
        elif pnl > 0:
            self._consecutive_losses = 0
        # pnl == 0 is a scratch: it breaks nothing and starts nothing. Treating
        # it as a loss would let a run of break-even trades trip the streak
        # limit, which is not what "consecutive losses" means.

    def _arm_cooldown(self, ts: datetime, minutes: float) -> None:
        until = ts + timedelta(minutes=minutes)
        if self._cooldown_until is None or until > self._cooldown_until:
            self._cooldown_until = until

    # --- queries -------------------------------------------------------

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def open_positions(self) -> int:
        return sum(self._open_count_by_symbol.values())

    def portfolio_heat(self) -> float:
        """Sum of open R in dollars."""
        return sum(self._open_risk.values())

    def realized_for_session(self, session_date: date) -> float:
        return self._realized_by_session.get(session_date, 0.0)

    def realized_for_week(self, session_date: date) -> float:
        return self._realized_by_week.get(session_date.isocalendar()[:2], 0.0)

    def trip_counts(self) -> dict[str, int]:
        """How often each limit blocked a bar. Feeds rejection analysis."""
        return dict(self._limit_trip_count)

    # --- the gate ------------------------------------------------------

    def check(
        self,
        *,
        ts: datetime,
        session_date: date,
        equity: float,
        symbol: str,
        prospective_risk: float = 0.0,
    ) -> LimitDecision:
        """May a new position be opened now? Checked in section 8's order."""
        cfg = self.config
        self.mark_equity(equity)

        drawdown = (
            (self.high_water_mark - equity) / self.high_water_mark
            if self.high_water_mark > 0
            else 0.0
        )
        day_pnl = self.realized_for_session(session_date)
        week_pnl = self.realized_for_week(session_date)
        heat = self.portfolio_heat()
        headroom = {
            "drawdown_pct": round(drawdown, 10),
            "drawdown_limit_pct": float(cfg.kill_switch_drawdown_pct),
            "session_realized": round(day_pnl, 10),
            "session_loss_limit": round(-self.starting_equity * cfg.daily_loss_limit_pct, 10),
            "week_realized": round(week_pnl, 10),
            "consecutive_losses": float(self._consecutive_losses),
            "open_positions": float(self.open_positions()),
            "portfolio_heat": round(heat, 10),
            "portfolio_heat_limit": round(equity * cfg.max_portfolio_heat_pct, 10),
        }

        def block(code: str, reason: WaitReason, detail: str, terminal: bool = False):
            self._limit_trip_count[code] = self._limit_trip_count.get(code, 0) + 1
            return LimitDecision(
                allowed=False, breach=code, wait_reason=reason,
                detail=detail, terminal=terminal, headroom=headroom,
            )

        # 1. kill switch -- terminal, and checked first so a destroyed account
        #    never reports a lesser reason.
        if self._killed:
            return block(LimitBreach.KILL_SWITCH, WaitReason.RISK_LIMIT,
                         self._kill_detail or "kill switch previously tripped", terminal=True)
        if drawdown >= cfg.kill_switch_drawdown_pct:
            detail = (
                f"equity {round(equity, 2)} is {round(drawdown * 100, 2)}% below the "
                f"high-water mark {round(self.high_water_mark, 2)}, at or past the "
                f"{cfg.kill_switch_drawdown_pct * 100}% kill-switch threshold"
            )
            if cfg.kill_switch_is_terminal:
                self._killed = True
                self._kill_detail = detail
            return block(LimitBreach.KILL_SWITCH, WaitReason.RISK_LIMIT, detail,
                         terminal=bool(cfg.kill_switch_is_terminal))

        # 2. daily loss limit. Measured against STARTING equity of the account,
        #    not current: a limit that shrinks as the account shrinks lets a
        #    bad run continue at ever-smaller absolute size rather than
        #    stopping, which is the opposite of a circuit breaker.
        day_limit = -self.starting_equity * cfg.daily_loss_limit_pct
        if day_pnl <= day_limit:
            if cfg.cooldown_minutes_after_limit > 0:
                self._arm_cooldown(ts, float(cfg.cooldown_minutes_after_limit))
            return block(
                LimitBreach.DAILY_LOSS_LIMIT, WaitReason.RISK_LIMIT,
                f"session {session_date} realized {round(day_pnl, 2)}, at or past the "
                f"daily limit {round(day_limit, 2)}",
            )

        week_limit = -self.starting_equity * cfg.weekly_loss_limit_pct
        if week_pnl <= week_limit:
            return block(
                LimitBreach.WEEKLY_LOSS_LIMIT, WaitReason.RISK_LIMIT,
                f"week of {session_date} realized {round(week_pnl, 2)}, at or past the "
                f"weekly limit {round(week_limit, 2)}",
            )

        # 3. consecutive losses
        if self._consecutive_losses >= cfg.max_consecutive_losses:
            return block(
                LimitBreach.MAX_CONSECUTIVE_LOSSES, WaitReason.RISK_LIMIT,
                f"{self._consecutive_losses} consecutive losses, at or past the limit "
                f"{cfg.max_consecutive_losses}",
            )

        # 4. cooldown
        if self._cooldown_until is not None and ts < self._cooldown_until:
            return block(
                LimitBreach.COOLDOWN_ACTIVE, WaitReason.RISK_LIMIT,
                f"cooldown active until {self._cooldown_until.isoformat()}",
            )

        # 5. position counts
        if self.open_positions() >= cfg.max_open_positions:
            return block(
                LimitBreach.MAX_OPEN_POSITIONS, WaitReason.RISK_LIMIT,
                f"{self.open_positions()} open positions, at the limit "
                f"{cfg.max_open_positions}",
            )
        if self._open_count_by_symbol.get(symbol, 0) >= cfg.max_positions_per_symbol:
            return block(
                LimitBreach.MAX_POSITIONS_PER_SYMBOL, WaitReason.RISK_LIMIT,
                f"{self._open_count_by_symbol.get(symbol, 0)} open in {symbol}, at the "
                f"per-symbol limit {cfg.max_positions_per_symbol}",
            )

        # 6. portfolio heat, INCLUDING the trade being proposed. Checking heat
        #    without the prospective risk would admit a position that breaches
        #    the limit the instant it opens.
        prospective_heat = heat + float(prospective_risk)
        heat_cap = equity * cfg.max_portfolio_heat_pct
        if prospective_heat > heat_cap * (1.0 + 1e-9):
            return block(
                LimitBreach.MAX_PORTFOLIO_HEAT, WaitReason.RISK_LIMIT,
                f"open heat {round(heat, 2)} plus proposed {round(float(prospective_risk), 2)} "
                f"= {round(prospective_heat, 2)} exceeds the cap {round(heat_cap, 2)}",
            )

        return LimitDecision(allowed=True, headroom=headroom)
