"""Equity-aware daily-loss assessment. OFF by default.

WHY THIS EXISTS

`RiskManager.evaluate_daily_loss` (app/risk/manager.py) answers the
question "have I realized more than X% of loss today?". An audit of it
found three things that make that a weaker guarantee than the name
suggests:

  1. OPEN POSITIONS ARE INVISIBLE. Only `Trade.pnl_usd` on trades whose
     `closed_at` falls inside today is summed. A book that is down $400
     unrealized reports a daily loss of $0 and the bot keeps buying. The
     limit only bites after the damage has been crystallised, which is
     exactly the moment it can no longer prevent anything.

  2. THE REFERENCE IS A CONSTANT, NOT EQUITY.
     `PORTFOLIO_STARTING_BALANCE_USD` is a fixed 1000.0 that never
     re-bases. After the account has fallen to $600 the "5% daily limit"
     is still $50 of the original $1000 - 8.3% of what is actually left.
     The limit therefore gets *more* permissive, in percentage-of-capital
     terms, precisely as the account gets smaller. That is backwards.

  3. IT IS NEVER CONSULTED ON THE BUY PATH. The only production caller is
     `_check_halt_conditions`, which runs after a close. Entries are gated
     only indirectly, by the halt flag a *previous* close happened to set.

What the audit did NOT find - and this matters, because "fix" it would
have been a real bug:

  FEES ARE NOT MISSING AND MUST NOT BE ADDED. `pnl_usd = proceeds -
  cost_basis` is computed from fill prices, and the paper fill model bakes
  the fee into the fill price itself (`total_cost = impact + spread + fee
  + adverse_drift`, app/execution/fill_model.py). The fee is already
  inside `pnl_usd`. Subtracting `Trade.fee_usd` on top would charge every
  fee twice and halt the bot on losses it never took.

WHAT THIS MODULE DOES INSTEAD

  drawdown = day_start_equity - current_equity

Equity is cash plus the marked-to-market book. That single subtraction
captures realized and unrealized P&L at once, and - the reason it is
written as a subtraction rather than a sum of parts - it *cannot* double
count. There are no two terms to add up wrongly. Fees are inside the fill
prices that moved cash and set entry prices, so they are counted exactly
once, by construction rather than by care.

Realized and unrealized are still reported, as diagnostics only. They are
not summed into the verdict. Note that they do not reconcile to the
drawdown for a position carried overnight: `unrealized_pnl_open_usd` is
measured from entry, while the drawdown is measured from midnight, and for
a position opened yesterday those are different reference points. That
discrepancy is not an error to fix - it is the reason the verdict comes
from equity and not from adding the two together.

WHAT IT DOES WITH A PRICE FEED THAT IS DOWN

`value_open_positions` values a position it cannot price at cost, which
*inflates* equity and therefore *understates* the drawdown - biased
towards letting the bot keep trading, which is the dangerous direction.
So the unpriced notional is treated as a band of uncertainty: if the
verdict would flip when that band is resolved against us, the assessment
returns `measurable=False` and breaches. The unpriced amount is reported
as unpriced, never as zero.

THE FLAG

`RISK_EQUITY_AWARE_DAILY_LOSS` defaults to false and nothing calls this
module while it is false. Turning it on is a change to live risk behavior
and mints a new strategy version (see app/strategy/version.py) so the
paper-collection dataset is split rather than silently pooled.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from app import models
from app.services import portfolio
from app.state import get_state, set_state

logger = logging.getLogger(__name__)

DAY_START_EQUITY_KEY = "risk_day_start_equity"
LAST_EQUITY_KEY = "risk_last_equity"

# How the day-start reference was arrived at. Recorded because the three
# are not equally trustworthy and a verdict should say which one it used.
SOURCE_STORED = "stored"                     # captured earlier today, still current
SOURCE_PREVIOUS_CLOSE = "previous_close"     # last equity seen before midnight
SOURCE_FIRST_OBSERVATION = "first_observation"  # no prior reading exists at all


@dataclass(frozen=True)
class DailyLossAssessment:
    """One day's loss picture. `breached` is the only field that gates."""

    breached: bool
    reason: str

    day_start_equity_usd: float
    current_equity_usd: float
    drawdown_usd: float
    limit_usd: float
    remaining_budget_usd: float

    # Diagnostics. Deliberately NOT summed into the verdict - see the
    # module docstring on why they do not reconcile to the drawdown.
    realized_pnl_today_usd: float
    unrealized_pnl_open_usd: float

    open_positions: int
    unpriced_positions: int
    unpriced_usd: float
    measurable: bool
    day_start_source: str

    def as_dict(self) -> dict:
        return asdict(self)


def _day_key(now: dt.datetime) -> str:
    return now.date().isoformat()


def _as_utc(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value


def _realized_today(db: Session, start: dt.datetime, end: dt.datetime) -> float:
    """Realized P&L booked today. Same query the legacy check uses, so the
    two agree on the one number they both compute."""
    rows = (
        db.query(models.Trade.pnl_usd)
        .filter(models.Trade.closed_at >= start, models.Trade.closed_at < end)
        .all()
    )
    return sum(pnl for (pnl,) in rows if pnl is not None)


def _open_cost_basis_usd(db: Session) -> float:
    rows = (
        db.query(models.Position.qty, models.Position.entry_price)
        .filter(models.Position.status == models.PositionStatus.OPEN.value)
        .all()
    )
    return sum(qty * entry for qty, entry in rows if qty and entry)


def _resolve_day_start_equity(
    db: Session, *, now: dt.datetime, current_equity: float, persist: bool
) -> tuple[float, str]:
    """The equity this day started at, and where that number came from.

    Preference order, best first:

      stored           - already captured during today. Reused verbatim so
                         the budget cannot be reset by a restart.
      previous_close   - the last equity observed before midnight. This is
                         a real measurement, just a slightly old one, and
                         it is the closest thing to a true midnight mark
                         that exists without a scheduler.
      first_observation - nothing has ever been recorded. Today's budget is
                         anchored to right now, which is the only honest
                         option; it is flagged so a reader knows the day
                         may already have been underway.
    """
    today = _day_key(now)
    stored = get_state(db, DAY_START_EQUITY_KEY, None)
    if isinstance(stored, dict) and stored.get("date") == today:
        equity = stored.get("equity_usd")
        if isinstance(equity, (int, float)):
            return float(equity), str(stored.get("source", SOURCE_STORED))

    last = get_state(db, LAST_EQUITY_KEY, None)
    day_start = current_equity
    source = SOURCE_FIRST_OBSERVATION
    if isinstance(last, dict):
        observed_at = last.get("observed_at")
        equity = last.get("equity_usd")
        if isinstance(equity, (int, float)) and isinstance(observed_at, str):
            try:
                seen = dt.datetime.fromisoformat(observed_at)
            except ValueError:
                seen = None
            if seen is not None and _day_key(_as_utc(seen) or now) != today:
                day_start = float(equity)
                source = SOURCE_PREVIOUS_CLOSE

    if persist:
        set_state(
            db,
            DAY_START_EQUITY_KEY,
            {
                "date": today,
                "equity_usd": day_start,
                "captured_at": now.isoformat(),
                "source": source,
            },
        )
    return day_start, source


async def assess(
    db: Session,
    *,
    daily_loss_limit_pct: float,
    now: dt.datetime | None = None,
    persist: bool = True,
) -> DailyLossAssessment:
    """Measure today's drawdown against the day's own starting equity.

    `persist=False` makes this a pure read - use it from the dashboard or
    any other caller that must not anchor the day as a side effect of
    looking at it.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(days=1)

    valuation = await portfolio.value_open_positions(db)
    cash = portfolio.get_cash_balance_usd(db)
    current_equity = cash + valuation.total_usd

    day_start_equity, source = _resolve_day_start_equity(
        db, now=now, current_equity=current_equity, persist=persist
    )

    if persist:
        set_state(
            db, LAST_EQUITY_KEY,
            {"equity_usd": current_equity, "observed_at": now.isoformat()},
        )

    drawdown = day_start_equity - current_equity
    limit = day_start_equity * daily_loss_limit_pct
    realized = _realized_today(db, start, end)
    unrealized = valuation.total_usd - _open_cost_basis_usd(db)

    def build(breached: bool, reason: str, measurable: bool = True) -> DailyLossAssessment:
        return DailyLossAssessment(
            breached=breached,
            reason=reason,
            day_start_equity_usd=day_start_equity,
            current_equity_usd=current_equity,
            drawdown_usd=drawdown,
            limit_usd=limit,
            remaining_budget_usd=max(limit - drawdown, 0.0),
            realized_pnl_today_usd=realized,
            unrealized_pnl_open_usd=unrealized,
            open_positions=valuation.positions,
            unpriced_positions=valuation.stale_positions,
            unpriced_usd=valuation.stale_usd,
            measurable=measurable,
            day_start_source=source,
        )

    # An account with nothing in it has no budget to spend, and a limit
    # computed as a percentage of zero would let every loss through.
    if day_start_equity <= 0:
        return build(
            True,
            f"day-start equity was ${day_start_equity:,.2f} - there is no capital to risk",
        )

    if drawdown >= limit:
        return build(
            True,
            f"equity is down ${drawdown:,.2f} today ({drawdown / day_start_equity * 100:.1f}% of "
            f"the ${day_start_equity:,.2f} it started at), at or past the "
            f"${limit:,.2f} daily limit",
        )

    # The unpriced band. Only fails closed when resolving it against us
    # would actually change the answer; a dead feed on a $5 position is
    # not a reason to stop trading.
    if valuation.stale_positions and (drawdown + valuation.stale_usd) >= limit:
        return build(
            True,
            f"cannot rule out a daily-loss breach: ${valuation.stale_usd:,.2f} across "
            f"{valuation.stale_positions} position(s) has no live price and is valued at cost. "
            f"Measured drawdown is ${drawdown:,.2f} against a ${limit:,.2f} limit, but the "
            f"unpriced amount is enough to cross it. Halting until the price feed recovers",
            measurable=False,
        )

    return build(
        False,
        f"equity drawdown ${drawdown:,.2f} of a ${limit:,.2f} daily budget "
        f"(${max(limit - drawdown, 0.0):,.2f} left)",
    )
