"""The global entry gate.

One question, asked before every new position: is the bot in a fit state to
be opening one at all?

This is not the per-trade risk check in app/risk/manager.py, which asks
"is THIS trade within limits?". It asks whether the machinery underneath
that question can be trusted right now. The two failure modes it exists for
are the ones that do not announce themselves:

    the bot keeps trading on data that stopped updating an hour ago
    the bot keeps sizing positions off a cash balance that is wrong

Both look completely normal in the logs. Both keep producing trades. Both
produce a record that is worthless afterwards, because you cannot tell
which results came from the strategy and which came from the fault.

FAIL CLOSED. Every check answers "may we trade?" and anything other than a
confident yes is a no. A check that cannot run - because the database
raised, because a health record is missing - counts as a failure, not as an
absence of evidence.

WHAT IT DOES NOT DO. It never closes existing positions. Stopping new
entries is safe; force-liquidating a book because a price feed hiccuped is
not, and would turn a data problem into a realised loss. Open positions
stay visible and stay managed by the monitor, which has its own stale-price
handling.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import settings
from app.risk.manager import is_trading_halted
from app.safety import reconcile as reconcile_mod
from app.services import api_health, portfolio

logger = logging.getLogger(__name__)

# Upstream services the bot cannot make a safe entry decision without.
# A degraded rug scanner is survivable (the engine fails closed on missing
# security data by itself); a dead price feed is not, because every size,
# stop and exit is computed from it.
CRITICAL_SERVICES: tuple[str, ...] = ("dexscreener",)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    @property
    def blocking(self) -> bool:
        return not self.passed


@dataclass
class Verdict:
    may_trade: bool
    checks: list[Check] = field(default_factory=list)
    # True when the switch was turned off rather than when it passed. The
    # distinction matters on a dashboard: "all integrity checks passed" and
    # "nobody checked" look identical to a reader and mean opposite things.
    disabled: bool = False

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.blocking]

    @property
    def reason(self) -> str:
        if self.disabled:
            return "kill switch disabled by configuration - integrity was NOT checked"
        if self.may_trade:
            return "all integrity checks passed"
        return "; ".join(c.detail for c in self.failures)

    def as_dict(self) -> dict:
        return {
            "may_trade": self.may_trade,
            "disabled": self.disabled,
            "reason": self.reason,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }

    def summary(self) -> str:
        if self.disabled:
            head = "ENTRIES ALLOWED (UNCHECKED)"
        else:
            head = "ENTRIES ALLOWED" if self.may_trade else "ENTRIES BLOCKED"
        lines = [f"{head}: {self.reason}", ""]
        for c in self.checks:
            lines.append(f"  {'ok  ' if c.passed else 'FAIL'}  {c.name:<26}{c.detail}")
        if not self.may_trade:
            lines.append("")
            lines.append("  Open positions are unaffected and remain managed by the monitor.")
        return "\n".join(lines)


def _check_manual_halt(db: Session) -> Check:
    if is_trading_halted(db):
        return Check("manual halt", False, "trading is halted (daily loss, loss streak, or manually)")
    return Check("manual halt", True, "not halted")


def _check_accounting(db: Session) -> Check:
    result = reconcile_mod.reconcile(db)
    if result.balanced:
        return Check(
            "accounting", True,
            f"ledger reconciles across {result.filled_trades} filled trades",
        )
    return Check(
        "accounting", False,
        f"cash ledger is off by ${result.discrepancy:+,.4f} against the trade record - "
        "refusing to size new positions off a balance known to be wrong",
    )


def _check_positions(db: Session) -> Check:
    problems = reconcile_mod.check_position_integrity(db)
    if not problems:
        return Check("position integrity", True, "open book is structurally sound")
    return Check(
        "position integrity", False,
        f"{len(problems)} structural problem(s) in the position book: {problems[0]}"
        + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
    )


def _check_data_freshness() -> Check:
    """Has every critical upstream succeeded recently enough to trust?

    A price feed that stopped responding is the failure this bot is least
    able to notice on its own: the last known price keeps being used, every
    position keeps being valued at it, and nothing looks wrong.
    """
    limit = settings.KILL_SWITCH_MAX_DATA_AGE_SECONDS
    now = dt.datetime.now(dt.timezone.utc)
    stale: list[str] = []

    for service in CRITICAL_SERVICES:
        record = api_health.get(service)
        if record is None or record.last_success_at is None:
            # Nothing has succeeded yet this process. On a fresh start that
            # is normal, so it is not treated as staleness - but a service
            # that has failed and never succeeded is caught below.
            if record is not None and record.consecutive_failures >= settings.KILL_SWITCH_MAX_CONSECUTIVE_FAILURES:
                stale.append(f"{service} has never succeeded and has failed {record.consecutive_failures}x")
            continue

        last = record.last_success_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=dt.timezone.utc)
        age = (now - last).total_seconds()
        if age > limit:
            stale.append(f"{service} last succeeded {age / 60:.0f} minutes ago")
        elif record.consecutive_failures >= settings.KILL_SWITCH_MAX_CONSECUTIVE_FAILURES:
            stale.append(f"{service} has failed {record.consecutive_failures} times in a row")

    if stale:
        return Check(
            "data freshness", False,
            "critical data source unusable: " + "; ".join(stale)
            + " - trading on a feed that stopped updating produces a record nobody can interpret",
        )
    return Check("data freshness", True, "every critical data source is responding")


async def _check_valuation(db: Session) -> Check:
    """Can enough of the open book be priced to size a new trade off it?

    Position size is a fraction of portfolio value, and portfolio value
    includes open positions. If most of the book is being valued at cost
    because its prices stopped arriving, that fraction is computed from a
    number that no longer means anything.
    """
    try:
        valuation = await portfolio.value_open_positions(db)
    except Exception as exc:
        return Check("valuation", False, f"could not value the open book: {exc}")

    if valuation.positions == 0 or valuation.fully_priced:
        return Check("valuation", True, f"{valuation.positions} open position(s), all priced")

    limit = settings.KILL_SWITCH_MAX_STALE_VALUATION_SHARE
    if valuation.stale_share > limit:
        return Check(
            "valuation", False,
            f"{valuation.stale_share * 100:.0f}% of the open book "
            f"(${valuation.stale_usd:,.0f}) is valued at cost because no live price came back, "
            f"above the {limit * 100:.0f}% limit - new position sizes would be computed off it",
        )
    return Check(
        "valuation", True,
        f"{valuation.stale_positions} of {valuation.positions} position(s) valued at cost, "
        "within tolerance",
    )


async def may_open_position(db: Session) -> Verdict:
    """The gate. Fails closed: any check that cannot run counts as failed."""
    if not settings.KILL_SWITCH_ENABLED:
        return Verdict(
            True, [Check("kill switch", True, "disabled by configuration")], disabled=True
        )

    checks: list[Check] = []
    for name, runner in (
        ("manual halt", lambda: _check_manual_halt(db)),
        ("accounting", lambda: _check_accounting(db)),
        ("position integrity", lambda: _check_positions(db)),
        ("data freshness", _check_data_freshness),
    ):
        try:
            checks.append(runner())
        except Exception as exc:
            logger.exception("kill-switch check %s raised", name)
            checks.append(Check(name, False, f"check could not run ({exc}) - treating as a failure"))

    try:
        checks.append(await _check_valuation(db))
    except Exception as exc:
        logger.exception("kill-switch valuation check raised")
        checks.append(Check("valuation", False, f"check could not run ({exc}) - treating as a failure"))

    return Verdict(may_trade=all(c.passed for c in checks), checks=checks)
