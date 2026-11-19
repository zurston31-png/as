"""Checks that prove (or disprove) the batch's identities, prices and membership."""

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.validation import trading_days
from app.universe.membership import DailyRow, MembershipRule, decide
from app.universe.models import Membership, Security, SecurityBar

#: Day-to-day raw close changes beyond these limits need an explanation.
JUMP_UP, JUMP_DOWN = 1.8, 0.55


@dataclass(frozen=True)
class Jump:
    security_id: int
    symbol: str
    session: date
    previous_close: float
    close: float
    explained_by_split: bool
    adjusted_also_jumps: bool | None


def adjustment_differences(session: Session, tolerance: float = 1e-3) -> dict[int, dict]:
    """Our split adjustment (raw / factor) against Alpaca's split-adjusted close."""
    result: dict[int, dict] = defaultdict(
        lambda: {"compared": 0, "within": 0, "worst": 0.0, "worst_session": None}
    )
    for bar in session.execute(select(SecurityBar)).scalars():
        if bar.alpaca_adjusted_close is None:
            continue
        ours = bar.close / bar.split_factor
        difference = abs(ours / bar.alpaca_adjusted_close - 1.0)
        entry = result[bar.security_id]
        entry["compared"] += 1
        entry["within"] += difference <= tolerance
        if difference > entry["worst"]:
            entry["worst"], entry["worst_session"] = difference, bar.session
    return dict(result)


def large_jumps(session: Session) -> list[Jump]:
    jumps = []
    previous: dict[int, SecurityBar] = {}
    for bar in session.execute(
        select(SecurityBar).order_by(SecurityBar.security_id, SecurityBar.session)
    ).scalars():
        before = previous.get(bar.security_id)
        previous[bar.security_id] = bar
        if before is None or before.close <= 0:
            continue
        ratio = bar.close / before.close
        if JUMP_DOWN <= ratio <= JUMP_UP:
            continue
        adjusted = None
        if bar.alpaca_adjusted_close and before.alpaca_adjusted_close:
            adjusted_ratio = bar.alpaca_adjusted_close / before.alpaca_adjusted_close
            adjusted = not (JUMP_DOWN <= adjusted_ratio <= JUMP_UP)
        jumps.append(
            Jump(
                bar.security_id,
                bar.symbol,
                bar.session,
                before.close,
                bar.close,
                not math.isclose(before.split_factor, bar.split_factor),
                adjusted,
            )
        )
    return jumps


def load_history(session: Session) -> tuple[dict[int, list[DailyRow]], dict[int, str]]:
    history: dict[int, list[DailyRow]] = defaultdict(list)
    for bar in session.execute(
        select(SecurityBar).order_by(SecurityBar.security_id, SecurityBar.session)
    ).scalars():
        history[bar.security_id].append(DailyRow(bar.session, bar.close, bar.volume))
    types = {s.security_id: s.declared_type for s in session.execute(select(Security)).scalars()}
    return dict(history), types


def stored_members(session: Session) -> dict[date, set[int]]:
    members: dict[date, set[int]] = defaultdict(set)
    for row in session.execute(select(Membership).where(Membership.member == 1)).scalars():
        members[row.decision_session].add(row.security_id)
    return dict(members)


def recompute_with_truncated_data(
    session: Session, rule: MembershipRule, data_start: date
) -> list[date]:
    """Recompute every decision from data cut off at that decision. Returns sessions that differ."""
    history, types = load_history(session)
    stored = stored_members(session)
    decisions = sorted({r.decision_session for r in session.execute(select(Membership)).scalars()})
    calendar = trading_days(data_start, max(decisions)) if decisions else []
    differ = []
    for when in decisions:
        truncated = {sid: [r for r in rows if r.session <= when] for sid, rows in history.items()}
        again = {
            d.security_id
            for d in decide(when, truncated, types, rule, [s for s in calendar if s <= when])
            if d.member
        }
        if again != stored.get(when, set()):
            differ.append(when)
    return differ


def tamper_future(session: Session, rule: MembershipRule, data_start: date, cut: date) -> bool:
    """Multiply every price and volume after ``cut`` by 10.

    No decision up to ``cut`` may change.
    """
    history, types = load_history(session)
    stored = stored_members(session)
    tampered = {
        sid: [
            r if r.session <= cut else replace(r, close=r.close * 10, volume=r.volume * 10)
            for r in rows
        ]
        for sid, rows in history.items()
    }
    decisions = sorted(d for d in stored if d <= cut)
    calendar = trading_days(data_start, cut)
    for when in decisions:
        again = {
            d.security_id
            for d in decide(when, tampered, types, rule, [s for s in calendar if s <= when])
            if d.member
        }
        if again != stored[when]:
            return False
    return True


def adjusted_price_would_change_eligibility(
    session: Session, rule: MembershipRule
) -> list[tuple[date, int, float, float]]:
    """Decisions where the price filter gives a different answer on split-adjusted prices.

    Returns (decision session, security id, raw close, adjusted close) for every
    eligible-on-raw security whose adjusted close is below the minimum price, and
    vice versa. Each is a case where adjusted prices would leak a later split.
    """
    cases = []
    bars = {(b.security_id, b.session): b for b in session.execute(select(SecurityBar)).scalars()}
    for row in session.execute(select(Membership)).scalars():
        bar = bars.get((row.security_id, row.decision_session))
        if bar is None:
            continue
        adjusted = bar.close / bar.split_factor
        if (bar.close >= rule.min_price) != (adjusted >= rule.min_price):
            cases.append((row.decision_session, row.security_id, bar.close, adjusted))
    return cases
