"""Point-in-time universe membership. Pure functions, no I/O.

The rule (fixed before any results were looked at):

* Decisions are made at the close of the last trading session of each month and
  take effect from the next session until the next decision.
* A security is eligible at decision session D when
    - it traded on D (a real trade, not a filler bar),
    - its RAW close on D is at least ``min_price`` (raw, because split-adjusted
      prices are rewritten whenever a later split happens and so carry future
      information),
    - it has at least ``min_history`` real trading sessions up to and including D,
    - it traded on at least ``min_active_sessions`` of the last ``window`` trading
      sessions, and
    - it is a common stock (its type, resolved from independent evidence by
      ``app.universe.sectypes``, is ``common``). ETFs, warrants, rights, units,
      preferred shares, funds, ADRs, partnerships and securities of unknown
      type are not eligible.
* Eligible securities are ranked by the median, over the last ``window`` trading
  sessions, of raw close x volume (sessions without a real trade count as 0).
  The ``size`` best are members; ties are broken by security id.

Everything a decision uses comes from sessions up to and including D. The
function never looks at a later bar, which the tests prove by removing and by
changing every bar after D.
"""

import hashlib
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date

from app.data.validation import trading_days
from app.universe.sectypes import UNIVERSE_TYPES


@dataclass(frozen=True)
class MembershipRule:
    size: int = 25
    min_price: float = 5.0
    min_history: int = 126
    window: int = 63
    min_active_sessions: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DailyRow:
    """One real trade of one security (raw prices)."""

    session: date
    close: float
    volume: int


@dataclass(frozen=True)
class Decision:
    session: date
    security_id: int
    eligible: bool
    reason: str | None  # why not eligible
    raw_close: float | None
    median_dollar_volume: float | None
    rank: int | None
    member: bool


def month_end_sessions(first: date, last: date) -> list[date]:
    """The last trading session of every month between two dates."""
    sessions = trading_days(first, last)
    return [
        s
        for i, s in enumerate(sessions)
        if i + 1 == len(sessions) or sessions[i + 1].month != s.month
    ]


def decide(
    decision_session: date,
    history: Mapping[int, Sequence[DailyRow]],
    types: Mapping[int, str],
    rule: MembershipRule,
    calendar: Sequence[date],
    held_out: Mapping[int, str] | None = None,
    dated_exclusions: Mapping[int, Callable[[date], str | None]] | None = None,
) -> list[Decision]:
    """Membership decided at ``decision_session``, from data up to that session only.

    ``calendar`` holds trading sessions up to and including ``decision_session``.
    ``held_out`` maps securities that must not be chosen (for example because a
    price adjustment awaits review) to the reason. ``dated_exclusions`` maps securities
    to a test of the decision date that returns a reason when the security may not be
    chosen on that date (for example while its issuer is a blank-check company). The
    test must only use facts known by that date.
    """
    held_out = held_out or {}
    dated_exclusions = dated_exclusions or {}
    window_sessions = [s for s in calendar if s <= decision_session][-rule.window :]
    window_set = set(window_sessions)
    scored: list[tuple[float, int, float]] = []
    decisions: dict[int, Decision] = {}
    for security_id, rows in history.items():
        past = [row for row in rows if row.session <= decision_session]
        today = past[-1] if past and past[-1].session == decision_session else None
        reason = None
        if types.get(security_id, "unknown") not in UNIVERSE_TYPES:
            reason = "not_common_stock"
        elif security_id in held_out:
            reason = held_out[security_id]
        elif security_id in dated_exclusions and (
            excluded := dated_exclusions[security_id](decision_session)
        ):
            reason = excluded
        elif today is None:
            reason = "no_trade_on_decision_session"
        elif today.close < rule.min_price:
            reason = "raw_price_below_minimum"
        elif len(past) < rule.min_history:
            reason = "history_too_short"
        else:
            in_window = {row.session: row for row in past if row.session in window_set}
            if len(in_window) < rule.min_active_sessions:
                reason = "too_few_active_sessions"
            else:
                values = [
                    in_window[s].close * in_window[s].volume if s in in_window else 0.0
                    for s in window_sessions
                ]
                median = statistics.median(values)
                scored.append((median, security_id, today.close))
                continue
        decisions[security_id] = Decision(
            decision_session,
            security_id,
            False,
            reason,
            today.close if today else None,
            None,
            None,
            False,
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    for rank, (median, security_id, close) in enumerate(scored, start=1):
        decisions[security_id] = Decision(
            decision_session, security_id, True, None, close, median, rank, rank <= rule.size
        )
    return [decisions[key] for key in sorted(decisions)]
