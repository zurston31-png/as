"""Trading-session calendar.

Every intraday statistic this system produces is bucketed by session, and
every entry filter is gated on one. If the calendar is wrong, nothing above
it can be right, and the error is invisible: a backtest with a session
boundary shifted by one hour still produces plausible numbers.

Two failure modes drive the design.

**Drift between backtest and live.** A calendar that reads a clock, hits a
network, or caches mutable state can classify the same instant differently
in two runs. `TradingCalendar` is therefore a pure function of
`(timestamp, InstrumentSpec)`: holiday sets are derived from arithmetic
rules at construction, extra dates are injected by the caller, and no
method consults `date.today()`. That is why this is a class with injected
holiday sets rather than a wrapper over a market-calendar library whose
data ships out of band.

**Timezone collapse.** 09:30 in New York is 14:30 UTC in January and 13:30
UTC in July. Comparing a UTC timestamp against a hardcoded UTC session
boundary silently shifts every session-relative feature by an hour for
roughly half the year. Every boundary here is expressed in exchange-local
time and resolved through `zoneinfo`, and naive datetimes are rejected
outright rather than assumed to be UTC.

Two further rules worth stating because callers get them wrong:

* RTH boundaries come from `spec.rth`, never from constants. GC runs
  08:20-13:30 while the equity index futures run 09:30-16:00, and a
  hardcoded 09:30 would misclassify every gold bar before 09:30.
* A half day closes early, and `rth_bounds` reports the *early* close. A
  flatten-at-session-close rule that used 16:00 on the day after
  Thanksgiving would hold a position for three hours after the market
  stopped trading, filling against prices that do not exist.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Iterable, Protocol
from zoneinfo import ZoneInfo

from flow_model.core.enums import Session
from flow_model.core.instruments import InstrumentSpec
from flow_model.data.base import SchemaError

MINUTES_PER_DAY = 24 * 60

MONDAY = 0
THURSDAY = 3

#: Juneteenth became a US federal holiday in 2021 but was first *observed*
#: by the exchanges in 2022. Back-dating it would close sessions that in
#: fact traded, and 2021-06-18 was a full trading day.
JUNETEENTH_FIRST_YEAR = 2022

#: Exchange-local boundaries of the non-RTH buckets. These are the CME
#: day structure (18:00 open, 17:00 close, one-hour maintenance break) and
#: are deliberately *not* derived from `spec`, because they describe the
#: clock rather than one instrument's regular session.
ASIA_START_MINUTES = 18 * 60
ASIA_END_MINUTES = 3 * 60
LONDON_END_MINUTES = 8 * 60
MAINTENANCE_START_MINUTES = 17 * 60

#: Length of the opening and closing sub-buckets inside RTH, measured from
#: the instrument's own open and close so a shortened session still has a
#: meaningful "last hour".
RTH_OPEN_MINUTES = 60
RTH_CLOSE_MINUTES = 60

#: Early close on a half day (13:00 exchange-local). Applied as a *cap*,
#: never an extension: an instrument whose regular close is already earlier
#: keeps its own close, because moving a close later would claim the market
#: traded when it did not.
HALF_DAY_CLOSE_MINUTES = 13 * 60

#: The reason string `is_tradable` returns when nothing blocks the bar.
TRADABLE_REASON = "ok"

#: Every rejection reason `is_tradable` can return. Exported so the signal
#: engine can assert the detail it attaches to `WaitReason.SESSION_CLOSED`
#: is one the calendar actually produces, rather than a typo that silently
#: becomes a new category in rejection analysis.
TRADABLE_REASONS = frozenset(
    {
        TRADABLE_REASON,
        "no_rth_window",
        "weekend",
        "holiday",
        "half_day",
        "maintenance_break",
        "not_rth",
        "within_open_skip",
        "within_close_skip",
    }
)


class SessionFilterLike(Protocol):
    """The five fields the calendar reads off a `SessionFilterConfig`.

    Declared structurally so the data layer does not import the config
    layer; it also pins down exactly which settings influence tradability,
    so a new config field cannot quietly change session filtering.
    """

    trade_rth_only: bool
    skip_first_minutes: float
    skip_last_minutes: float
    skip_holidays: bool
    skip_half_days: bool


# ---------------------------------------------------------------------------
# Date arithmetic
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def easter_sunday(year: int) -> date:
    """Western (Gregorian) Easter via the anonymous Gregorian algorithm.

    Computed rather than tabulated because Good Friday is the one moving
    market holiday, and a table silently runs out: the first year past the
    end of a hardcoded list becomes a trading day that was not.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def good_friday(year: int) -> date:
    """The Friday before Easter Sunday."""
    return easter_sunday(year) - timedelta(days=2)


def _month_length(year: int, month: int) -> int:
    first = date(year, month, 1)
    next_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return (next_first - first).days


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The `n`-th `weekday` of a month, with `weekday` in Python's
    Monday=0..Sunday=6 convention.

    Raises when the occurrence does not exist (a month with only four
    Mondays has no fifth). Clamping to the last occurrence instead would
    turn "5th Monday" into a different, plausible-looking date, and a
    holiday on the wrong date is worse than a loud failure.
    """
    if not 1 <= n <= 5:
        raise ValueError(f"n must be in 1..5, got {n}")
    if not 0 <= weekday <= 6:
        raise ValueError(f"weekday must be in 0..6 (Monday=0), got {weekday}")
    first = date(year, month, 1)
    day_of_month = 1 + (weekday - first.weekday()) % 7 + 7 * (n - 1)
    if day_of_month > _month_length(year, month):
        raise ValueError(
            f"{year}-{month:02d} has no {n}th weekday {weekday} (Monday=0)"
        )
    return date(year, month, day_of_month)


def last_weekday(year: int, month: int, weekday: int) -> date:
    """The final `weekday` of a month (Monday=0..Sunday=6)."""
    if not 0 <= weekday <= 6:
        raise ValueError(f"weekday must be in 0..6 (Monday=0), got {weekday}")
    last = date(year, month, _month_length(year, month))
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed(day: date) -> date:
    """Weekend observance: Saturday shifts back to Friday, Sunday forward
    to Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def us_market_holidays(year: int) -> frozenset[date]:
    """Observed US equity/futures market holidays for the given year.

    The dates returned are *observed* dates, so a New Year's Day falling on
    a Saturday yields 31 December of the preceding year -- the returned set
    is keyed on the holiday's nominal year, not on the year of every date
    in it. `TradingCalendar` therefore consults both `year` and `year + 1`
    when classifying a day, and a caller doing its own membership test must
    do the same.
    """
    days = [
        observed(date(year, 1, 1)),  # New Year's Day
        nth_weekday(year, 1, MONDAY, 3),  # Martin Luther King Jr. Day
        nth_weekday(year, 2, MONDAY, 3),  # Washington's Birthday
        good_friday(year),
        last_weekday(year, 5, MONDAY),  # Memorial Day
        observed(date(year, 7, 4)),  # Independence Day
        nth_weekday(year, 9, MONDAY, 1),  # Labor Day
        nth_weekday(year, 11, THURSDAY, 4),  # Thanksgiving
        observed(date(year, 12, 25)),  # Christmas
    ]
    if year >= JUNETEENTH_FIRST_YEAR:
        days.append(observed(date(year, 6, 19)))
    return frozenset(days)


@lru_cache(maxsize=None)
def us_half_days(year: int) -> frozenset[date]:
    """Early-close sessions: the day after Thanksgiving, Christmas Eve, and
    3 July, each only when it is a weekday.

    Dates that are already full holidays are removed. When Christmas falls
    on a Saturday the observed holiday *is* 24 December, and when
    Independence Day falls on a Saturday it is 3 July; reporting those as
    half days would mark a closed market as open for four hours.
    """
    holidays = us_market_holidays(year)
    candidates = {
        nth_weekday(year, 11, THURSDAY, 4) + timedelta(days=1),  # Black Friday
        date(year, 12, 24),
        date(year, 7, 3),
    }
    return frozenset(d for d in candidates if d.weekday() < 5 and d not in holidays)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _as_date_set(days: Iterable[date], label: str) -> frozenset[date]:
    """Reject datetimes, which would silently never match.

    `datetime` is a subclass of `date`, so a `datetime` in a holiday set
    passes every type check and matches nothing -- the holiday would simply
    stop existing. Coercing it is no better: which calendar day a
    `datetime` falls on depends on a timezone the caller did not supply.
    """
    out: set[date] = set()
    for day in days:
        if isinstance(day, datetime) or not isinstance(day, date):
            raise TypeError(f"{label} must contain date objects, got {day!r}")
        out.add(day)
    return frozenset(out)


class TradingCalendar:
    """US equity/futures session calendar, pure in `(timestamp, spec)`."""

    __slots__ = (
        "first_year",
        "last_year",
        "_extra_holidays",
        "_extra_half_days",
        "_holidays",
        "_half_days",
    )

    def __init__(
        self,
        extra_holidays: Iterable[date] = frozenset(),
        extra_half_days: Iterable[date] = frozenset(),
        first_year: int = 2010,
        last_year: int = 2035,
    ) -> None:
        if first_year > last_year:
            raise ValueError(f"first_year {first_year} exceeds last_year {last_year}")
        self.first_year = first_year
        self.last_year = last_year
        self._extra_holidays = _as_date_set(extra_holidays, "extra_holidays")
        self._extra_half_days = _as_date_set(extra_half_days, "extra_half_days")
        years = range(first_year, last_year + 1)
        self._holidays = frozenset(
            d for y in years for d in us_market_holidays(y)
        ) | self._extra_holidays
        self._half_days = frozenset(
            d for y in years for d in us_half_days(y)
        ) | self._extra_half_days

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"TradingCalendar(first_year={self.first_year}, last_year={self.last_year}, "
            f"extra_holidays={len(self._extra_holidays)}, "
            f"extra_half_days={len(self._extra_half_days)})"
        )

    @property
    def holidays(self) -> frozenset[date]:
        """Observed holidays across the configured span, plus injected ones.

        Queries are not limited to this span: `is_trading_day` answers for
        any year by re-deriving the rules, because returning "trading day"
        for 2040-12-25 would be a fabricated answer rather than a missing
        one.
        """
        return self._holidays

    @property
    def half_days(self) -> frozenset[date]:
        return self._half_days

    # --- day-level ------------------------------------------------------

    def _is_holiday(self, day: date) -> bool:
        if day in self._extra_holidays:
            return True
        # year + 1 catches a New Year's Day observance that shifted back
        # across the year boundary onto 31 December.
        return day in us_market_holidays(day.year) or day in us_market_holidays(day.year + 1)

    def _is_half_day(self, day: date) -> bool:
        return day in self._extra_half_days or day in us_half_days(day.year)

    def is_trading_day(self, day: date, spec: InstrumentSpec) -> bool:
        """True when `day` is a session date the market is open on.

        `spec` is accepted but unused: every instrument this system trades
        is US-listed and shares one holiday calendar. It stays in the
        signature so routing a non-US instrument to a different calendar
        later is a change of implementation, not of every call site.
        """
        if day.weekday() >= 5:
            return False
        return not self._is_holiday(day)

    def is_half_day(self, day: date, spec: InstrumentSpec) -> bool:
        """True when `day` trades but closes early. A full holiday is not a
        half day."""
        return self.is_trading_day(day, spec) and self._is_half_day(day)

    # --- timestamp-level ------------------------------------------------

    def _local(self, ts: datetime, spec: InstrumentSpec) -> datetime:
        if not isinstance(ts, datetime):
            raise SchemaError(f"expected datetime, got {type(ts).__name__}: {ts!r}")
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            raise SchemaError(
                f"naive datetime {ts!r}: the calendar requires timezone-aware "
                "timestamps. Assuming UTC for an exchange timestamp shifts every "
                "session boundary by the exchange's offset."
            )
        return ts.astimezone(ZoneInfo(spec.timezone))

    def session_date(self, ts: datetime, spec: InstrumentSpec) -> date:
        """The trade date `ts` belongs to.

        For an instrument whose ETH window wraps midnight, the trade date
        rolls at the ETH open (18:00 exchange-local for CME), so a bar at
        Monday 19:00 and a bar at Tuesday 02:00 both belong to Tuesday.
        Bucketing the Monday-evening bar under Monday would split one
        continuous session across two rows of every daily statistic.

        The result is pure calendar reckoning and is *not* rolled forward
        over weekends or holidays: Friday 19:00 maps to Saturday, which
        `is_trading_day` then reports as closed. Rolling it to Monday would
        assert that an instant belongs to a session it is not part of.
        """
        return self._session_date_local(self._local(ts, spec), spec)

    def _session_date_local(self, local: datetime, spec: InstrumentSpec) -> date:
        eth = spec.eth
        if eth is None or not eth.wraps_midnight:
            return local.date()
        if local.hour * 60 + local.minute >= eth.start_minutes:
            return local.date() + timedelta(days=1)
        return local.date()

    def rth_bounds(
        self, day: date, spec: InstrumentSpec
    ) -> tuple[datetime, datetime] | None:
        """UTC open and close of `day`'s regular session, or None.

        None means there is no regular session to speak of: `day` is not a
        trading day, or the instrument declares no `rth` window. On a half
        day the close is capped at `HALF_DAY_CLOSE_MINUTES`, which is the
        close a flatten rule must use.

        Built from exchange-local wall times and converted, so the returned
        instants move with DST instead of drifting an hour twice a year.
        """
        window = spec.rth
        if window is None or not self.is_trading_day(day, spec):
            return None

        start = window.start_minutes
        end = window.end_minutes
        end_day = day + timedelta(days=1) if window.wraps_midnight else day
        if not window.wraps_midnight and self._is_half_day(day):
            if HALF_DAY_CLOSE_MINUTES <= start:
                return None
            end = min(end, HALF_DAY_CLOSE_MINUTES)

        tz = ZoneInfo(spec.timezone)
        open_local = datetime(day.year, day.month, day.day, start // 60, start % 60, tzinfo=tz)
        close_local = datetime(
            end_day.year, end_day.month, end_day.day, end // 60, end % 60, tzinfo=tz
        )
        return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)

    def _bounds_for(self, ts: datetime, spec: InstrumentSpec) -> tuple[datetime, datetime] | None:
        return self.rth_bounds(self.session_date(ts, spec), spec)

    def minutes_into_rth(self, ts: datetime, spec: InstrumentSpec) -> float | None:
        """Minutes elapsed since the regular open, or None outside RTH.

        None rather than a negative number, so a pre-open bar cannot be
        read as "minute -30 of the session" by a feature that only checks
        the magnitude.
        """
        bounds = self._bounds_for(ts, spec)
        if bounds is None:
            return None
        open_ts, close_ts = bounds
        if not open_ts <= ts < close_ts:
            return None
        return (ts - open_ts).total_seconds() / 60.0

    def minutes_to_rth_close(self, ts: datetime, spec: InstrumentSpec) -> float | None:
        """Minutes remaining until the real close (early on a half day), or
        None outside RTH."""
        bounds = self._bounds_for(ts, spec)
        if bounds is None:
            return None
        open_ts, close_ts = bounds
        if not open_ts <= ts < close_ts:
            return None
        return (close_ts - ts).total_seconds() / 60.0

    def is_rth(self, ts: datetime, spec: InstrumentSpec) -> bool:
        """True inside the regular session, half-open: the open is included,
        the close is not. A bar stamped at the close belongs to the closing
        auction, not to the next minute of trading."""
        return self.minutes_into_rth(ts, spec) is not None

    def session_of(self, ts: datetime, spec: InstrumentSpec) -> Session:
        """Bucket `ts` into a `Session`.

        The three RTH buckets are measured from the instrument's own open
        and close; everything else follows the CME clock structure. A
        timestamp on a non-trading day, and every timestamp for an
        instrument with no declared `rth`, is `CLOSED`: without a regular
        session there is no honest PRE/POST distinction to draw, and
        inventing a 09:30 open for an instrument that never declared one is
        the kind of silent substitution this layer exists to prevent.
        """
        local = self._local(ts, spec)
        day = self._session_date_local(local, spec)
        if not self.is_trading_day(day, spec):
            return Session.CLOSED

        bounds = self.rth_bounds(day, spec)
        if bounds is None:
            return Session.CLOSED
        open_ts, close_ts = bounds

        if open_ts <= ts < close_ts:
            length = (close_ts - open_ts).total_seconds() / 60.0
            offset = (ts - open_ts).total_seconds() / 60.0
            # On a session shorter than open+close, the opening bucket wins
            # the overlap and the middle bucket simply does not occur.
            open_len = min(float(RTH_OPEN_MINUTES), length)
            close_start = max(open_len, length - RTH_CLOSE_MINUTES)
            if offset < open_len:
                return Session.RTH_OPEN
            if offset >= close_start:
                return Session.RTH_CLOSE
            return Session.RTH_MID

        minute = local.hour * 60 + local.minute
        if MAINTENANCE_START_MINUTES <= minute < ASIA_START_MINUTES:
            return Session.CLOSED
        if minute >= ASIA_START_MINUTES or minute < ASIA_END_MINUTES:
            return Session.ASIA
        if minute < LONDON_END_MINUTES:
            return Session.LONDON
        # Between the London window and the maintenance break, but outside
        # RTH: either side of this instrument's own cash session.
        return Session.PRE_RTH if minute < spec.rth.start_minutes else Session.POST_RTH

    # --- filtering ------------------------------------------------------

    def is_tradable(
        self, ts: datetime, spec: InstrumentSpec, session_cfg: SessionFilterLike
    ) -> tuple[bool, str]:
        """Apply `SessionFilterConfig` to one timestamp.

        Returns `(True, "ok")` or `(False, reason)` where `reason` is one of
        `TRADABLE_REASONS`. The reason is carried into
        `WaitReason.SESSION_CLOSED` detail, so the categories must stay
        distinguishable: "rejected by the session filter" aggregated into
        one bucket makes it impossible to tell a calendar bug from a
        correctly-configured open skip.

        `skip_holidays=False` disables the holiday *filter* only. It does
        not assert that a session existed: `rth_bounds` still reports no
        regular session on a holiday, so `trade_rth_only` rejects the
        timestamp anyway, and `session_of` still reports CLOSED.
        """
        if spec.rth is None:
            return False, "no_rth_window"

        local = self._local(ts, spec)
        day = self._session_date_local(local, spec)
        if day.weekday() >= 5:
            return False, "weekend"
        if session_cfg.skip_holidays and self._is_holiday(day):
            return False, "holiday"
        if session_cfg.skip_half_days and self._is_half_day(day):
            return False, "half_day"

        minute = local.hour * 60 + local.minute
        if MAINTENANCE_START_MINUTES <= minute < ASIA_START_MINUTES:
            return False, "maintenance_break"

        into = self.minutes_into_rth(ts, spec)
        if into is None:
            if session_cfg.trade_rth_only:
                return False, "not_rth"
            return True, TRADABLE_REASON

        if into < session_cfg.skip_first_minutes:
            return False, "within_open_skip"
        remaining = self.minutes_to_rth_close(ts, spec)
        if remaining is not None and remaining <= session_cfg.skip_last_minutes:
            return False, "within_close_skip"
        return True, TRADABLE_REASON
