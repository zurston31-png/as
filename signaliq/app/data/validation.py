"""Bar validation, the US equity trading calendar and stored-data checks.

Validation never repairs data. A bar that fails a check is reported (and, at
ingestion, rejected) exactly as it was received: no price is filled in and no
candle is ever invented.

The module has four parts:

1. Trading calendar - which days the US stock market is open.
2. Record validation - the row-level checks, evaluated with Polars.
3. Session-gap detection - expected trading sessions that have no bar.
4. Stored-bar validation - re-checks what is already in the database and
   records the outcome in ``data_quality_checks``.
"""

import math
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
from sqlalchemy import bindparam, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from app.config import SOURCE_ALPACA, Settings
from app.models import Bar, CheckStage, CheckStatus, DataQualityCheck, utc_now

# =============================================================================
# Check names
# =============================================================================

CHECK_MISSING_FIELDS = "missing_required_fields"
CHECK_NON_POSITIVE_PRICE = "non_positive_price"
CHECK_HIGH_BELOW_LOW = "high_below_low"
CHECK_OPEN_CLOSE_RANGE = "open_close_outside_range"
CHECK_NEGATIVE_VOLUME = "negative_volume"
CHECK_DUPLICATE_TIMESTAMP = "duplicate_timestamp"
CHECK_INVALID_TIMESTAMP = "invalid_timestamp"
CHECK_UNEXPECTED_SYMBOL = "unexpected_symbol"
CHECK_SOURCE_METADATA = "invalid_source_metadata"
CHECK_SESSION_GAP = "trading_session_gap"
CHECK_CONFLICTING_RECORD = "conflicting_record"

#: Checks evaluated for every individual bar.
ROW_CHECKS: tuple[str, ...] = (
    CHECK_MISSING_FIELDS,
    CHECK_NON_POSITIVE_PRICE,
    CHECK_HIGH_BELOW_LOW,
    CHECK_OPEN_CLOSE_RANGE,
    CHECK_NEGATIVE_VOLUME,
    CHECK_DUPLICATE_TIMESTAMP,
    CHECK_INVALID_TIMESTAMP,
    CHECK_UNEXPECTED_SYMBOL,
    CHECK_SOURCE_METADATA,
)

CHECK_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        CHECK_MISSING_FIELDS: "Required fields are present and well formed",
        CHECK_NON_POSITIVE_PRICE: "Open, high, low and close are greater than zero",
        CHECK_HIGH_BELOW_LOW: "High is not lower than low",
        CHECK_OPEN_CLOSE_RANGE: "Open and close lie inside the low-high range",
        CHECK_NEGATIVE_VOLUME: "Volume is not negative",
        CHECK_DUPLICATE_TIMESTAMP: "No duplicate timestamps per symbol and timeframe",
        CHECK_INVALID_TIMESTAMP: "Timestamps are valid and explicitly UTC",
        CHECK_UNEXPECTED_SYMBOL: "Symbols belong to the expected set",
        CHECK_SOURCE_METADATA: "Source, feed and adjustment metadata are as configured",
        CHECK_SESSION_GAP: "No unexpected gaps in trading sessions",
        CHECK_CONFLICTING_RECORD: "Incoming bars agree with bars already stored",
    }
)

# =============================================================================
# 1. Trading calendar (New York Stock Exchange / Nasdaq full-day closures)
# =============================================================================

NEW_YORK = ZoneInfo("America/New_York")

#: A session's daily candle is treated as final only after this New York time
#: on the session date: the end of extended-hours trading (20:00) plus a
#: margin. Before that moment the candle can still change and is not requested.
DAILY_BAR_FINAL_TIME = time(20, 15)

#: Timeframes for which session-gap detection is implemented.
GAP_CHECK_TIMEFRAMES: tuple[str, ...] = ("1Day",)

#: The holiday rules below were cross-checked, day by day, against an
#: independent exchange calendar for this range (see README).
CALENDAR_VERIFIED_FROM = date(2000, 1, 1)
CALENDAR_VERIFIED_TO = date(2027, 12, 31)

#: One-off full-day closures that no rule can predict. A future unscheduled
#: closure must be added here; until then it would be reported as a gap.
UNSCHEDULED_CLOSURES: Mapping[date, str] = MappingProxyType(
    {
        date(2001, 9, 11): "September 11 attacks",
        date(2001, 9, 12): "September 11 attacks",
        date(2001, 9, 13): "September 11 attacks",
        date(2001, 9, 14): "September 11 attacks",
        date(2004, 6, 11): "National Day of Mourning for President Reagan",
        date(2007, 1, 2): "National Day of Mourning for President Ford",
        date(2012, 10, 29): "Hurricane Sandy",
        date(2012, 10, 30): "Hurricane Sandy",
        date(2018, 12, 5): "National Day of Mourning for President George H. W. Bush",
        date(2025, 1, 9): "National Day of Mourning for President Carter",
    }
)

_MONDAY, _THURSDAY, _SATURDAY, _SUNDAY = 0, 3, 5, 6


def easter_sunday(year: int) -> date:
    """Return Easter Sunday for ``year`` (anonymous Gregorian algorithm)."""
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


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the ``n``-th given weekday of a month (n starts at 1)."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last given weekday of a month."""
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    last_day = next_month - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() - weekday) % 7)


def _observed(holiday: date) -> date:
    """Exchange observance: Saturday holidays close Friday, Sunday ones Monday."""
    if holiday.weekday() == _SATURDAY:
        return holiday - timedelta(days=1)
    if holiday.weekday() == _SUNDAY:
        return holiday + timedelta(days=1)
    return holiday


@cache
def market_holidays(year: int) -> Mapping[date, str]:
    """Return the full-day market closures of ``year`` (weekdays only)."""
    holidays: dict[date, str] = {}

    # New Year's Day. When it falls on a Saturday the exchange does NOT close
    # on the preceding Friday (31 December stays a normal trading day).
    new_year = date(year, 1, 1)
    if new_year.weekday() == _SUNDAY:
        holidays[date(year, 1, 2)] = "New Year's Day (observed)"
    elif new_year.weekday() != _SATURDAY:
        holidays[new_year] = "New Year's Day"

    holidays[_nth_weekday(year, 1, _MONDAY, 3)] = "Martin Luther King Jr. Day"
    holidays[_nth_weekday(year, 2, _MONDAY, 3)] = "Washington's Birthday"
    holidays[easter_sunday(year) - timedelta(days=2)] = "Good Friday"
    holidays[_last_weekday(year, 5, _MONDAY)] = "Memorial Day"
    if year >= 2022:  # first observed by the exchanges in 2022
        holidays[_observed(date(year, 6, 19))] = "Juneteenth National Independence Day"
    holidays[_observed(date(year, 7, 4))] = "Independence Day"
    holidays[_nth_weekday(year, 9, _MONDAY, 1)] = "Labor Day"
    holidays[_nth_weekday(year, 11, _THURSDAY, 4)] = "Thanksgiving Day"
    holidays[_observed(date(year, 12, 25))] = "Christmas Day"

    for closure, reason in UNSCHEDULED_CLOSURES.items():
        if closure.year == year:
            holidays[closure] = reason
    return MappingProxyType(holidays)


def closure_reason(day: date) -> str | None:
    """Explain why the market is closed on ``day`` (None if it is open)."""
    if day.weekday() >= _SATURDAY:
        return "weekend"
    return market_holidays(day.year).get(day)


def is_trading_day(day: date) -> bool:
    """True when the US stock market holds a regular session on ``day``."""
    return closure_reason(day) is None


#: Regular-session closing times in New York. An early close ends the regular
#: session at 13:00; the day is still a trading session with its own daily bar.
REGULAR_CLOSE_TIME = time(16, 0)
EARLY_CLOSE_TIME = time(13, 0)

#: The early-close rules below were cross-checked against an independent
#: exchange calendar for this range. Before 2004 the exchanges used one-off
#: schedules that these rules do not reproduce (see README).
EARLY_CLOSES_VERIFIED_FROM = date(2004, 1, 1)
EARLY_CLOSES_VERIFIED_TO = date(2027, 12, 31)


@cache
def market_early_closes(year: int) -> Mapping[date, str]:
    """Return the scheduled 13:00 early closes of ``year``.

    An early close shortens the session; it does not remove it. Such a day is
    a trading day, a daily bar is expected for it, and a missing bar is
    reported as a gap like on any other session.
    """
    early: dict[date, str] = {}
    # 3 July and 24 December close early only when they fall on a Monday to
    # Thursday. On a Friday they are the observed holiday itself, and on a
    # weekend there is no session to shorten.
    if date(year, 7, 3).weekday() <= _THURSDAY:
        early[date(year, 7, 3)] = "Day before Independence Day"
    # The Friday after Thanksgiving closes early every year.
    early[_nth_weekday(year, 11, _THURSDAY, 4) + timedelta(days=1)] = "Day after Thanksgiving"
    if date(year, 12, 24).weekday() <= _THURSDAY:
        early[date(year, 12, 24)] = "Christmas Eve"
    return MappingProxyType(early)


def early_close_reason(day: date) -> str | None:
    """Explain why ``day`` is a shortened session (None for a full session or a closed day)."""
    return market_early_closes(day.year).get(day)


def session_close_time(day: date) -> time | None:
    """Return the New York time the regular session ends on ``day`` (None when closed)."""
    if not is_trading_day(day):
        return None
    return EARLY_CLOSE_TIME if early_close_reason(day) else REGULAR_CLOSE_TIME


def trading_days(start: date, end: date) -> list[date]:
    """Return every trading day from ``start`` to ``end`` (both inclusive)."""
    days: list[date] = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def session_date(timestamp: datetime) -> date:
    """Return the New York calendar date a UTC bar timestamp belongs to."""
    if timestamp.tzinfo is None:
        raise ValueError("session_date() needs a timezone-aware timestamp")
    return timestamp.astimezone(NEW_YORK).date()


def last_completed_session(now: datetime) -> date:
    """Return the most recent trading day whose daily candle is final at ``now``."""
    if now.tzinfo is None:
        raise ValueError("last_completed_session() needs a timezone-aware time")
    now_new_york = now.astimezone(NEW_YORK)
    day = now_new_york.date()
    if is_trading_day(day) and now_new_york.time() >= DAILY_BAR_FINAL_TIME:
        return day
    day -= timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def session_window_utc(first_day: date, last_day: date) -> tuple[datetime, datetime]:
    """Return the UTC instants covering New York dates ``first_day``..``last_day``.

    The window runs from 00:00:00 New York time on the first day to 23:59:59
    New York time on the last day, so it contains every bar of those sessions
    and nothing from the following day.
    """
    start = datetime.combine(first_day, time(0, 0, 0), tzinfo=NEW_YORK)
    end = datetime.combine(last_day, time(23, 59, 59), tzinfo=NEW_YORK)
    return start.astimezone(UTC), end.astimezone(UTC)


# =============================================================================
# 2. Record validation
# =============================================================================

PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
VALUE_FIELDS: tuple[str, ...] = (*PRICE_FIELDS, "volume", "vwap", "trade_count")
METADATA_FIELDS: tuple[str, ...] = ("source", "feed", "adjustment")
_INTEGER_FIELDS = frozenset({"volume", "trade_count"})

#: Alpaca bar key -> canonical field name.
ALPACA_FIELD_MAP: Mapping[str, str] = MappingProxyType(
    {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "n": "trade_count",
    }
)

_INT64_LIMIT = 2**63

_FRAME_SCHEMA: dict[str, Any] = {
    "row_index": pl.UInt32,
    "symbol": pl.String,
    "timeframe": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "timestamp_error": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "vwap": pl.Float64,
    "trade_count": pl.Int64,
    "source": pl.String,
    "feed": pl.String,
    "adjustment": pl.String,
    "problem_count": pl.UInt32,
}


class TimestampError(ValueError):
    """A timestamp is malformed or not explicitly UTC."""


def parse_utc_timestamp(value: object) -> datetime:
    """Parse a timestamp that must be valid and explicitly UTC.

    Accepts RFC 3339 / ISO-8601 text such as ``2024-01-02T05:00:00Z``. A
    timestamp without a timezone, or with an offset other than zero, is
    rejected instead of being guessed at.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise TimestampError("timestamp is empty")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            raise TimestampError("not a valid ISO-8601 timestamp") from None
    else:
        raise TimestampError(f"timestamp must be text, got {type(value).__name__}")

    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None:
        raise TimestampError("no timezone given; an explicit UTC designator (Z) is required")
    if offset != timedelta(0):
        raise TimestampError(f"not in UTC (UTC offset is {parsed.strftime('%z')})")
    return parsed.astimezone(UTC)


def _as_float(value: object) -> tuple[float | None, str | None]:
    """Return ``(number, problem)``; exactly one of the two is None."""
    if value is None:
        return None, "missing"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, f"not a number ({type(value).__name__})"
    number = float(value)
    if not math.isfinite(number):
        return None, "not a finite number"
    return number, None


def _as_int(value: object) -> tuple[int | None, str | None]:
    """Return ``(whole number, problem)``; exactly one of the two is None."""
    if value is None:
        return None, "missing"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, f"not a number ({type(value).__name__})"
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, "not a finite number"
        if not value.is_integer():
            return None, "not a whole number"
        value = int(value)
    if abs(value) >= _INT64_LIMIT:
        return None, "out of range"
    return value, None


def _as_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _json_safe(value: Any) -> Any:
    """Convert a value into something that can be stored as JSON."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    return repr(value)


def _build_record(
    row_index: int,
    *,
    symbol: object,
    timeframe: object,
    raw_timestamp: object,
    raw_values: Mapping[str, object],
    source: object,
    feed: object,
    adjustment: object,
    raw: Any,
) -> dict[str, Any]:
    """Build one canonical record. Nothing is repaired: bad values become None
    and the reason is noted in ``problems``."""
    problems: dict[str, str] = {}

    timestamp: datetime | None = None
    timestamp_error: str | None = None
    if raw_timestamp is None:
        problems["timestamp"] = "missing"
    else:
        try:
            timestamp = parse_utc_timestamp(raw_timestamp)
        except TimestampError as exc:
            timestamp_error = str(exc)

    record: dict[str, Any] = {
        "row_index": row_index,
        "symbol": _as_text(symbol),
        "timeframe": _as_text(timeframe),
        "timestamp": timestamp,
        "timestamp_raw": raw_timestamp
        if isinstance(raw_timestamp, str)
        else _json_safe(raw_timestamp),
        "timestamp_error": timestamp_error,
        "source": _as_text(source),
        "feed": _as_text(feed),
        "adjustment": _as_text(adjustment),
    }
    for name in VALUE_FIELDS:
        converter = _as_int if name in _INTEGER_FIELDS else _as_float
        record[name], problem = converter(raw_values.get(name))
        if problem is not None:
            problems[name] = problem

    record["problems"] = problems
    record["problem_count"] = len(problems)
    record["raw"] = _json_safe(raw)
    return record


def normalize_alpaca_bars(
    bars_by_symbol: Mapping[str, Sequence[Any]],
    *,
    timeframe: str,
    source: str,
    feed: str,
    adjustment: str,
) -> list[dict[str, Any]]:
    """Convert provider bars into canonical records, in the order received.

    ``source``, ``feed`` and ``adjustment`` describe how the data was requested
    and are attached to every record as provenance.
    """
    records: list[dict[str, Any]] = []
    for symbol, items in bars_by_symbol.items():
        for raw in items:
            payload = raw if isinstance(raw, Mapping) else {}
            records.append(
                _build_record(
                    len(records),
                    symbol=symbol,
                    timeframe=timeframe,
                    raw_timestamp=payload.get("t"),
                    raw_values={name: payload.get(key) for key, name in ALPACA_FIELD_MAP.items()},
                    source=source,
                    feed=feed,
                    adjustment=adjustment,
                    raw=raw,
                )
            )
    return records


def normalize_stored_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw ``bars`` table rows into canonical records."""
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            _build_record(
                len(records),
                symbol=row.get("symbol"),
                timeframe=row.get("timeframe"),
                raw_timestamp=row.get("timestamp"),
                raw_values={name: row.get(name) for name in VALUE_FIELDS},
                source=row.get("source"),
                feed=row.get("feed"),
                adjustment=row.get("adjustment"),
                raw=dict(row),
            )
        )
    return records


@dataclass(frozen=True)
class Finding:
    """One data-quality finding (``failed`` or ``warning``)."""

    check_name: str
    status: str
    message: str
    symbol: str | None = None
    timeframe: str | None = None
    bar_timestamp: datetime | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchValidation:
    """Outcome of the row-level checks over one batch of records."""

    records: list[dict[str, Any]]
    findings: list[Finding]
    #: Rows that must not be stored because a check failed.
    rejected_rows: frozenset[int]
    #: Extra copies of an identical duplicate; dropped, the first copy is kept.
    duplicate_rows: frozenset[int]

    @property
    def records_checked(self) -> int:
        return len(self.records)

    @property
    def accepted_records(self) -> list[dict[str, Any]]:
        """Records that passed every check, without duplicate extras."""
        skip = self.rejected_rows | self.duplicate_rows
        return [record for record in self.records if record["row_index"] not in skip]

    def failures(self, check_name: str) -> list[Finding]:
        return [finding for finding in self.findings if finding.check_name == check_name]


def _frame(records: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        [{name: record[name] for name in _FRAME_SCHEMA} for record in records],
        schema=_FRAME_SCHEMA,
        orient="row",
    )


def _check_expressions(
    allowed_symbols: Sequence[str],
    expected_source: str,
    expected_feed: str,
    expected_adjustment: str,
) -> dict[str, pl.Expr]:
    """Boolean Polars expressions: True marks a row that violates the check."""
    open_, high, low, close = (pl.col(name) for name in PRICE_FIELDS)
    key_present = pl.all_horizontal(
        pl.col("symbol").is_not_null(),
        pl.col("timeframe").is_not_null(),
        pl.col("timestamp").is_not_null(),
    )
    expressions = {
        CHECK_MISSING_FIELDS: pl.col("problem_count") > 0,
        CHECK_NON_POSITIVE_PRICE: pl.any_horizontal(*(pl.col(name) <= 0 for name in PRICE_FIELDS)),
        CHECK_HIGH_BELOW_LOW: high < low,
        # Only meaningful when the range itself is valid (high >= low).
        CHECK_OPEN_CLOSE_RANGE: (high >= low)
        & ((open_ < low) | (open_ > high) | (close < low) | (close > high)),
        CHECK_NEGATIVE_VOLUME: pl.col("volume") < 0,
        CHECK_DUPLICATE_TIMESTAMP: key_present
        & pl.struct("symbol", "timeframe", "timestamp").is_duplicated(),
        CHECK_INVALID_TIMESTAMP: pl.col("timestamp_error").is_not_null(),
        CHECK_UNEXPECTED_SYMBOL: pl.col("symbol").is_null()
        | ~pl.col("symbol").is_in(list(allowed_symbols)),
        CHECK_SOURCE_METADATA: pl.col("source").is_null()
        | (pl.col("source") != expected_source)
        | pl.col("feed").is_null()
        | (pl.col("feed") != expected_feed)
        | pl.col("adjustment").is_null()
        | (pl.col("adjustment") != expected_adjustment),
    }
    # A comparison with a missing value is "unknown", which is not a violation
    # of that particular check (the missing value is reported separately).
    return {name: expression.fill_null(False) for name, expression in expressions.items()}


def _finding_for(
    check_name: str,
    record: Mapping[str, Any],
    *,
    allowed_symbols: Sequence[str],
    expected: Mapping[str, str],
) -> Finding:
    """Describe one violated row-level check in plain words."""
    details: dict[str, Any] = {"record": record["raw"]}

    if check_name == CHECK_MISSING_FIELDS:
        problems = record["problems"]
        listing = ", ".join(f"{name} ({reason})" for name, reason in problems.items())
        message = f"Required field(s) missing or malformed: {listing}"
        details["problems"] = dict(problems)
    elif check_name == CHECK_NON_POSITIVE_PRICE:
        bad = {
            name: record[name]
            for name in PRICE_FIELDS
            if record[name] is not None and record[name] <= 0
        }
        listing = ", ".join(f"{name}={value}" for name, value in bad.items())
        message = f"Non-positive price(s): {listing}"
        details["non_positive"] = bad
    elif check_name == CHECK_HIGH_BELOW_LOW:
        message = f"High {record['high']} is lower than low {record['low']}"
    elif check_name == CHECK_OPEN_CLOSE_RANGE:
        low, high = record["low"], record["high"]
        outside = {
            name: record[name]
            for name in ("open", "close")
            if record[name] is not None and not low <= record[name] <= high
        }
        listing = " and ".join(f"{name} {value}" for name, value in outside.items())
        message = f"{listing.capitalize()} outside the low-high range [{low}, {high}]"
        details["outside_range"] = outside
    elif check_name == CHECK_NEGATIVE_VOLUME:
        message = f"Negative volume: {record['volume']}"
    elif check_name == CHECK_INVALID_TIMESTAMP:
        message = f"Invalid timestamp {record['timestamp_raw']!r}: {record['timestamp_error']}"
    elif check_name == CHECK_UNEXPECTED_SYMBOL:
        shown = ", ".join(allowed_symbols[:12]) + (", ..." if len(allowed_symbols) > 12 else "")
        message = f"Unexpected symbol {record['symbol']!r}; expected one of: {shown}"
    elif check_name == CHECK_SOURCE_METADATA:
        wrong = {
            name: {"found": record[name], "expected": expected[name]}
            for name in METADATA_FIELDS
            if record[name] != expected[name]
        }
        listing = ", ".join(
            f"{name}={value['found']!r} (expected {value['expected']!r})"
            for name, value in wrong.items()
        )
        message = f"Invalid or missing source metadata: {listing}"
        details["metadata"] = wrong
    else:  # pragma: no cover - guarded by ROW_CHECKS
        raise ValueError(f"Unknown row check: {check_name}")

    return Finding(
        check_name=check_name,
        status=CheckStatus.FAILED,
        message=message,
        symbol=record["symbol"],
        timeframe=record["timeframe"],
        bar_timestamp=record["timestamp"],
        details=details,
    )


_DUPLICATE_COMPARE_FIELDS: tuple[str, ...] = (*VALUE_FIELDS, *METADATA_FIELDS)


def validate_records(
    records: list[dict[str, Any]],
    *,
    allowed_symbols: Sequence[str],
    expected_feed: str,
    expected_adjustment: str,
    expected_source: str = SOURCE_ALPACA,
) -> BatchValidation:
    """Run every row-level check over ``records``.

    Returns the findings plus which rows must be rejected. Duplicates are
    handled as follows: when all copies of a timestamp are identical the first
    copy is kept and the extras are dropped; when the copies differ every copy
    is rejected, because the correct one cannot be determined.
    """
    if not records:
        return BatchValidation(
            records=[], findings=[], rejected_rows=frozenset(), duplicate_rows=frozenset()
        )

    allowed = list(allowed_symbols)
    expected = {"source": expected_source, "feed": expected_feed, "adjustment": expected_adjustment}
    expressions = _check_expressions(allowed, expected_source, expected_feed, expected_adjustment)
    flags = _frame(records).select(
        "row_index", *(expression.alias(name) for name, expression in expressions.items())
    )

    findings: list[Finding] = []
    rejected: set[int] = set()
    duplicates: set[int] = set()

    for check_name in ROW_CHECKS:
        offending = flags.filter(pl.col(check_name)).get_column("row_index").to_list()
        if check_name == CHECK_DUPLICATE_TIMESTAMP:
            groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
            for index in offending:
                record = records[index]
                groups[(record["symbol"], record["timeframe"], record["timestamp"])].append(index)
            for (symbol, timeframe, timestamp), indices in groups.items():
                signatures = {
                    tuple(records[index][name] for name in _DUPLICATE_COMPARE_FIELDS)
                    for index in indices
                }
                identical = len(signatures) == 1
                if identical:
                    duplicates.update(indices[1:])
                    outcome = "the copies are identical, so one is kept and the rest are dropped"
                else:
                    rejected.update(indices)
                    outcome = (
                        "the copies differ, so all of them are rejected because the correct "
                        "one cannot be determined"
                    )
                findings.append(
                    Finding(
                        check_name=CHECK_DUPLICATE_TIMESTAMP,
                        status=CheckStatus.FAILED,
                        message=f"Timestamp appears {len(indices)} times; {outcome}",
                        symbol=symbol,
                        timeframe=timeframe,
                        bar_timestamp=timestamp,
                        details={
                            "occurrences": len(indices),
                            "identical": identical,
                            "records": [records[index]["raw"] for index in indices],
                        },
                    )
                )
            continue

        for index in offending:
            findings.append(
                _finding_for(check_name, records[index], allowed_symbols=allowed, expected=expected)
            )
            rejected.add(index)

    return BatchValidation(
        records=records,
        findings=findings,
        rejected_rows=frozenset(rejected),
        duplicate_rows=frozenset(duplicates - rejected),
    )


# =============================================================================
# 3. Session-gap detection
# =============================================================================

GAP_LEADING = "leading"
GAP_INTERIOR = "interior"
GAP_TRAILING = "trailing"
GAP_NO_DATA = "no_data"

#: Longest list of individual dates stored with one gap finding.
_MAX_LISTED_SESSIONS = 100


@dataclass(frozen=True)
class SessionGap:
    """A run of consecutive expected sessions that have no bar."""

    kind: str
    sessions: tuple[date, ...]

    @property
    def first(self) -> date:
        return self.sessions[0]

    @property
    def last(self) -> date:
        return self.sessions[-1]


@dataclass(frozen=True)
class GapAnalysis:
    """Comparison of stored session dates with the trading calendar."""

    window_start: date
    window_end: date
    expected_sessions: int
    present_sessions: int
    gaps: tuple[SessionGap, ...]
    #: Bars found on days the calendar says the market is closed.
    unexpected_dates: tuple[date, ...]

    @property
    def missing_sessions(self) -> int:
        return sum(len(gap.sessions) for gap in self.gaps)

    @property
    def interior_missing_sessions(self) -> int:
        return sum(len(gap.sessions) for gap in self.gaps if gap.kind == GAP_INTERIOR)


def analyse_session_gaps(
    present_dates: Iterable[date], window_start: date, window_end: date
) -> GapAnalysis:
    """Find expected trading sessions in the window that have no bar.

    Weekends and market holidays are never expected, so they are never
    reported. Missing sessions are classified as:

    * ``leading``  - before the first stored bar (history may start later);
    * ``interior`` - between stored bars (a genuine hole in the data);
    * ``trailing`` - after the last stored bar (data not updated yet);
    * ``no_data``  - nothing stored in the window at all.
    """
    expected = trading_days(window_start, window_end)
    expected_set = set(expected)
    present = {day for day in present_dates if window_start <= day <= window_end}
    present_expected = present & expected_set
    unexpected = tuple(sorted(present - expected_set))

    gaps: list[SessionGap] = []
    if expected and not present_expected:
        gaps.append(SessionGap(GAP_NO_DATA, tuple(expected)))
    elif expected:
        first, last = min(present_expected), max(present_expected)
        leading = tuple(day for day in expected if day < first)
        trailing = tuple(day for day in expected if day > last)
        if leading:
            gaps.append(SessionGap(GAP_LEADING, leading))
        run: list[date] = []
        for day in expected:
            if day < first or day > last:
                continue
            if day in present_expected:
                if run:
                    gaps.append(SessionGap(GAP_INTERIOR, tuple(run)))
                    run = []
            else:
                run.append(day)
        if trailing:
            gaps.append(SessionGap(GAP_TRAILING, trailing))

    return GapAnalysis(
        window_start=window_start,
        window_end=window_end,
        expected_sessions=len(expected),
        present_sessions=len(present_expected),
        gaps=tuple(gaps),
        unexpected_dates=unexpected,
    )


@dataclass(frozen=True)
class SessionCoverage:
    """What is stored for one symbol, measured against the trading calendar.

    ``gap_free_from`` is the first session of the uninterrupted run of sessions
    that ends at ``last_session``. Everything from that date on has a bar for
    every expected trading session; anything stored before it is separated
    from the run by at least one missing session.
    """

    first_session: date
    last_session: date
    #: Trading sessions with a bar, from ``first_session`` to ``last_session``.
    sessions_present: int
    #: Trading sessions the calendar expects in that range.
    sessions_expected: int
    #: Expected sessions in that range that have no bar.
    sessions_missing: int
    gap_free_from: date
    #: Sessions (all with a bar) from ``gap_free_from`` to ``last_session``.
    gap_free_sessions: int


def summarise_session_coverage(present_dates: Iterable[date]) -> SessionCoverage | None:
    """Summarise stored session dates against the trading calendar.

    Returns None when no bar falls on a trading session. Nothing is filled in:
    the summary only describes what is stored.
    """
    present = {day for day in present_dates if is_trading_day(day)}
    if not present:
        return None
    first, last = min(present), max(present)
    expected = trading_days(first, last)
    gap_free_from = last
    gap_free_sessions = 0
    for day in reversed(expected):
        if day not in present:
            break
        gap_free_from = day
        gap_free_sessions += 1
    return SessionCoverage(
        first_session=first,
        last_session=last,
        sessions_present=len(present),
        sessions_expected=len(expected),
        sessions_missing=len(expected) - len(present),
        gap_free_from=gap_free_from,
        gap_free_sessions=gap_free_sessions,
    )


def stored_session_coverage(
    database: Session | Connection, symbols: Sequence[str] | None = None
) -> dict[tuple[str, str], SessionCoverage]:
    """Return the session coverage of stored bars per ``(symbol, timeframe)``.

    Only timeframes with calendar support (daily bars) are summarised; other
    timeframes are left out. This reads the database and changes nothing.
    """
    statement = select(Bar.symbol, Bar.timeframe, Bar.timestamp).where(
        Bar.timeframe.in_(GAP_CHECK_TIMEFRAMES)
    )
    if symbols is not None:
        statement = statement.where(Bar.symbol.in_(symbols))
    dates: dict[tuple[str, str], list[date]] = defaultdict(list)
    for symbol, timeframe, timestamp in database.execute(statement):
        dates[(symbol, timeframe)].append(session_date(timestamp))
    coverage: dict[tuple[str, str], SessionCoverage] = {}
    for key, days in dates.items():
        summary = summarise_session_coverage(days)
        if summary is not None:
            coverage[key] = summary
    return coverage


def _sessions_text(gap: SessionGap) -> str:
    count = len(gap.sessions)
    if count == 1:
        return f"1 expected trading session ({gap.first.isoformat()})"
    return f"{count} expected trading sessions ({gap.first.isoformat()} to {gap.last.isoformat()})"


def session_gap_findings(analysis: GapAnalysis, *, symbol: str, timeframe: str) -> list[Finding]:
    """Turn a :class:`GapAnalysis` into findings.

    Interior gaps and bars on closed days are failures. Leading, trailing and
    no-data gaps are warnings: they usually mean the provider's history starts
    later than requested, or that the database simply needs updating.
    """
    findings: list[Finding] = []
    for gap in analysis.gaps:
        what = _sessions_text(gap)
        if gap.kind == GAP_INTERIOR:
            status = CheckStatus.FAILED
            message = f"Missing data: no bar for {what} between stored bars"
        elif gap.kind == GAP_LEADING:
            status = CheckStatus.WARNING
            message = (
                f"History starts later than requested: no bar for {what} before the first "
                "stored bar. The provider's history for this feed may simply begin later."
            )
        elif gap.kind == GAP_TRAILING:
            status = CheckStatus.WARNING
            message = (
                f"Data is not up to date: no bar for {what} after the last stored bar. "
                "Run the ingestion again to update."
            )
        else:
            status = CheckStatus.WARNING
            message = f"No data: no bar is stored for {what} in the window"

        listed = gap.sessions[:_MAX_LISTED_SESSIONS]
        findings.append(
            Finding(
                check_name=CHECK_SESSION_GAP,
                status=status,
                message=message,
                symbol=symbol,
                timeframe=timeframe,
                details={
                    "gap_type": gap.kind,
                    "first_missing_session": gap.first.isoformat(),
                    "last_missing_session": gap.last.isoformat(),
                    "missing_session_count": len(gap.sessions),
                    "missing_sessions": [day.isoformat() for day in listed],
                    "missing_sessions_truncated": len(gap.sessions) > len(listed),
                    "window_start": analysis.window_start.isoformat(),
                    "window_end": analysis.window_end.isoformat(),
                },
            )
        )

    for day in analysis.unexpected_dates:
        findings.append(
            Finding(
                check_name=CHECK_SESSION_GAP,
                status=CheckStatus.FAILED,
                message=(
                    f"Bar found on {day.isoformat()}, which is not a trading session "
                    f"({closure_reason(day)})"
                ),
                symbol=symbol,
                timeframe=timeframe,
                details={"gap_type": "unexpected_session", "session": day.isoformat()},
            )
        )
    return findings


# =============================================================================
# Recording check outcomes
# =============================================================================


def new_check_run_id() -> str:
    """Return an identifier that groups the rows of one validation pass."""
    return uuid.uuid4().hex


def build_check_rows(
    *,
    check_run_id: str,
    stage: str,
    symbol: str | None,
    timeframe: str | None,
    check_names: Sequence[str],
    records_checked: int,
    findings: Sequence[Finding],
    ingestion_run_id: int | None = None,
    skipped: Mapping[str, str] | None = None,
    records_checked_by_check: Mapping[str, int] | None = None,
) -> list[DataQualityCheck]:
    """Create ``data_quality_checks`` rows for one symbol and timeframe.

    Every finding becomes its own row. A check that found nothing gets a single
    ``passed`` row, and a check that could not run gets a ``skipped`` row, so
    the table shows what was verified and not only what went wrong.

    ``records_checked`` is the number of records each check looked at;
    ``records_checked_by_check`` overrides it for individual checks (the gap
    check, for example, counts expected sessions rather than bars).
    """
    skipped = skipped or {}
    records_checked_by_check = records_checked_by_check or {}
    created_at = utc_now()
    rows: list[DataQualityCheck] = []

    def add(check_name: str, status: str, message: str, **extra: Any) -> None:
        rows.append(
            DataQualityCheck(
                created_at=created_at,
                check_run_id=check_run_id,
                stage=stage,
                ingestion_run_id=ingestion_run_id,
                check_name=check_name,
                status=status,
                message=message,
                **extra,
            )
        )

    for finding in findings:
        add(
            finding.check_name,
            finding.status,
            finding.message,
            symbol=finding.symbol if finding.symbol is not None else symbol,
            timeframe=finding.timeframe if finding.timeframe is not None else timeframe,
            bar_timestamp=finding.bar_timestamp,
            details=_json_safe(finding.details) or None,
        )

    with_findings = {finding.check_name for finding in findings}
    for check_name in check_names:
        if check_name in skipped:
            add(
                check_name,
                CheckStatus.SKIPPED,
                skipped[check_name],
                symbol=symbol,
                timeframe=timeframe,
            )
        elif check_name not in with_findings:
            add(
                check_name,
                CheckStatus.PASSED,
                f"{CHECK_DESCRIPTIONS[check_name]}: no problems found",
                symbol=symbol,
                timeframe=timeframe,
                records_checked=records_checked_by_check.get(check_name, records_checked),
            )
    return rows


# =============================================================================
# 4. Validation of bars already stored in the database
# =============================================================================


@dataclass(frozen=True)
class StoredValidationSummary:
    """Result of re-validating stored bars."""

    check_run_id: str
    bars_checked: int
    #: (symbol, timeframe) groups that were validated.
    groups: tuple[tuple[str, str], ...]
    #: check name -> status -> number of rows recorded.
    counts: dict[str, dict[str, int]]
    findings: list[Finding]

    def total(self, status: str) -> int:
        return sum(by_status.get(status, 0) for by_status in self.counts.values())

    @property
    def failed(self) -> int:
        return self.total(CheckStatus.FAILED)

    @property
    def warnings(self) -> int:
        return self.total(CheckStatus.WARNING)


_STORED_BARS_QUERY = (
    "SELECT symbol, timeframe, timestamp, open, high, low, close, volume, vwap, "
    "trade_count, source, feed, adjustment FROM bars"
)


def _read_stored_rows(
    session: Session, symbols: Sequence[str] | None, timeframe: str | None
) -> list[dict[str, Any]]:
    """Read bars as raw values (no type conversion) so that damaged rows are
    reported by the checks instead of crashing the reader."""
    conditions: list[str] = []
    parameters: dict[str, Any] = {}
    if symbols is not None:
        conditions.append("symbol IN :symbols")
        parameters["symbols"] = list(symbols)
    if timeframe is not None:
        conditions.append("timeframe = :timeframe")
        parameters["timeframe"] = timeframe
    query = _STORED_BARS_QUERY
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY symbol, timeframe, timestamp"

    statement = text(query)
    if symbols is not None:
        statement = statement.bindparams(bindparam("symbols", expanding=True))
    return [dict(row) for row in session.execute(statement, parameters).mappings()]


def validate_stored_bars(
    session: Session,
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    timeframe: str | None = None,
    now: datetime | None = None,
) -> StoredValidationSummary:
    """Re-run every check over bars stored in the database.

    The outcome is written to ``data_quality_checks`` (stage ``stored``) inside
    the caller's transaction; the caller commits. ``symbols`` limits the run to
    those symbols; otherwise everything stored is validated.
    """
    now = now or utc_now()
    check_run_id = new_check_run_id()
    rows = _read_stored_rows(session, symbols, timeframe)
    records = normalize_stored_rows(rows)
    batch = validate_records(
        records,
        allowed_symbols=settings.symbols,
        expected_feed=settings.alpaca_feed,
        expected_adjustment=settings.alpaca_adjustment,
    )

    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[(record["symbol"] or "", record["timeframe"] or "")].append(record)
    grouped_findings: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    for finding in batch.findings:
        grouped_findings[(finding.symbol or "", finding.timeframe or "")].append(finding)

    # Symbols that were asked for explicitly are reported even with no bars.
    groups = set(grouped_records)
    if symbols is not None:
        requested_timeframe = timeframe or settings.default_timeframe
        stored_symbols = {symbol for symbol, _ in groups}
        groups.update(
            (symbol, requested_timeframe) for symbol in symbols if symbol not in stored_symbols
        )

    window_start = settings.history_start
    window_end = last_completed_session(now)
    all_findings: list[Finding] = list(batch.findings)
    check_rows: list[DataQualityCheck] = []

    for symbol, group_timeframe in sorted(groups):
        group_records = grouped_records.get((symbol, group_timeframe), [])
        group_findings = list(grouped_findings.get((symbol, group_timeframe), []))
        skipped: dict[str, str] = {}
        counts_by_check: dict[str, int] = {}

        if group_timeframe not in GAP_CHECK_TIMEFRAMES:
            skipped[CHECK_SESSION_GAP] = (
                f"Session-gap detection is implemented for {', '.join(GAP_CHECK_TIMEFRAMES)} "
                f"bars only; not run for timeframe {group_timeframe!r}"
            )
        elif symbol not in settings.symbols:
            skipped[CHECK_SESSION_GAP] = (
                f"Session-gap detection not run: {symbol!r} is not in the configured universe"
            )
        elif window_start > window_end:
            skipped[CHECK_SESSION_GAP] = (
                "Session-gap detection not run: the configured history start is after the "
                "last completed session"
            )
        else:
            present = [
                session_date(record["timestamp"])
                for record in group_records
                if record["timestamp"] is not None
            ]
            analysis = analyse_session_gaps(present, window_start, window_end)
            counts_by_check[CHECK_SESSION_GAP] = analysis.expected_sessions
            gap_findings = session_gap_findings(analysis, symbol=symbol, timeframe=group_timeframe)
            group_findings.extend(gap_findings)
            all_findings.extend(gap_findings)

        check_rows.extend(
            build_check_rows(
                check_run_id=check_run_id,
                stage=CheckStage.STORED,
                symbol=symbol or None,
                timeframe=group_timeframe or None,
                check_names=(*ROW_CHECKS, CHECK_SESSION_GAP),
                records_checked=len(group_records),
                findings=group_findings,
                skipped=skipped,
                records_checked_by_check=counts_by_check,
            )
        )

    session.add_all(check_rows)
    session.flush()

    counts: dict[str, dict[str, int]] = {}
    for row in check_rows:
        by_status = counts.setdefault(row.check_name, {})
        by_status[row.status] = by_status.get(row.status, 0) + 1

    return StoredValidationSummary(
        check_run_id=check_run_id,
        bars_checked=len(records),
        groups=tuple(sorted(groups)),
        counts=counts,
        findings=all_findings,
    )
