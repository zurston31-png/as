"""Validation tests: row-level checks, trading calendar, gap detection and
validation of stored bars.

Every bar here is a SYNTHETIC TEST FIXTURE with invented values (see
``conftest.py``). Nothing in this file is real market data.
"""

from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text

from app.data.validation import (
    CHECK_DUPLICATE_TIMESTAMP,
    CHECK_HIGH_BELOW_LOW,
    CHECK_INVALID_TIMESTAMP,
    CHECK_MISSING_FIELDS,
    CHECK_NEGATIVE_VOLUME,
    CHECK_NON_POSITIVE_PRICE,
    CHECK_OPEN_CLOSE_RANGE,
    CHECK_SESSION_GAP,
    CHECK_SOURCE_METADATA,
    CHECK_UNEXPECTED_SYMBOL,
    GAP_INTERIOR,
    GAP_LEADING,
    GAP_NO_DATA,
    GAP_TRAILING,
    ROW_CHECKS,
    TimestampError,
    analyse_session_gaps,
    build_check_rows,
    closure_reason,
    easter_sunday,
    is_trading_day,
    last_completed_session,
    market_holidays,
    new_check_run_id,
    normalize_alpaca_bars,
    parse_utc_timestamp,
    session_date,
    session_gap_findings,
    session_window_utc,
    trading_days,
    validate_records,
    validate_stored_bars,
)
from app.database import session_scope
from app.models import CheckStage, CheckStatus, DataQualityCheck

ALLOWED = ("TESTA", "TESTB")


def run_checks(bars_by_symbol, *, feed="iex", adjustment="split", source="alpaca"):
    """Normalise provider-shaped fixture bars and run the row-level checks."""
    records = normalize_alpaca_bars(
        bars_by_symbol, timeframe="1Day", source=source, feed=feed, adjustment=adjustment
    )
    return validate_records(
        records, allowed_symbols=ALLOWED, expected_feed="iex", expected_adjustment="split"
    )


def failed_checks(result) -> set[str]:
    return {finding.check_name for finding in result.findings}


def weekdays(first: date, last: date, *, closed: tuple[date, ...] = ()) -> list[date]:
    """Weekdays in a range minus explicitly listed closures.

    Written out by hand so that gap tests do not depend on the calendar code
    they are testing.
    """
    days = []
    current = first
    while current <= last:
        if current.weekday() < 5 and current not in closed:
            days.append(current)
        current += timedelta(days=1)
    return days


# =============================================================================
# Valid data
# =============================================================================


def test_valid_bars_pass_every_check(make_bar):
    days = [date(2024, 1, day) for day in (2, 3, 4, 5, 8)]

    result = run_checks({"TESTA": [make_bar(day) for day in days]})

    assert result.findings == []
    assert result.rejected_rows == frozenset()
    assert result.duplicate_rows == frozenset()
    assert result.records_checked == 5
    assert len(result.accepted_records) == 5


def test_empty_batch_is_valid():
    result = run_checks({})

    assert result.records_checked == 0
    assert result.findings == []
    assert result.accepted_records == []


def test_records_are_normalised_without_altering_values(make_bar):
    raw = make_bar(date(2024, 1, 2), o=10.25, h=11.75, l=9.5, c=10.5, v=123456, n=789, vw=10.123456)

    record = run_checks({"TESTA": [raw]}).accepted_records[0]

    assert record["symbol"] == "TESTA"
    assert record["timeframe"] == "1Day"
    assert record["timestamp"] == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert (record["open"], record["high"], record["low"], record["close"]) == (10.25, 11.75, 9.5, 10.5)
    assert (record["volume"], record["trade_count"], record["vwap"]) == (123456, 789, 10.123456)
    assert (record["source"], record["feed"], record["adjustment"]) == ("alpaca", "iex", "split")
    assert record["raw"] == raw


@pytest.mark.parametrize(
    "overrides",
    [
        {"o": 11.0},  # open equal to high
        {"c": 9.0},  # close equal to low
        {"o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},  # completely flat bar
        {"v": 0},  # zero volume is unusual but not negative
        {"v": 1000.0},  # whole number delivered as a float
        {"o": 0.0001, "h": 0.0002, "l": 0.0001, "c": 0.0002},  # tiny but positive
    ],
)
def test_boundary_values_are_accepted(make_bar, overrides):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), **overrides)]})

    assert result.findings == []
    assert len(result.accepted_records) == 1


# =============================================================================
# 1. Missing required fields
# =============================================================================


@pytest.mark.parametrize(
    ("provider_key", "field_name"),
    [
        ("t", "timestamp"),
        ("o", "open"),
        ("h", "high"),
        ("l", "low"),
        ("c", "close"),
        ("v", "volume"),
        ("n", "trade_count"),
        ("vw", "vwap"),
    ],
)
def test_missing_field_is_reported_and_the_bar_rejected(make_bar, provider_key, field_name):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), omit=(provider_key,))]})

    assert failed_checks(result) == {CHECK_MISSING_FIELDS}
    finding = result.findings[0]
    assert finding.status == CheckStatus.FAILED
    assert f"{field_name} (missing)" in finding.message
    assert finding.details["problems"] == {field_name: "missing"}
    assert result.rejected_rows == {0}
    assert result.accepted_records == []


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "missing"),
        ("10.5", "not a number (str)"),
        (True, "not a number (bool)"),
        ([10.5], "not a number (list)"),
        (float("nan"), "not a finite number"),
        (float("inf"), "not a finite number"),
    ],
)
def test_malformed_price_counts_as_missing_and_is_never_guessed(make_bar, value, reason):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), c=value)]})

    assert failed_checks(result) == {CHECK_MISSING_FIELDS}
    assert result.findings[0].details["problems"] == {"close": reason}
    assert result.rejected_rows == {0}


def test_bar_that_is_not_an_object_is_rejected():
    result = run_checks({"TESTA": ["fixture garbage"]})

    assert failed_checks(result) == {CHECK_MISSING_FIELDS}
    assert set(result.findings[0].details["problems"]) == {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "trade_count",
    }
    assert result.findings[0].details["record"] == "fixture garbage"
    assert result.rejected_rows == {0}


def test_original_payload_is_kept_with_the_finding_for_investigation(make_bar):
    raw = make_bar(date(2024, 1, 2), omit=("vw",))

    finding = run_checks({"TESTA": [raw]}).findings[0]

    assert finding.details["record"] == raw
    assert finding.symbol == "TESTA"
    assert finding.timeframe == "1Day"
    assert finding.bar_timestamp == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)


# =============================================================================
# 2-4. OHLC validation
# =============================================================================


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"l": 0}, {CHECK_NON_POSITIVE_PRICE}),
        ({"l": -1.0}, {CHECK_NON_POSITIVE_PRICE}),
        ({"o": 0}, {CHECK_NON_POSITIVE_PRICE, CHECK_OPEN_CLOSE_RANGE}),
        ({"c": 0}, {CHECK_NON_POSITIVE_PRICE, CHECK_OPEN_CLOSE_RANGE}),
        ({"h": 0}, {CHECK_NON_POSITIVE_PRICE, CHECK_HIGH_BELOW_LOW}),
        ({"o": -10.0, "h": -9.0, "l": -11.0, "c": -10.5}, {CHECK_NON_POSITIVE_PRICE}),
        ({"o": 0, "h": 0, "l": 0, "c": 0}, {CHECK_NON_POSITIVE_PRICE}),
    ],
)
def test_non_positive_prices_are_rejected(make_bar, overrides, expected):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), **overrides)]})

    assert failed_checks(result) == expected
    assert result.rejected_rows == {0}
    non_positive = result.failures(CHECK_NON_POSITIVE_PRICE)[0]
    assert non_positive.message.startswith("Non-positive price(s):")


def test_non_positive_finding_names_each_offending_price(make_bar):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), o=-10.0, h=-9.0, l=-11.0, c=-10.5)]})

    finding = result.failures(CHECK_NON_POSITIVE_PRICE)[0]
    assert finding.details["non_positive"] == {
        "open": -10.0,
        "high": -9.0,
        "low": -11.0,
        "close": -10.5,
    }


def test_high_lower_than_low_is_rejected(make_bar):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), h=8.0, l=9.0)]})

    # Reported once, as the high/low problem; the range check needs a valid range.
    assert failed_checks(result) == {CHECK_HIGH_BELOW_LOW}
    assert result.findings[0].message == "High 8.0 is lower than low 9.0"
    assert result.rejected_rows == {0}


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"o": 12.0}, "Open 12.0 outside the low-high range [9.0, 11.0]"),
        ({"o": 8.5}, "Open 8.5 outside the low-high range [9.0, 11.0]"),
        ({"c": 11.5}, "Close 11.5 outside the low-high range [9.0, 11.0]"),
        ({"c": 8.0}, "Close 8.0 outside the low-high range [9.0, 11.0]"),
        ({"o": 12.0, "c": 8.0}, "Open 12.0 and close 8.0 outside the low-high range [9.0, 11.0]"),
    ],
)
def test_open_or_close_outside_the_high_low_range_is_rejected(make_bar, overrides, expected_message):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), **overrides)]})

    assert failed_checks(result) == {CHECK_OPEN_CLOSE_RANGE}
    assert result.findings[0].message == expected_message
    assert result.rejected_rows == {0}


def test_only_the_invalid_bars_are_rejected(make_bar):
    bars = [
        make_bar(date(2024, 1, 2)),
        make_bar(date(2024, 1, 3), h=8.0),  # high below low
        make_bar(date(2024, 1, 4)),
        make_bar(date(2024, 1, 5), c=50.0),  # close above high
        make_bar(date(2024, 1, 8)),
    ]

    result = run_checks({"TESTA": bars})

    assert result.rejected_rows == {1, 3}
    assert [record["row_index"] for record in result.accepted_records] == [0, 2, 4]
    # Accepted bars are passed through untouched: nothing is repaired or filled.
    assert [record["raw"] for record in result.accepted_records] == [bars[0], bars[2], bars[4]]


# =============================================================================
# 5. Invalid volume
# =============================================================================


@pytest.mark.parametrize("volume", [-1, -1000, -1.0])
def test_negative_volume_is_rejected(make_bar, volume):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), v=volume)]})

    assert failed_checks(result) == {CHECK_NEGATIVE_VOLUME}
    assert result.findings[0].message == f"Negative volume: {int(volume)}"
    assert result.rejected_rows == {0}


@pytest.mark.parametrize(
    ("volume", "reason"),
    [
        (1000.5, "not a whole number"),
        ("1000", "not a number (str)"),
        (None, "missing"),
        (float("nan"), "not a finite number"),
        (2**63, "out of range"),
    ],
)
def test_malformed_volume_is_rejected(make_bar, volume, reason):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), v=volume)]})

    assert failed_checks(result) == {CHECK_MISSING_FIELDS}
    assert result.findings[0].details["problems"] == {"volume": reason}
    assert result.rejected_rows == {0}


# =============================================================================
# 6. Duplicate timestamps
# =============================================================================


def test_identical_duplicates_keep_one_copy(make_bar):
    bars = [make_bar(date(2024, 1, 2)), make_bar(date(2024, 1, 3)), make_bar(date(2024, 1, 2))]

    result = run_checks({"TESTA": bars})

    assert failed_checks(result) == {CHECK_DUPLICATE_TIMESTAMP}
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.details["occurrences"] == 2
    assert finding.details["identical"] is True
    assert finding.bar_timestamp == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
    assert result.duplicate_rows == {2}  # the extra copy is dropped
    assert result.rejected_rows == frozenset()
    assert [record["row_index"] for record in result.accepted_records] == [0, 1]


def test_three_identical_copies_drop_two(make_bar):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2)) for _ in range(3)]})

    assert result.findings[0].details["occurrences"] == 3
    assert result.duplicate_rows == {1, 2}
    assert len(result.accepted_records) == 1


def test_conflicting_duplicates_are_all_rejected(make_bar):
    bars = [make_bar(date(2024, 1, 2)), make_bar(date(2024, 1, 2), c=10.6), make_bar(date(2024, 1, 3))]

    result = run_checks({"TESTA": bars})

    assert failed_checks(result) == {CHECK_DUPLICATE_TIMESTAMP}
    finding = result.findings[0]
    assert finding.details["identical"] is False
    assert finding.details["records"] == [bars[0], bars[1]]
    assert "all of them are rejected" in finding.message
    # Neither copy is trusted, and the unrelated bar is unaffected.
    assert result.rejected_rows == {0, 1}
    assert result.duplicate_rows == frozenset()
    assert [record["row_index"] for record in result.accepted_records] == [2]


def test_same_timestamp_for_different_symbols_is_not_a_duplicate(make_bar):
    result = run_checks(
        {"TESTA": [make_bar(date(2024, 1, 2))], "TESTB": [make_bar(date(2024, 1, 2))]}
    )

    assert result.findings == []
    assert len(result.accepted_records) == 2


def test_equivalent_spellings_of_one_instant_are_detected_as_duplicates(make_bar):
    bars = [
        make_bar(date(2024, 1, 2), t="2024-01-02T05:00:00Z"),
        make_bar(date(2024, 1, 2), t="2024-01-02T05:00:00.000+00:00"),
    ]

    result = run_checks({"TESTA": bars})

    assert failed_checks(result) == {CHECK_DUPLICATE_TIMESTAMP}
    assert result.duplicate_rows == {1}


# =============================================================================
# 7. Non-UTC or invalid timestamps
# =============================================================================


@pytest.mark.parametrize(
    ("timestamp", "reason"),
    [
        ("2024-01-02T05:00:00", "no timezone given"),
        ("2024-01-02", "no timezone given"),
        ("2024-01-02T00:00:00-05:00", "not in UTC"),
        ("2024-01-02T06:00:00+01:00", "not in UTC"),
        ("not-a-timestamp", "not a valid ISO-8601 timestamp"),
        ("2024-13-45T99:00:00Z", "not a valid ISO-8601 timestamp"),
        ("", "timestamp is empty"),
        (1704171600, "timestamp must be text"),
    ],
)
def test_invalid_or_non_utc_timestamps_are_rejected(make_bar, timestamp, reason):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), t=timestamp)]})

    assert failed_checks(result) == {CHECK_INVALID_TIMESTAMP}
    finding = result.findings[0]
    assert reason in finding.message
    assert finding.bar_timestamp is None  # an invalid time is never guessed
    assert result.rejected_rows == {0}


@pytest.mark.parametrize(
    "timestamp",
    [
        "2024-01-02T05:00:00Z",
        "2024-01-02T05:00:00+00:00",
        "2024-01-02T05:00:00.000000Z",
        "2024-01-02T05:00:00.000000000Z",
    ],
)
def test_utc_timestamps_are_accepted_in_their_documented_forms(make_bar, timestamp):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2), t=timestamp)]})

    assert result.findings == []
    assert result.accepted_records[0]["timestamp"] == datetime(2024, 1, 2, 5, 0, tzinfo=UTC)


def test_parse_utc_timestamp_returns_timezone_aware_utc():
    parsed = parse_utc_timestamp("2024-07-01T04:00:00Z")

    assert parsed == datetime(2024, 7, 1, 4, 0, tzinfo=UTC)
    assert parsed.utcoffset() == timedelta(0)


def test_parse_utc_timestamp_handles_datetime_objects():
    assert parse_utc_timestamp(datetime(2024, 1, 2, 5, 0, tzinfo=UTC)) == datetime(
        2024, 1, 2, 5, 0, tzinfo=UTC
    )
    with pytest.raises(TimestampError, match="no timezone given"):
        parse_utc_timestamp(datetime(2024, 1, 2, 5, 0))
    with pytest.raises(TimestampError, match="not in UTC"):
        parse_utc_timestamp(datetime(2024, 1, 2, 0, 0, tzinfo=timezone(timedelta(hours=-5))))


# =============================================================================
# 8. Unexpected symbols
# =============================================================================


def test_unexpected_symbol_is_rejected(make_bar):
    result = run_checks(
        {"TESTA": [make_bar(date(2024, 1, 2))], "ZZZZ": [make_bar(date(2024, 1, 2))]}
    )

    assert failed_checks(result) == {CHECK_UNEXPECTED_SYMBOL}
    finding = result.findings[0]
    assert finding.symbol == "ZZZZ"
    assert finding.message == "Unexpected symbol 'ZZZZ'; expected one of: TESTA, TESTB"
    assert result.rejected_rows == {1}
    assert [record["symbol"] for record in result.accepted_records] == ["TESTA"]


@pytest.mark.parametrize("symbol", ["testa", "TESTA ", "", "TEST A"])
def test_symbol_matching_is_exact(make_bar, symbol):
    result = run_checks({symbol: [make_bar(date(2024, 1, 2))]})

    # " TESTA " is trimmed to a known symbol; everything else is unexpected.
    if symbol.strip() == "TESTA":
        assert result.findings == []
    else:
        assert CHECK_UNEXPECTED_SYMBOL in failed_checks(result)
        assert result.rejected_rows == {0}


# =============================================================================
# 9. Invalid or missing source metadata
# =============================================================================


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"feed": "sip"}, "feed"),
        ({"feed": ""}, "feed"),
        ({"adjustment": "raw"}, "adjustment"),
        ({"adjustment": None}, "adjustment"),
        ({"source": "somewhere-else"}, "source"),
        ({"source": None}, "source"),
    ],
)
def test_invalid_or_missing_source_metadata_is_rejected(make_bar, kwargs, field_name):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2))]}, **kwargs)

    assert failed_checks(result) == {CHECK_SOURCE_METADATA}
    finding = result.findings[0]
    assert set(finding.details["metadata"]) == {field_name}
    assert field_name in finding.message
    assert result.rejected_rows == {0}


def test_metadata_finding_states_found_and_expected_values(make_bar):
    finding = run_checks({"TESTA": [make_bar(date(2024, 1, 2))]}, feed="sip").findings[0]

    assert finding.details["metadata"] == {"feed": {"found": "sip", "expected": "iex"}}
    assert finding.message == "Invalid or missing source metadata: feed='sip' (expected 'iex')"


# =============================================================================
# Several problems at once
# =============================================================================


def test_every_deliberately_introduced_error_is_caught(make_bar):
    bars = {
        "TESTA": [
            make_bar(date(2024, 1, 2)),  # 0 valid
            make_bar(date(2024, 1, 3), omit=("vw",)),  # 1 missing field
            make_bar(date(2024, 1, 4), l=-1.0),  # 2 non-positive price
            make_bar(date(2024, 1, 5), h=8.0),  # 3 high below low
            make_bar(date(2024, 1, 8), o=12.0),  # 4 open outside range
            make_bar(date(2024, 1, 9), v=-5),  # 5 negative volume
            make_bar(date(2024, 1, 10)),  # 6 duplicate (first copy, kept)
            make_bar(date(2024, 1, 10)),  # 7 duplicate (extra copy, dropped)
            make_bar(date(2024, 1, 11), t="2024-01-11T00:00:00-05:00"),  # 8 non-UTC
            make_bar(date(2024, 1, 12)),  # 9 valid
        ],
        "ZZZZ": [make_bar(date(2024, 1, 2))],  # 10 unexpected symbol
    }

    result = run_checks(bars)

    assert failed_checks(result) == {
        CHECK_MISSING_FIELDS,
        CHECK_NON_POSITIVE_PRICE,
        CHECK_HIGH_BELOW_LOW,
        CHECK_OPEN_CLOSE_RANGE,
        CHECK_NEGATIVE_VOLUME,
        CHECK_DUPLICATE_TIMESTAMP,
        CHECK_INVALID_TIMESTAMP,
        CHECK_UNEXPECTED_SYMBOL,
    }
    assert result.rejected_rows == {1, 2, 3, 4, 5, 8, 10}
    assert result.duplicate_rows == {7}
    assert [record["row_index"] for record in result.accepted_records] == [0, 6, 9]


def test_findings_are_reported_in_a_stable_check_order(make_bar):
    result = run_checks(
        {"TESTA": [make_bar(date(2024, 1, 2), v=-5), make_bar(date(2024, 1, 3), h=8.0)]}
    )

    assert [finding.check_name for finding in result.findings] == [
        CHECK_HIGH_BELOW_LOW,
        CHECK_NEGATIVE_VOLUME,
    ]
    assert [check for check in ROW_CHECKS if check in failed_checks(result)] == [
        CHECK_HIGH_BELOW_LOW,
        CHECK_NEGATIVE_VOLUME,
    ]


# =============================================================================
# Trading calendar
# =============================================================================

#: Full-day closures on weekdays, 2016-2027, taken from an independent
#: reference calendar (the ``exchange_calendars`` XNYS calendar) and NOT from
#: the code under test.
REFERENCE_WEEKDAY_CLOSURES = {
    2016: ["01-01", "01-18", "02-15", "03-25", "05-30", "07-04", "09-05", "11-24", "12-26"],
    2017: ["01-02", "01-16", "02-20", "04-14", "05-29", "07-04", "09-04", "11-23", "12-25"],
    2018: ["01-01", "01-15", "02-19", "03-30", "05-28", "07-04", "09-03", "11-22", "12-05", "12-25"],
    2019: ["01-01", "01-21", "02-18", "04-19", "05-27", "07-04", "09-02", "11-28", "12-25"],
    2020: ["01-01", "01-20", "02-17", "04-10", "05-25", "07-03", "09-07", "11-26", "12-25"],
    2021: ["01-01", "01-18", "02-15", "04-02", "05-31", "07-05", "09-06", "11-25", "12-24"],
    2022: ["01-17", "02-21", "04-15", "05-30", "06-20", "07-04", "09-05", "11-24", "12-26"],
    2023: ["01-02", "01-16", "02-20", "04-07", "05-29", "06-19", "07-04", "09-04", "11-23", "12-25"],
    2024: ["01-01", "01-15", "02-19", "03-29", "05-27", "06-19", "07-04", "09-02", "11-28", "12-25"],
    2025: [
        "01-01", "01-09", "01-20", "02-17", "04-18", "05-26", "06-19", "07-04", "09-01", "11-27",
        "12-25",
    ],
    2026: ["01-01", "01-19", "02-16", "04-03", "05-25", "06-19", "07-03", "09-07", "11-26", "12-25"],
    2027: ["01-01", "01-18", "02-15", "03-26", "05-31", "06-18", "07-05", "09-06", "11-25", "12-24"],
}  # fmt: skip

#: Number of trading sessions per year from the same independent reference.
REFERENCE_SESSIONS_PER_YEAR = {
    2000: 252, 2001: 248, 2002: 252, 2003: 252, 2004: 252, 2005: 252, 2006: 251,
    2007: 251, 2008: 253, 2009: 252, 2010: 252, 2011: 252, 2012: 250, 2013: 252,
    2014: 252, 2015: 252, 2016: 252, 2017: 251, 2018: 251, 2019: 252, 2020: 253,
    2021: 252, 2022: 251, 2023: 250, 2024: 252, 2025: 250, 2026: 251, 2027: 251,
}  # fmt: skip


@pytest.mark.parametrize("year", sorted(REFERENCE_WEEKDAY_CLOSURES))
def test_calendar_closures_match_the_independent_reference(year):
    closures = [
        day.strftime("%m-%d")
        for day in weekdays(date(year, 1, 1), date(year, 12, 31))
        if not is_trading_day(day)
    ]

    assert closures == REFERENCE_WEEKDAY_CLOSURES[year]
    assert sorted(day.strftime("%m-%d") for day in market_holidays(year)) == closures


@pytest.mark.parametrize("year", sorted(REFERENCE_SESSIONS_PER_YEAR))
def test_trading_session_count_matches_the_independent_reference(year):
    assert len(trading_days(date(year, 1, 1), date(year, 12, 31))) == REFERENCE_SESSIONS_PER_YEAR[year]


def test_weekends_are_never_trading_days():
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(366)]

    assert not any(is_trading_day(day) for day in days if day.weekday() >= 5)
    assert closure_reason(date(2024, 1, 6)) == "weekend"


@pytest.mark.parametrize(
    ("day", "is_open"),
    [
        (date(2021, 12, 31), True),  # New Year's Day 2022 fell on a Saturday: no Friday closure
        (date(2022, 1, 3), True),
        (date(2021, 6, 18), True),  # Juneteenth was not yet a market holiday in 2021
        (date(2022, 6, 20), False),  # first Juneteenth closure (observed on Monday)
        (date(2018, 12, 5), False),  # one-off closure
        (date(2025, 1, 9), False),  # one-off closure
        (date(2024, 3, 29), False),  # Good Friday
        (date(2024, 11, 29), True),  # day after Thanksgiving is a (short) session
        (date(2024, 12, 24), True),  # Christmas Eve is a (short) session
        (date(2023, 1, 2), False),  # New Year's Day observed on Monday
    ],
)
def test_notable_calendar_days(day, is_open):
    assert is_trading_day(day) is is_open


def test_closure_reason_names_the_holiday():
    assert closure_reason(date(2024, 7, 4)) == "Independence Day"
    assert closure_reason(date(2024, 3, 29)) == "Good Friday"
    assert closure_reason(date(2018, 12, 5)).startswith("National Day of Mourning")
    assert closure_reason(date(2024, 7, 5)) is None


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2016, date(2016, 3, 27)),
        (2019, date(2019, 4, 21)),
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2027, date(2027, 3, 28)),
    ],
)
def test_easter_sunday(year, expected):
    assert easter_sunday(year) == expected


def test_trading_days_is_inclusive_and_ordered():
    assert trading_days(date(2024, 7, 1), date(2024, 7, 8)) == [
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
    ]
    assert trading_days(date(2024, 7, 6), date(2024, 7, 7)) == []  # a weekend
    assert trading_days(date(2024, 7, 8), date(2024, 7, 1)) == []  # empty range


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (datetime(2024, 1, 2, 5, 0, tzinfo=UTC), date(2024, 1, 2)),  # winter: midnight is 05:00Z
        (datetime(2024, 7, 1, 4, 0, tzinfo=UTC), date(2024, 7, 1)),  # summer: midnight is 04:00Z
        (datetime(2024, 7, 1, 3, 59, tzinfo=UTC), date(2024, 6, 30)),
        (datetime(2024, 1, 2, 23, 0, tzinfo=UTC), date(2024, 1, 2)),
    ],
)
def test_session_date_uses_new_york_time(timestamp, expected):
    assert session_date(timestamp) == expected


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # Friday 1 March 2024, 10:00 New York: the session is still running.
        (datetime(2024, 3, 1, 15, 0, tzinfo=UTC), date(2024, 2, 29)),
        # Friday 16:30 New York: regular trading is over but the candle is not final yet.
        (datetime(2024, 3, 1, 21, 30, tzinfo=UTC), date(2024, 2, 29)),
        # Friday 20:14 New York: one minute before the candle is treated as final.
        (datetime(2024, 3, 2, 1, 14, tzinfo=UTC), date(2024, 2, 29)),
        # Friday 20:15 New York: the candle is final.
        (datetime(2024, 3, 2, 1, 15, tzinfo=UTC), date(2024, 3, 1)),
        # Saturday and Sunday: Friday remains the last completed session.
        (datetime(2024, 3, 2, 17, 0, tzinfo=UTC), date(2024, 3, 1)),
        (datetime(2024, 3, 3, 17, 0, tzinfo=UTC), date(2024, 3, 1)),
        # Monday morning before the open.
        (datetime(2024, 3, 4, 13, 0, tzinfo=UTC), date(2024, 3, 1)),
        # Tuesday after the Martin Luther King Jr. Day closure (Monday 15 January 2024).
        (datetime(2024, 1, 16, 13, 0, tzinfo=UTC), date(2024, 1, 12)),
        # Evening of a holiday (4 July 2024): the previous session is the last one.
        (datetime(2024, 7, 5, 1, 30, tzinfo=UTC), date(2024, 7, 3)),
        # Summer time: 20:15 New York is 00:15 UTC on the next day.
        (datetime(2024, 7, 2, 0, 15, tzinfo=UTC), date(2024, 7, 1)),
        (datetime(2024, 7, 2, 0, 14, tzinfo=UTC), date(2024, 6, 28)),
    ],
)
def test_last_completed_session_never_returns_an_unfinished_candle(now, expected):
    assert last_completed_session(now) == expected


def test_last_completed_session_requires_an_aware_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        last_completed_session(datetime(2024, 3, 1, 15, 0))


def test_session_window_utc_covers_whole_new_york_days():
    assert session_window_utc(date(2024, 1, 2), date(2024, 1, 12)) == (
        datetime(2024, 1, 2, 5, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 13, 4, 59, 59, tzinfo=UTC),
    )
    # During daylight saving time New York is four hours behind UTC.
    assert session_window_utc(date(2024, 7, 1), date(2024, 7, 5)) == (
        datetime(2024, 7, 1, 4, 0, 0, tzinfo=UTC),
        datetime(2024, 7, 6, 3, 59, 59, tzinfo=UTC),
    )


# =============================================================================
# 10. Missing data detection (gaps in trading sessions)
# =============================================================================


def test_complete_data_has_no_gaps():
    days = weekdays(date(2024, 1, 2), date(2024, 1, 12))

    analysis = analyse_session_gaps(days, date(2024, 1, 2), date(2024, 1, 12))

    assert analysis.gaps == ()
    assert analysis.unexpected_dates == ()
    assert (analysis.expected_sessions, analysis.present_sessions, analysis.missing_sessions) == (9, 9, 0)


@pytest.mark.parametrize(
    ("first", "last", "closed", "sessions"),
    [
        # Independence Day (Thursday 4 July 2024) inside the window.
        (date(2024, 7, 1), date(2024, 7, 5), (date(2024, 7, 4),), 4),
        # A weekend plus Martin Luther King Jr. Day (Monday 15 January 2024).
        (date(2024, 1, 8), date(2024, 1, 19), (date(2024, 1, 15),), 9),
        # Good Friday (29 March 2024) followed by a weekend.
        (date(2024, 3, 25), date(2024, 4, 2), (date(2024, 3, 29),), 6),
        # Thanksgiving (28 November 2024); the Friday after is a session.
        (date(2024, 11, 25), date(2024, 11, 29), (date(2024, 11, 28),), 4),
        # Christmas and New Year across a year boundary.
        (date(2024, 12, 23), date(2025, 1, 3), (date(2024, 12, 25), date(2025, 1, 1)), 8),
        # One-off closure on Thursday 9 January 2025.
        (date(2025, 1, 6), date(2025, 1, 10), (date(2025, 1, 9),), 4),
        # Juneteenth (Wednesday 19 June 2024).
        (date(2024, 6, 17), date(2024, 6, 21), (date(2024, 6, 19),), 4),
    ],
)
def test_weekends_and_market_holidays_are_not_reported_as_gaps(first, last, closed, sessions):
    present = weekdays(first, last, closed=closed)

    analysis = analyse_session_gaps(present, first, last)

    assert analysis.gaps == ()
    assert analysis.unexpected_dates == ()
    assert analysis.expected_sessions == analysis.present_sessions == sessions
    assert session_gap_findings(analysis, symbol="TESTA", timeframe="1Day") == []


def test_a_missing_session_is_detected():
    present = [day for day in weekdays(date(2024, 1, 2), date(2024, 1, 12)) if day != date(2024, 1, 9)]

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 1, 12))

    assert [(gap.kind, gap.sessions) for gap in analysis.gaps] == [
        (GAP_INTERIOR, (date(2024, 1, 9),))
    ]
    assert analysis.missing_sessions == analysis.interior_missing_sessions == 1

    finding = session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")[0]
    assert finding.check_name == CHECK_SESSION_GAP
    assert finding.status == CheckStatus.FAILED
    assert finding.message == (
        "Missing data: 1 expected trading session (2024-01-09) between stored bars have no bar"
    )
    assert finding.details["missing_sessions"] == ["2024-01-09"]
    assert finding.details["gap_type"] == GAP_INTERIOR


def test_consecutive_missing_sessions_across_a_weekend_form_one_gap():
    missing = {date(2024, 1, 5), date(2024, 1, 8)}  # Friday and the following Monday
    present = [day for day in weekdays(date(2024, 1, 2), date(2024, 1, 12)) if day not in missing]

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 1, 12))

    assert [(gap.kind, gap.sessions) for gap in analysis.gaps] == [
        (GAP_INTERIOR, (date(2024, 1, 5), date(2024, 1, 8)))
    ]
    finding = session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")[0]
    assert "2 expected trading sessions (2024-01-05 to 2024-01-08)" in finding.message
    assert finding.details["missing_session_count"] == 2


def test_separate_gaps_are_reported_separately():
    missing = {date(2024, 1, 4), date(2024, 1, 10), date(2024, 1, 11)}
    present = [day for day in weekdays(date(2024, 1, 2), date(2024, 1, 12)) if day not in missing]

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 1, 12))

    assert [gap.sessions for gap in analysis.gaps] == [
        (date(2024, 1, 4),),
        (date(2024, 1, 10), date(2024, 1, 11)),
    ]
    assert analysis.missing_sessions == 3
    assert len(session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")) == 2


def test_leading_gap_is_a_warning_because_history_may_start_later():
    present = weekdays(date(2024, 1, 8), date(2024, 1, 12))

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 1, 12))

    assert [(gap.kind, len(gap.sessions)) for gap in analysis.gaps] == [(GAP_LEADING, 4)]
    assert analysis.interior_missing_sessions == 0
    finding = session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")[0]
    assert finding.status == CheckStatus.WARNING
    assert "History starts later than requested" in finding.message
    assert finding.details["first_missing_session"] == "2024-01-02"
    assert finding.details["last_missing_session"] == "2024-01-05"


def test_trailing_gap_is_a_warning_that_data_needs_updating():
    present = weekdays(date(2024, 1, 2), date(2024, 1, 9))

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 1, 12))

    assert [(gap.kind, len(gap.sessions)) for gap in analysis.gaps] == [(GAP_TRAILING, 3)]
    finding = session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")[0]
    assert finding.status == CheckStatus.WARNING
    assert "Data is not up to date" in finding.message


def test_no_data_at_all_is_reported():
    analysis = analyse_session_gaps([], date(2024, 1, 2), date(2024, 1, 12))

    assert [(gap.kind, len(gap.sessions)) for gap in analysis.gaps] == [(GAP_NO_DATA, 9)]
    finding = session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")[0]
    assert finding.status == CheckStatus.WARNING
    assert finding.message.startswith("No data:")


def test_leading_interior_and_trailing_gaps_together():
    present = [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 9), date(2024, 1, 10)]

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 1, 12))

    assert [(gap.kind, gap.sessions) for gap in analysis.gaps] == [
        (GAP_LEADING, (date(2024, 1, 2), date(2024, 1, 3))),
        (GAP_INTERIOR, (date(2024, 1, 8),)),
        (GAP_TRAILING, (date(2024, 1, 11), date(2024, 1, 12))),
    ]
    statuses = [
        finding.status for finding in session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")
    ]
    assert statuses == [CheckStatus.WARNING, CheckStatus.FAILED, CheckStatus.WARNING]


def test_bars_on_closed_days_are_reported():
    present = [*weekdays(date(2024, 7, 1), date(2024, 7, 5)), date(2024, 7, 6)]  # 4 July and a Saturday

    analysis = analyse_session_gaps(present, date(2024, 7, 1), date(2024, 7, 8))

    assert analysis.unexpected_dates == (date(2024, 7, 4), date(2024, 7, 6))
    findings = [
        finding
        for finding in session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")
        if finding.details["gap_type"] == "unexpected_session"
    ]
    assert [finding.status for finding in findings] == [CheckStatus.FAILED, CheckStatus.FAILED]
    assert "not a trading session (Independence Day)" in findings[0].message
    assert "not a trading session (weekend)" in findings[1].message


def test_dates_outside_the_window_are_ignored():
    present = weekdays(date(2023, 12, 1), date(2024, 2, 29), closed=(date(2023, 12, 25), date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19)))

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 1, 12))

    assert analysis.gaps == ()
    assert analysis.present_sessions == 9


def test_window_without_any_trading_day_has_nothing_to_report():
    analysis = analyse_session_gaps([], date(2024, 7, 6), date(2024, 7, 7))

    assert analysis.expected_sessions == 0
    assert analysis.gaps == ()


def test_long_gaps_list_a_limited_number_of_dates():
    present = [date(2024, 12, 31)]

    analysis = analyse_session_gaps(present, date(2024, 1, 2), date(2024, 12, 31))
    finding = session_gap_findings(analysis, symbol="TESTA", timeframe="1Day")[0]

    assert finding.details["missing_session_count"] == 251
    assert len(finding.details["missing_sessions"]) == 100
    assert finding.details["missing_sessions_truncated"] is True


# =============================================================================
# Recording check outcomes
# =============================================================================


def test_build_check_rows_records_findings_passes_and_skips(make_bar):
    result = run_checks({"TESTA": [make_bar(date(2024, 1, 2)), make_bar(date(2024, 1, 3), v=-5)]})
    check_run_id = new_check_run_id()

    rows = build_check_rows(
        check_run_id=check_run_id,
        stage=CheckStage.INGESTION,
        symbol="TESTA",
        timeframe="1Day",
        check_names=(*ROW_CHECKS, CHECK_SESSION_GAP),
        records_checked=result.records_checked,
        findings=result.findings,
        skipped={CHECK_SESSION_GAP: "fixture reason"},
    )

    by_check = {row.check_name: row for row in rows}
    assert set(by_check) == {*ROW_CHECKS, CHECK_SESSION_GAP}
    assert {row.check_run_id for row in rows} == {check_run_id}
    assert {row.stage for row in rows} == {CheckStage.INGESTION}

    failed = by_check[CHECK_NEGATIVE_VOLUME]
    assert failed.status == CheckStatus.FAILED
    assert failed.bar_timestamp == datetime(2024, 1, 3, 5, 0, tzinfo=UTC)
    assert failed.details["record"]["v"] == -5
    assert failed.records_checked is None

    passed = by_check[CHECK_HIGH_BELOW_LOW]
    assert passed.status == CheckStatus.PASSED
    assert passed.records_checked == 2
    assert passed.symbol == "TESTA"

    skipped = by_check[CHECK_SESSION_GAP]
    assert skipped.status == CheckStatus.SKIPPED
    assert skipped.message == "fixture reason"


# =============================================================================
# Validation of bars already stored in the database
# =============================================================================

JAN_FEB_2024_CLOSURES = (date(2024, 1, 15), date(2024, 2, 19))


def january_february_2024() -> list[date]:
    """Every trading session from 2 January to 29 February 2024 (41 sessions)."""
    return weekdays(date(2024, 1, 2), date(2024, 2, 29), closed=JAN_FEB_2024_CLOSURES)


def run_stored_validation(session_factory, settings, now, **kwargs):
    with session_scope(session_factory) as session:
        return validate_stored_bars(session, settings, now=now, **kwargs)


def stored_checks(session_factory) -> list[DataQualityCheck]:
    with session_scope(session_factory) as session:
        return list(session.execute(select(DataQualityCheck).order_by(DataQualityCheck.id)).scalars())


def corrupt(engine, statement: str, **parameters) -> None:
    """Deliberately damage stored fixture data behind the application's back."""
    with engine.begin() as connection:
        result = connection.execute(text(statement), parameters)
        assert result.rowcount >= 1, "the corruption statement did not change anything"


def test_valid_stored_bars_pass_every_check(session_factory, settings, fixed_now, store_bars):
    store_bars("TESTA", january_february_2024())

    summary = run_stored_validation(session_factory, settings, fixed_now)

    assert summary.bars_checked == 41
    assert summary.groups == (("TESTA", "1Day"),)
    assert summary.failed == 0
    assert summary.warnings == 0
    assert summary.findings == []

    rows = stored_checks(session_factory)
    assert {row.check_name for row in rows} == {*ROW_CHECKS, CHECK_SESSION_GAP}
    assert {row.status for row in rows} == {CheckStatus.PASSED}
    assert {row.stage for row in rows} == {CheckStage.STORED}
    assert {row.check_run_id for row in rows} == {summary.check_run_id}
    assert {row.records_checked for row in rows} == {41}
    assert {row.ingestion_run_id for row in rows} == {None}


@pytest.mark.parametrize(
    ("statement", "expected_check"),
    [
        ("UPDATE bars SET high = 8.0 WHERE timestamp = :ts", CHECK_HIGH_BELOW_LOW),
        ("UPDATE bars SET low = -1.0 WHERE timestamp = :ts", CHECK_NON_POSITIVE_PRICE),
        ("UPDATE bars SET open = 50.0 WHERE timestamp = :ts", CHECK_OPEN_CLOSE_RANGE),
        ("UPDATE bars SET close = 8.0 WHERE timestamp = :ts", CHECK_OPEN_CLOSE_RANGE),
        ("UPDATE bars SET volume = -5 WHERE timestamp = :ts", CHECK_NEGATIVE_VOLUME),
        ("UPDATE bars SET volume = 'fixture text' WHERE timestamp = :ts", CHECK_MISSING_FIELDS),
        ("UPDATE bars SET vwap = 'fixture text' WHERE timestamp = :ts", CHECK_MISSING_FIELDS),
        ("UPDATE bars SET feed = 'sip' WHERE timestamp = :ts", CHECK_SOURCE_METADATA),
        ("UPDATE bars SET adjustment = 'raw' WHERE timestamp = :ts", CHECK_SOURCE_METADATA),
        ("UPDATE bars SET source = '' WHERE timestamp = :ts", CHECK_SOURCE_METADATA),
    ],
)
def test_deliberately_corrupted_stored_bar_is_caught(
    session_factory, settings, fixed_now, store_bars, engine, statement, expected_check
):
    store_bars("TESTA", january_february_2024())
    corrupt(engine, statement, ts="2024-01-10T05:00:00.000000Z")

    summary = run_stored_validation(session_factory, settings, fixed_now)

    assert summary.failed == 1
    finding = summary.findings[0]
    assert finding.check_name == expected_check
    assert finding.status == CheckStatus.FAILED
    assert finding.symbol == "TESTA"
    assert finding.bar_timestamp == datetime(2024, 1, 10, 5, 0, tzinfo=UTC)

    failed_rows = [row for row in stored_checks(session_factory) if row.status == CheckStatus.FAILED]
    assert [(row.check_name, row.symbol) for row in failed_rows] == [(expected_check, "TESTA")]
    assert failed_rows[0].details["record"]["timestamp"] == "2024-01-10T05:00:00.000000Z"


@pytest.mark.parametrize(
    "bad_timestamp",
    ["2024-01-10 05:00:00", "2024-01-10T00:00:00-05:00", "fixture-not-a-time"],
)
def test_corrupted_stored_timestamp_is_caught_without_crashing(
    session_factory, settings, fixed_now, store_bars, engine, bad_timestamp
):
    store_bars("TESTA", january_february_2024())
    corrupt(
        engine,
        "UPDATE bars SET timestamp = :bad WHERE timestamp = :ts",
        bad=bad_timestamp,
        ts="2024-01-10T05:00:00.000000Z",
    )

    summary = run_stored_validation(session_factory, settings, fixed_now)

    by_check = {finding.check_name: finding for finding in summary.findings}
    # The unreadable timestamp is reported, and its session shows up as missing.
    assert set(by_check) == {CHECK_INVALID_TIMESTAMP, CHECK_SESSION_GAP}
    assert bad_timestamp in by_check[CHECK_INVALID_TIMESTAMP].message
    assert by_check[CHECK_SESSION_GAP].details["missing_sessions"] == ["2024-01-10"]


def test_unexpected_symbol_in_the_database_is_caught(
    session_factory, settings, fixed_now, store_bars
):
    store_bars("TESTA", january_february_2024())
    store_bars("ZZZZ", [date(2024, 1, 10)])

    summary = run_stored_validation(session_factory, settings, fixed_now)

    assert [(finding.check_name, finding.symbol) for finding in summary.findings] == [
        (CHECK_UNEXPECTED_SYMBOL, "ZZZZ")
    ]
    gap_rows = [
        row
        for row in stored_checks(session_factory)
        if row.symbol == "ZZZZ" and row.check_name == CHECK_SESSION_GAP
    ]
    assert [row.status for row in gap_rows] == [CheckStatus.SKIPPED]


def test_deleted_stored_bar_is_reported_as_missing_data(
    session_factory, settings, fixed_now, store_bars, engine
):
    store_bars("TESTA", january_february_2024())
    corrupt(engine, "DELETE FROM bars WHERE timestamp = :ts", ts="2024-02-07T05:00:00.000000Z")

    summary = run_stored_validation(session_factory, settings, fixed_now)

    assert summary.failed == 1
    finding = summary.findings[0]
    assert finding.check_name == CHECK_SESSION_GAP
    assert finding.status == CheckStatus.FAILED
    assert finding.details["missing_sessions"] == ["2024-02-07"]
    assert finding.details["gap_type"] == GAP_INTERIOR


def test_stored_validation_does_not_flag_weekends_or_holidays(
    session_factory, settings, fixed_now, store_bars
):
    # 41 sessions; the window also contains 8 weekends and 2 market holidays.
    store_bars("TESTA", january_february_2024())

    summary = run_stored_validation(session_factory, settings, fixed_now)

    gap_rows = [row for row in stored_checks(session_factory) if row.check_name == CHECK_SESSION_GAP]
    assert [row.status for row in gap_rows] == [CheckStatus.PASSED]
    assert summary.counts[CHECK_SESSION_GAP] == {CheckStatus.PASSED: 1}


def test_stale_and_late_starting_data_produce_warnings_not_failures(
    session_factory, settings, fixed_now, store_bars
):
    store_bars("TESTA", weekdays(date(2024, 1, 8), date(2024, 2, 16), closed=JAN_FEB_2024_CLOSURES))

    summary = run_stored_validation(session_factory, settings, fixed_now)

    assert summary.failed == 0
    assert summary.warnings == 2
    assert [finding.details["gap_type"] for finding in summary.findings] == [GAP_LEADING, GAP_TRAILING]
    assert summary.findings[0].details["missing_session_count"] == 4  # 2-5 January
    assert summary.findings[1].details["missing_session_count"] == 8  # 20-29 February


def test_stored_validation_can_be_limited_to_symbols(
    session_factory, settings, fixed_now, store_bars
):
    store_bars("TESTA", january_february_2024())
    store_bars("TESTB", january_february_2024(), volume=-1)

    only_a = run_stored_validation(session_factory, settings, fixed_now, symbols=["TESTA"])
    everything = run_stored_validation(session_factory, settings, fixed_now)

    assert only_a.groups == (("TESTA", "1Day"),)
    assert only_a.failed == 0
    assert everything.groups == (("TESTA", "1Day"), ("TESTB", "1Day"))
    assert everything.failed == 41
    assert {finding.check_name for finding in everything.findings} == {CHECK_NEGATIVE_VOLUME}


def test_requested_symbol_without_stored_bars_is_reported(session_factory, settings, fixed_now):
    summary = run_stored_validation(session_factory, settings, fixed_now, symbols=["TESTB"])

    assert summary.bars_checked == 0
    assert summary.groups == (("TESTB", "1Day"),)
    assert summary.failed == 0
    assert [finding.details["gap_type"] for finding in summary.findings] == [GAP_NO_DATA]


def test_empty_database_validates_cleanly(session_factory, settings, fixed_now):
    summary = run_stored_validation(session_factory, settings, fixed_now)

    assert summary.bars_checked == 0
    assert summary.groups == ()
    assert summary.failed == summary.warnings == 0
    assert stored_checks(session_factory) == []


def test_gap_detection_is_marked_skipped_for_unsupported_timeframes(
    session_factory, settings, fixed_now, store_bars
):
    store_bars("TESTA", [date(2024, 1, 10)], timeframe="1Hour")

    summary = run_stored_validation(session_factory, settings, fixed_now)

    assert summary.failed == 0
    assert summary.counts[CHECK_SESSION_GAP] == {CheckStatus.SKIPPED: 1}
    skipped = [row for row in stored_checks(session_factory) if row.status == CheckStatus.SKIPPED]
    assert "implemented for 1Day bars only" in skipped[0].message


def test_each_validation_run_gets_its_own_identifier(
    session_factory, settings, fixed_now, store_bars
):
    store_bars("TESTA", january_february_2024())

    first = run_stored_validation(session_factory, settings, fixed_now)
    second = run_stored_validation(session_factory, settings, fixed_now)

    assert first.check_run_id != second.check_run_id
    rows = stored_checks(session_factory)
    assert len(rows) == 20  # ten checks per run
    assert {row.check_run_id for row in rows} == {first.check_run_id, second.check_run_id}


def test_validation_never_modifies_the_stored_bars(
    session_factory, settings, fixed_now, store_bars, engine
):
    store_bars("TESTA", january_february_2024())
    corrupt(engine, "UPDATE bars SET high = 8.0 WHERE timestamp = :ts", ts="2024-01-10T05:00:00.000000Z")
    with engine.connect() as connection:
        before = connection.execute(text("SELECT * FROM bars ORDER BY timestamp")).all()

    run_stored_validation(session_factory, settings, fixed_now)

    with engine.connect() as connection:
        after = connection.execute(text("SELECT * FROM bars ORDER BY timestamp")).all()
    assert after == before  # invalid data is reported, never repaired or removed


# =============================================================================
# Command line: scripts/validate.py
# =============================================================================


def test_cli_passes_for_valid_stored_bars(run_script, session_factory, store_bars):
    store_bars("TESTA", january_february_2024())

    exit_code, output, _ = run_script("validate")

    assert exit_code == 0
    assert "Validated 41 stored bar(s) in 1 symbol/timeframe group(s)." in output
    assert "RESULT: PASSED - no problems found" in output
    table = {line.split()[0]: line.split()[1:] for line in output.splitlines() if "_" in line.split(" ")[0]}
    assert table["high_below_low"] == ["1", "0", "0", "0"]
    assert table["negative_volume"] == ["1", "0", "0", "0"]
    # The fixture data ends in February 2024, so it is (correctly) reported as not up to date.
    assert table["trading_session_gap"] == ["0", "0", "1", "0"]
    assert "Data is not up to date" in output
    assert len(stored_checks(session_factory)) == 10


def test_cli_fails_when_stored_bars_were_corrupted(run_script, session_factory, store_bars, engine):
    store_bars("TESTA", january_february_2024())
    corrupt(engine, "UPDATE bars SET high = 8.0 WHERE timestamp = :ts", ts="2024-01-10T05:00:00.000000Z")
    corrupt(engine, "UPDATE bars SET volume = -5 WHERE timestamp = :ts", ts="2024-02-07T05:00:00.000000Z")

    exit_code, output, _ = run_script("validate")

    assert exit_code == 1
    assert "RESULT: FAILED - 2 problem(s) found" in output
    assert "Nothing was changed or repaired." in output
    assert (
        "[high_below_low] TESTA 1Day 2024-01-10T05:00:00.000000Z - High 8.0 is lower than low 9.0"
        in output
    )
    assert "[negative_volume] TESTA 1Day 2024-02-07T05:00:00.000000Z - Negative volume: -5" in output
    failed = [row for row in stored_checks(session_factory) if row.status == CheckStatus.FAILED]
    assert {row.check_name for row in failed} == {CHECK_HIGH_BELOW_LOW, CHECK_NEGATIVE_VOLUME}


def test_cli_can_validate_selected_symbols(run_script, store_bars):
    store_bars("TESTA", january_february_2024())
    store_bars("TESTB", january_february_2024(), volume=-1)

    code_a, output_a, _ = run_script("validate", "--symbol", "testa")
    code_b, output_b, _ = run_script("validate", "--symbol", "TESTB", "--max-issues", "3")

    assert code_a == 0
    assert "in 1 symbol/timeframe group(s)" in output_a
    assert code_b == 1
    assert "Failed checks (showing 3 of 41):" in output_b
    assert output_b.count("[negative_volume]") == 3


def test_cli_on_an_empty_database(run_script, engine):
    exit_code, output, _ = run_script("validate")

    assert exit_code == 0
    assert "Nothing to validate: no bars are stored yet." in output


def test_cli_explains_a_missing_database(run_script):
    # No ``engine`` fixture: the database file was never created.
    exit_code, _, errors = run_script("validate")

    assert exit_code == 2
    assert "database not found" in errors
    assert "python -m app.database" in errors


@pytest.mark.parametrize(
    ("argv", "expected"),
    [(("--symbol", "bad symbol!"), "invalid symbol"), (("--timeframe", "1D"), "invalid timeframe")],
)
def test_cli_rejects_invalid_arguments(run_script, engine, argv, expected):
    exit_code, _, errors = run_script("validate", *argv)

    assert exit_code == 2
    assert expected in errors
