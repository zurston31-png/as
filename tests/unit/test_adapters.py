"""File and in-memory adapters.

The load boundary is where a timestamp convention, a timezone, or a blank
cell either gets handled or gets baked into every number the system later
reports. The emphasis here is on the bar-open/bar-close conversion (a
silent one-interval lookahead), on timezone conversion across a DST
boundary, and on the error messages: a cryptic KeyError at this layer costs
an hour of hunting, so the messages are asserted, not just the exceptions.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from flow_model.data.adapters import (
    ColumnMap,
    CsvAdapter,
    CsvAdapterConfig,
    InMemoryAdapter,
    ParquetAdapter,
    ParquetAdapterConfig,
    series_content_hash,
)
from flow_model.data.base import DataLayerError, DataSourceAdapter, Feed, SchemaError
from flow_model.data.series import (
    BarSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    to_ns,
    to_ns_array,
)
from flow_model.data.store import DataStore

HEADER = "timestamp,open,high,low,close,volume"
UTC = timezone.utc
EASTERN = "America/New_York"

WIDE_START = date(2000, 1, 1)
WIDE_END = date(2100, 1, 1)


# --- local fixtures --------------------------------------------------------


@pytest.fixture
def write_csv(tmp_path):
    """Write a CSV under a fresh root and return (root, path).

    Each file gets its own symbol so one test's malformed file cannot be
    picked up by another's glob.
    """

    def _write(text: str, symbol: str = "NQ", interval: int = 300) -> tuple[str, object]:
        path = tmp_path / f"{symbol}_{interval}s.csv"
        path.write_text(text, encoding="utf-8")
        return str(tmp_path), path

    return _write


@pytest.fixture
def csv_adapter(write_csv):
    """CsvAdapter over a file written from `rows`."""

    def _make(rows: str, header: str = HEADER, interval: int = 300, **config) -> CsvAdapter:
        root, _ = write_csv(f"{header}\n{rows}", interval=interval)
        return CsvAdapter(CsvAdapterConfig(root_path=root, **config))

    return _make


def iso_rows(n: int = 3, start: str = "2020-01-02T14:30:00Z", interval: int = 300) -> str:
    base = datetime.fromisoformat(start)
    lines = []
    for i in range(n):
        ts = (base + timedelta(seconds=interval * i)).isoformat()
        price = 100.0 + i
        lines.append(f"{ts},{price},{price + 1},{price - 1},{price + 0.5},{1000 + i}")
    return "\n".join(lines) + "\n"


def make_bar_series(
    n: int = 10,
    symbol: str = "NQ",
    interval: int = 300,
    start: datetime | None = None,
    base_price: float = 100.0,
) -> BarSeries:
    start = start or datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
    ts = to_ns_array(start + timedelta(seconds=interval * i) for i in range(n))
    close = np.arange(n, dtype=np.float64) + base_price
    return BarSeries(
        symbol=symbol,
        ts_ns=ts,
        interval_seconds=interval,
        columns={
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
    )


# --- round trip ------------------------------------------------------------


def test_round_trip_bars_from_csv(csv_adapter):
    adapter = csv_adapter(iso_rows(3))
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert len(series) == 3
    assert series.symbol == "NQ"
    assert series.interval_seconds == 300
    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
    assert series.col("close").tolist() == [100.5, 101.5, 102.5]
    assert series.col("volume").tolist() == [1000.0, 1001.0, 1002.0]
    assert series.meta["source"] == "csv"


def test_optional_columns_are_loaded_when_mapped(csv_adapter):
    adapter = csv_adapter(
        "2020-01-02T14:30:00Z,100,101,99,100.5,1000,42,100.25\n",
        header=f"{HEADER},trades,vwap",
        column_map=ColumnMap(trades="trades", vwap="vwap"),
    )
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.has("trades") and series.has("vwap")
    bar = series.bar_at(0)
    assert bar.trades == 42
    assert bar.vwap == pytest.approx(100.25)


def test_optional_columns_are_absent_when_unmapped(csv_adapter):
    adapter = csv_adapter(
        "2020-01-02T14:30:00Z,100,101,99,100.5,1000,42,100.25\n",
        header=f"{HEADER},trades,vwap",
    )
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert not series.has("trades")
    assert not series.has("vwap")


def test_vendor_column_names_are_remapped(csv_adapter):
    adapter = csv_adapter(
        "2020-01-02T14:30:00Z,100,101,99,100.5,1000\n",
        header="Date,O,H,L,C,Vol",
        column_map=ColumnMap(
            timestamp="Date", open="O", high="H", low="L", close="C", volume="Vol"
        ),
    )
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.col("open").tolist() == [100.0]
    assert series.col("volume").tolist() == [1000.0]


def test_utf8_bom_header_is_tolerated(write_csv):
    root, path = write_csv("")
    path.write_text(f"﻿{HEADER}\n{iso_rows(1)}", encoding="utf-8")
    series = CsvAdapter(CsvAdapterConfig(root_path=root)).load_bars(
        "NQ", 300, WIDE_START, WIDE_END
    )

    assert len(series) == 1


def test_header_only_file_yields_an_empty_series(csv_adapter):
    series = csv_adapter("").load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert len(series) == 0
    assert series.interval_seconds == 300


def test_empty_file_names_the_missing_header(write_csv):
    root, _ = write_csv("")
    with pytest.raises(SchemaError, match="expected a header row"):
        CsvAdapter(CsvAdapterConfig(root_path=root)).load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_repeated_loads_are_identical(csv_adapter):
    adapter = csv_adapter(iso_rows(5))
    first = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)
    second = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert np.array_equal(first.ts_ns, second.ts_ns)
    assert np.array_equal(first.col("close"), second.col("close"))
    assert series_content_hash(first) == series_content_hash(second)


# --- bar-open vs bar-close -------------------------------------------------


def test_bar_open_timestamps_are_shifted_by_exactly_one_interval(write_csv):
    """The same file read both ways differs by exactly `interval_seconds`.

    If an open-stamped file is read as close-stamped, every bar becomes
    visible one interval early and every signal gets a head start that looks
    like skill.
    """
    root, _ = write_csv(f"{HEADER}\n{iso_rows(4)}")
    as_close = CsvAdapter(CsvAdapterConfig(root_path=root)).load_bars(
        "NQ", 300, WIDE_START, WIDE_END
    )
    as_open = CsvAdapter(
        CsvAdapterConfig(root_path=root, timestamp_is_bar_open=True)
    ).load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert len(as_open) == len(as_close) == 4
    assert np.array_equal(as_open.ts_ns - as_close.ts_ns, np.full(4, 300 * 10**9))
    assert np.array_equal(as_open.col("close"), as_close.col("close"))
    assert as_open.timestamps()[0] == datetime(2020, 1, 2, 14, 35, tzinfo=UTC)


def test_bar_open_shift_follows_the_requested_interval(write_csv):
    root, _ = write_csv(f"{HEADER}\n{iso_rows(2, interval=900)}", interval=900)
    series = CsvAdapter(
        CsvAdapterConfig(
            root_path=root,
            timestamp_is_bar_open=True,
            filename_template="{symbol}_{interval}s.csv",
        )
    ).load_bars("NQ", 900, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 45, tzinfo=UTC)


def test_bar_open_flag_is_boolean_with_no_guess_value():
    """There is deliberately no third "detect it" state.

    The two conventions are indistinguishable from the file's contents, so a
    detector would be a coin flip on a one-interval lookahead.
    """
    field = CsvAdapterConfig.model_fields["timestamp_is_bar_open"]
    assert field.annotation is bool
    assert field.default is False


def test_bar_open_and_bar_close_are_different_datasets(write_csv):
    root, _ = write_csv(f"{HEADER}\n{iso_rows(3)}")
    close_fp = CsvAdapter(CsvAdapterConfig(root_path=root)).fingerprint(
        "NQ", 300, WIDE_START, WIDE_END, 3
    )
    open_fp = CsvAdapter(
        CsvAdapterConfig(root_path=root, timestamp_is_bar_open=True)
    ).fingerprint("NQ", 300, WIDE_START, WIDE_END, 3)

    assert close_fp.data_hash != open_fp.data_hash


# --- timestamp units -------------------------------------------------------

EPOCH_SECONDS = int(datetime(2020, 1, 2, 14, 30, tzinfo=UTC).timestamp())
UNIT_VALUES = {
    "s": EPOCH_SECONDS,
    "ms": EPOCH_SECONDS * 1_000,
    "us": EPOCH_SECONDS * 1_000_000,
    "ns": EPOCH_SECONDS * 1_000_000_000,
}


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_explicit_epoch_unit(csv_adapter, unit):
    adapter = csv_adapter(
        f"{UNIT_VALUES[unit]},100,101,99,100.5,1000\n", timestamp_unit=unit
    )
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 30, tzinfo=UTC)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_auto_detects_each_epoch_unit(csv_adapter, unit):
    adapter = csv_adapter(f"{UNIT_VALUES[unit]},100,101,99,100.5,1000\n")
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 30, tzinfo=UTC)


def test_auto_detects_iso_8601(csv_adapter):
    adapter = csv_adapter("2020-01-02 14:30:00,100,101,99,100.5,1000\n")
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 30, tzinfo=UTC)


def test_auto_detection_ignores_a_zero_first_row(csv_adapter):
    """Unit inference reads the whole sample, not row one.

    A sentinel 0 in the first row would otherwise pick seconds for a
    millisecond file and move every bar to 1970.
    """
    rows = (
        "0,100,101,99,100.5,1000\n"
        f"{UNIT_VALUES['ms']},100,101,99,100.5,1000\n"
    )
    series = csv_adapter(rows).load_bars("NQ", 300, date(1970, 1, 1), WIDE_END)

    assert series.timestamps() == (
        datetime(1970, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 2, 14, 30, tzinfo=UTC),
    )


def test_nanosecond_epoch_keeps_sub_microsecond_resolution(csv_adapter):
    value = UNIT_VALUES["ns"] + 123
    series = csv_adapter(f"{value},100,101,99,100.5,1000\n", timestamp_unit="ns").load_bars(
        "NQ", 300, WIDE_START, WIDE_END
    )

    assert int(series.ts_ns[0]) == value


def test_explicit_iso_unit_rejects_an_epoch_number(csv_adapter):
    adapter = csv_adapter(f"{EPOCH_SECONDS},100,101,99,100.5,1000\n", timestamp_unit="iso")
    with pytest.raises(SchemaError, match="not ISO-8601"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_unparseable_timestamp_names_the_line(csv_adapter):
    adapter = csv_adapter(
        "2020-01-02T14:30:00Z,100,101,99,100.5,1000\n01/02/2020,100,101,99,100.5,1000\n"
    )
    with pytest.raises(SchemaError, match="line 3.*not ISO-8601"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_empty_timestamp_cell_is_rejected(csv_adapter):
    adapter = csv_adapter(",100,101,99,100.5,1000\n", timestamp_unit="iso")
    with pytest.raises(SchemaError, match="timestamp is empty"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_date_only_timestamps_become_local_midnight(csv_adapter):
    adapter = csv_adapter(
        "2020-01-02,100,101,99,100.5,1000\n",
        interval=86400,
        input_timezone=EASTERN,
        timestamp_unit="iso",
    )
    series = adapter.load_bars("NQ", 86400, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 5, 0, tzinfo=UTC)


def test_timestamp_unit_is_validated_by_the_config():
    with pytest.raises(ValidationError):
        CsvAdapterConfig(root_path="x", timestamp_unit="minutes")


def test_unreadable_timestamp_column_asks_for_an_explicit_unit(csv_adapter):
    adapter = csv_adapter(",,,,,\n,100,101,99,100.5,1000\n")
    with pytest.raises(SchemaError, match="set timestamp_unit explicitly"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


# --- timezones -------------------------------------------------------------


def test_naive_local_input_is_read_in_the_declared_zone(csv_adapter):
    """The one place naive input is acceptable: the config declares the zone."""
    adapter = csv_adapter(
        "2020-01-02 09:30:00,100,101,99,100.5,1000\n", input_timezone=EASTERN
    )
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 30, tzinfo=UTC)


def test_dst_boundary_lands_on_the_right_utc_instant(csv_adapter):
    """Two local stamps two hours apart are one hour apart in real time.

    A fixed -5 offset would place 03:30 at 08:30Z and insert an hour of
    fictional time into the session.
    """
    rows = (
        "2021-03-14 01:30:00,100,101,99,100.5,1000\n"
        "2021-03-14 03:30:00,100,101,99,100.5,1000\n"
    )
    series = csv_adapter(rows, input_timezone=EASTERN).load_bars(
        "NQ", 300, WIDE_START, WIDE_END
    )

    assert series.timestamps() == (
        datetime(2021, 3, 14, 6, 30, tzinfo=UTC),
        datetime(2021, 3, 14, 7, 30, tzinfo=UTC),
    )
    assert int(series.ts_ns[1] - series.ts_ns[0]) == 3600 * 10**9


def test_iso_offset_overrides_the_configured_zone(csv_adapter):
    adapter = csv_adapter(
        "2020-01-02T09:30:00-05:00,100,101,99,100.5,1000\n", input_timezone="Asia/Tokyo"
    )
    series = adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 30, tzinfo=UTC)


def test_nonexistent_local_time_is_rejected(csv_adapter):
    """A stamp in the spring-forward gap means the declared zone is wrong."""
    adapter = csv_adapter(
        "2021-03-14 02:30:00,100,101,99,100.5,1000\n", input_timezone=EASTERN
    )
    with pytest.raises(SchemaError, match="does not exist in America/New_York"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_unknown_timezone_is_rejected_at_construction(tmp_path):
    with pytest.raises(DataLayerError, match="unknown input_timezone 'Mars/Olympus'"):
        CsvAdapter(CsvAdapterConfig(root_path=str(tmp_path), input_timezone="Mars/Olympus"))


# --- schema and numeric validation -----------------------------------------


def test_missing_required_column_names_it_and_lists_what_was_found(csv_adapter):
    adapter = csv_adapter("1577975400,100,101,99,100.5\n", header="ts,open,high,low,close")
    with pytest.raises(SchemaError) as excinfo:
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    message = str(excinfo.value)
    assert "'timestamp'" in message
    assert "'volume'" in message
    assert "['ts', 'open', 'high', 'low', 'close']" in message


def test_missing_mapped_optional_column_also_raises(csv_adapter):
    adapter = csv_adapter(iso_rows(1), column_map=ColumnMap(vwap="VWAP"))
    with pytest.raises(SchemaError, match="'VWAP' \\(mapped to 'vwap'\\)"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_unparseable_numeric_raises_naming_row_and_value(csv_adapter):
    rows = iso_rows(1) + "2020-01-02T14:35:00+00:00,100,101,99,n/a,1000\n"
    adapter = csv_adapter(rows)
    with pytest.raises(SchemaError, match="line 3.*'close'.*not a number: 'n/a'"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_blank_numeric_is_not_coerced_to_nan(csv_adapter):
    adapter = csv_adapter("2020-01-02T14:30:00Z,100,101,99,,1000\n")
    with pytest.raises(SchemaError, match="is empty"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_nan_token_is_rejected(csv_adapter):
    adapter = csv_adapter("2020-01-02T14:30:00Z,100,101,99,nan,1000\n")
    with pytest.raises(SchemaError, match="not finite"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_wrong_field_count_names_the_line(csv_adapter):
    adapter = csv_adapter(iso_rows(1) + "2020-01-02T14:35:00Z,100,101,99,100.5\n")
    with pytest.raises(SchemaError, match="line 3 has 5 field"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_duplicate_header_name_is_rejected(csv_adapter):
    adapter = csv_adapter(
        "2020-01-02T14:30:00Z,100,101,99,100.5,1000,7\n", header=f"{HEADER},close"
    )
    with pytest.raises(SchemaError, match="duplicate column name"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_entirely_empty_rows_are_dropped(csv_adapter):
    rows = iso_rows(1) + "\n" + ",,,,,\n" + "   ,,,,,\n" + iso_rows(
        1, start="2020-01-02T14:35:00+00:00"
    )
    series = csv_adapter(rows).load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert len(series) == 2


def test_ohlc_violation_is_rejected(csv_adapter):
    adapter = csv_adapter("2020-01-02T14:30:00Z,100,99,98,100.5,1000\n")
    with pytest.raises(SchemaError, match="OHLC ordering violated"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_negative_volume_is_rejected(csv_adapter):
    adapter = csv_adapter("2020-01-02T14:30:00Z,100,101,99,100.5,-5\n")
    with pytest.raises(SchemaError, match="negative values"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


# --- ordering --------------------------------------------------------------


def test_duplicate_timestamp_raises_naming_the_line(csv_adapter):
    rows = iso_rows(1) + iso_rows(1)
    adapter = csv_adapter(rows)
    with pytest.raises(SchemaError, match="duplicate timestamp at line 3"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_out_of_order_timestamp_raises_naming_the_line(csv_adapter):
    rows = (
        "2020-01-02T14:35:00Z,100,101,99,100.5,1000\n"
        "2020-01-02T14:30:00Z,100,101,99,100.5,1000\n"
    )
    adapter = csv_adapter(rows)
    with pytest.raises(SchemaError, match="out-of-order timestamp at line 3"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_out_of_order_file_is_not_silently_sorted(csv_adapter):
    """Sorting would produce a plausible series nobody knows is wrong."""
    rows = (
        "2020-01-02T14:40:00Z,100,101,99,100.5,1000\n"
        "2020-01-02T14:30:00Z,100,101,99,100.5,1000\n"
        "2020-01-02T14:35:00Z,100,101,99,100.5,1000\n"
    )
    adapter = csv_adapter(rows)
    with pytest.raises(SchemaError, match="does not sort or de-duplicate"):
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)


def test_ordering_is_checked_outside_the_requested_range(csv_adapter):
    """A broken export is a broken export even where the range does not reach."""
    rows = iso_rows(2, start="2020-01-02T14:30:00+00:00") + (
        "2020-01-03T14:35:00Z,100,101,99,100.5,1000\n"
        "2020-01-03T14:30:00Z,100,101,99,100.5,1000\n"
    )
    adapter = csv_adapter(rows)
    with pytest.raises(SchemaError, match="out-of-order"):
        adapter.load_bars("NQ", 300, date(2020, 1, 2), date(2020, 1, 3))


# --- date range ------------------------------------------------------------


def test_start_is_inclusive_and_end_is_exclusive(csv_adapter):
    rows = "".join(
        f"2020-01-0{day}T00:00:00Z,100,101,99,100.5,1000\n" for day in (1, 2, 3, 4)
    )
    series = csv_adapter(rows).load_bars("NQ", 300, date(2020, 1, 2), date(2020, 1, 4))

    assert series.timestamps() == (
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    )


def test_a_bar_exactly_at_the_end_boundary_is_excluded(csv_adapter):
    rows = (
        "2020-01-02T23:59:59Z,100,101,99,100.5,1000\n"
        "2020-01-03T00:00:00Z,100,101,99,100.5,1000\n"
    )
    series = csv_adapter(rows).load_bars("NQ", 300, date(2020, 1, 2), date(2020, 1, 3))

    assert series.timestamps() == (datetime(2020, 1, 2, 23, 59, 59, tzinfo=UTC),)


def test_range_boundaries_use_the_declared_timezone(csv_adapter):
    """19:00 Eastern on Jan 2 is 00:00 UTC on Jan 3, and still a Jan-2 bar."""
    adapter = csv_adapter(
        "2020-01-02 19:00:00,100,101,99,100.5,1000\n", input_timezone=EASTERN
    )
    series = adapter.load_bars("NQ", 300, date(2020, 1, 2), date(2020, 1, 3))

    assert series.timestamps() == (datetime(2020, 1, 3, tzinfo=UTC),)


def test_range_outside_the_file_yields_an_empty_series(csv_adapter):
    series = csv_adapter(iso_rows(3)).load_bars("NQ", 300, date(2021, 1, 1), date(2021, 2, 1))

    assert len(series) == 0
    assert series.interval_seconds == 300


def test_reversed_range_is_rejected(csv_adapter):
    adapter = csv_adapter(iso_rows(1))
    with pytest.raises(DataLayerError, match="ends before it starts"):
        adapter.load_bars("NQ", 300, date(2020, 2, 1), date(2020, 1, 1))


# --- disk, feeds, provenance ----------------------------------------------


def test_available_feeds_reflects_the_disk(tmp_path):
    adapter = CsvAdapter(CsvAdapterConfig(root_path=str(tmp_path)))
    assert adapter.available_feeds("NQ") == frozenset()

    (tmp_path / "NQ_300s.csv").write_text(f"{HEADER}\n{iso_rows(1)}", encoding="utf-8")
    assert adapter.available_feeds("NQ") == frozenset({Feed.BARS})
    assert adapter.available_feeds("ES") == frozenset()


def test_available_feeds_finds_any_interval(tmp_path):
    (tmp_path / "NQ_900s.csv").write_text(f"{HEADER}\n{iso_rows(1)}", encoding="utf-8")
    adapter = CsvAdapter(CsvAdapterConfig(root_path=str(tmp_path)))

    assert adapter.available_feeds("NQ") == frozenset({Feed.BARS})
    assert [p.name for p in adapter.bar_files("NQ")] == ["NQ_900s.csv"]


def test_csv_adapter_never_claims_a_feed_it_cannot_read(csv_adapter):
    adapter = csv_adapter(iso_rows(1))

    assert adapter.available_feeds("NQ") == frozenset({Feed.BARS})
    assert adapter.load_quotes("NQ", WIDE_START, WIDE_END) is None
    assert adapter.load_tick_aggregates("NQ", 300, WIDE_START, WIDE_END) is None
    assert adapter.load_options("NQ", WIDE_START, WIDE_END) is None


def test_missing_file_error_names_the_expected_path(tmp_path):
    adapter = CsvAdapter(CsvAdapterConfig(root_path=str(tmp_path)))
    with pytest.raises(DataLayerError) as excinfo:
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert str(tmp_path / "NQ_300s.csv") in str(excinfo.value)


def test_filename_template_is_honoured(tmp_path):
    (tmp_path / "bars-NQ-300.csv").write_text(f"{HEADER}\n{iso_rows(2)}", encoding="utf-8")
    adapter = CsvAdapter(
        CsvAdapterConfig(root_path=str(tmp_path), filename_template="bars-{symbol}-{interval}.csv")
    )

    assert adapter.available_feeds("NQ") == frozenset({Feed.BARS})
    assert len(adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)) == 2


def test_filename_template_with_an_unknown_placeholder_is_reported(tmp_path):
    adapter = CsvAdapter(
        CsvAdapterConfig(root_path=str(tmp_path), filename_template="{sym}_{interval}s.csv")
    )
    with pytest.raises(DataLayerError, match="unknown placeholder"):
        adapter.available_feeds("NQ")


def test_fingerprint_tracks_the_file_bytes(write_csv):
    root, path = write_csv(f"{HEADER}\n{iso_rows(3)}")
    adapter = CsvAdapter(CsvAdapterConfig(root_path=root))
    before = adapter.fingerprint("NQ", 300, WIDE_START, WIDE_END, 3)

    path.write_text(f"{HEADER}\n{iso_rows(3, start='2020-01-03T14:30:00+00:00')}")
    after = adapter.fingerprint("NQ", 300, WIDE_START, WIDE_END, 3)

    assert before.data_hash != after.data_hash
    assert before.source == "csv"
    assert before.feeds == (Feed.BARS,)


def test_csv_adapter_satisfies_the_adapter_contract(csv_adapter):
    adapter = csv_adapter(iso_rows(1))
    assert isinstance(adapter, DataSourceAdapter)
    assert adapter.name == "csv"


# --- in-memory adapter -----------------------------------------------------


def test_in_memory_adapter_reports_only_what_it_holds():
    adapter = InMemoryAdapter(bars={("NQ", 300): make_bar_series()})

    assert adapter.available_feeds("NQ") == frozenset({Feed.BARS})
    assert adapter.available_feeds("ES") == frozenset()
    assert adapter.name == "memory"


def test_in_memory_adapter_slices_to_the_requested_range():
    series = make_bar_series(n=600, start=datetime(2020, 1, 2, tzinfo=UTC))
    adapter = InMemoryAdapter(bars={("NQ", 300): series})

    sliced = adapter.load_bars("NQ", 300, date(2020, 1, 3), date(2020, 1, 4))

    assert len(sliced) > 0
    assert sliced.first_ts >= datetime(2020, 1, 3, tzinfo=UTC)
    assert sliced.last_ts < datetime(2020, 1, 4, tzinfo=UTC)


def test_in_memory_adapter_returns_none_for_absent_feeds():
    adapter = InMemoryAdapter(bars={("NQ", 300): make_bar_series()})

    assert adapter.load_quotes("NQ", WIDE_START, WIDE_END) is None
    assert adapter.load_tick_aggregates("NQ", 300, WIDE_START, WIDE_END) is None
    assert adapter.load_options("NQ", WIDE_START, WIDE_END) is None


def test_in_memory_adapter_does_not_substitute_another_interval():
    adapter = InMemoryAdapter(
        bars={("NQ", 300): make_bar_series()},
        ticks={("NQ", 60): _tick_series()},
    )

    assert adapter.load_tick_aggregates("NQ", 300, WIDE_START, WIDE_END) is None
    assert adapter.load_tick_aggregates("NQ", 60, WIDE_START, WIDE_END) is not None


def test_in_memory_adapter_rejects_a_miskeyed_symbol():
    with pytest.raises(DataLayerError, match="under another symbol's name"):
        InMemoryAdapter(bars={("ES", 300): make_bar_series(symbol="NQ")})


def test_in_memory_adapter_rejects_an_interval_mismatch():
    with pytest.raises(DataLayerError, match="interval_seconds=300"):
        InMemoryAdapter(bars={("NQ", 60): make_bar_series(interval=300)})


def test_in_memory_adapter_lists_what_it_holds_for_an_unknown_interval():
    adapter = InMemoryAdapter(bars={("NQ", 300): make_bar_series()})
    with pytest.raises(DataLayerError, match=r"no in-memory bars at 900s"):
        adapter.load_bars("NQ", 900, WIDE_START, WIDE_END)


def test_in_memory_fingerprint_depends_on_contents():
    one = InMemoryAdapter(bars={("NQ", 300): make_bar_series(base_price=100.0)})
    two = InMemoryAdapter(bars={("NQ", 300): make_bar_series(base_price=200.0)})
    args = ("NQ", 300, WIDE_START, WIDE_END, 10)

    assert one.fingerprint(*args).data_hash != two.fingerprint(*args).data_hash


def _quote_series(n: int = 10, symbol: str = "NQ") -> QuoteSeries:
    start = datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
    return QuoteSeries(
        symbol=symbol,
        ts_ns=to_ns_array(start + timedelta(seconds=30 * i) for i in range(n)),
        columns={
            "bid": np.full(n, 99.75),
            "ask": np.full(n, 100.25),
            "bid_size": np.full(n, 10.0),
            "ask_size": np.full(n, 12.0),
        },
    )


def _tick_series(n: int = 10, symbol: str = "NQ", interval: int = 60) -> TickSeries:
    start = datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
    return TickSeries(
        symbol=symbol,
        ts_ns=to_ns_array(start + timedelta(seconds=interval * i) for i in range(n)),
        columns={"buy_volume": np.full(n, 600.0), "sell_volume": np.full(n, 400.0)},
        meta={"classification_method": "test"},
    )


def _options_series(n: int = 4, symbol: str = "NQ") -> OptionsSeries:
    start = datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
    return OptionsSeries(
        symbol=symbol,
        ts_ns=to_ns_array(start + timedelta(seconds=3600 * i) for i in range(n)),
        columns={
            "call_volume": np.full(n, 1000.0),
            "put_volume": np.full(n, 800.0),
            "call_premium": np.full(n, 2.0e6),
            "put_premium": np.full(n, 1.5e6),
            "call_oi": np.full(n, 5000.0),
            "put_oi": np.full(n, 4000.0),
        },
        meta={"is_intraday": True, "source": "test"},
    )


def test_in_memory_adapter_loads_through_the_data_store():
    adapter = InMemoryAdapter(
        bars={("NQ", 300): make_bar_series(n=20)},
        quotes={"NQ": _quote_series()},
        ticks={("NQ", 300): _tick_series(interval=300)},
        options={"NQ": _options_series()},
    )
    store = DataStore()
    data = store.load(adapter, "NQ", date(2020, 1, 1), date(2020, 1, 3), primary_interval=300)

    assert len(data.primary_bars) == 20
    assert data.available_feeds() == frozenset(
        {Feed.BARS, Feed.QUOTES, Feed.TICK_AGGREGATE, Feed.OPTIONS_SNAPSHOT}
    )
    assert data.fingerprint is not None and data.fingerprint.source == "memory"


def test_in_memory_adapter_feeds_a_point_in_time_view():
    adapter = InMemoryAdapter(bars={("NQ", 300): make_bar_series(n=20)})
    store = DataStore()
    store.load(adapter, "NQ", date(2020, 1, 1), date(2020, 1, 3), primary_interval=300)

    view = store.view("NQ", datetime(2020, 1, 2, 15, 0, tzinfo=UTC))
    last = view.last_bar()

    assert last is not None
    assert last.close_ts == datetime(2020, 1, 2, 15, 0, tzinfo=UTC)


def test_in_memory_adapter_honours_a_custom_name():
    adapter = InMemoryAdapter(bars={("NQ", 300): make_bar_series()}, name="generator")
    assert adapter.name == "generator"
    assert adapter.fingerprint("NQ", 300, WIDE_START, WIDE_END, 10).source == "generator"


# --- parquet ---------------------------------------------------------------


def test_parquet_adapter_reports_missing_pyarrow_clearly(tmp_path, monkeypatch):
    """Absent pyarrow is an explicit failure, not a quiet fallback."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)

    with pytest.raises(DataLayerError, match="requires the optional 'pyarrow' package"):
        ParquetAdapter(ParquetAdapterConfig(root_path=str(tmp_path)))


def test_parquet_config_defaults_to_a_parquet_filename():
    assert ParquetAdapterConfig(root_path="x").filename_template == "{symbol}_{interval}s.parquet"


def test_parquet_round_trip_matches_csv(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    base = datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
    stamps = [base + timedelta(seconds=300 * i) for i in range(3)]
    table = pa.table(
        {
            "timestamp": pa.array(stamps, type=pa.timestamp("us", tz="UTC")),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1001.0, 1002.0],
        }
    )
    pq.write_table(table, tmp_path / "NQ_300s.parquet")
    (tmp_path / "NQ_300s.csv").write_text(f"{HEADER}\n{iso_rows(3)}", encoding="utf-8")

    from_parquet = ParquetAdapter(ParquetAdapterConfig(root_path=str(tmp_path))).load_bars(
        "NQ", 300, WIDE_START, WIDE_END
    )
    from_csv = CsvAdapter(CsvAdapterConfig(root_path=str(tmp_path))).load_bars(
        "NQ", 300, WIDE_START, WIDE_END
    )

    assert np.array_equal(from_parquet.ts_ns, from_csv.ts_ns)
    assert np.array_equal(from_parquet.col("close"), from_csv.col("close"))


def test_parquet_bar_open_shift_applies(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    base = datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
    stamps = [base + timedelta(seconds=300 * i) for i in range(2)]
    table = pa.table(
        {
            "timestamp": pa.array(stamps, type=pa.timestamp("us", tz="UTC")),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1001.0],
        }
    )
    pq.write_table(table, tmp_path / "NQ_300s.parquet")

    config = ParquetAdapterConfig(root_path=str(tmp_path), timestamp_is_bar_open=True)
    series = ParquetAdapter(config).load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert series.timestamps()[0] == datetime(2020, 1, 2, 14, 35, tzinfo=UTC)


def test_parquet_missing_file_names_the_expected_path(tmp_path):
    pytest.importorskip("pyarrow")
    adapter = ParquetAdapter(ParquetAdapterConfig(root_path=str(tmp_path)))

    with pytest.raises(DataLayerError) as excinfo:
        adapter.load_bars("NQ", 300, WIDE_START, WIDE_END)

    assert str(tmp_path / "NQ_300s.parquet") in str(excinfo.value)


# --- column map ------------------------------------------------------------


def test_column_map_mapping_covers_required_and_mapped_optional():
    mapping = ColumnMap(vwap="VWAP").mapping()

    assert mapping["timestamp"] == "timestamp"
    assert mapping["vwap"] == "VWAP"
    assert "trades" not in mapping


def test_column_map_rejects_an_unknown_canonical_name():
    with pytest.raises(KeyError):
        ColumnMap().vendor_for("spread")


def test_column_map_is_frozen():
    with pytest.raises(ValidationError):
        ColumnMap().vwap = "VWAP"


def test_series_content_hash_is_sensitive_to_timestamps():
    one = make_bar_series(n=5)
    two = make_bar_series(n=5, start=datetime(2020, 1, 3, 14, 30, tzinfo=UTC))

    assert series_content_hash(one) != series_content_hash(two)
    assert to_ns(one.timestamps()[0]) == int(one.ts_ns[0])
