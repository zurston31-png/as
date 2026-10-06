"""File-backed and in-memory data adapters.

These are the entry points where bytes become a `BarSeries`. Four decisions
in here determine whether every number produced downstream is honest.

**1. Bar-open vs bar-close timestamps are declared, never guessed.**
`BarSeries.ts_ns` is the bar *close* instant, and `MarketView` makes a bar
visible at exactly that instant. Vendors disagree: some stamp a bar by the
minute it opened. Reading an open-stamped file as if it were close-stamped
makes every bar visible `interval_seconds` early, which hands the whole
backtest a five-minute head start on every signal -- a lookahead that shows
up as skill, not as an error. There is deliberately no "detect it" option:
the two conventions are indistinguishable from the file's contents alone
(`09:30` is a plausible open stamp *and* a plausible close stamp), so any
auto-detection would be a coin flip on the single most damaging bug the data
layer can have. `timestamp_is_bar_open` is therefore an explicit config
field, and when it is True the adapter adds `interval_seconds` to convert.

**2. A naive timestamp is interpreted in `input_timezone`, and nowhere else.**
`series.to_ns` rejects naive datetimes precisely so that an exchange-local
timestamp cannot be silently read as UTC. A file is the one place where
naive input is acceptable, because the config *declares* which zone the
vendor wrote it in. Integer epoch timestamps are absolute instants and the
zone is not applied to them. A local time that does not exist in the
declared zone (the spring-forward gap) is rejected: it means the declared
zone is not the zone the file was written in.

**3. Nothing is filled, sorted, substituted, or coerced.**
Duplicate or out-of-order timestamps raise `SchemaError` naming the offending
row; the adapter does not sort, because a file whose rows are interleaved or
duplicated is an upstream bug and sorting it would hide that. A blank or
unparseable numeric raises instead of becoming NaN, because a NaN that
reaches a feature is indistinguishable from a measurement. Gaps are neither
filled nor interpolated -- the returned series contains exactly the rows the
file holds, and counting gaps belongs to the cleaning layer. A bars-only
file yields a bars-only adapter: `load_quotes`/`load_tick_aggregates`/
`load_options` return None rather than a derived stand-in.

**4. `available_feeds` reports the disk, not the class.**
It globs for files that actually exist, so a missing file is reported as an
absent feed before a run starts, rather than as a load failure halfway in.

Row numbering in error messages: CSV errors name the `line` as a text editor
numbers it (the header is line 1, so the first data row is line 2). Parquet
errors name the 1-based data `row`, since a Parquet file has no header line.

Date ranges are half-open `[start, end)` and are evaluated against the
*canonical close* timestamp, in `input_timezone` (the calendar the vendor's
file is expressed in; with the default UTC that is the UTC date). Ranges here
mean visibility ranges, which is what the point-in-time layer above consumes,
so a bar stamped by its open on the last day of the range can legitimately
close outside it.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from pydantic import Field

from flow_model.core.determinism import hash_file, stable_hash
from flow_model.core.model import FrozenModel
from flow_model.data.base import (
    DataLayerError,
    DataSourceAdapter,
    DatasetFingerprint,
    Feed,
    SchemaError,
)
from flow_model.data.series import (
    NS_PER_SECOND,
    BarSeries,
    ColumnSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    from_ns,
    to_ns,
)
from flow_model.utils.logging import get_logger

logger = get_logger("data.adapters")

UTC = timezone.utc

#: Nanoseconds per unit of an integer epoch timestamp.
EPOCH_UNITS: dict[str, int] = {"s": NS_PER_SECOND, "ms": 1_000_000, "us": 1_000, "ns": 1}

#: Upper magnitude bound (exclusive) that identifies each epoch unit. A
#: second-epoch stays below 1e11 until the year 5138, a millisecond-epoch
#: below 1e14, a microsecond-epoch below 1e17; anything larger is
#: nanoseconds. The bands are wide enough that no plausible market timestamp
#: lands near a boundary.
_EPOCH_MAGNITUDE_BANDS: tuple[tuple[float, str], ...] = (
    (1e11, "s"),
    (1e14, "ms"),
    (1e17, "us"),
)

_INT64_MAX = 2**63 - 1
_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")
_UNIT_PATTERN = "^(auto|s|ms|us|ns|iso)$"
_DETECTION_SAMPLE = 256

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: tuple[str, ...] = ("trades", "vwap")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class ColumnMap(FrozenModel):
    """Maps vendor column names onto our canonical names.

    A mapped optional column is a promise that the column exists and parses:
    naming `vwap` and then finding it absent or blank raises, rather than
    producing a column of NaN that a feature would later read as a price.
    """

    timestamp: str = "timestamp"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    trades: str | None = None
    vwap: str | None = None

    def vendor_for(self, canonical: str) -> str | None:
        if canonical not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            raise KeyError(f"{canonical!r} is not a canonical column name")
        return getattr(self, canonical)

    def mapping(self) -> dict[str, str]:
        """Canonical -> vendor for every column this map expects to find."""
        pairs = [(name, getattr(self, name)) for name in REQUIRED_COLUMNS]
        pairs += [
            (name, getattr(self, name))
            for name in OPTIONAL_COLUMNS
            if getattr(self, name) is not None
        ]
        return dict(pairs)


class CsvAdapterConfig(FrozenModel):
    """Everything needed to read one vendor's CSV layout.

    `timestamp_is_bar_open` has no "guess" value on purpose -- see the module
    docstring. It is the one field here whose default being wrong produces a
    backtest that looks better rather than broken.
    """

    root_path: str
    column_map: ColumnMap = ColumnMap()
    input_timezone: str = "UTC"
    timestamp_is_bar_open: bool = Field(
        description=(
            "REQUIRED, with no default. Our BarSeries contract is bar CLOSE. "
            "Most OHLCV exports stamp bars by the interval START, and reading "
            "one of those as a close makes every bar visible exactly one "
            "interval early -- a silent head start on every signal that looks "
            "like skill. A default of False would hand that reading to any "
            "config that simply omits the field, so there is no default: a "
            "config that does not declare its convention fails validation "
            "instead of guessing."
        )
    )
    timestamp_unit: str = Field(default="auto", pattern=_UNIT_PATTERN)
    filename_template: str = "{symbol}_{interval}s.csv"


class ParquetAdapterConfig(CsvAdapterConfig):
    """`CsvAdapterConfig` with a Parquet filename default.

    Separate only so the extension default is right; every other field means
    exactly what it means for CSV.
    """

    filename_template: str = "{symbol}_{interval}s.parquet"


# ---------------------------------------------------------------------------
# timezone / value parsing
# ---------------------------------------------------------------------------


def resolve_timezone(name: str) -> ZoneInfo:
    """IANA zone for `name`, or a `DataLayerError` naming what was asked for.

    Resolved at adapter construction rather than at first read: a typo in a
    zone name should fail before a sweep starts, not after it has loaded four
    symbols.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ValueError, OSError) as exc:
        raise DataLayerError(
            f"unknown input_timezone {name!r} ({exc}); expected an IANA key such as "
            "'UTC', 'America/New_York' or 'Europe/London'"
        ) from None


def _is_blank(value: object) -> bool:
    """True for an absent cell: a null, or text that is empty once stripped."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


def _localize(naive: datetime, zone: ZoneInfo, where: str) -> datetime:
    """Attach `zone` to a naive wall-clock timestamp.

    A local time inside the spring-forward gap does not exist, so reading one
    means the file was not written in the declared zone. Accepting it would
    silently shift that row by an hour.
    """
    local = naive.replace(tzinfo=zone)
    if local.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != naive:
        raise SchemaError(
            f"{where}: local time {naive.isoformat()} does not exist in "
            f"{getattr(zone, 'key', zone)!s} (it falls in the daylight-saving gap), so "
            "input_timezone is not the zone this file was written in"
        )
    return local


def _scale_epoch(value: int, multiplier: int, where: str, unit: str) -> int:
    if abs(value) > _INT64_MAX // multiplier:
        raise SchemaError(
            f"{where}: epoch timestamp {value} is out of range when read as {unit!r}; "
            "check timestamp_unit"
        )
    return value * multiplier


def _detect_timestamp_unit(values: Sequence[object], where: str) -> str:
    """Decide between ISO-8601 text and an integer epoch, and the epoch unit.

    Decided from the largest magnitude in a sample rather than from the first
    row, because a single zero or sentinel first row would otherwise pick the
    wrong unit for the whole file.
    """
    magnitude = 0
    saw_number = False
    for value in values[:_DETECTION_SAMPLE]:
        if _is_blank(value):
            continue
        if isinstance(value, (datetime, date)):
            return "native"
        if isinstance(value, str):
            text = value.strip()
            if not _INTEGER_TEXT.match(text):
                return "iso"
            saw_number = True
            magnitude = max(magnitude, abs(int(text)))
            continue
        if isinstance(value, bool):
            break
        if isinstance(value, (int, float, np.integer, np.floating)):
            saw_number = True
            magnitude = max(magnitude, int(abs(value)))
            continue
        break
    if not saw_number:
        raise SchemaError(
            f"{where}: cannot tell what the timestamp column holds; set "
            "timestamp_unit explicitly to one of s | ms | us | ns | iso"
        )
    for bound, unit in _EPOCH_MAGNITUDE_BANDS:
        if magnitude < bound:
            return unit
    return "ns"


def _timestamp_to_ns(value: object, unit: str, zone: ZoneInfo, where: str) -> int:
    """One raw timestamp cell to UTC nanoseconds."""
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else _localize(value, zone, where)
        return to_ns(aware)
    if isinstance(value, date):
        return to_ns(_localize(datetime(value.year, value.month, value.day), zone, where))
    if _is_blank(value):
        raise SchemaError(f"{where}: timestamp is empty")

    if isinstance(value, str):
        text = value.strip()
        if unit == "iso":
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                raise SchemaError(
                    f"{where}: timestamp {value!r} is not ISO-8601 "
                    "(expected e.g. '2020-01-02T09:30:00' or '2020-01-02 09:30:00-05:00')"
                ) from None
            aware = parsed if parsed.tzinfo is not None else _localize(parsed, zone, where)
            return to_ns(aware)
        try:
            number: float | int = int(text) if _INTEGER_TEXT.match(text) else float(text)
        except ValueError:
            raise SchemaError(
                f"{where}: timestamp {value!r} is not a {unit!r} epoch number"
            ) from None
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if unit == "iso":
            raise SchemaError(
                f"{where}: timestamp_unit='iso' but the column holds the number {value!r}"
            )
        number = value.item() if isinstance(value, (np.integer, np.floating)) else value
    else:
        raise SchemaError(f"{where}: unsupported timestamp value {value!r}")

    multiplier = EPOCH_UNITS.get(unit)
    if multiplier is None:
        raise SchemaError(
            f"{where}: the timestamp column mixes value types (found {value!r} after a "
            "native timestamp); set timestamp_unit explicitly"
        )
    if isinstance(number, int):
        return _scale_epoch(number, multiplier, where, unit)
    if not np.isfinite(number):
        raise SchemaError(f"{where}: timestamp {value!r} is not finite")
    return _scale_epoch(int(round(number * multiplier)), 1, where, unit)


def _timestamp_column(
    values: Sequence[object],
    rows: Sequence[int],
    *,
    unit: str,
    zone: ZoneInfo,
    path: str,
    row_word: str,
) -> np.ndarray:
    """The timestamp column as int64 UTC nanoseconds."""
    if not len(values):
        return np.empty(0, dtype=np.int64)
    where = f"{path}: timestamp column"
    resolved = _detect_timestamp_unit(values, where) if unit == "auto" else unit

    if resolved in EPOCH_UNITS:
        fast = _as_int64(values)
        if fast is not None:
            multiplier = EPOCH_UNITS[resolved]
            if fast.size and int(np.max(np.abs(fast))) > _INT64_MAX // multiplier:
                raise SchemaError(
                    f"{path}: an epoch timestamp is out of range when read as "
                    f"{resolved!r}; check timestamp_unit"
                )
            return fast * multiplier

    out = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        out[i] = _timestamp_to_ns(
            value, resolved, zone, f"{path}: {row_word} {rows[i]}, timestamp"
        )
    return out


def _as_int64(values: Sequence[object]) -> np.ndarray | None:
    """Vectorized int64 conversion, or None when the slow path is needed.

    A float column is deliberately refused rather than cast: `np.asarray`
    truncates floats towards zero, which would drop the sub-second part of a
    fractional epoch without a word.
    """
    try:
        probe = np.asarray(values)
    except (TypeError, ValueError):
        return None
    if probe.dtype.kind in "iu":
        return probe.astype(np.int64, copy=False)
    if probe.dtype.kind in "US":
        try:
            return probe.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _float_column(
    values: Sequence[object],
    rows: Sequence[int],
    *,
    canonical: str,
    vendor: str,
    path: str,
    row_word: str,
) -> np.ndarray:
    """One numeric column as float64, or `SchemaError` naming the bad cell.

    The vectorized path is the normal one; it is abandoned the moment anything
    is not a finite number, because NumPy turns a null into NaN and a NaN that
    survives into a feature is indistinguishable from a real measurement.
    """
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        out = None
    if out is None or not bool(np.all(np.isfinite(out))):
        _raise_bad_number(values, rows, canonical=canonical, vendor=vendor,
                          path=path, row_word=row_word)
    return out


def _raise_bad_number(
    values: Sequence[object],
    rows: Sequence[int],
    *,
    canonical: str,
    vendor: str,
    path: str,
    row_word: str,
) -> None:
    """Locate the first unusable cell and raise. Always raises."""
    for i, value in enumerate(values):
        where = f"{path}: {row_word} {rows[i]}, column {vendor!r} (mapped to {canonical!r})"
        if _is_blank(value):
            raise SchemaError(
                f"{where} is empty. A blank numeric is not coerced to NaN, because a "
                "NaN that reaches a feature is indistinguishable from a measurement."
            )
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise SchemaError(f"{where} is not a number: {value!r}") from None
        if not np.isfinite(number):
            raise SchemaError(f"{where} is not finite: {value!r}")
    raise SchemaError(
        f"{path}: column {vendor!r} (mapped to {canonical!r}) could not be read as "
        "float64"
    )


def _check_strictly_ascending(
    ts_ns: np.ndarray, rows: Sequence[int], path: str, row_word: str
) -> None:
    """Reject duplicate or out-of-order timestamps, naming the row.

    The adapter raises here rather than handing the problem to `BarSeries`,
    because only the adapter still knows which line of the file is at fault --
    and an hour of hunting for a cryptic error is the usual cost of not
    saying. Sorting is not an option: an interleaved or doubled export is an
    upstream bug, and sorting it produces a plausible-looking series that no
    one knows is wrong.
    """
    if ts_ns.size < 2:
        return
    diffs = np.diff(ts_ns)
    offenders = np.flatnonzero(diffs <= 0)
    if not offenders.size:
        return
    i = int(offenders[0])
    kind = "duplicate" if diffs[i] == 0 else "out-of-order"
    raise SchemaError(
        f"{path}: {kind} timestamp at {row_word} {rows[i + 1]} -- "
        f"{from_ns(int(ts_ns[i + 1])).isoformat()} does not come after "
        f"{from_ns(int(ts_ns[i])).isoformat()} at {row_word} {rows[i]}. The adapter does "
        "not sort or de-duplicate: an unordered file means the export is wrong, and "
        "silently fixing it would mask interleaved or doubled sessions."
    )


def series_content_hash(series: ColumnSeries) -> str:
    """Digest of a series' actual contents.

    Explicit little-endian casts so the digest is the same on any platform.
    Used where there is no file to hash, so that two different datasets of the
    same shape cannot share one `data_hash` and have their results attributed
    to the same dataset.
    """
    digest = hashlib.sha256()
    digest.update(str(series.symbol).encode("utf-8"))
    digest.update(np.ascontiguousarray(series.ts_ns, dtype="<i8").tobytes())
    for name in series.columns:
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(series.col(name), dtype="<f8").tobytes())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# raw tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RawTable:
    """Unparsed columns plus the row numbers to quote in error messages."""

    path: str
    column_names: tuple[str, ...]
    columns: dict[str, list[object]]
    rows: list[int]
    row_word: str
    blank_rows_dropped: int = 0

    def __len__(self) -> int:
        return len(self.rows)


def _reject_duplicate_headers(names: Sequence[str], path: str) -> None:
    duplicates = sorted({name for name in names if list(names).count(name) > 1})
    if duplicates:
        raise SchemaError(
            f"{path}: duplicate column name(s) {duplicates}; a column_map entry would "
            "be ambiguous about which one it meant"
        )


# ---------------------------------------------------------------------------
# file adapters
# ---------------------------------------------------------------------------


class _FileBarAdapter(DataSourceAdapter):
    """Shared schema, timestamp and range logic for file-backed bar feeds.

    Subclasses supply only `_read_table`, so the parsing rules that decide
    whether the data is honest exist once and are tested once.
    """

    extension = ""

    def __init__(self, config: CsvAdapterConfig) -> None:
        self.config = config
        self.root = Path(config.root_path)
        self.zone = resolve_timezone(config.input_timezone)

    # --- paths ---------------------------------------------------------

    def _render_filename(self, symbol: str, interval: object) -> str:
        try:
            return self.config.filename_template.format(symbol=symbol, interval=interval)
        except (KeyError, IndexError) as exc:
            raise DataLayerError(
                f"filename_template {self.config.filename_template!r} uses an unknown "
                f"placeholder ({exc}); only {{symbol}} and {{interval}} are substituted"
            ) from None

    def bar_path(self, symbol: str, interval_seconds: int) -> Path:
        """Where this adapter expects `symbol`'s bars to be."""
        return self.root / self._render_filename(symbol, int(interval_seconds))

    def bar_files(self, symbol: str) -> tuple[Path, ...]:
        """Bar files present on disk for `symbol`, at any interval."""
        pattern = self._render_filename(symbol, "*")
        return tuple(sorted(self.root.glob(pattern)))

    # --- interface -----------------------------------------------------

    def available_feeds(self, symbol: str) -> frozenset[Feed]:
        """Feeds backed by files that exist right now.

        Only bars: a bar file cannot answer a quote, tick or options
        question, and claiming a feed this adapter cannot read would make the
        store attempt a load that can only come back empty.
        """
        return frozenset({Feed.BARS}) if self.bar_files(symbol) else frozenset()

    def load_bars(
        self, symbol: str, interval_seconds: int, start: date, end: date
    ) -> BarSeries:
        interval = int(interval_seconds)
        if interval <= 0:
            raise DataLayerError(f"interval_seconds must be positive, got {interval_seconds!r}")
        if end < start:
            raise DataLayerError(
                f"{symbol}: requested range {start} -> {end} ends before it starts"
            )

        path = self.bar_path(symbol, interval)
        if not path.is_file():
            raise DataLayerError(
                f"{symbol}: no {self.extension or 'data'} file for {interval}s bars at "
                f"{path}. Expected filename_template "
                f"{self.config.filename_template!r} under root_path "
                f"{self.config.root_path!r}."
            )

        table = self._read_table(path)
        if table.blank_rows_dropped:
            logger.info(
                "%s: dropped %d entirely-empty row(s)", path, table.blank_rows_dropped
            )
        mapping = self._validate_schema(table)

        ts_ns = _timestamp_column(
            table.columns[mapping["timestamp"]],
            table.rows,
            unit=self.config.timestamp_unit,
            zone=self.zone,
            path=table.path,
            row_word=table.row_word,
        )
        if self.config.timestamp_is_bar_open:
            ts_ns = ts_ns + interval * NS_PER_SECOND
        _check_strictly_ascending(ts_ns, table.rows, table.path, table.row_word)

        columns = {
            canonical: _float_column(
                table.columns[vendor],
                table.rows,
                canonical=canonical,
                vendor=vendor,
                path=table.path,
                row_word=table.row_word,
            )
            for canonical, vendor in mapping.items()
            if canonical != "timestamp"
        }

        lo = np.searchsorted(ts_ns, self._midnight_ns(start), side="left")
        hi = np.searchsorted(ts_ns, self._midnight_ns(end), side="left")
        window = slice(int(lo), int(hi))

        return BarSeries(
            symbol=symbol,
            ts_ns=ts_ns[window],
            interval_seconds=interval,
            columns={name: values[window] for name, values in columns.items()},
            meta={
                "source": self.name,
                "path": str(path),
                "timestamp_is_bar_open": bool(self.config.timestamp_is_bar_open),
            },
        )

    def fingerprint(
        self, symbol: str, interval_seconds: int, start: date, end: date, rows: int
    ) -> DatasetFingerprint:
        """Provenance keyed to the file's bytes.

        The base implementation hashes only (symbol, range, rows, source),
        which would give an edited file the same `data_hash` as the original
        and attribute two different results to one dataset. The bar-open flag
        is part of the identity too: the same bytes read under the other
        convention are a different dataset.
        """
        path = self.bar_path(symbol, int(interval_seconds))
        return DatasetFingerprint(
            symbol=symbol,
            source=self.name,
            interval_seconds=int(interval_seconds),
            start=start,
            end=end,
            rows=rows,
            data_hash=stable_hash(
                {
                    "source": self.name,
                    "symbol": symbol,
                    "interval_seconds": int(interval_seconds),
                    "start": start,
                    "end": end,
                    "rows": rows,
                    "file": str(path),
                    "file_hash": hash_file(path) if path.is_file() else "",
                    "timestamp_is_bar_open": bool(self.config.timestamp_is_bar_open),
                    "input_timezone": self.config.input_timezone,
                }
            ),
            feeds=tuple(sorted(self.available_feeds(symbol), key=lambda f: f.value)),
        )

    # --- internals -----------------------------------------------------

    def _midnight_ns(self, day: date) -> int:
        return to_ns(datetime(day.year, day.month, day.day, tzinfo=self.zone))

    def _validate_schema(self, table: _RawTable) -> dict[str, str]:
        mapping = self.config.column_map.mapping()
        present = set(table.column_names)
        missing = [(c, v) for c, v in mapping.items() if v not in present]
        if missing:
            wanted = ", ".join(f"{v!r} (mapped to {c!r})" for c, v in missing)
            raise SchemaError(
                f"{table.path}: missing column(s) {wanted}; columns found in the file: "
                f"{list(table.column_names)}. Set CsvAdapterConfig.column_map to this "
                "vendor's spelling."
            )
        return mapping

    def _read_table(self, path: Path) -> _RawTable:
        raise NotImplementedError


class CsvAdapter(_FileBarAdapter):
    """Reads bars from delimited text files.

    Parsed with the stdlib `csv` module rather than a DataFrame reader so
    that every cell is seen as written: there is no type inference to turn a
    blank into NaN, and `csv.reader.line_num` gives the real file line to
    quote when a cell is unusable.
    """

    name = "csv"
    extension = "csv"

    def _read_table(self, path: Path) -> _RawTable:
        # utf-8-sig: a byte-order mark would otherwise become part of the
        # first column's name and turn a correct column_map into a
        # missing-column error.
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                raise SchemaError(
                    f"{path}: file is empty; expected a header row naming the columns"
                ) from None
            names = tuple(name.strip() for name in header)
            _reject_duplicate_headers(names, str(path))

            columns: dict[str, list[object]] = {name: [] for name in names}
            rows: list[int] = []
            blank = 0
            for record in reader:
                line = reader.line_num
                if not record or all(_is_blank(cell) for cell in record):
                    blank += 1
                    continue
                if len(record) != len(names):
                    raise SchemaError(
                        f"{path}: line {line} has {len(record)} field(s) but the header "
                        f"declares {len(names)}: {list(names)}"
                    )
                for name, cell in zip(names, record):
                    columns[name].append(cell)
                rows.append(line)

        return _RawTable(
            path=str(path),
            column_names=names,
            columns=columns,
            rows=rows,
            row_word="line",
            blank_rows_dropped=blank,
        )


def _require_pyarrow_parquet():
    """The pyarrow.parquet module, or a `DataLayerError` saying what is missing."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DataLayerError(
            "ParquetAdapter requires the optional 'pyarrow' package, which is not "
            f"importable ({exc}). Install pyarrow, or convert the files to CSV and use "
            "CsvAdapter -- the adapter does not quietly read some other file instead."
        ) from None
    return parquet


class ParquetAdapter(_FileBarAdapter):
    """Reads bars from Parquet files, with CsvAdapter's semantics.

    Parquet can carry a real timestamp type, in which case the stored values
    are used as they are (a tz-aware column keeps its instant; a naive one is
    interpreted in `input_timezone`, like naive text). `timestamp_unit` still
    applies when the column holds integers or strings.

    Resolution caveat: a native timestamp column reaches us as Python
    `datetime`, which stores microseconds, so a nanosecond-resolution Parquet
    column loses its last three digits. A feed that genuinely needs
    nanoseconds should be stored as an integer epoch column with
    `timestamp_unit='ns'`, which is carried through exactly.
    """

    name = "parquet"
    extension = "parquet"

    def __init__(self, config: CsvAdapterConfig) -> None:
        super().__init__(config)
        self._parquet = _require_pyarrow_parquet()

    def _read_table(self, path: Path) -> _RawTable:
        table = self._parquet.read_table(path)
        names = tuple(str(name) for name in table.column_names)
        _reject_duplicate_headers(names, str(path))
        raw = {name: table.column(index).to_pylist() for index, name in enumerate(names)}

        total = int(table.num_rows)
        keep = [
            index
            for index in range(total)
            if not all(_is_blank(raw[name][index]) for name in names)
        ]
        if len(keep) == total:
            return _RawTable(
                path=str(path),
                column_names=names,
                columns=raw,
                rows=list(range(1, total + 1)),
                row_word="row",
            )
        return _RawTable(
            path=str(path),
            column_names=names,
            columns={name: [raw[name][i] for i in keep] for name in names},
            rows=[i + 1 for i in keep],
            row_word="row",
            blank_rows_dropped=total - len(keep),
        )


# ---------------------------------------------------------------------------
# in-memory adapter
# ---------------------------------------------------------------------------


class InMemoryAdapter(DataSourceAdapter):
    """Serves pre-built series through the normal adapter interface.

    For tests, and for wiring a generator into the stack without writing
    files. It reports exactly the feeds it was handed: a series it does not
    hold is absent, not approximated from one it does hold.

    Date ranges are evaluated in UTC here, because an in-memory series is
    already canonical UTC and no vendor calendar is involved.
    """

    def __init__(
        self,
        bars: Mapping[tuple[str, int], BarSeries],
        quotes: Mapping[str, QuoteSeries] | None = None,
        ticks: Mapping[tuple[str, int], TickSeries] | None = None,
        options: Mapping[str, OptionsSeries] | None = None,
        name: str = "memory",
    ) -> None:
        self.name = str(name)
        self.bars = {(str(s), int(i)): v for (s, i), v in dict(bars).items()}
        self.ticks = {(str(s), int(i)): v for (s, i), v in dict(ticks or {}).items()}
        self.quotes = {str(s): v for s, v in dict(quotes or {}).items()}
        self.options = {str(s): v for s, v in dict(options or {}).items()}

        for (symbol, interval), series in self.bars.items():
            self._check_symbol(symbol, series, f"bars[{symbol!r}, {interval}]")
            if series.interval_seconds != interval:
                raise DataLayerError(
                    f"bars[{symbol!r}, {interval}] holds a series declaring "
                    f"interval_seconds={series.interval_seconds}; the key and the series "
                    "must agree or the backtest clock would step at the wrong rate"
                )
        for (symbol, interval), series in self.ticks.items():
            self._check_symbol(symbol, series, f"ticks[{symbol!r}, {interval}]")
        for symbol, series in self.quotes.items():
            self._check_symbol(symbol, series, f"quotes[{symbol!r}]")
        for symbol, series in self.options.items():
            self._check_symbol(symbol, series, f"options[{symbol!r}]")

    @staticmethod
    def _check_symbol(symbol: str, series: ColumnSeries, where: str) -> None:
        if series.symbol != symbol:
            raise DataLayerError(
                f"{where} holds a series for {series.symbol!r}; a mis-keyed series would "
                "serve one symbol's prices under another symbol's name"
            )

    # --- interface -----------------------------------------------------

    def available_feeds(self, symbol: str) -> frozenset[Feed]:
        feeds: set[Feed] = set()
        if any(key[0] == symbol and len(series) for key, series in self.bars.items()):
            feeds.add(Feed.BARS)
        if any(key[0] == symbol and len(series) for key, series in self.ticks.items()):
            feeds.add(Feed.TICK_AGGREGATE)
        if len(self.quotes.get(symbol) or ()):
            feeds.add(Feed.QUOTES)
        if len(self.options.get(symbol) or ()):
            feeds.add(Feed.OPTIONS_SNAPSHOT)
        return frozenset(feeds)

    def load_bars(
        self, symbol: str, interval_seconds: int, start: date, end: date
    ) -> BarSeries:
        series = self.bars.get((symbol, int(interval_seconds)))
        if series is None:
            raise DataLayerError(
                f"{symbol!r}: no in-memory bars at {int(interval_seconds)}s; held: "
                f"{sorted(self.bars)}"
            )
        return self._slice(series, start, end)

    def load_quotes(self, symbol: str, start: date, end: date) -> QuoteSeries | None:
        series = self.quotes.get(symbol)
        return None if series is None else self._slice(series, start, end)

    def load_tick_aggregates(
        self, symbol: str, interval_seconds: int, start: date, end: date
    ) -> TickSeries | None:
        series = self.ticks.get((symbol, int(interval_seconds)))
        return None if series is None else self._slice(series, start, end)

    def load_options(self, symbol: str, start: date, end: date) -> OptionsSeries | None:
        series = self.options.get(symbol)
        return None if series is None else self._slice(series, start, end)

    def fingerprint(
        self, symbol: str, interval_seconds: int, start: date, end: date, rows: int
    ) -> DatasetFingerprint:
        """Provenance that depends on the series' contents, not just its shape."""
        series = self.bars.get((symbol, int(interval_seconds)))
        return DatasetFingerprint(
            symbol=symbol,
            source=self.name,
            interval_seconds=int(interval_seconds),
            start=start,
            end=end,
            rows=rows,
            data_hash=stable_hash(
                {
                    "source": self.name,
                    "symbol": symbol,
                    "interval_seconds": int(interval_seconds),
                    "start": start,
                    "end": end,
                    "rows": rows,
                    "content": series_content_hash(series) if series is not None else "",
                }
            ),
            feeds=tuple(sorted(self.available_feeds(symbol), key=lambda f: f.value)),
        )

    # --- internals -----------------------------------------------------

    @staticmethod
    def _midnight_ns(day: date) -> int:
        return to_ns(datetime(day.year, day.month, day.day, tzinfo=UTC))

    def _slice(self, series, start: date, end: date):
        if end < start:
            raise DataLayerError(f"requested range {start} -> {end} ends before it starts")
        return series.slice_range(self._midnight_ns(start), self._midnight_ns(end))
