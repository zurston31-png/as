"""Data-layer interfaces.

`Feed` is re-exported from `core.enums` for convenience; it lives there
because `config.schema.FeedRequirement` references it and config must not
import from the data layer.

This module freezes the contract between the data layer and everything
above it. Adapters, cleaners, calendars and quality graders are written
against these types; nothing above the data layer imports an adapter
directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import Field, computed_field

from flow_model.core.enums import DataQuality, Feed, Session
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel


class FeedStatus(FrozenModel):
    """Grade for one feed at one point in time.

    `coverage` is the fraction of expected observations actually present in
    the requested window; `age_seconds` is the staleness of the most recent
    observation relative to the evaluation timestamp. Both drive the
    GOOD/DEGRADED/STALE/MISSING grade, and both are reported so a degraded
    grade can be explained rather than just asserted.
    """

    feed: Feed
    quality: DataQuality
    rows: int = Field(ge=0, default=0)
    coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    age_seconds: float | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    note: str = ""

    @property
    def present(self) -> bool:
        return self.quality is not DataQuality.MISSING


class DatasetFingerprint(FrozenModel):
    """Provenance for a loaded dataset slice.

    `data_hash` goes onto every TradeRecord, so a performance number can be
    tied to the exact bytes it was computed from.
    """

    symbol: str
    source: str
    interval_seconds: int
    start: date
    end: date
    rows: int = Field(ge=0)
    data_hash: str
    feeds: tuple[Feed, ...] = ()


class CleanReport(FrozenModel):
    """What cleaning changed, itemized.

    Cleaning silently is how a dataset acquires properties nobody knows
    about. Every removal is counted, and quarantined bars are retained by
    index so they can be inspected.
    """

    symbol: str
    interval_seconds: int
    rows_in: int = Field(ge=0)
    rows_out: int = Field(ge=0)
    duplicates_dropped: int = Field(ge=0, default=0)
    out_of_order_dropped: int = Field(ge=0, default=0)
    malformed_dropped: int = Field(ge=0, default=0)
    partial_bars_dropped: int = Field(ge=0, default=0)
    zero_volume_flagged: int = Field(ge=0, default=0)
    outliers_quarantined: int = Field(ge=0, default=0)
    gaps_detected: int = Field(ge=0, default=0)
    largest_gap_bars: int = Field(ge=0, default=0)
    quarantined_timestamps: tuple[datetime, ...] = ()
    notes: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    @computed_field  # type: ignore[prop-decorator]
    @property
    def retention(self) -> float:
        return round(self.rows_out / self.rows_in, 6) if self.rows_in else 0.0


@runtime_checkable
class SessionCalendarProtocol(Protocol):
    """Trading-session calendar.

    Implementations must be pure functions of the timestamp and the
    instrument spec -- no network, no mutable state -- so that a backtest
    and a live run classify an identical timestamp identically.
    """

    def is_trading_day(self, day: date, spec: InstrumentSpec) -> bool: ...

    def is_half_day(self, day: date, spec: InstrumentSpec) -> bool: ...

    def session_of(self, ts: datetime, spec: InstrumentSpec) -> Session: ...

    def is_rth(self, ts: datetime, spec: InstrumentSpec) -> bool: ...


class DataSourceAdapter(ABC):
    """Loads raw series for one provider.

    An adapter reports what it *has* (`available_feeds`) and returns only
    what it has. It must never synthesize a feed it lacks: a bar-only
    provider returns None for tick aggregates rather than deriving a fake
    delta from bar volume, because that estimator is known to be bad and a
    silent substitution would be indistinguishable from real data
    downstream.
    """

    name: str = "abstract"

    @abstractmethod
    def available_feeds(self, symbol: str) -> frozenset[Feed]:
        """Feeds this adapter can supply for `symbol`."""

    @abstractmethod
    def load_bars(
        self,
        symbol: str,
        interval_seconds: int,
        start: date,
        end: date,
    ):  # -> BarSeries
        """Bars in [start, end). Must be returned in ascending timestamp order."""

    def load_quotes(self, symbol: str, start: date, end: date):  # -> QuoteSeries | None
        return None

    def load_tick_aggregates(
        self, symbol: str, interval_seconds: int, start: date, end: date
    ):  # -> TickSeries | None
        return None

    def load_options(self, symbol: str, start: date, end: date):  # -> OptionsSeries | None
        return None

    def fingerprint(
        self, symbol: str, interval_seconds: int, start: date, end: date, rows: int
    ) -> DatasetFingerprint:
        from flow_model.core.determinism import hash_dataset

        return DatasetFingerprint(
            symbol=symbol,
            source=self.name,
            interval_seconds=interval_seconds,
            start=start,
            end=end,
            rows=rows,
            data_hash=hash_dataset(symbol, start, end, rows, self.name),
            feeds=tuple(sorted(self.available_feeds(symbol), key=lambda f: f.value)),
        )


class DataLayerError(RuntimeError):
    """Base class for data-layer failures."""


class SchemaError(DataLayerError):
    """Raised when raw input does not match the expected schema."""


class MonotonicityError(DataLayerError):
    """Raised when a series is not in strictly ascending timestamp order.

    Not recoverable by sorting at read time: an out-of-order series means
    the upstream file or query is wrong, and silently sorting it would mask
    duplicated or interleaved sessions.
    """
