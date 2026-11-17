"""SQLAlchemy ORM models: ``bars``, ``ingestion_runs`` and ``data_quality_checks``.

All timestamps are stored in UTC through :class:`UTCDateTime`, which refuses
naive datetimes so that an ambiguous time can never reach the database.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

#: Fixed-width text format used for every stored timestamp. The trailing ``Z``
#: makes the UTC meaning explicit to any tool that reads the database, and the
#: fixed width keeps text ordering identical to chronological ordering.
UTC_STORAGE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` converted to UTC, rejecting naive datetimes."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Naive datetime rejected: timestamps must be timezone-aware so they can be "
            "stored unambiguously in UTC."
        )
    return value.astimezone(UTC)


def format_utc(value: datetime) -> str:
    """Render an aware datetime in the stored UTC text format."""
    return ensure_utc(value).strftime(UTC_STORAGE_FORMAT)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetime stored as ISO-8601 text ending in ``Z``.

    * Writing a naive datetime raises ``ValueError``.
    * Writing an aware datetime in another zone converts it to UTC.
    * Reading always returns a timezone-aware UTC datetime.
    """

    impl = String(27)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"Expected a datetime, got {type(value).__name__}.")
        return format_utc(value)

    def process_result_value(self, value: str | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError(f"Stored timestamp {value!r} has no UTC designator.")
        return parsed.astimezone(UTC)


class RunStatus:
    """Allowed values of ``ingestion_runs.status``."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    COMPLETED_WITH_ISSUES = "completed_with_issues"
    FAILED = "failed"

    ALL = (RUNNING, SUCCEEDED, COMPLETED_WITH_ISSUES, FAILED)


class CheckStatus:
    """Allowed values of ``data_quality_checks.status``.

    * ``passed``  - the check ran over ``records_checked`` records and found nothing.
    * ``failed``  - one specific data-integrity finding (one row per finding).
    * ``warning`` - needs attention but is not an integrity violation
      (for example the provider's history starting later than requested).
    * ``skipped`` - the check does not apply (for example gap detection for a
      timeframe that is not supported yet).
    """

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

    ALL = (PASSED, FAILED, WARNING, SKIPPED)


class CheckStage:
    """Where a quality check ran."""

    INGESTION = "ingestion"  # on a batch received from the provider
    STORED = "stored"  # on bars already in the database

    ALL = (INGESTION, STORED)


class Base(DeclarativeBase):
    """Declarative base for all SignalIQ tables."""


class Bar(Base):
    """One OHLCV bar. Unique per (symbol, timeframe, timestamp)."""

    __tablename__ = "bars"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(16), primary_key=True)
    #: Start of the bar in UTC. Daily bars from Alpaca start at midnight
    #: America/New_York, i.e. 04:00Z or 05:00Z depending on daylight saving.
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), primary_key=True)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    vwap: Mapped[float] = mapped_column(Float, nullable=False)
    trade_count: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Provenance: where the bar came from and how it was requested.
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    feed: Mapped[str] = mapped_column(String(16), nullable=False)
    adjustment: Mapped[str] = mapped_column(String(16), nullable=False)

    # Lineage: which ingestion run first stored this bar, and when.
    ingestion_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_runs.id"), nullable=True, index=True
    )
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    def __repr__(self) -> str:
        return f"Bar({self.symbol!r}, {self.timeframe!r}, {self.timestamp!r}, close={self.close!r})"


class IngestionRun(Base):
    """History of one download attempt for one symbol and timeframe."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (Index("ix_ingestion_runs_symbol_started", "symbol", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=RunStatus.RUNNING)

    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    feed: Mapped[str] = mapped_column(String(16), nullable=False)
    adjustment: Mapped[str] = mapped_column(String(16), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)

    #: The window actually sent to the provider (UTC, both ends inclusive).
    requested_start: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    requested_end: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)

    # Counters. bars_fetched always equals the sum of the five outcome counters.
    pages_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bars_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bars_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bars_unchanged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bars_conflicting: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bars_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bars_duplicate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Expected trading sessions inside the requested window that are not stored.
    missing_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"IngestionRun(id={self.id!r}, symbol={self.symbol!r}, status={self.status!r})"


class DataQualityCheck(Base):
    """Outcome of a data-quality check.

    A ``passed`` or ``skipped`` row summarises one check over a set of records.
    A ``failed`` or ``warning`` row describes one specific finding, with the
    offending values kept in ``details`` so the problem can be investigated.
    """

    __tablename__ = "data_quality_checks"
    __table_args__ = (Index("ix_data_quality_checks_symbol_check", "symbol", "check_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    #: Groups every row written by the same validation pass.
    check_run_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(16), nullable=False)
    ingestion_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_runs.id"), nullable=True, index=True
    )

    check_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(16), nullable=True)
    bar_timestamp: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    records_checked: Mapped[int | None] = mapped_column(Integer, nullable=True)

    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"DataQualityCheck(id={self.id!r}, check_name={self.check_name!r}, "
            f"status={self.status!r}, symbol={self.symbol!r})"
        )
