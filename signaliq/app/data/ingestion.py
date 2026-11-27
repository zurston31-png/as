"""Idempotent ingestion of historical bars from Alpaca into SQLite.

Every download attempt for a symbol is recorded as one ``ingestion_runs`` row.
The steps are:

1. Work out the request window. It never includes a daily candle that is
   still forming.
2. Download every page of bars from the provider.
3. Validate the batch. Invalid bars are rejected and recorded with their
   original payload; they are never repaired or filled in.
4. Insert bars that are new. Bars already stored with identical values are
   skipped, which is what makes repeated imports safe. Bars that differ from
   what is stored are NOT overwritten: the difference is recorded for
   investigation and the stored record is kept.
5. Check the stored data for missing trading sessions inside the window.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import SOURCE_ALPACA, SUPPORTED_INGESTION_TIMEFRAMES, Settings, is_valid_timeframe
from app.data.alpaca_client import (
    AlpacaAPIError,
    AlpacaAuthenticationError,
    AlpacaClient,
    AlpacaNetworkError,
    AlpacaPermissionError,
    AlpacaRateLimitError,
    AlpacaServerError,
)
from app.data.validation import (
    CHECK_CONFLICTING_RECORD,
    CHECK_SESSION_GAP,
    GAP_CHECK_TIMEFRAMES,
    METADATA_FIELDS,
    ROW_CHECKS,
    VALUE_FIELDS,
    Finding,
    analyse_session_gaps,
    build_check_rows,
    last_completed_session,
    new_check_run_id,
    normalize_alpaca_bars,
    session_date,
    session_gap_findings,
    session_window_utc,
    validate_records,
)
from app.database import (
    create_db_engine,
    create_session_factory,
    init_db,
    refuse_protected,
    session_scope,
)
from app.models import Bar, CheckStage, CheckStatus, IngestionRun, RunStatus, utc_now

logger = logging.getLogger(__name__)

#: Errors that would hit every remaining symbol in the same way, so a batch
#: stops instead of repeating the failure (and its retries) for each symbol.
_BATCH_STOPPING_ERRORS = (
    AlpacaAuthenticationError,
    AlpacaPermissionError,
    AlpacaNetworkError,
    AlpacaRateLimitError,
    AlpacaServerError,
)

#: Fields compared to decide whether an incoming bar matches the stored one.
_COMPARED_FIELDS: tuple[str, ...] = (*VALUE_FIELDS, *METADATA_FIELDS)


class IngestionError(ValueError):
    """The ingestion request itself is invalid (nothing was downloaded)."""


@dataclass(frozen=True)
class RequestWindow:
    """The time window sent to the provider."""

    first_session: date
    last_session: date
    start: datetime  # UTC, inclusive
    end: datetime  # UTC, inclusive
    #: True when the requested end was moved back to the last completed session.
    clamped: bool


@dataclass(frozen=True)
class SymbolIngestionResult:
    """What happened when one symbol was ingested."""

    symbol: str
    timeframe: str
    status: str
    run_id: int | None = None
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    pages_fetched: int = 0
    bars_fetched: int = 0
    bars_inserted: int = 0
    bars_unchanged: int = 0
    bars_conflicting: int = 0
    bars_rejected: int = 0
    bars_duplicate: int = 0
    missing_sessions: int = 0
    #: Number of data-quality findings with status ``failed`` / ``warning``.
    failures: int = 0
    warnings: int = 0
    warning_messages: tuple[str, ...] = ()
    error: str | None = None
    #: True when the error would affect every other symbol too.
    stops_batch: bool = False

    @property
    def ok(self) -> bool:
        return self.status != RunStatus.FAILED


@dataclass(frozen=True)
class BatchIngestionResult:
    """Results for a list of symbols."""

    results: tuple[SymbolIngestionResult, ...]
    #: Symbols skipped because an earlier error would have failed them as well.
    not_attempted: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.not_attempted and all(result.ok for result in self.results)


def _differences(stored: Bar, record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the fields whose stored value differs from the incoming value."""
    return {
        name: {"stored": getattr(stored, name), "incoming": record[name]}
        for name in _COMPARED_FIELDS
        if getattr(stored, name) != record[name]
    }


class IngestionService:
    """Downloads, validates and stores bars for the configured symbols."""

    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        client: AlpacaClient,
        *,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._client = client
        self._now = now

    # -- request preparation ------------------------------------------------------

    def resolve_window(self, start: date | None = None, end: date | None = None) -> RequestWindow:
        """Turn optional start/end dates into the window sent to the provider.

        The end never goes beyond the last *completed* trading session, so an
        unfinished daily candle is never requested.
        """
        last_completed = last_completed_session(self._now())
        first = start or self._settings.history_start
        if end is not None and first > end:
            raise IngestionError(
                f"Start date {first.isoformat()} is after end date {end.isoformat()}."
            )
        last = last_completed if end is None else min(end, last_completed)
        if first > last:
            raise IngestionError(
                f"No completed trading session on or after {first.isoformat()}: the last "
                f"completed session is {last_completed.isoformat()}. A daily candle is only "
                "requested once it is final."
            )
        window_start, window_end = session_window_utc(first, last)
        return RequestWindow(
            first_session=first,
            last_session=last,
            start=window_start,
            end=window_end,
            clamped=end is not None and end > last_completed,
        )

    def _normalise_symbol(self, symbol: str) -> str:
        normalised = symbol.strip().upper()
        if normalised not in self._settings.symbols:
            raise IngestionError(
                f"Symbol {symbol!r} is not in the configured universe: "
                f"{', '.join(self._settings.symbols)}."
            )
        return normalised

    def _check_timeframe(self, timeframe: str | None) -> str:
        timeframe = timeframe or self._settings.default_timeframe
        if not is_valid_timeframe(timeframe):
            raise IngestionError(f"Invalid timeframe {timeframe!r}.")
        if timeframe not in SUPPORTED_INGESTION_TIMEFRAMES:
            raise IngestionError(
                f"Timeframe {timeframe!r} is not supported yet. Milestone 1 ingests: "
                f"{', '.join(SUPPORTED_INGESTION_TIMEFRAMES)}."
            )
        return timeframe

    # -- public API ----------------------------------------------------------------

    def ingest_symbol(
        self,
        symbol: str,
        *,
        timeframe: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> SymbolIngestionResult:
        """Download, validate and store bars for one symbol."""
        symbol = self._normalise_symbol(symbol)
        timeframe = self._check_timeframe(timeframe)
        return self._ingest(symbol, timeframe, self.resolve_window(start, end))

    def ingest_symbols(
        self,
        symbols: Sequence[str],
        *,
        timeframe: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> BatchIngestionResult:
        """Ingest several symbols one after another.

        A problem with one symbol does not stop the others, except for errors
        that would fail every symbol (bad credentials, no network, rate limit,
        provider outage). All inputs are checked before anything is downloaded.
        """
        ordered = list(dict.fromkeys(self._normalise_symbol(symbol) for symbol in symbols))
        if not ordered:
            raise IngestionError("No symbols were given.")
        timeframe = self._check_timeframe(timeframe)
        window = self.resolve_window(start, end)

        results: list[SymbolIngestionResult] = []
        for position, symbol in enumerate(ordered):
            result = self._ingest(symbol, timeframe, window)
            results.append(result)
            if result.stops_batch:
                remaining = tuple(ordered[position + 1 :])
                if remaining:
                    logger.error(
                        "Stopping the batch after %s: the same error would affect %s",
                        symbol,
                        ", ".join(remaining),
                    )
                return BatchIngestionResult(results=tuple(results), not_attempted=remaining)
        return BatchIngestionResult(results=tuple(results))

    # -- one symbol ---------------------------------------------------------------

    def _ingest(self, symbol: str, timeframe: str, window: RequestWindow) -> SymbolIngestionResult:
        run_id = self._open_run(symbol, timeframe, window)
        logger.info(
            "Ingestion run %d started: %s %s sessions %s to %s (feed=%s, adjustment=%s)%s",
            run_id,
            symbol,
            timeframe,
            window.first_session.isoformat(),
            window.last_session.isoformat(),
            self._client.feed,
            self._client.adjustment,
            " [end moved back to the last completed session]" if window.clamped else "",
        )

        pages_fetched = 0
        bars_by_symbol: dict[str, list[Any]] = {}
        try:
            for page in self._client.iter_bar_pages(
                [symbol], timeframe=timeframe, start=window.start, end=window.end
            ):
                pages_fetched += 1
                for page_symbol, items in page.bars.items():
                    bars_by_symbol.setdefault(page_symbol, []).extend(items)
        except AlpacaAPIError as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._fail_run(run_id, message, pages_fetched)
            logger.error("Ingestion run %d failed for %s: %s", run_id, symbol, message)
            return SymbolIngestionResult(
                symbol=symbol,
                timeframe=timeframe,
                status=RunStatus.FAILED,
                run_id=run_id,
                requested_start=window.start,
                requested_end=window.end,
                pages_fetched=pages_fetched,
                error=message,
                stops_batch=isinstance(exc, _BATCH_STOPPING_ERRORS),
            )

        try:
            with session_scope(self._session_factory) as session:
                result = self._store(
                    session, run_id, symbol, timeframe, window, bars_by_symbol, pages_fetched
                )
        except Exception as exc:
            # Nothing from this run was stored (the transaction was rolled
            # back). Record the failure, then let the error surface.
            self._fail_run(run_id, f"{type(exc).__name__}: {exc}", pages_fetched)
            raise

        logger.info(
            "Ingestion run %d finished for %s: status=%s fetched=%d inserted=%d unchanged=%d "
            "conflicting=%d rejected=%d duplicate=%d missing_sessions=%d",
            run_id,
            symbol,
            result.status,
            result.bars_fetched,
            result.bars_inserted,
            result.bars_unchanged,
            result.bars_conflicting,
            result.bars_rejected,
            result.bars_duplicate,
            result.missing_sessions,
        )
        return result

    def _open_run(self, symbol: str, timeframe: str, window: RequestWindow) -> int:
        """Record that a run has started (committed straight away)."""
        with session_scope(self._session_factory) as session:
            run = IngestionRun(
                started_at=self._now(),
                status=RunStatus.RUNNING,
                symbol=symbol,
                timeframe=timeframe,
                source=SOURCE_ALPACA,
                feed=self._client.feed,
                adjustment=self._client.adjustment,
                currency=self._client.currency,
                requested_start=window.start,
                requested_end=window.end,
            )
            session.add(run)
            session.flush()
            return run.id

    def _fail_run(self, run_id: int, message: str, pages_fetched: int) -> None:
        with session_scope(self._session_factory) as session:
            run = session.get(IngestionRun, run_id)
            assert run is not None
            run.status = RunStatus.FAILED
            run.error_message = message
            run.pages_fetched = pages_fetched
            run.finished_at = self._now()

    def _store(
        self,
        session: Session,
        run_id: int,
        symbol: str,
        timeframe: str,
        window: RequestWindow,
        bars_by_symbol: dict[str, list[Any]],
        pages_fetched: int,
    ) -> SymbolIngestionResult:
        """Validate the downloaded bars and store them in one transaction."""
        feed, adjustment = self._client.feed, self._client.adjustment
        records = normalize_alpaca_bars(
            bars_by_symbol,
            timeframe=timeframe,
            source=SOURCE_ALPACA,
            feed=feed,
            adjustment=adjustment,
        )
        # Only the requested symbol is acceptable in this response.
        batch = validate_records(
            records, allowed_symbols=(symbol,), expected_feed=feed, expected_adjustment=adjustment
        )
        accepted = batch.accepted_records

        # Compare with what is already stored: insert, skip or record a conflict.
        existing = self._load_existing(session, symbol, timeframe, accepted)
        ingested_at = self._now()
        new_rows: list[dict[str, Any]] = []
        unchanged = 0
        conflicts: list[Finding] = []
        for record in accepted:
            stored = existing.get(record["timestamp"])
            if stored is None:
                row = {name: record[name] for name in ("symbol", "timeframe", "timestamp")}
                row.update({name: record[name] for name in _COMPARED_FIELDS})
                row.update({"ingestion_run_id": run_id, "ingested_at": ingested_at})
                new_rows.append(row)
                continue
            differences = _differences(stored, record)
            if not differences:
                unchanged += 1
                continue
            conflicts.append(
                Finding(
                    check_name=CHECK_CONFLICTING_RECORD,
                    status=CheckStatus.FAILED,
                    message=(
                        "Stored bar differs from the provider's current data in "
                        f"{', '.join(differences)}; the stored record was kept unchanged"
                    ),
                    symbol=symbol,
                    timeframe=timeframe,
                    bar_timestamp=record["timestamp"],
                    details={
                        "differences": differences,
                        "stored_by_ingestion_run_id": stored.ingestion_run_id,
                        "incoming_record": record["raw"],
                    },
                )
            )
        if new_rows:
            session.execute(insert(Bar), new_rows)

        # Missing trading sessions, judged on what is now stored in the window.
        gap_findings: list[Finding] = []
        skipped: dict[str, str] = {}
        counts_by_check = {CHECK_CONFLICTING_RECORD: len(accepted)}
        missing_sessions = 0
        if timeframe in GAP_CHECK_TIMEFRAMES:
            stored_timestamps = session.execute(
                select(Bar.timestamp).where(
                    Bar.symbol == symbol,
                    Bar.timeframe == timeframe,
                    Bar.timestamp >= window.start,
                    Bar.timestamp <= window.end,
                )
            ).scalars()
            analysis = analyse_session_gaps(
                (session_date(timestamp) for timestamp in stored_timestamps),
                window.first_session,
                window.last_session,
            )
            gap_findings = session_gap_findings(analysis, symbol=symbol, timeframe=timeframe)
            missing_sessions = analysis.missing_sessions
            counts_by_check[CHECK_SESSION_GAP] = analysis.expected_sessions
        else:  # pragma: no cover - ingestion only accepts supported timeframes
            skipped[CHECK_SESSION_GAP] = f"Session-gap detection is not implemented for {timeframe}"

        findings = [*batch.findings, *conflicts, *gap_findings]
        session.add_all(
            build_check_rows(
                check_run_id=new_check_run_id(),
                stage=CheckStage.INGESTION,
                ingestion_run_id=run_id,
                symbol=symbol,
                timeframe=timeframe,
                check_names=(*ROW_CHECKS, CHECK_CONFLICTING_RECORD, CHECK_SESSION_GAP),
                records_checked=len(records),
                findings=findings,
                skipped=skipped,
                records_checked_by_check=counts_by_check,
            )
        )

        failures = sum(1 for finding in findings if finding.status == CheckStatus.FAILED)
        warning_messages = tuple(
            finding.message for finding in findings if finding.status == CheckStatus.WARNING
        )
        status = RunStatus.COMPLETED_WITH_ISSUES if failures else RunStatus.SUCCEEDED

        run = session.get(IngestionRun, run_id)
        assert run is not None
        run.status = status
        run.finished_at = self._now()
        run.pages_fetched = pages_fetched
        run.bars_fetched = len(records)
        run.bars_inserted = len(new_rows)
        run.bars_unchanged = unchanged
        run.bars_conflicting = len(conflicts)
        run.bars_rejected = len(batch.rejected_rows)
        run.bars_duplicate = len(batch.duplicate_rows)
        run.missing_sessions = missing_sessions

        return SymbolIngestionResult(
            symbol=symbol,
            timeframe=timeframe,
            status=status,
            run_id=run_id,
            requested_start=window.start,
            requested_end=window.end,
            pages_fetched=pages_fetched,
            bars_fetched=len(records),
            bars_inserted=len(new_rows),
            bars_unchanged=unchanged,
            bars_conflicting=len(conflicts),
            bars_rejected=len(batch.rejected_rows),
            bars_duplicate=len(batch.duplicate_rows),
            missing_sessions=missing_sessions,
            failures=failures,
            warnings=len(warning_messages),
            warning_messages=warning_messages,
        )

    @staticmethod
    def _load_existing(
        session: Session, symbol: str, timeframe: str, accepted: list[dict[str, Any]]
    ) -> dict[datetime, Bar]:
        """Load stored bars that could collide with the accepted records."""
        if not accepted:
            return {}
        timestamps = [record["timestamp"] for record in accepted]
        stored = session.execute(
            select(Bar).where(
                Bar.symbol == symbol,
                Bar.timeframe == timeframe,
                Bar.timestamp >= min(timestamps),
                Bar.timestamp <= max(timestamps),
            )
        ).scalars()
        return {bar.timestamp: bar for bar in stored}


def run_ingestion(
    settings: Settings,
    symbols: Sequence[str],
    *,
    timeframe: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> BatchIngestionResult:
    """Ingest ``symbols`` using the real database and the real Alpaca API.

    Creates the database tables if they do not exist yet. Raises
    ``ConfigurationError`` when the Alpaca credentials are missing and
    ``IngestionError`` when the request itself is invalid.
    """
    refuse_protected(settings.database_url)
    engine = create_db_engine(settings.database_url)
    try:
        init_db(engine)
        with AlpacaClient(settings) as client:
            service = IngestionService(settings, create_session_factory(engine), client)
            return service.ingest_symbols(symbols, timeframe=timeframe, start=start, end=end)
    finally:
        engine.dispose()
