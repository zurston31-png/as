"""Read-only HTTP endpoints over the stored market data.

The API only reads from the local database. It never calls the data provider,
and it offers no trading signals, predictions or recommendations.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import __version__
from app.config import Settings, is_valid_symbol, is_valid_timeframe
from app.data.validation import CHECK_DESCRIPTIONS, NEW_YORK
from app.models import (
    Bar,
    CheckStage,
    CheckStatus,
    DataQualityCheck,
    IngestionRun,
    RunStatus,
    utc_now,
)

router = APIRouter()

UNIVERSE_NOTE = (
    "This is the curated Milestone 1 starter universe. It does not represent the "
    "entire US stock market."
)

_DATE_HELP = (
    "use a date (YYYY-MM-DD, interpreted as a New York market date) or a date-time with a "
    "UTC offset such as 2024-01-02T14:30:00Z"
)


# =============================================================================
# Dependencies
# =============================================================================


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[Session, Depends(get_session)]


# =============================================================================
# Response models
# =============================================================================


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    version: str
    time_utc: datetime


class BarOut(BaseModel):
    """One stored bar, exactly as it is in the database."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int
    source: str
    feed: str
    adjustment: str


class BarsResponse(BaseModel):
    symbol: str
    timeframe: str
    #: The filters that were applied, in UTC (null when not given).
    start: datetime | None
    end: datetime | None
    #: Number of stored bars matching the filters.
    total: int
    #: Number of bars in this response.
    count: int
    limit: int
    offset: int
    bars: list[BarOut]


class TimeframeCoverage(BaseModel):
    timeframe: str
    bar_count: int
    first_timestamp: datetime
    last_timestamp: datetime


class SymbolInfo(BaseModel):
    symbol: str
    bar_count: int
    coverage: list[TimeframeCoverage]


class SymbolsResponse(BaseModel):
    count: int
    note: str
    symbols: list[SymbolInfo]


class IngestionRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    started_at: datetime
    finished_at: datetime | None
    status: str
    symbol: str
    timeframe: str
    source: str
    feed: str
    adjustment: str
    currency: str
    requested_start: datetime
    requested_end: datetime
    pages_fetched: int
    bars_fetched: int
    bars_inserted: int
    bars_unchanged: int
    bars_conflicting: int
    bars_rejected: int
    bars_duplicate: int
    missing_sessions: int
    error_message: str | None


class IngestionRunsResponse(BaseModel):
    total: int
    count: int
    limit: int
    offset: int
    runs: list[IngestionRunOut]


class StatusCounts(BaseModel):
    passed: int = 0
    failed: int = 0
    warning: int = 0
    skipped: int = 0


class CheckSummary(StatusCounts):
    check_name: str
    description: str


class ValidationRunSummary(StatusCounts):
    check_run_id: str
    created_at: datetime


class QualityIssue(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    check_run_id: str
    stage: str
    ingestion_run_id: int | None
    check_name: str
    status: str
    symbol: str | None
    timeframe: str | None
    bar_timestamp: datetime | None
    message: str
    details: dict[str, Any] | None


class QualityReport(BaseModel):
    generated_at: datetime
    #: Filters that were applied (null when not given).
    symbol: str | None
    stage: str | None
    bars_stored: int
    #: Every recorded check outcome matching the filters, by status.
    totals: StatusCounts
    checks: list[CheckSummary]
    #: The most recent validation of stored bars (scripts/validate.py).
    latest_stored_validation: ValidationRunSummary | None
    #: Number of failed and warning rows matching the filters.
    issue_count: int
    #: The most recent failed and warning rows, newest first.
    issues: list[QualityIssue]


# =============================================================================
# Input validation helpers
# =============================================================================


def _invalid(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=message)


def _known_symbol(raw: str, settings: Settings) -> str:
    """Normalise a symbol and make sure it belongs to the configured universe."""
    symbol = raw.strip().upper()
    if not is_valid_symbol(symbol):
        raise _invalid(
            f"Invalid symbol {raw!r}: a symbol is 1-10 letters or digits, starting with a "
            "letter (for example AAPL)."
        )
    if symbol not in settings.symbols:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown symbol {symbol!r}. Supported symbols: {', '.join(settings.symbols)}.",
        )
    return symbol


def _timeframe(raw: str | None, settings: Settings) -> str:
    timeframe = (raw or settings.default_timeframe).strip()
    if not is_valid_timeframe(timeframe):
        raise _invalid(
            f"Invalid timeframe {raw!r}. Examples: 1Min, 5Min, 15Min, 1Hour, 1Day, 1Week, 1Month."
        )
    return timeframe


def _time_bound(raw: str | None, *, name: str, end_of_day: bool) -> datetime | None:
    """Parse a ``start`` or ``end`` filter into a UTC datetime.

    A plain date is read as a New York market date: ``start`` begins at
    00:00:00 and ``end`` finishes at 23:59:59.999999 New York time, so the
    range ``start=2024-01-02&end=2024-01-05`` contains those four sessions.
    """
    if raw is None:
        return None
    value = raw.strip()
    if len(value) == 10:
        try:
            day = date.fromisoformat(value)
        except ValueError:
            raise _invalid(f"Invalid {name} {raw!r}: {_DATE_HELP}.") from None
        boundary = time(23, 59, 59, 999999) if end_of_day else time(0, 0, 0)
        return datetime.combine(day, boundary, tzinfo=NEW_YORK).astimezone(UTC)
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise _invalid(f"Invalid {name} {raw!r}: {_DATE_HELP}.") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise _invalid(
            f"Invalid {name} {raw!r}: a date-time needs a UTC offset. Write a trailing Z for "
            "UTC, or URL-encode the plus sign of an offset as %2B."
        )
    return moment.astimezone(UTC)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/", include_in_schema=False)
def index() -> dict[str, Any]:
    """A small signpost for anyone opening the base address in a browser."""
    return {
        "name": "SignalIQ",
        "version": __version__,
        "description": "Validated historical US stock bars (read-only).",
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/symbols",
            "/bars/{symbol}",
            "/ingestion/runs",
            "/quality/report",
        ],
    }


@router.get("/health", summary="Service and database health")
def health(session: SessionDep) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="The database is not reachable.") from None
    return HealthResponse(status="ok", database="ok", version=__version__, time_utc=utc_now())


@router.get("/symbols", summary="The symbol universe and what is stored for each symbol")
def list_symbols(session: SessionDep, settings: SettingsDep) -> SymbolsResponse:
    rows = session.execute(
        select(
            Bar.symbol,
            Bar.timeframe,
            func.count().label("bar_count"),
            func.min(Bar.timestamp).label("first_timestamp"),
            func.max(Bar.timestamp).label("last_timestamp"),
        )
        .where(Bar.symbol.in_(settings.symbols))
        .group_by(Bar.symbol, Bar.timeframe)
        .order_by(Bar.symbol, Bar.timeframe)
    ).all()

    coverage: dict[str, list[TimeframeCoverage]] = {symbol: [] for symbol in settings.symbols}
    for row in rows:
        coverage[row.symbol].append(
            TimeframeCoverage(
                timeframe=row.timeframe,
                bar_count=row.bar_count,
                first_timestamp=row.first_timestamp,
                last_timestamp=row.last_timestamp,
            )
        )
    symbols = [
        SymbolInfo(
            symbol=symbol,
            bar_count=sum(item.bar_count for item in coverage[symbol]),
            coverage=coverage[symbol],
        )
        for symbol in settings.symbols
    ]
    return SymbolsResponse(count=len(symbols), note=UNIVERSE_NOTE, symbols=symbols)


@router.get("/bars/{symbol}", summary="Stored bars for one symbol")
def get_bars(
    symbol: str,
    session: SessionDep,
    settings: SettingsDep,
    timeframe: Annotated[
        str | None, Query(description="Bar timeframe, for example 1Day. Defaults to 1Day.")
    ] = None,
    start: Annotated[
        str | None,
        Query(description="First date or date-time to include (YYYY-MM-DD or RFC 3339)."),
    ] = None,
    end: Annotated[
        str | None,
        Query(description="Last date or date-time to include (YYYY-MM-DD or RFC 3339)."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=10_000, description="Maximum bars to return.")] = 1000,
    offset: Annotated[int, Query(ge=0, description="Bars to skip, for paging.")] = 0,
    order: Annotated[
        Literal["asc", "desc"], Query(description="Sort by timestamp: asc or desc.")
    ] = "asc",
) -> BarsResponse:
    symbol = _known_symbol(symbol, settings)
    timeframe = _timeframe(timeframe, settings)
    start_at = _time_bound(start, name="start", end_of_day=False)
    end_at = _time_bound(end, name="end", end_of_day=True)
    if start_at is not None and end_at is not None and start_at > end_at:
        raise _invalid(f"start ({start}) must not be after end ({end}).")

    conditions = [Bar.symbol == symbol, Bar.timeframe == timeframe]
    if start_at is not None:
        conditions.append(Bar.timestamp >= start_at)
    if end_at is not None:
        conditions.append(Bar.timestamp <= end_at)

    total = session.execute(select(func.count()).select_from(Bar).where(*conditions)).scalar_one()
    ordering = Bar.timestamp.asc() if order == "asc" else Bar.timestamp.desc()
    bars = (
        session.execute(select(Bar).where(*conditions).order_by(ordering).limit(limit).offset(offset))
        .scalars()
        .all()
    )
    return BarsResponse(
        symbol=symbol,
        timeframe=timeframe,
        start=start_at,
        end=end_at,
        total=total,
        count=len(bars),
        limit=limit,
        offset=offset,
        bars=[BarOut.model_validate(bar) for bar in bars],
    )


@router.get("/ingestion/runs", summary="Ingestion history, newest first")
def list_ingestion_runs(
    session: SessionDep,
    settings: SettingsDep,
    symbol: Annotated[str | None, Query(description="Only runs for this symbol.")] = None,
    status: Annotated[
        str | None, Query(description=f"Only runs with this status: {', '.join(RunStatus.ALL)}.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum runs to return.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Runs to skip, for paging.")] = 0,
) -> IngestionRunsResponse:
    conditions = []
    if symbol is not None:
        conditions.append(IngestionRun.symbol == _known_symbol(symbol, settings))
    if status is not None:
        if status not in RunStatus.ALL:
            raise _invalid(f"Invalid status {status!r}. Expected one of: {', '.join(RunStatus.ALL)}.")
        conditions.append(IngestionRun.status == status)

    total = session.execute(
        select(func.count()).select_from(IngestionRun).where(*conditions)
    ).scalar_one()
    runs = (
        session.execute(
            select(IngestionRun)
            .where(*conditions)
            .order_by(IngestionRun.id.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return IngestionRunsResponse(
        total=total,
        count=len(runs),
        limit=limit,
        offset=offset,
        runs=[IngestionRunOut.model_validate(run) for run in runs],
    )


def _status_counts(session: Session, conditions: list[Any]) -> dict[str, int]:
    rows = session.execute(
        select(DataQualityCheck.status, func.count())
        .where(*conditions)
        .group_by(DataQualityCheck.status)
    ).all()
    return {status: count for status, count in rows if status in CheckStatus.ALL}


@router.get("/quality/report", summary="Data-quality check outcomes and open issues")
def quality_report(
    session: SessionDep,
    settings: SettingsDep,
    symbol: Annotated[str | None, Query(description="Only checks for this symbol.")] = None,
    stage: Annotated[
        str | None,
        Query(description=f"Only checks from this stage: {', '.join(CheckStage.ALL)}."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum issues to list.")] = 100,
) -> QualityReport:
    conditions: list[Any] = []
    bar_conditions: list[Any] = []
    if symbol is not None:
        symbol = _known_symbol(symbol, settings)
        conditions.append(DataQualityCheck.symbol == symbol)
        bar_conditions.append(Bar.symbol == symbol)
    if stage is not None:
        if stage not in CheckStage.ALL:
            raise _invalid(f"Invalid stage {stage!r}. Expected one of: {', '.join(CheckStage.ALL)}.")
        conditions.append(DataQualityCheck.stage == stage)

    bars_stored = session.execute(
        select(func.count()).select_from(Bar).where(*bar_conditions)
    ).scalar_one()

    # Outcomes per check and status.
    per_check: dict[str, dict[str, int]] = {}
    for check_name, status, count in session.execute(
        select(DataQualityCheck.check_name, DataQualityCheck.status, func.count())
        .where(*conditions)
        .group_by(DataQualityCheck.check_name, DataQualityCheck.status)
    ):
        if status in CheckStatus.ALL:
            per_check.setdefault(check_name, {})[status] = count
    known_order = {name: position for position, name in enumerate(CHECK_DESCRIPTIONS)}
    checks = [
        CheckSummary(
            check_name=name,
            description=CHECK_DESCRIPTIONS.get(name, name),
            **per_check[name],
        )
        for name in sorted(per_check, key=lambda name: (known_order.get(name, len(known_order)), name))
    ]

    # The latest validation of stored bars shows the current state of the data.
    latest: ValidationRunSummary | None = None
    if stage in (None, CheckStage.STORED):
        newest = session.execute(
            select(DataQualityCheck.check_run_id, DataQualityCheck.created_at)
            .where(DataQualityCheck.stage == CheckStage.STORED)
            .order_by(DataQualityCheck.id.desc())
            .limit(1)
        ).first()
        if newest is not None:
            run_conditions = [DataQualityCheck.check_run_id == newest.check_run_id]
            if symbol is not None:
                run_conditions.append(DataQualityCheck.symbol == symbol)
            latest = ValidationRunSummary(
                check_run_id=newest.check_run_id,
                created_at=newest.created_at,
                **_status_counts(session, run_conditions),
            )

    issue_conditions = [
        *conditions,
        DataQualityCheck.status.in_((CheckStatus.FAILED, CheckStatus.WARNING)),
    ]
    issue_count = session.execute(
        select(func.count()).select_from(DataQualityCheck).where(*issue_conditions)
    ).scalar_one()
    issues = (
        session.execute(
            select(DataQualityCheck)
            .where(*issue_conditions)
            .order_by(DataQualityCheck.id.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )

    return QualityReport(
        generated_at=utc_now(),
        symbol=symbol,
        stage=stage,
        bars_stored=bars_stored,
        totals=StatusCounts(**_status_counts(session, conditions)),
        checks=checks,
        latest_stored_validation=latest,
        issue_count=issue_count,
        issues=[QualityIssue.model_validate(issue) for issue in issues],
    )
