"""Tables of the SIP research database. Separate from the market data database.

``ResearchBase`` has its own metadata, so these tables are never created in, or
migrated with, ``data/signaliq.db``.
"""

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Date, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models import UTCDateTime, utc_now


class ResearchBase(DeclarativeBase):
    """Declarative base of the SIP research database only."""


class BatchSymbol(ResearchBase):
    """A ticker requested for the validation batch, and why it was chosen."""

    __tablename__ = "batch_symbols"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    #: How it was chosen: ``known_case`` (listed by hand, with a reason) or
    #: ``random:<source>:seed=<n>`` (drawn from historical corporate-action events).
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    #: Security type declared in the batch definition (``unknown`` when not declared).
    declared_type: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Added because a corporate action linked it to another batch ticker.
    discovered_from: Mapped[str | None] = mapped_column(String(16), nullable=True)


class CorporateAction(ResearchBase):
    """One Alpaca corporate action, as received, with the fields used here pulled out."""

    __tablename__ = "corporate_actions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    process_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    ex_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    old_symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    new_symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    acquiree_symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    acquirer_symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    old_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)


class TickerBar(ResearchBase):
    """A daily SIP bar exactly as Alpaca returned it for a literal ticker (``asof=-``), raw prices.

    Filler bars (no trade) are kept here as evidence but never used as prices.
    """

    __tablename__ = "ticker_bars"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    session: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    vwap: Mapped[float] = mapped_column(Float, nullable=False)
    trade_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    is_filler: Mapped[bool] = mapped_column(Integer, nullable=False)


class Security(ResearchBase):
    """One security with a permanent internal id, whatever tickers it traded under."""

    __tablename__ = "securities"

    security_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_session: Mapped[date] = mapped_column(Date, nullable=False)
    last_trade_session: Mapped[date] = mapped_column(Date, nullable=False)
    last_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    #: ``trading`` when it still trades at the end of the data, otherwise ``ended``.
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    #: How it ended: none, cash_merger, stock_merger, stock_and_cash_merger,
    #: bankruptcy_otc (renamed to a Q ticker that does not trade on SIP), unexplained.
    exit_type: Mapped[str] = mapped_column(String(32), nullable=False)
    exit_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exit_cash_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    acquirer_symbol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    #: What a holder is assumed to receive at the exit, per share, and why.
    assumed_exit_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_assumption: Mapped[str | None] = mapped_column(Text, nullable=True)
    declared_type: Mapped[str] = mapped_column(String(16), nullable=False)
    real_trade_sessions: Mapped[int] = mapped_column(Integer, nullable=False)
    #: common, etf, warrant, right, unit, preferred, fund, adr, partnership, unknown
    security_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="unknown", server_default="unknown"
    )
    #: confirmed (two sources agree), single_source, conflict, unknown
    type_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="unknown", server_default="unknown"
    )
    #: The resolved outcome for an ended security (see app.universe.outcomes).
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: confirmed, single_source, conflict, unresolved (None while trading)
    outcome_status: Mapped[str | None] = mapped_column(String(16), nullable=True)


class TickerHistory(ResearchBase):
    """Which ticker a security traded under, and when (first and last real trade)."""

    __tablename__ = "ticker_history"

    security_id: Mapped[int] = mapped_column(ForeignKey("securities.security_id"), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    valid_from: Mapped[date] = mapped_column(Date, primary_key=True)
    valid_to: Mapped[date] = mapped_column(Date, nullable=False)
    #: ``first listing`` or the id of the name_change event that led here.
    linked_by: Mapped[str] = mapped_column(String(64), nullable=False)


class SecurityBar(ResearchBase):
    """A real trade of a security: raw prices, and split-adjusted prices two ways."""

    __tablename__ = "security_bars"

    security_id: Mapped[int] = mapped_column(ForeignKey("securities.security_id"), primary_key=True)
    session: Mapped[date] = mapped_column(Date, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    vwap: Mapped[float] = mapped_column(Float, nullable=False)
    trade_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: Product of the ratios of this security's later splits (adjusted = raw / factor).
    split_factor: Mapped[float] = mapped_column(Float, nullable=False)
    #: Product of later spin-off factors (adjusted = raw / (split_factor x distribution_factor)).
    distribution_factor: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0, server_default="1.0"
    )
    #: Alpaca's own split-adjusted close for the same session, for comparison.
    alpaca_adjusted_close: Mapped[float | None] = mapped_column(Float, nullable=True)


class Membership(ResearchBase):
    """One security at one monthly decision: eligible or not, rank, member or not."""

    __tablename__ = "membership"

    decision_session: Mapped[date] = mapped_column(Date, primary_key=True)
    security_id: Mapped[int] = mapped_column(ForeignKey("securities.security_id"), primary_key=True)
    rule_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    eligible: Mapped[bool] = mapped_column(Integer, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(48), nullable=True)
    raw_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_dollar_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    member: Mapped[bool] = mapped_column(Integer, nullable=False)


class ExternalResponse(ResearchBase):
    """A response from an outside reference source, stored as received.

    Reruns read these instead of asking again, so a report can be reproduced
    exactly and the sources' rate limits are respected. ``--refresh`` replaces them.
    """

    __tablename__ = "external_responses"

    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    request_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    response: Mapped[Any] = mapped_column(JSON, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)


class ExitEvidence(ResearchBase):
    """One piece of evidence about how a security stopped trading, from one source."""

    __tablename__ = "exit_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(
        ForeignKey("securities.security_id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    #: acquired, merged, bankruptcy, failure, renamed, delisted_no_reason, resumed, fragment
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    reference: Mapped[str | None] = mapped_column(String(128), nullable=True)


class TypeEvidence(ResearchBase):
    """One piece of evidence about what kind of security it is, from one source."""

    __tablename__ = "type_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(
        ForeignKey("securities.security_id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    security_type: Mapped[str] = mapped_column(String(16), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)


class SplitCheck(ResearchBase):
    """Whether a recorded split agrees with the prices on either side of it."""

    __tablename__ = "split_checks"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    security_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False)
    ratio: Mapped[float] = mapped_column(Float, nullable=False)
    close_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_on: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: consistent, contradicted (not applied), no_prices (applied, unverified)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)


class SpinOff(ResearchBase):
    """A distribution of shares of another company to a security's holders."""

    __tablename__ = "spin_offs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(ForeignKey("securities.security_id"), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # alpaca or curated
    reference: Mapped[str] = mapped_column(Text, nullable=False)
    parent_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False)
    #: JSON list of {"symbol", "ratio", "close_on_ex_date"} for each distributed company.
    children: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    parent_close_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    parent_close_on: Mapped[float | None] = mapped_column(Float, nullable=True)
    distribution_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Raw prices before the ex-date are divided by this to make the series continuous.
    factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Holder's return over the ex-date including the distributed shares.
    holder_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: price_consistent, price_inconsistent, prices_missing
    status: Mapped[str] = mapped_column(String(24), nullable=False)
