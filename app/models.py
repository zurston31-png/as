"""ORM models. Every signal, rug-check result, trade, position, risk event,
and daily P&L snapshot is persisted here for auditing.

Relationships are intentionally modeled as plain foreign-key columns (no
SQLAlchemy `relationship()` wiring) to keep the schema easy to reason about
and query directly for the dashboard / audits.
"""
import datetime as dt
import enum

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class SignalType(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, enum.Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    FAILED = "failed"


class TradeMode(str, enum.Enum):
    PAPER = "paper"
    LIVE = "live"


class PositionStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class Signal(Base):
    """Every alert received from TradingView, whether or not it led to a trade."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    received_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    token_address: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    chain: Mapped[str] = mapped_column(String(32), default="solana")
    signal_type: Mapped[str] = mapped_column(String(16))
    price: Mapped[float] = mapped_column(Float)
    tv_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema9: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema21: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_sma: Mapped[float | None] = mapped_column(Float, nullable=True)
    breakout_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)


class RugCheckResult(Base):
    """Outcome of the pre-trade rug-pull / scam filter for a given signal."""

    __tablename__ = "rug_check_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    checked_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    passed: Mapped[bool] = mapped_column(Boolean)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    ownership_renounced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    mint_disabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    liquidity_locked: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_honeypot: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    top10_holder_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Which scanner the verdict came from, which chain it was screened as,
    # and what every source consulted actually returned. Without these,
    # reviewing a paper trade cannot answer "why did the bot buy this?" —
    # the same "passed" row could come from a rich RugCheck report or a
    # sparse GoPlus one, and a chain misroute is invisible.
    scanner_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chain_screened: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lookup_outcomes: Mapped[list] = mapped_column(JSON, default=list)
    dev_wallet_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class Trade(Base):
    """A single executed (or attempted) buy or sell leg."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    token_address: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chain: Mapped[str] = mapped_column(String(32), default="solana")
    side: Mapped[str] = mapped_column(String(8))                       # buy | sell
    status: Mapped[str] = mapped_column(String(16), default=TradeStatus.PENDING.value)
    mode: Mapped[str] = mapped_column(String(8), default=TradeMode.PAPER.value)
    size_usd: Mapped[float] = mapped_column(Float, default=0.0)
    qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    tx_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    opened_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Position(Base):
    """An open (or recently closed) holding. One row per round-trip trade.

    `qty` shrinks in place when a partial profit-take fires; `initial_qty`
    keeps the original size around so "sell 50% of the position" always
    means 50% of what was actually bought, not 50% of whatever is left
    after an earlier partial exit.
    """

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    token_address: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chain: Mapped[str] = mapped_column(String(32), default="solana")
    qty: Mapped[float] = mapped_column(Float)
    initial_qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default=PositionStatus.OPEN.value, index=True)
    mode: Mapped[str] = mapped_column(String(8), default=TradeMode.PAPER.value)
    dev_wallet_pct_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    close_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- smart-exit tracking (Stage 4) ---
    highest_price_since_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_stop_active: Mapped[bool] = mapped_column(Boolean, default=False)
    break_even_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    partial_exit_taken: Mapped[bool] = mapped_column(Boolean, default=False)
    realized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    # Rolling buffer of [iso_timestamp, price] samples taken on each monitor
    # tick, capped in app/exits/manager.py. This is what momentum-loss and
    # trend-reversal exits read — the bot has no live intrabar OHLCV feed for
    # memecoins wired in yet, so these are built from actual observed prices
    # during the trade rather than invented indicator data.
    recent_prices: Mapped[list] = mapped_column(JSON, default=list)


class RiskEvent(Base):
    """Audit log of every risk-driven decision: rejections, halts, resumes."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    event_type: Mapped[str] = mapped_column(String(64))
    details: Mapped[str] = mapped_column(Text)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class DailyPnL(Base):
    """One row per day, written by the daily summary job."""

    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[str] = mapped_column(String(10), unique=True, index=True)  # YYYY-MM-DD
    realized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    portfolio_value_usd: Mapped[float] = mapped_column(Float, default=0.0)
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BotState(Base):
    """Small key/value store for runtime state: cash ledger, halt flag, etc."""

    __tablename__ = "bot_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
