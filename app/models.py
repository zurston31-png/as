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
    # Live 0-100 composite score (app/signals/scoring.py via
    # app/signals/live_gate.py), populated for buy signals when
    # LIVE_SIGNAL_SCORE_ENABLED and live candle data was available - persisted
    # regardless of pass/fail so a passing signal's score stays visible in
    # the journal, same principle as RugCheckResult.rug_risk_score.
    signal_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_score_reliable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    signal_score_factors: Mapped[list] = mapped_column(JSON, default=list)
    # Where this signal came from: "tradingview" (a webhook alert) or
    # "scanner" (auto-discovered by app/scanner/). Both take the identical
    # path through _handle_buy_signal; this only records which one found it.
    source: Mapped[str] = mapped_column(String(32), default="tradingview", index=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    # Market quality 0-100 (app/signals/market_quality.py) - "can this be
    # traded?", distinct from the signal score's "is this a good setup?"
    # and the rug score's "will this rug?". Persisted regardless of
    # pass/fail so a rejection is explainable after the fact.
    market_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_quality_factors: Mapped[list] = mapped_column(JSON, default=list)


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
    # Composite 0-100 Rug Risk Score (app/rugcheck/risk_score.py), attached
    # regardless of pass/fail so risk level stays visible in the trade
    # journal even for a token that passed every binary check.
    rug_risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rug_risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rug_risk_factors: Mapped[list] = mapped_column(JSON, default=list)


class Trade(Base):
    """A single executed (or attempted) buy or sell leg."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # Which Position this leg belongs to - set on both the entry (buy) and
    # every exit (sell) leg, including partial exits. Without this there is
    # no way to join an exit trade back to the entry's signal/rug-check
    # context or the position's close_reason, since a monitor-triggered
    # exit (stop-loss/take-profit/smart-exit) carries no signal_id of its
    # own - only a TradingView-driven sell does.
    position_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
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
    # Why THIS leg closed - set on every sell trade, full or partial. A
    # position can have several partial-exit legs each with a different
    # reason ("partial profit-take" then later "trend reversal" for the
    # rest); Position.close_reason alone only ever captures the LAST one.
    close_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # --- execution-cost accounting (paper fill model) ---
    # What this leg actually cost beyond the mid price, per
    # app/execution/fill_model.py: the fee in dollars, the total adverse
    # move (impact + spread + drift + fee) as a fraction, and the simulated
    # confirmation delay. Without these, "total fees paid" and "average
    # slippage" are unanswerable and the analytics can only show gross P&L.
    fee_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    execution_cost_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fill_delay_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Which strategy configuration produced this trade (StrategyVersion
    # label). Results from materially different configurations must never
    # be silently pooled - a stats table mixing v1 and v3 trades answers a
    # question nobody asked.
    strategy_version: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)


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
    # The other half of the path. Without a low-water mark a post-mortem can
    # only report where a trade ENDED, so a +5% winner that first went -30%
    # reads identically to one that never dipped - and those are not the same
    # trade. Together these give per-position MFE/MAE.
    lowest_price_since_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Pool depth when the position was opened, and the lowest depth seen
    # since. A liquidity pull while holding is a total loss and nothing else
    # in the exit stack is watching for it: price can look fine right up to
    # the moment there is nothing left to sell into.
    liquidity_at_entry_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    lowest_liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
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


class ScannedToken(Base):
    """One row per token the auto-scanner has ever evaluated.

    Serves two jobs at once. First, deduplication: a scan cycle re-surfaces
    the same few hundred newly listed tokens every minute, and without a
    persisted record the bot would re-run the full rug-check + candle-fetch
    pipeline on all of them forever. `last_evaluated_at` plus
    SCANNER_RECHECK_MINUTES is what stops that.

    Second, an audit trail: `last_stage` records how far each candidate got
    before being rejected (prescreen / signal_score / rug_check / traded),
    so "the scanner found 300 tokens and traded none" is answerable rather
    than mysterious - the same reasoning behind RugCheckResult's
    lookup_outcomes.
    """

    __tablename__ = "scanned_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_address: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(64))
    chain: Mapped[str] = mapped_column(String(32), default="solana")
    discovery_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    first_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_evaluated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    evaluation_count: Mapped[int] = mapped_column(Integer, default=0)
    last_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_24h_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    times_traded: Mapped[int] = mapped_column(Integer, default=0)


class StrategyVersion(Base):
    """One row per distinct strategy configuration the bot has ever run.

    Pooling results across materially different configurations answers a
    question nobody asked: if the score threshold moved from 75 to 65
    halfway through, a combined win rate describes a strategy that never
    existed. Every Signal and Trade records the version label active when
    it happened, so analytics can be filtered to one configuration -
    and the dashboard can say plainly when the numbers on screen span more
    than one.

    The label is a short hash of the settings that actually change trading
    behavior (app/strategy/version.py decides which). Changing a cosmetic
    setting - a log level, a dashboard password - does NOT mint a new
    version, because that would fragment history for no analytical gain.
    """

    __tablename__ = "strategy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class ApiHealth(Base):
    """Rolling health of each external data source, for the dashboard's
    system-health panel and for answering "is the bot quiet because the
    market is quiet, or because an API has been down for six hours?".
    """

    __tablename__ = "api_health"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    service: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    last_success_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_failure_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)


class BotState(Base):
    """Small key/value store for runtime state: cash ledger, halt flag, etc."""

    __tablename__ = "bot_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class PipelineEvent(Base):
    """One token's passage through one pipeline stage. APPEND ONLY.

    ScannedToken already tracks each token's CURRENT state, which is what
    the scanner list needs. This is the opposite: an immutable log of what
    happened and when, which is what research needs. Overwriting
    `last_stage` answers "where is this token now?" and destroys "how many
    tokens died at each stage last Tuesday, and why?" - and the second
    question is the one that tells you which filter to change.

    The row carries the SCORE where the stage produced one, including for
    tokens that were then rejected. That is deliberate and it is the whole
    basis of the score-distribution and score-calibration analyses: a
    dataset of only the setups that passed the threshold cannot tell you
    whether the threshold is in the right place. Recording the score of
    every token the engine scored - and only then filtering - is what makes
    the question answerable.

    `detail` holds the stage's inputs and computed values as JSON so a
    decision can be re-read months later without re-fetching data that no
    longer exists.
    """

    __tablename__ = "pipeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    occurred_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    # Canonical identity is the mint; the symbol is carried alongside for
    # display only. See app/identity.py.
    token_address: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(64))
    chain: Mapped[str] = mapped_column(String(32), default="solana")
    stage: Mapped[str] = mapped_column(String(32), index=True)
    passed: Mapped[bool] = mapped_column(Boolean, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The 0-100 score this stage produced, where it produced one. NULL means
    # the stage does not score, never "scored zero".
    score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    strategy_version: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    signal_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)


class ForwardReturn(Base):
    """What a scored candidate actually did afterwards, whether or not it
    was traded.

    This is the answer key for score calibration. A score is only useful if
    higher scores precede better outcomes, and that cannot be measured from
    trades alone: the bot only trades what it already believed in, so the
    trade record is a censored sample by construction. Measuring the
    forward return of REJECTED candidates too is what turns "the score is
    complicated" into "the score does, or does not, predict anything".

    One row per (candidate, horizon). `price_at_horizon` is NULL when the
    horizon has not elapsed yet or the price could not be fetched - never
    zero, and never back-filled with the last known price.
    """

    __tablename__ = "forward_returns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline_event_id: Mapped[int] = mapped_column(Integer, index=True)
    token_address: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(64))
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    price_at_signal: Mapped[float] = mapped_column(Float)
    horizon_minutes: Mapped[int] = mapped_column(Integer, index=True)
    due_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    price_at_horizon: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Why the horizon could not be measured, when it could not be. Kept so a
    # gap in the calibration dataset is explained rather than mysterious.
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    # --- path, not just endpoint ---
    # The close-to-close return hides everything that happened in between,
    # and what happened in between is what a stop would have hit. A +5%
    # horizon return that first went -30% is not a +5% trade; it is a
    # stopped-out loss. MFE/MAE are what make the two distinguishable.
    max_favorable_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_adverse_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Coarse label for grouping. NULL until the horizon resolves.
    outcome: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)
    # The Early Opportunity Score at the moment of the signal, carried here
    # so calibration can group by it without joining back through the
    # pipeline event. NULL for candidates the early engine never scored.
    early_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    late_entry_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_class: Mapped[str | None] = mapped_column(String(24), nullable=True, index=True)
    # The early feature values AS THEY WERE at signal time. Stored rather
    # than recomputed, because the whole point of ablation is to ask what
    # the engine could have known when it decided - re-extracting features
    # later would score the token on candles that had not happened yet.
    # none_as_null so a row with no features is SQL NULL rather than the JSON
    # value `null`. Without it `early_features IS NOT NULL` matches every row
    # and any count of "rows that stored features" is silently the row count.
    early_features: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )


class TokenObservation(Base):
    """One stored market snapshot for a token, at a point in time.

    The Early Signal Engine needs to know whether activity is
    ACCELERATING, and several of those measurements have no other source.
    DexScreener reports transaction counts only over 1h and 24h windows,
    so "transactions per minute, and is that rate rising?" cannot be read
    from a single response at any granularity that matters. It can only be
    computed by differencing successive observations - which means the bot
    has to keep them.

    Candle-derived features (volume and price acceleration, compression,
    VWAP, EMA/RSI/MACD) come from 1m/5m/15m OHLCV instead and do NOT need
    this table. Only the flow features do: transaction rate, buy-pressure
    change and persistence, and liquidity growth.

    Rows are written by the watchlist re-evaluation loop and pruned by age,
    since their whole value is recency.
    """

    __tablename__ = "token_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_address: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(64))
    observed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    price_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_cap_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_5m_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_1h_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_24h_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    buys_1h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sells_1h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    buys_24h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sells_24h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    price_change_5m_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_change_1h_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class WatchlistEntry(Base):
    """A token the Early Signal Engine is tracking, and its state.

    The state machine exists because "promising" and "ready" are different
    facts, and collapsing them forces a bad choice: either trade every
    promising token immediately (chasing) or discard it (missing the move
    entirely). WATCH is the third option - keep looking, and enter only if
    confirmation arrives BEFORE the token becomes overextended.

        DISCOVERED -> WATCH -> CONFIRMED -> PAPER_BUY -> EXIT
        DISCOVERED -> WATCH -> FAILED -> SKIP

    `score_history` is an append-only JSON list of {at, early, technical,
    late_risk, stage} points. Keeping it is what makes it possible to ask
    whether an IMPROVING score predicts better outcomes than a high static
    one - a question that cannot be asked from a single snapshot per token.
    """

    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_address: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(64))
    chain: Mapped[str] = mapped_column(String(32), default="solana")

    state: Mapped[str] = mapped_column(String(16), default="WATCH", index=True)
    stage: Mapped[str | None] = mapped_column(String(16), nullable=True)   # EARLY/DEVELOPING/CONFIRMED/LATE/OVEREXTENDED
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    first_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_evaluated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    evaluations: Mapped[int] = mapped_column(Integer, default=0)

    # Price when the token first entered WATCH. Lead-time analysis measures
    # every later move against this, so it must be the price at DETECTION,
    # not at entry - otherwise a signal that fired early but was acted on
    # late would score as if it had been late.
    price_at_first_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_signal_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    early_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    technical_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    security_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    late_entry_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_class: Mapped[str | None] = mapped_column(String(24), nullable=True)

    best_early_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_history: Mapped[list] = mapped_column(JSON, default=list)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    # Why a WATCH candidate ultimately failed, from the taxonomy in
    # app/analysis/early_calibration.py. NULL while still live.
    failure_category: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
