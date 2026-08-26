"""Detailed trade analytics: costs, holding time, and per-bucket breakdowns.

The headline numbers (win rate, profit factor, expectancy, drawdown) live
in app/dashboard/analytics.py. This module answers the follow-up questions
that decide what to change next:

    Where did the money actually go?     total fees, total execution cost
    How long is capital tied up?         holding-time distribution
    What kind of setup works?            breakdowns by signal score, market
                                         quality, liquidity, pool age
    What is being rejected, and why?     rejection-reason distribution

The breakdowns exist to answer "which filter should move?" with evidence
instead of intuition. If the 65-75 signal-score bucket loses money and the
85+ bucket makes it, that is an argument for raising the threshold - and
one the bot can only make once it has enough trades in each bucket, which
is why every bucket carries its own sample count and no bucket is
interpreted for the operator.

Two rules run through all of it:

  MISSING IS NOT ZERO. A trade written before execution costs were
  recorded has fee_usd = None, which is "not recorded", not "free". Those
  rows are counted and reported separately rather than being summed as 0,
  because a fee total that quietly includes unrecorded trades understates
  costs - the direction an error must never point.

  NEVER POOL STRATEGY VERSIONS SILENTLY. Every function takes the trades
  it is given, and `split_by_strategy_version` exists so the caller can
  make that split deliberate.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app import models


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def closed_trades(trades: list[models.Trade]) -> list[models.Trade]:
    """Realized trades only, oldest close first."""
    closed = [t for t in trades if t.pnl_usd is not None and t.closed_at is not None]
    return sorted(closed, key=lambda t: t.closed_at)


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    """SQLite hands back naive datetimes; treat them as UTC rather than
    raising when they meet an aware one."""
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def entry_leg_by_position(trades: list[models.Trade]) -> dict[int, models.Trade]:
    """position_id -> the buy leg that opened it.

    Every per-attribute breakdown below asks a question about the ENTRY
    ("did a higher score trade better?") but is computed over the EXIT
    legs, because only an exit carries realized P&L. The entry context -
    signal_id, opened_at - lives on the buy leg and is absent from the
    sell, so the two have to be joined, and position_id is the only thing
    that joins them.
    """
    out: dict[int, models.Trade] = {}
    for t in trades:
        if t.side != "buy" or t.position_id is None:
            continue
        existing = out.get(t.position_id)
        if existing is None:
            out[t.position_id] = t
        elif t.created_at and existing.created_at and t.created_at < existing.created_at:
            out[t.position_id] = t
    return out


def entry_signal_id(
    trade: models.Trade, entries: dict[int, models.Trade]
) -> int | None:
    """The signal that opened this trade's position.

    A sell leg carries a signal_id ONLY when a TradingView alert asked for
    it; every stop-loss, take-profit and smart exit is raised by the
    position monitor and has none (see the note on Trade.position_id in
    app/models.py). Reading `trade.signal_id` on an exit therefore returns
    None for the majority of real trades, and every breakdown keyed on it
    silently reported "not recorded" for the whole book.
    """
    if trade.signal_id is not None:
        return trade.signal_id
    entry = entries.get(trade.position_id)
    return entry.signal_id if entry is not None else None


def holding_time_hours(
    trade: models.Trade, entries: dict[int, models.Trade] | None = None
) -> float | None:
    """How long the position was held.

    `opened_at` is stamped on the BUY leg at fill; an exit leg only ever
    gets `closed_at`. Measuring both off the exit therefore yielded None
    for every trade, which is why the holding-time panel read "-h"
    regardless of how many trades had closed. `entries` supplies the buy
    leg so the span can actually be measured.
    """
    opened, closed = _aware(trade.opened_at), _aware(trade.closed_at)
    if opened is None and entries is not None:
        entry = entries.get(trade.position_id)
        if entry is not None:
            opened = _aware(entry.opened_at)
    if opened is None or closed is None:
        return None
    return (closed - opened).total_seconds() / 3600


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


# ---------------------------------------------------------------------------
# execution costs
# ---------------------------------------------------------------------------

@dataclass
class CostSummary:
    """What trading cost, beyond being wrong about direction."""

    total_fees_usd: float
    total_execution_cost_usd: float      # fees + spread + impact + drift
    total_slippage_usd: float            # execution cost excluding the fee
    avg_execution_cost_pct: float | None
    avg_fill_delay_seconds: float | None
    legs_counted: int
    legs_missing_cost_data: int          # not zero-cost - simply unrecorded

    @property
    def cost_data_complete(self) -> bool:
        return self.legs_missing_cost_data == 0

    @property
    def coverage_pct(self) -> float:
        total = self.legs_counted + self.legs_missing_cost_data
        return (self.legs_counted / total * 100) if total else 0.0


def summarize_costs(trades: list[models.Trade]) -> CostSummary:
    """Total what execution cost across every filled leg, buys and sells.

    Costs are per LEG, not per round trip: a buy and its matching sell each
    pay. Legs with no recorded cost are counted separately instead of being
    treated as free, so the total is never quietly understated.
    """
    fees = 0.0
    execution_cost = 0.0
    cost_pcts: list[float] = []
    delays: list[float] = []
    counted = 0
    missing = 0

    costed_notional = 0.0
    for t in trades:
        if t.status != models.TradeStatus.FILLED.value:
            continue
        # A leg is "costed" only if it has an execution cost rate. A fee
        # alone is not cost coverage: the fee is the one component already
        # known from configuration, while spread, price impact and
        # confirmation drift - the parts worth measuring - are exactly
        # what execution_cost_pct carries. Counting a fee-only leg as
        # covered inflated coverage_pct and let cost_data_complete report
        # True while total_execution_cost_usd silently omitted that leg's
        # unknown slippage. Same failure the comment below describes for a
        # missing notional. CLAUDE.md: unmeasurable is never zero.
        #
        # Its fee is still summed - that part WAS measured - but it counts
        # as a leg missing cost data.
        if t.execution_cost_pct is None:
            if t.fee_usd is not None:
                fees += t.fee_usd
            missing += 1
            continue

        # execution_cost_pct is a FRACTION, so turning it into dollars needs
        # the notional. A leg with a cost percentage but no recorded size
        # would otherwise contribute exactly $0 to the total while still
        # counting as covered - understating costs and reporting 100%
        # coverage while doing it, which is the precise failure this module
        # exists to avoid. Such a leg is counted as unmeasured instead.
        notional = t.size_usd or ((t.qty or 0.0) * (t.exit_price or t.entry_price or 0.0))
        if not notional:
            missing += 1
            continue

        counted += 1
        if t.fee_usd is not None:
            fees += t.fee_usd
        cost_pcts.append(t.execution_cost_pct)
        execution_cost += t.execution_cost_pct * notional
        costed_notional += notional
        if t.fill_delay_seconds is not None:
            delays.append(t.fill_delay_seconds)

    return CostSummary(
        total_fees_usd=fees,
        total_execution_cost_usd=execution_cost,
        # Slippage is everything that isn't the protocol fee: spread, price
        # impact and drift during confirmation. Clamped at zero because a
        # favourable drift can make one leg's cost negative, and reporting
        # "negative slippage" as a portfolio total invites misreading.
        total_slippage_usd=max(execution_cost - fees, 0.0),
        # Weighted by notional, not a flat mean over legs. The dollar
        # totals above are already correct; this is the rate that goes
        # with them, and dividing the same numerator by the same
        # denominator is what makes the two agree. A per-leg mean lets a
        # $10 leg outvote a $10,000 one, so the headline rate could
        # contradict the dollar figure printed beside it.
        avg_execution_cost_pct=(
            execution_cost / costed_notional if costed_notional else None
        ),
        avg_fill_delay_seconds=(sum(delays) / len(delays)) if delays else None,
        legs_counted=counted,
        legs_missing_cost_data=missing,
    )


# ---------------------------------------------------------------------------
# holding time
# ---------------------------------------------------------------------------

@dataclass
class HoldingTimeSummary:
    avg_hours: float | None
    median_hours: float | None
    shortest_hours: float | None
    longest_hours: float | None
    avg_winner_hours: float | None
    avg_loser_hours: float | None
    trades_counted: int

    @property
    def winners_held_longer(self) -> bool | None:
        """Cutting winners short while riding losers is the classic failure
        mode, and it is invisible in a win rate. None when unknowable."""
        if self.avg_winner_hours is None or self.avg_loser_hours is None:
            return None
        return self.avg_winner_hours > self.avg_loser_hours


def summarize_holding_time(trades: list[models.Trade]) -> HoldingTimeSummary:
    durations: list[float] = []
    winners: list[float] = []
    losers: list[float] = []

    for t in closed_trades(trades):
        hours = holding_time_hours(t)
        if hours is None:
            continue
        durations.append(hours)
        (winners if (t.pnl_usd or 0) > 0 else losers).append(hours)

    return HoldingTimeSummary(
        avg_hours=(sum(durations) / len(durations)) if durations else None,
        median_hours=_median(durations),
        shortest_hours=min(durations) if durations else None,
        longest_hours=max(durations) if durations else None,
        avg_winner_hours=(sum(winners) / len(winners)) if winners else None,
        avg_loser_hours=(sum(losers) / len(losers)) if losers else None,
        trades_counted=len(durations),
    )


# ---------------------------------------------------------------------------
# extremes
# ---------------------------------------------------------------------------

@dataclass
class Extremes:
    largest_win_usd: float | None
    largest_win_symbol: str | None
    largest_loss_usd: float | None
    largest_loss_symbol: str | None
    largest_win_pct: float | None
    largest_loss_pct: float | None
    best_trade_share_of_profit: float | None

    @property
    def profit_depends_on_one_trade(self) -> bool:
        """True when a single trade produced most of the gross profit.

        A strategy whose entire edge is one lucky trade has not been shown
        to work; it has been shown to have got lucky once. This is the
        cheapest available check for that, and it is why the flag is
        surfaced next to the headline P&L rather than buried.
        """
        return self.best_trade_share_of_profit is not None and self.best_trade_share_of_profit > 0.5


def find_extremes(trades: list[models.Trade]) -> Extremes:
    closed = closed_trades(trades)
    if not closed:
        return Extremes(None, None, None, None, None, None, None)

    best = max(closed, key=lambda t: t.pnl_usd or 0.0)
    worst = min(closed, key=lambda t: t.pnl_usd or 0.0)
    gross_profit = sum(t.pnl_usd for t in closed if (t.pnl_usd or 0) > 0)

    share = None
    if gross_profit > 0 and (best.pnl_usd or 0) > 0:
        share = (best.pnl_usd or 0.0) / gross_profit

    return Extremes(
        largest_win_usd=best.pnl_usd if (best.pnl_usd or 0) > 0 else None,
        largest_win_symbol=best.symbol if (best.pnl_usd or 0) > 0 else None,
        largest_loss_usd=worst.pnl_usd if (worst.pnl_usd or 0) < 0 else None,
        largest_loss_symbol=worst.symbol if (worst.pnl_usd or 0) < 0 else None,
        largest_win_pct=best.pnl_pct if (best.pnl_usd or 0) > 0 else None,
        largest_loss_pct=worst.pnl_pct if (worst.pnl_usd or 0) < 0 else None,
        best_trade_share_of_profit=share,
    )


# ---------------------------------------------------------------------------
# bucketed breakdowns
# ---------------------------------------------------------------------------

# A bucket with fewer trades than this says nothing. It is still shown -
# hiding it would make the table look more decisive than the data is - but
# it is flagged, and no automated decision may use it.
MIN_TRADES_FOR_A_MEANINGFUL_BUCKET = 20


@dataclass
class Bucket:
    label: str
    trade_count: int
    win_count: int
    total_pnl_usd: float
    expectancy_usd: float

    @property
    def win_rate(self) -> float:
        return (self.win_count / self.trade_count * 100) if self.trade_count else 0.0

    @property
    def meaningful(self) -> bool:
        return self.trade_count >= MIN_TRADES_FOR_A_MEANINGFUL_BUCKET

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "win_rate": round(self.win_rate, 1),
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "expectancy_usd": round(self.expectancy_usd, 2),
            "meaningful": self.meaningful,
        }


@dataclass
class Breakdown:
    dimension: str
    buckets: list[Bucket] = field(default_factory=list)
    unknown_count: int = 0        # trades whose bucket value wasn't recorded

    @property
    def has_any_meaningful_bucket(self) -> bool:
        return any(b.meaningful for b in self.buckets)

    def as_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "unknown_count": self.unknown_count,
            "buckets": [b.as_dict() for b in self.buckets],
            "has_any_meaningful_bucket": self.has_any_meaningful_bucket,
        }


def _bucket_label(value: float, edges: list[float], unit: str = "") -> str:
    """Label for the half-open interval `value` falls in."""
    for i, edge in enumerate(edges):
        if value < edge:
            low = edges[i - 1] if i else None
            return f"<{edge:g}{unit}" if low is None else f"{low:g}-{edge:g}{unit}"
    return f"{edges[-1]:g}{unit}+"


def _build_breakdown(
    dimension: str,
    pairs: list[tuple[float | None, models.Trade]],
    edges: list[float],
    unit: str = "",
) -> Breakdown:
    grouped: dict[str, list[models.Trade]] = defaultdict(list)
    unknown = 0
    for value, trade in pairs:
        if value is None:
            unknown += 1
            continue
        grouped[_bucket_label(value, edges, unit)].append(trade)

    def sort_key(label: str) -> tuple[float, int]:
        """Order buckets by their lower bound, with "<X" ahead of "X-Y".

        Both parse to the same number - "<1h" and "1-4h" each yield 1 - so
        without the tiebreak the open-ended first bucket lands in the
        middle of the table and the column reads as unsorted.
        """
        digits = label.lstrip("<").rstrip("+").split("-")[0].rstrip(unit or " ")
        try:
            value = float(digits)
        except ValueError:
            return (float("inf"), 0)
        return (value, 0 if label.startswith("<") else 1)

    buckets = []
    for label in sorted(grouped, key=sort_key):
        rows = grouped[label]
        total = sum(t.pnl_usd or 0.0 for t in rows)
        buckets.append(
            Bucket(
                label=label,
                trade_count=len(rows),
                win_count=sum(1 for t in rows if (t.pnl_usd or 0) > 0),
                total_pnl_usd=total,
                expectancy_usd=total / len(rows),
            )
        )
    return Breakdown(dimension=dimension, buckets=buckets, unknown_count=unknown)


def breakdown_by_signal_score(
    trades: list[models.Trade], signals: dict[int, models.Signal]
) -> Breakdown:
    """Does a higher entry score actually produce better trades?

    The single most useful question for tuning MIN_SIGNAL_SCORE_TO_ENTER,
    and the one that must be answered from the record rather than assumed.
    """
    entries = entry_leg_by_position(trades)
    pairs = [
        (getattr(signals.get(entry_signal_id(t, entries)), "signal_score", None), t)
        for t in closed_trades(trades)
    ]
    return _build_breakdown("signal score", pairs, [50, 60, 70, 80, 90])


def breakdown_by_market_quality(
    trades: list[models.Trade], signals: dict[int, models.Signal]
) -> Breakdown:
    entries = entry_leg_by_position(trades)
    pairs = [
        (getattr(signals.get(entry_signal_id(t, entries)), "market_quality_score", None), t)
        for t in closed_trades(trades)
    ]
    return _build_breakdown("market quality", pairs, [40, 55, 70, 85])


def breakdown_by_liquidity(
    trades: list[models.Trade], liquidity_by_signal: dict[int, float | None]
) -> Breakdown:
    entries = entry_leg_by_position(trades)
    pairs = [
        (liquidity_by_signal.get(entry_signal_id(t, entries)), t)
        for t in closed_trades(trades)
    ]
    return _build_breakdown("entry liquidity USD", pairs, [25_000, 50_000, 100_000, 250_000])


def breakdown_by_token_age(
    trades: list[models.Trade], age_hours_by_signal: dict[int, float | None]
) -> Breakdown:
    """Does the bot do better on brand-new pools or established ones?

    The edges bracket the windows that actually differ in character: the
    first hour is launch chaos, the first day is where most rugs happen,
    and past a week a memecoin has either found holders or died.
    """
    entries = entry_leg_by_position(trades)
    pairs = [
        (age_hours_by_signal.get(entry_signal_id(t, entries)), t)
        for t in closed_trades(trades)
    ]
    return _build_breakdown("token age", pairs, [1, 6, 24, 168], unit="h")


def breakdown_by_market_cap(
    trades: list[models.Trade], mcap_by_signal: dict[int, float | None]
) -> Breakdown:
    entries = entry_leg_by_position(trades)
    pairs = [
        (mcap_by_signal.get(entry_signal_id(t, entries)), t)
        for t in closed_trades(trades)
    ]
    return _build_breakdown("market cap USD", pairs, [100_000, 500_000, 2_000_000, 10_000_000])


def breakdown_by_rug_score(
    trades: list[models.Trade], rug_by_signal: dict[int, float | None]
) -> Breakdown:
    """Do the trades that cleared security by a wide margin do better?

    A binary pass/fail hides this entirely. If the 60-79 bucket performs as
    well as the 0-19 one, the rug score is not carrying information about
    outcomes and its threshold is doing all the work.
    """
    entries = entry_leg_by_position(trades)
    pairs = [
        (rug_by_signal.get(entry_signal_id(t, entries)), t)
        for t in closed_trades(trades)
    ]
    return _build_breakdown("rug risk score", pairs, [20, 40, 60, 80])


def breakdown_by_holding_time(trades: list[models.Trade]) -> Breakdown:
    entries = entry_leg_by_position(trades)
    pairs = [(holding_time_hours(t, entries), t) for t in closed_trades(trades)]
    return _build_breakdown("holding time", pairs, [1, 4, 12, 24, 72], unit="h")


def breakdown_by_exit_reason(trades: list[models.Trade]) -> Breakdown:
    """Which exit mechanism is actually earning its place.

    Not a numeric range, so it builds its buckets directly - a trailing
    stop that only ever fires below the entry is worth knowing about.
    """
    grouped: dict[str, list[models.Trade]] = defaultdict(list)
    unknown = 0
    for t in closed_trades(trades):
        if not t.close_reason:
            unknown += 1
            continue
        grouped[t.close_reason.strip()].append(t)

    buckets = []
    for label, rows in grouped.items():
        total = sum(r.pnl_usd or 0.0 for r in rows)
        buckets.append(
            Bucket(
                label=label,
                trade_count=len(rows),
                win_count=sum(1 for r in rows if (r.pnl_usd or 0) > 0),
                total_pnl_usd=total,
                expectancy_usd=total / len(rows),
            )
        )
    buckets.sort(key=lambda b: b.trade_count, reverse=True)
    return Breakdown(dimension="exit reason", buckets=buckets, unknown_count=unknown)


# ---------------------------------------------------------------------------
# rejections
# ---------------------------------------------------------------------------

@dataclass
class RejectionSummary:
    total: int
    by_reason: list[tuple[str, int]]      # most common first

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "by_reason": [
                {"reason": r, "count": c, "share_pct": round(c / self.total * 100, 1) if self.total else 0.0}
                for r, c in self.by_reason
            ],
        }


def summarize_rejections(events: list[models.RiskEvent]) -> RejectionSummary:
    """Where candidates die in the pipeline.

    Reported by event_type rather than by the free-text detail, because the
    detail embeds symbols and numbers and would produce one "category" per
    rejection. Reading this backwards is the main risk it carries: a filter
    rejecting a lot is doing its job, and the correct response to "too few
    trades" is better candidates, never a lower filter.
    """
    counts = Counter(e.event_type for e in events if e.event_type.endswith(("_rejected", "_blocked", "_unavailable")))
    return RejectionSummary(total=sum(counts.values()), by_reason=counts.most_common())


# ---------------------------------------------------------------------------
# strategy versions
# ---------------------------------------------------------------------------

def split_by_strategy_version(trades: list[models.Trade]) -> dict[str, list[models.Trade]]:
    """Group trades by the configuration that produced them.

    Pooling versions is the default failure mode of every trading journal:
    a combined win rate across a threshold change describes a strategy that
    never existed. Trades written before versioning are grouped under
    "unversioned" rather than being folded into the current label.
    """
    grouped: dict[str, list[models.Trade]] = defaultdict(list)
    for t in trades:
        grouped[t.strategy_version or "unversioned"].append(t)
    return dict(grouped)
