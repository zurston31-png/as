"""Everything the bot knows about one token, assembled in one place.

The question this answers is "why did the bot do (or not do) that?", asked
about a specific mint weeks after the fact. Answering it previously meant
joining five tables by hand and hoping the data was still there.

Identity is the MINT ADDRESS throughout, never the symbol. Symbols are not
unique and are trivially spoofed - a scam token can call itself anything an
existing one calls itself, and looking up "BONK" could otherwise merge two
completely different assets into one history. The symbol is displayed;
the address is what is matched on.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models


@dataclass
class TokenEvent:
    """One thing that happened to this token, for a single timeline."""

    at: dt.datetime
    kind: str            # discovered | signal | rejected | buy | sell | position
    detail: str
    outcome: str = ""    # "" | "ok" | "rejected" | "win" | "loss"


@dataclass
class TokenDetail:
    token_address: str
    symbol: str | None = None
    chain: str | None = None
    scanned: models.ScannedToken | None = None
    signals: list[models.Signal] = field(default_factory=list)
    rug_checks: list[models.RugCheckResult] = field(default_factory=list)
    positions: list[models.Position] = field(default_factory=list)
    trades: list[models.Trade] = field(default_factory=list)
    rejections: list[models.RiskEvent] = field(default_factory=list)
    timeline: list[TokenEvent] = field(default_factory=list)
    watchlist: "models.WatchlistEntry | None" = None
    observations: list = field(default_factory=list)

    @property
    def found(self) -> bool:
        """False when the bot has never seen this address at all - which is
        a real answer ("it was never discovered"), not an error."""
        return bool(
            self.scanned or self.signals or self.positions or self.trades or self.rejections
        )

    @property
    def realized_pnl_usd(self) -> float:
        return sum(t.pnl_usd or 0.0 for t in self.trades if t.pnl_usd is not None)

    @property
    def times_traded(self) -> int:
        return sum(
            1 for t in self.trades
            if t.side == "buy" and t.status == models.TradeStatus.FILLED.value
        )

    @property
    def latest_signal(self) -> models.Signal | None:
        return self.signals[-1] if self.signals else None

    @property
    def latest_rug_check(self) -> models.RugCheckResult | None:
        return self.rug_checks[-1] if self.rug_checks else None

    @property
    def verdict(self) -> str:
        """One line saying what the bot ultimately did, and why."""
        if not self.found:
            return "never seen - this address has not been discovered or submitted"
        if self.times_traded:
            return f"traded {self.times_traded}x, realized ${self.realized_pnl_usd:,.2f}"
        if self.rejections:
            last = self.rejections[-1]
            return f"never traded - last rejected at the {last.event_type} stage"
        if self.scanned and self.scanned.last_stage == "prescreen":
            return f"never traded - stopped at the pre-screen: {self.scanned.last_reason or 'no reason recorded'}"
        return "seen but never traded, and no rejection was recorded"


def _aware(moment: dt.datetime | None) -> dt.datetime:
    """Naive timestamps come back from SQLite; sorting a mix of naive and
    aware datetimes raises, which would break the timeline entirely."""
    if moment is None:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def build_token_detail(db: Session, token_address: str) -> TokenDetail:
    """Gather every record touching one mint address."""
    detail = TokenDetail(token_address=token_address)

    # The early-signal watchlist entry, if this token was ever watched. Its
    # score history is the only place the bot records how a candidate's
    # assessment CHANGED over time, which is what makes "was it improving?"
    # answerable at all.
    detail.watchlist = (
        db.query(models.WatchlistEntry).filter_by(token_address=token_address).first()
    )
    detail.observations = (
        db.query(models.TokenObservation)
        .filter(models.TokenObservation.token_address == token_address)
        .order_by(models.TokenObservation.observed_at.asc())
        .limit(200)
        .all()
    )

    detail.scanned = (
        db.query(models.ScannedToken).filter_by(token_address=token_address).first()
    )
    detail.signals = (
        db.query(models.Signal)
        .filter(models.Signal.token_address == token_address)
        .order_by(models.Signal.received_at.asc())
        .all()
    )
    signal_ids = [s.id for s in detail.signals]

    if signal_ids:
        detail.rug_checks = (
            db.query(models.RugCheckResult)
            .filter(models.RugCheckResult.signal_id.in_(signal_ids))
            .order_by(models.RugCheckResult.id.asc())
            .all()
        )
        detail.rejections = (
            db.query(models.RiskEvent)
            .filter(models.RiskEvent.signal_id.in_(signal_ids))
            .order_by(models.RiskEvent.created_at.asc())
            .all()
        )

    detail.positions = (
        db.query(models.Position)
        .filter(models.Position.token_address == token_address)
        .order_by(models.Position.opened_at.asc())
        .all()
    )
    detail.trades = (
        db.query(models.Trade)
        .filter(models.Trade.token_address == token_address)
        .order_by(models.Trade.created_at.asc())
        .all()
    )

    # Symbol and chain are display-only, taken from whatever the bot last
    # recorded. They are never used to look anything up.
    for source in (detail.scanned, *reversed(detail.signals), *reversed(detail.trades)):
        if source is not None:
            detail.symbol = detail.symbol or getattr(source, "symbol", None)
            detail.chain = detail.chain or getattr(source, "chain", None)

    detail.timeline = _build_timeline(detail)
    return detail


def _build_timeline(detail: TokenDetail) -> list[TokenEvent]:
    events: list[TokenEvent] = []

    if detail.scanned and detail.scanned.first_seen_at:
        events.append(TokenEvent(
            _aware(detail.scanned.first_seen_at), "discovered",
            f"discovered via {detail.scanned.discovery_source or 'unknown source'}"
            + (f" - liquidity ${detail.scanned.liquidity_usd:,.0f}" if detail.scanned.liquidity_usd else ""),
        ))

    for s in detail.signals:
        bits = [f"{s.signal_type} signal from {s.source or 'unknown'}"]
        if s.signal_score is not None:
            bits.append(f"signal {s.signal_score:.0f}/100")
        if s.market_quality_score is not None:
            bits.append(f"quality {s.market_quality_score:.0f}/100")
        events.append(TokenEvent(_aware(s.received_at), "signal", " · ".join(bits)))

    for e in detail.rejections:
        events.append(TokenEvent(
            _aware(e.created_at), "rejected",
            f"{e.event_type}: {e.details or 'no detail recorded'}",
            outcome="rejected",
        ))

    for t in detail.trades:
        if t.side == "buy":
            events.append(TokenEvent(
                _aware(t.created_at), "buy",
                f"bought ${t.size_usd:,.0f} at ${t.entry_price:.10g}"
                if t.entry_price else f"buy attempt ({t.status})",
                outcome="ok" if t.status == models.TradeStatus.FILLED.value else "rejected",
            ))
        else:
            pnl = t.pnl_usd
            events.append(TokenEvent(
                _aware(t.closed_at or t.created_at), "sell",
                f"sold: {t.close_reason or 'no reason recorded'}"
                + (f" — ${pnl:,.2f}" if pnl is not None else ""),
                outcome=("win" if (pnl or 0) > 0 else "loss") if pnl is not None else "",
            ))

    events.sort(key=lambda e: e.at)
    return events
