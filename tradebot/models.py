"""Core domain types shared by every layer of the pipeline.

The whole bot is a one-way flow:

    market data -> features -> strategy verdict -> AI confirmation
                -> risk decision -> TradeSignal -> notification / paper fill

Each arrow in that chain is one of the dataclasses below.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Iterable, Optional
from collections import deque


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "NO_TRADE"

    @property
    def sign(self) -> int:
        return {Side.LONG: 1, Side.SHORT: -1, Side.FLAT: 0}[self]

    @property
    def opposite(self) -> "Side":
        return {Side.LONG: Side.SHORT, Side.SHORT: Side.LONG, Side.FLAT: Side.FLAT}[self]


class Decision(str, Enum):
    """What the AI confirmation layer came back with."""

    CONFIRM = "confirm"
    REJECT = "reject"
    WAIT = "wait"
    SKIPPED = "skipped"      # layer disabled
    UNAVAILABLE = "unavailable"  # layer enabled but the call failed


@dataclass(slots=True)
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    closed: bool = True

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def bullish(self) -> bool:
        return self.close >= self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = _iso(self.ts)
        return d


class Series:
    """A bounded, append-only OHLCV series for one symbol/timeframe.

    ``upsert`` is intentionally tolerant: feeds re-send the forming candle many
    times per second, and we only want one row per timestamp.
    """

    def __init__(self, symbol: str, timeframe: str, maxlen: int = 1500) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles: Deque[Candle] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self.candles)

    def __iter__(self) -> Iterable[Candle]:
        return iter(self.candles)

    def upsert(self, candle: Candle) -> bool:
        """Add or replace a candle. Returns True when a *new* bar was appended."""
        if self.candles and self.candles[-1].ts == candle.ts:
            self.candles[-1] = candle
            return False
        if self.candles and candle.ts < self.candles[-1].ts:
            return False  # out-of-order replay tick, ignore
        self.candles.append(candle)
        return True

    def extend(self, candles: Iterable[Candle]) -> None:
        for c in candles:
            self.upsert(c)

    @property
    def last(self) -> Optional[Candle]:
        return self.candles[-1] if self.candles else None

    def closed_candles(self) -> list[Candle]:
        return [c for c in self.candles if c.closed]

    def field(self, name: str, n: Optional[int] = None) -> list[float]:
        vals = [getattr(c, name) for c in self.candles]
        return vals if n is None else vals[-n:]

    @property
    def closes(self) -> list[float]:
        return self.field("close")

    @property
    def highs(self) -> list[float]:
        return self.field("high")

    @property
    def lows(self) -> list[float]:
        return self.field("low")

    @property
    def volumes(self) -> list[float]:
        return self.field("volume")


@dataclass(slots=True)
class RuleResult:
    name: str
    passed: bool
    detail: str = ""
    weight: float = 1.0
    required: bool = True
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "weight": self.weight,
            "required": self.required,
            "value": self.value,
        }


@dataclass(slots=True)
class StrategyVerdict:
    side: Side
    passed: bool
    score: float                      # 0..1 weighted share of rules that passed
    checks: list[RuleResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failed_required(self) -> list[str]:
        return [c.name for c in self.checks if c.required and not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "passed": self.passed,
            "score": round(self.score, 4),
            "checks": [c.to_dict() for c in self.checks],
            "notes": self.notes,
            "failed_required": self.failed_required,
        }


@dataclass(slots=True)
class AIVerdict:
    decision: Decision
    confidence: float = 0.0
    rationale: str = ""
    risk_flags: list[str] = field(default_factory=list)
    source: str = "structured"        # "structured" | "vision" | "none"
    latency_ms: int = 0
    model: str = ""

    @property
    def blocks_trade(self) -> bool:
        return self.decision in (Decision.REJECT, Decision.WAIT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "risk_flags": self.risk_flags,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "model": self.model,
        }


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    reason: str = ""
    qty: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "qty": self.qty,
            "risk_amount": round(self.risk_amount, 2),
            "risk_pct": round(self.risk_pct, 4),
            "violations": self.violations,
        }


@dataclass(slots=True)
class TradeSignal:
    """The thing the user actually sees on screen."""

    symbol: str
    timeframe: str
    side: Side
    entry: float
    stop: float
    tp1: float
    tp2: float
    ts: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    qty: float = 0.0
    risk_pct: float = 0.0
    risk_amount: float = 0.0
    rr1: float = 0.0
    rr2: float = 0.0
    strategy: str = ""
    strategy_verdict: Optional[StrategyVerdict] = None
    ai: Optional[AIVerdict] = None
    risk: Optional[RiskDecision] = None
    actionable: bool = False          # survived every gate
    blocked_by: str = ""              # first gate that said no
    reasons: list[str] = field(default_factory=list)

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": _iso(self.ts),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side.value,
            "entry": self.entry,
            "stop": self.stop,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "qty": self.qty,
            "risk_pct": round(self.risk_pct, 4),
            "risk_amount": round(self.risk_amount, 2),
            "stop_distance": round(self.stop_distance, 6),
            "rr1": round(self.rr1, 2),
            "rr2": round(self.rr2, 2),
            "strategy": self.strategy,
            "actionable": self.actionable,
            "blocked_by": self.blocked_by,
            "reasons": self.reasons,
            "strategy_verdict": self.strategy_verdict.to_dict() if self.strategy_verdict else None,
            "ai": self.ai.to_dict() if self.ai else None,
            "risk": self.risk.to_dict() if self.risk else None,
        }

    def headline(self) -> str:
        """The one-line form used for notifications and the terminal."""
        if not self.actionable:
            return f"NO TRADE {self.symbol} - {self.blocked_by}"
        return (
            f"{self.side.value} {self.symbol} @ {self.entry:g} | "
            f"SL {self.stop:g} | TP1 {self.tp1:g} | TP2 {self.tp2:g} | "
            f"risk {self.risk_pct * 100:.2f}%"
        )


@dataclass(slots=True)
class Fill:
    ts: datetime
    price: float
    qty: float
    kind: str  # entry | tp1 | tp2 | stop | manual


@dataclass(slots=True)
class Position:
    signal: TradeSignal
    qty: float
    entry_price: float
    opened_at: datetime
    stop: float
    tp1: float
    tp2: float
    remaining: float = 0.0
    realized: float = 0.0
    tp1_hit: bool = False
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.remaining:
            self.remaining = self.qty

    @property
    def side(self) -> Side:
        return self.signal.side

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.remaining * self.side.sign

    def to_dict(self, price: Optional[float] = None) -> dict[str, Any]:
        return {
            "signal_id": self.signal.id,
            "symbol": self.signal.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "remaining": self.remaining,
            "entry_price": self.entry_price,
            "opened_at": _iso(self.opened_at),
            "stop": self.stop,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp1_hit": self.tp1_hit,
            "realized": round(self.realized, 2),
            "unrealized": round(self.unrealized(price), 2) if price is not None else None,
        }


@dataclass(slots=True)
class ClosedTrade:
    signal_id: str
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    r_multiple: float
    opened_at: datetime
    closed_at: datetime
    exit_reason: str

    @property
    def win(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "pnl": round(self.pnl, 2),
            "r_multiple": round(self.r_multiple, 3),
            "opened_at": _iso(self.opened_at),
            "closed_at": _iso(self.closed_at),
            "exit_reason": self.exit_reason,
            "win": self.win,
        }
