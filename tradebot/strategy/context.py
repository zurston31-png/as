"""The decision-time view of the market.

Two guarantees this module exists to provide:

1. **No lookahead.** A context is built from closed candles at or before a
   stated `decision_ts`, and it keeps no reference to anything later. Every
   indicator here is causal (a prefix function of the series), so a value at
   bar *i* can only ever depend on bars <= *i*. `build()` refuses a series that
   would violate this rather than silently trimming.

2. **Config-driven features.** Rules ask for what they need - `ctx.ema(9)`,
   `ctx.rsi(14, )`, `ctx.sweep(width=2)` - and the answer is computed once and
   cached. That's what lets rule parameters live in YAML: nothing here has to
   know in advance which periods a strategy wants.

The cache doubles as an audit record: `used_features()` reports exactly which
indicator values the decision actually consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..indicators import atr, crossed_above, crossed_below, ema, rsi, session_vwap, sma
from ..models import Candle, Series
from ..structure import (
    LiquiditySweep,
    StructureBreak,
    classify_trend,
    detect_break,
    detect_liquidity_sweep,
    recent_swings,
)


class LookaheadError(AssertionError):
    """Raised when a context would be built over data from the future."""


def _tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError):
        return ZoneInfo("UTC")


def _parse_hm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


@dataclass(slots=True)
class FeedHealth:
    """How current the data behind this decision is."""

    bar_age_seconds: float = 0.0        # decision bar's close vs. wall clock
    expected_bar_seconds: int = 0
    received_at: Optional[datetime] = None
    stale: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar_age_seconds": round(self.bar_age_seconds, 1),
            "expected_bar_seconds": self.expected_bar_seconds,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "stale": self.stale,
            "reason": self.reason,
        }


class MarketContext:
    """Everything a rule, the AI layer or the dashboard can ask about *now*."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        candles: list[Candle],
        decision_ts: datetime,
        tz: ZoneInfo,
        session_start: time,
        in_session: bool,
        structure_window: int = 60,
        feed: Optional[FeedHealth] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._candles = candles
        self.decision_ts = decision_ts
        self.tz = tz
        self.session_start = session_start
        self.in_session = in_session
        self.structure_window = structure_window
        self.feed = feed or FeedHealth()
        self._cache: dict[str, Any] = {}
        self._used: dict[str, Any] = {}

    # ------------------------------------------------------------ the bar

    @property
    def candle(self) -> Candle:
        return self._candles[-1]

    @property
    def price(self) -> float:
        """The decision price: the close of the last *closed* bar."""
        return self._candles[-1].close

    @property
    def bars(self) -> int:
        return len(self._candles)

    @property
    def candles(self) -> Sequence[Candle]:
        return tuple(self._candles)

    @property
    def local_time(self) -> str:
        return self.decision_ts.astimezone(self.tz).strftime("%Y-%m-%d %H:%M %Z")

    def warm(self, need: int) -> bool:
        return len(self._candles) >= need

    # ------------------------------------------------------------ features

    def _memo(self, key: str, compute, record=None) -> Any:
        if key not in self._cache:
            self._cache[key] = compute()
            self._used[key] = record(self._cache[key]) if record else _scalar(self._cache[key])
        return self._cache[key]

    def _closes(self) -> list[float]:
        return self._memo("_closes", lambda: [c.close for c in self._candles])

    def _series_window(self) -> list[Candle]:
        return self._candles[-self.structure_window:]

    def ema(self, period: int) -> Optional[float]:
        return _last(self.ema_series(period))

    def ema_series(self, period: int) -> list[Optional[float]]:
        return self._memo(f"ema_{period}", lambda: ema(self._closes(), period))

    def rsi(self, period: int = 14) -> Optional[float]:
        return _last(self._memo(f"rsi_{period}", lambda: rsi(self._closes(), period)))

    def atr(self, period: int = 14) -> Optional[float]:
        def compute():
            return atr(
                [c.high for c in self._candles],
                [c.low for c in self._candles],
                self._closes(),
                period,
            )
        return _last(self._memo(f"atr_{period}", compute))

    def vwap(self) -> Optional[float]:
        def compute():
            return session_vwap(
                [c.high for c in self._candles],
                [c.low for c in self._candles],
                self._closes(),
                [c.volume for c in self._candles],
                self._session_flags(),
            )
        return _last(self._memo("vwap", compute))

    def volume_ma(self, period: int = 20) -> Optional[float]:
        return _last(self._memo(
            f"volume_ma_{period}",
            lambda: sma([c.volume for c in self._candles], period),
        ))

    def volume_ratio(self, period: int = 20) -> Optional[float]:
        average = self.volume_ma(period)
        if not average:
            return None
        return self.candle.volume / average

    def bars_since_cross(self, fast: int, slow: int, direction: str, lookback: int) -> Optional[int]:
        key = f"cross_{direction}_{fast}_{slow}_{lookback}"
        fn = crossed_above if direction == "up" else crossed_below
        return self._memo(
            key, lambda: fn(self.ema_series(fast), self.ema_series(slow), lookback + 1)
        )

    def trend(self, width: int = 2, lookback: int = 60) -> str:
        return self._memo(
            f"trend_{width}_{lookback}",
            lambda: classify_trend(self._series_window(), width, lookback),
        )

    def structure(self, width: int = 2, lookback: int = 5) -> StructureBreak:
        return self._memo(
            f"structure_{width}_{lookback}",
            lambda: detect_break(self._series_window(), width, lookback),
            record=lambda sb: {"kind": sb.kind, "direction": sb.direction, "level": sb.level},
        )

    def sweep(self, width: int = 2, lookback: int = 5, min_wick_ratio: float = 0.5) -> LiquiditySweep:
        return self._memo(
            f"sweep_{width}_{lookback}_{min_wick_ratio}",
            lambda: detect_liquidity_sweep(self._series_window(), width, lookback, min_wick_ratio),
            record=lambda sw: {"happened": sw.happened, "direction": sw.direction, "level": sw.level},
        )

    def swing(self, kind: str, width: int = 2, back: int = 0) -> Optional[float]:
        """`back=0` is the most recent confirmed swing of that kind."""
        swings = self._memo(
            f"swings_{kind}_{width}",
            lambda: [s.price for s in recent_swings(self._series_window(), kind, 4, width)],
        )
        if len(swings) <= back:
            return None
        return swings[-(back + 1)]

    def _session_flags(self) -> list[bool]:
        flags: list[bool] = []
        prev_day = None
        for c in self._candles:
            local = c.ts.astimezone(self.tz)
            day = local.date()
            started = day != prev_day and local.time() >= self.session_start
            if day != prev_day:
                prev_day = day
            flags.append(started or not flags)
        return flags

    # -------------------------------------------------------------- output

    def used_features(self) -> dict[str, Any]:
        """Exactly the indicator values this decision consulted - for the audit log."""
        return {k: v for k, v in self._used.items() if not k.startswith("_")}

    def to_dict(self, display: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision_ts": self.decision_ts.isoformat(),
            "ts": self.candle.ts.isoformat(),
            "local_time": self.local_time,
            "price": self.price,
            "volume": self.candle.volume,
            "in_session": self.in_session,
            "bars": self.bars,
            "feed": self.feed.to_dict(),
            "features": self.used_features(),
            "display": display or [],
        }


def _last(values: Sequence[Optional[float]]) -> Optional[float]:
    return values[-1] if values else None


def _scalar(value: Any) -> Any:
    if isinstance(value, list):
        tail = value[-1] if value else None
        return round(tail, 6) if isinstance(tail, float) else tail
    if isinstance(value, float):
        return round(value, 6)
    return value


class ContextBuilder:
    """Builds a `MarketContext`, enforcing the no-lookahead contract."""

    def __init__(self, config) -> None:
        self.cfg = config
        self.tz = _tz(config.market.session_tz)
        self.session_start = _parse_hm(config.market.session_start)
        self.session_end = _parse_hm(config.market.session_end)

    def in_session(self, ts: datetime) -> bool:
        if not self.cfg.market.trade_session_only:
            return True
        local = ts.astimezone(self.tz).time()
        if self.session_start <= self.session_end:
            return self.session_start <= local <= self.session_end
        return local >= self.session_start or local <= self.session_end  # overnight

    def build(
        self,
        series: Series,
        decision_ts: Optional[datetime] = None,
        feed: Optional[FeedHealth] = None,
        structure_window: int = 60,
    ) -> Optional[MarketContext]:
        """Build the view as of `decision_ts` (default: the last closed bar).

        Only *closed* candles at or before `decision_ts` are included. A forming
        bar is never visible to a rule - its high, low and close are all still
        moving, and letting a rule see them is the single easiest way to build a
        backtest that cannot be reproduced live.
        """
        closed = [c for c in series.candles if c.closed]
        if not closed:
            return None

        if decision_ts is None:
            decision_ts = closed[-1].ts
        visible = [c for c in closed if c.ts <= decision_ts]
        if not visible:
            return None

        # Two integrity checks rather than trusting the filter above. The first
        # is belt-and-braces against a future refactor. The second catches a real
        # condition: a feed that replays an old bar late leaves the series out of
        # order, and every indicator here assumes time order - a silently
        # mis-ordered window produces values that cannot be reproduced.
        latest = visible[-1].ts
        if latest > decision_ts:
            raise LookaheadError(
                f"context for {decision_ts.isoformat()} would include a bar from "
                f"{latest.isoformat()}"
            )
        for earlier, later in zip(visible, visible[1:]):
            if later.ts < earlier.ts:
                raise LookaheadError(
                    f"bars are out of order: {later.ts.isoformat()} follows "
                    f"{earlier.ts.isoformat()}; indicators would read the future"
                )

        return MarketContext(
            symbol=series.symbol,
            timeframe=series.timeframe,
            candles=visible,
            decision_ts=decision_ts,
            tz=self.tz,
            session_start=self.session_start,
            in_session=self.in_session(decision_ts),
            structure_window=structure_window,
            feed=feed,
        )
