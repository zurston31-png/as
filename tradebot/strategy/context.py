"""Turns a raw candle series into the structured feature set the rules read.

This is the layer that makes the AI's job easy: by the time anything reaches a
model, "is price above VWAP" is a boolean, not something to squint at on a
screenshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import Config
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


def _tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError):
        return ZoneInfo("UTC")


def _parse_hm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


@dataclass
class MarketContext:
    """Everything the rules, the AI layer and the dashboard need about *now*."""

    symbol: str
    timeframe: str
    candle: Candle
    price: float
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_fast_prev: Optional[float] = None
    ema_slow_prev: Optional[float] = None
    bars_since_bull_cross: Optional[int] = None
    bars_since_bear_cross: Optional[int] = None
    vwap: Optional[float] = None
    vwap_distance_pct: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    volume: float = 0.0
    volume_ma: Optional[float] = None
    volume_ratio: Optional[float] = None
    trend: str = "range"
    structure: StructureBreak = field(default_factory=lambda: StructureBreak("none", "none"))
    sweep: LiquiditySweep = field(default_factory=lambda: LiquiditySweep(False))
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    prev_swing_high: Optional[float] = None
    prev_swing_low: Optional[float] = None
    in_session: bool = True
    bars: int = 0
    warm: bool = False               # enough history for every indicator
    local_time: str = ""

    def to_dict(self) -> dict[str, Any]:
        def r(v: Optional[float], nd: int = 4) -> Optional[float]:
            return round(v, nd) if isinstance(v, (int, float)) else None

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "ts": self.candle.ts.isoformat(),
            "local_time": self.local_time,
            "price": r(self.price),
            "ema_fast": r(self.ema_fast),
            "ema_slow": r(self.ema_slow),
            "ema_stack": self.ema_stack,
            "bars_since_bull_cross": self.bars_since_bull_cross,
            "bars_since_bear_cross": self.bars_since_bear_cross,
            "vwap": r(self.vwap),
            "vwap_side": self.vwap_side,
            "vwap_distance_pct": r(self.vwap_distance_pct, 3),
            "rsi": r(self.rsi, 2),
            "atr": r(self.atr),
            "atr_pct": r(self.atr_pct, 3),
            "volume": self.volume,
            "volume_ma": r(self.volume_ma, 2),
            "volume_ratio": r(self.volume_ratio, 2),
            "trend": self.trend,
            "structure": {
                "kind": self.structure.kind,
                "direction": self.structure.direction,
                "level": r(self.structure.level),
                "bars_ago": self.structure.bars_ago,
            },
            "liquidity_sweep": {
                "happened": self.sweep.happened,
                "direction": self.sweep.direction,
                "level": r(self.sweep.level),
                "bars_ago": self.sweep.bars_ago,
            },
            "swing_high": r(self.swing_high),
            "swing_low": r(self.swing_low),
            "in_session": self.in_session,
            "bars": self.bars,
            "warm": self.warm,
        }

    @property
    def ema_stack(self) -> str:
        if self.ema_fast is None or self.ema_slow is None:
            return "unknown"
        return "bullish" if self.ema_fast > self.ema_slow else "bearish"

    @property
    def vwap_side(self) -> str:
        if self.vwap is None:
            return "unknown"
        return "above" if self.price > self.vwap else "below"


class ContextBuilder:
    """Computes a `MarketContext` from a series. Stateless and cheap to re-run."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.tz = _tz(config.market.session_tz)
        self.session_start = _parse_hm(config.market.session_start)
        self.session_end = _parse_hm(config.market.session_end)

    def min_bars(self) -> int:
        s = self.cfg.strategy
        return max(s.ema_slow, s.rsi_period, s.atr_period, s.volume_ma_period) + 5

    def session_flags(self, candles: list[Candle]) -> list[bool]:
        """True on the first bar of each trading session (the VWAP anchor)."""
        flags: list[bool] = []
        prev_day = None
        for c in candles:
            local = c.ts.astimezone(self.tz)
            day = local.date()
            started = day != prev_day and local.time() >= self.session_start
            if day != prev_day:
                prev_day = day
            flags.append(started or not flags)
        return flags

    def in_session(self, ts: datetime) -> bool:
        if not self.cfg.market.trade_session_only:
            return True
        local = ts.astimezone(self.tz).time()
        if self.session_start <= self.session_end:
            return self.session_start <= local <= self.session_end
        return local >= self.session_start or local <= self.session_end  # overnight

    def build(self, series: Series) -> Optional[MarketContext]:
        candles = list(series.candles)
        if not candles:
            return None
        s = self.cfg.strategy
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]

        fast = ema(closes, s.ema_fast)
        slow = ema(closes, s.ema_slow)
        rsi_vals = rsi(closes, s.rsi_period)
        atr_vals = atr(highs, lows, closes, s.atr_period)
        vwap_vals = session_vwap(highs, lows, closes, vols, self.session_flags(candles))
        vol_ma = sma(vols, s.volume_ma_period)

        # Swing/structure detection is O(window); feeding it the whole series
        # would make every bar cost more as the session went on, for no benefit -
        # structure older than `structure_lookback` bars is not what these rules
        # are about.
        window = candles[-s.structure_lookback:]

        ctx = MarketContext(
            symbol=series.symbol,
            timeframe=series.timeframe,
            candle=last,
            price=last.close,
            ema_fast=fast[-1],
            ema_slow=slow[-1],
            ema_fast_prev=fast[-2] if len(fast) > 1 else None,
            ema_slow_prev=slow[-2] if len(slow) > 1 else None,
            bars_since_bull_cross=crossed_above(fast, slow, s.ema_cross_lookback + 1),
            bars_since_bear_cross=crossed_below(fast, slow, s.ema_cross_lookback + 1),
            vwap=vwap_vals[-1],
            rsi=rsi_vals[-1],
            atr=atr_vals[-1],
            volume=last.volume,
            volume_ma=vol_ma[-1],
            trend=classify_trend(window, s.swing_width, s.structure_lookback),
            structure=detect_break(window, s.swing_width, s.sweep_lookback),
            sweep=detect_liquidity_sweep(window, s.swing_width, s.sweep_lookback),
            in_session=self.in_session(last.ts),
            bars=len(candles),
            warm=len(candles) >= self.min_bars(),
            local_time=last.ts.astimezone(self.tz).strftime("%Y-%m-%d %H:%M %Z"),
        )

        if ctx.vwap:
            ctx.vwap_distance_pct = (ctx.price - ctx.vwap) / ctx.vwap * 100.0
        if ctx.atr and ctx.price:
            ctx.atr_pct = ctx.atr / ctx.price * 100.0
        if ctx.volume_ma:
            ctx.volume_ratio = ctx.volume / ctx.volume_ma if ctx.volume_ma else None

        sh = recent_swings(window, "high", 2, s.swing_width)
        sl = recent_swings(window, "low", 2, s.swing_width)
        ctx.swing_high = sh[-1].price if sh else None
        ctx.prev_swing_high = sh[0].price if len(sh) > 1 else None
        ctx.swing_low = sl[-1].price if sl else None
        ctx.prev_swing_low = sl[0].price if len(sl) > 1 else None
        return ctx
