"""The individual conditions a setup has to satisfy.

Every rule is a pure function of (context, side, params). Params come from the
strategy config, so periods and thresholds are never hard-coded here - the
defaults below are only what applies when the config stays silent.

Adding your own rule is a function plus an entry in `RULES`; it is then
immediately usable from YAML by name.
"""

from __future__ import annotations

from typing import Any, Callable

from ..models import RuleResult, Side
from .context import MarketContext

Params = dict[str, Any]
RuleFn = Callable[[MarketContext, Side, Params], RuleResult]


def _f(params: Params, key: str, default: float) -> float:
    value = params.get(key, default)
    return float(value) if value is not None else float(default)


def _i(params: Params, key: str, default: int) -> int:
    value = params.get(key, default)
    return int(value) if value is not None else int(default)


def ema_stack(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """Fast EMA on the right side of the slow EMA."""
    fast_n, slow_n = _i(p, "fast", 9), _i(p, "slow", 21)
    fast, slow = ctx.ema(fast_n), ctx.ema(slow_n)
    if fast is None or slow is None:
        return RuleResult("ema_stack", False, "EMAs not warm yet")
    ok = fast > slow if side is Side.LONG else fast < slow
    rel = ">" if fast > slow else "<"
    return RuleResult(
        "ema_stack", ok,
        f"EMA{fast_n} {fast:.2f} {rel} EMA{slow_n} {slow:.2f}",
        value="bullish" if fast > slow else "bearish",
    )


def ema_cross_fresh(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """The crossover is recent enough to still be the reason we're entering."""
    fast_n, slow_n = _i(p, "fast", 9), _i(p, "slow", 21)
    max_bars = _i(p, "max_bars", _i(p, "lookback", 5))
    direction = "up" if side is Side.LONG else "down"
    bars = ctx.bars_since_cross(fast_n, slow_n, direction, max_bars)
    if bars is None:
        return RuleResult(
            "ema_cross_fresh", False,
            f"no {'bullish' if side is Side.LONG else 'bearish'} cross in the last "
            f"{max_bars} bars",
        )
    return RuleResult(
        "ema_cross_fresh", bars <= max_bars, f"cross was {bars} bar(s) ago", value=bars
    )


def vwap_position(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    vwap = ctx.vwap()
    if vwap is None:
        return RuleResult("vwap_position", False, "VWAP unavailable")
    ok = ctx.price > vwap if side is Side.LONG else ctx.price < vwap
    distance = (ctx.price - vwap) / vwap * 100.0 if vwap else 0.0
    where = "above" if ctx.price > vwap else "below"
    max_distance = p.get("max_distance_pct")
    if max_distance is not None and abs(distance) > float(max_distance):
        return RuleResult(
            "vwap_position", False,
            f"price is {abs(distance):.2f}% from VWAP, beyond the "
            f"{float(max_distance):g}% limit", value=where,
        )
    return RuleResult(
        "vwap_position", ok,
        f"price {ctx.price:.2f} {where} VWAP {vwap:.2f} ({distance:+.2f}%)",
        value=where,
    )


def rsi_window(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """Momentum confirms, but isn't already exhausted."""
    period = _i(p, "period", 14)
    value = ctx.rsi(period)
    if value is None:
        return RuleResult("rsi_window", False, "RSI not warm yet")
    if side is Side.LONG:
        lo, hi = _f(p, "long_min", 50.0), _f(p, "long_max", 72.0)
    else:
        lo, hi = _f(p, "short_min", 28.0), _f(p, "short_max", 50.0)
    ok = lo <= value <= hi
    return RuleResult(
        "rsi_window", ok,
        f"RSI({period}) {value:.1f} vs window {lo:g}-{hi:g}", value=round(value, 2),
    )


def market_structure(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """Don't fight the swing structure. A range is tolerated; the opposite isn't."""
    width, lookback = _i(p, "width", 2), _i(p, "lookback", 60)
    trend = ctx.trend(width, lookback)
    against = "bearish" if side is Side.LONG else "bullish"
    allow_range = bool(p.get("allow_range", True))
    ok = trend != against and (allow_range or trend != "range")
    return RuleResult(
        "market_structure", ok,
        f"structure is {trend} (blocks on {against}"
        f"{'' if allow_range else ' and range'})",
        value=trend,
    )


def structure_break(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """A BOS or CHoCH in our direction."""
    width, lookback = _i(p, "width", 2), _i(p, "lookback", 5)
    sb = ctx.structure(width, lookback)
    want = "up" if side is Side.LONG else "down"
    kinds = p.get("kinds") or ["BOS", "CHoCH"]
    ok = sb.kind in kinds and sb.direction == want and sb.bars_ago <= lookback
    detail = (
        f"{sb.kind} {sb.direction} at {sb.level:.2f}, {sb.bars_ago} bar(s) ago"
        if sb.level is not None else "no recent structure break"
    )
    return RuleResult("structure_break", ok, detail, value=sb.kind)


def liquidity_sweep(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """Stops run and rejected on the side we're trading away from."""
    width = _i(p, "width", 2)
    lookback = _i(p, "lookback", 5)
    ratio = _f(p, "min_wick_ratio", 0.5)
    sw = ctx.sweep(width, lookback, ratio)
    ok = sw.bullish if side is Side.LONG else sw.bearish
    detail = (
        f"swept {sw.direction}side liquidity at {sw.level:.2f}, {sw.bars_ago} bar(s) ago"
        if sw.happened and sw.level is not None else "no recent sweep"
    )
    return RuleResult("liquidity_sweep", ok, detail, value=sw.direction)


def volume_confirmation(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    period = _i(p, "period", 20)
    multiple = _f(p, "multiple", 1.1)
    average = ctx.volume_ma(period)
    if average is None:
        return RuleResult("volume_confirmation", False, "volume average not warm yet")
    if average == 0:
        # Index and FX feeds often report no volume; don't veto every trade.
        return RuleResult("volume_confirmation", True, "feed reports no volume - rule skipped")
    ratio = ctx.candle.volume / average
    return RuleResult(
        "volume_confirmation", ratio >= multiple,
        f"volume {ratio:.2f}x the {period}-bar average (need {multiple:g}x)",
        value=round(ratio, 2),
    )


def candle_direction(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """The trigger bar itself has to agree."""
    c = ctx.candle
    ok = c.bullish if side is Side.LONG else not c.bullish
    min_body = p.get("min_body_ratio")
    if ok and min_body is not None and c.range > 0:
        ratio = c.body / c.range
        if ratio < float(min_body):
            return RuleResult(
                "candle_direction", False,
                f"body is only {ratio:.2f} of the bar's range "
                f"(need {float(min_body):g})", weight=1.0, value=round(ratio, 2),
            )
    return RuleResult(
        "candle_direction", ok,
        f"trigger candle is {'bullish' if c.bullish else 'bearish'}",
        value=c.bullish,
    )


def atr_range(ctx: MarketContext, side: Side, p: Params) -> RuleResult:
    """Volatility filter - skip setups when the market is dead or unhinged."""
    period = _i(p, "period", 14)
    value = ctx.atr(period)
    if value is None or not ctx.price:
        return RuleResult("atr_range", False, "ATR not warm yet")
    pct = value / ctx.price * 100.0
    lo, hi = _f(p, "min_pct", 0.0), _f(p, "max_pct", 100.0)
    return RuleResult(
        "atr_range", lo <= pct <= hi,
        f"ATR({period}) is {pct:.3f}% of price (window {lo:g}-{hi:g}%)",
        value=round(pct, 4),
    )


RULES: dict[str, RuleFn] = {
    "ema_stack": ema_stack,
    "ema_cross_fresh": ema_cross_fresh,
    "vwap_position": vwap_position,
    "rsi_window": rsi_window,
    "market_structure": market_structure,
    "structure_break": structure_break,
    "liquidity_sweep": liquidity_sweep,
    "volume_confirmation": volume_confirmation,
    "candle_direction": candle_direction,
    "atr_range": atr_range,
}
