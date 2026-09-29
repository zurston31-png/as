"""The individual conditions a setup has to satisfy.

Each rule is a small pure function of (context, side, config) so it can be unit
tested in isolation, reordered, or swapped out without touching the engine. Add
your own by writing a function and registering it in `RULES`.
"""

from __future__ import annotations

from typing import Callable

from ..config import StrategyConfig
from ..models import RuleResult, Side
from .context import MarketContext

RuleFn = Callable[[MarketContext, Side, StrategyConfig], RuleResult]


def ema_stack(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    """9 EMA on the right side of the 21 EMA."""
    if ctx.ema_fast is None or ctx.ema_slow is None:
        return RuleResult("ema_stack", False, "EMAs not warm yet")
    ok = ctx.ema_fast > ctx.ema_slow if side is Side.LONG else ctx.ema_fast < ctx.ema_slow
    rel = ">" if ctx.ema_fast > ctx.ema_slow else "<"
    return RuleResult(
        "ema_stack",
        ok,
        f"EMA{cfg.ema_fast} {ctx.ema_fast:.2f} {rel} EMA{cfg.ema_slow} {ctx.ema_slow:.2f}",
        value=ctx.ema_stack,
    )


def ema_cross_fresh(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    """The crossover happened recently enough to still be the reason we're in."""
    bars = ctx.bars_since_bull_cross if side is Side.LONG else ctx.bars_since_bear_cross
    if bars is None:
        return RuleResult(
            "ema_cross_fresh",
            False,
            f"no {'bullish' if side is Side.LONG else 'bearish'} cross in last "
            f"{cfg.ema_cross_lookback} bars",
        )
    ok = bars <= cfg.ema_cross_lookback
    return RuleResult(
        "ema_cross_fresh", ok, f"cross was {bars} bar(s) ago", value=bars
    )


def vwap_position(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    if ctx.vwap is None:
        return RuleResult("vwap_position", False, "VWAP unavailable")
    ok = ctx.price > ctx.vwap if side is Side.LONG else ctx.price < ctx.vwap
    return RuleResult(
        "vwap_position",
        ok,
        f"price {ctx.price:.2f} {ctx.vwap_side} VWAP {ctx.vwap:.2f} "
        f"({ctx.vwap_distance_pct:+.2f}%)" if ctx.vwap_distance_pct is not None else "",
        value=ctx.vwap_side,
    )


def rsi_window(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    """Momentum confirms, but isn't already exhausted."""
    if ctx.rsi is None:
        return RuleResult("rsi_window", False, "RSI not warm yet")
    if side is Side.LONG:
        lo, hi = cfg.rsi_long_min, cfg.rsi_long_max
    else:
        lo, hi = cfg.rsi_short_min, cfg.rsi_short_max
    ok = lo <= ctx.rsi <= hi
    return RuleResult(
        "rsi_window", ok, f"RSI {ctx.rsi:.1f} vs window {lo:g}-{hi:g}", value=round(ctx.rsi, 2)
    )


def market_structure(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    """Don't fight the swing structure. A range is tolerated, the opposite isn't."""
    want = "bullish" if side is Side.LONG else "bearish"
    against = "bearish" if side is Side.LONG else "bullish"
    ok = ctx.trend != against
    detail = f"structure is {ctx.trend} (want {want}, blocks on {against})"
    return RuleResult("market_structure", ok, detail, value=ctx.trend)


def structure_break(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    """A BOS/CHoCH in our direction - confirmation, not a hard requirement."""
    want = "up" if side is Side.LONG else "down"
    sb = ctx.structure
    ok = sb.kind != "none" and sb.direction == want and sb.bars_ago <= cfg.sweep_lookback
    detail = (
        f"{sb.kind} {sb.direction} at {sb.level:.2f}, {sb.bars_ago} bar(s) ago"
        if sb.level is not None
        else "no recent structure break"
    )
    return RuleResult("structure_break", ok, detail, value=sb.kind)


def liquidity_sweep(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    """Stops run and rejected on the side we're about to trade away from."""
    sw = ctx.sweep
    ok = sw.bullish if side is Side.LONG else sw.bearish
    detail = (
        f"swept {sw.direction}side liquidity at {sw.level:.2f}, {sw.bars_ago} bar(s) ago"
        if sw.happened and sw.level is not None
        else "no recent sweep"
    )
    return RuleResult("liquidity_sweep", ok, detail, value=sw.direction)


def volume_confirmation(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    if ctx.volume_ratio is None:
        return RuleResult("volume_confirmation", False, "volume average not warm yet")
    if ctx.volume_ma == 0:
        # Index/forex feeds without volume shouldn't veto every trade.
        return RuleResult("volume_confirmation", True, "feed reports no volume - rule skipped")
    ok = ctx.volume_ratio >= cfg.volume_multiple
    return RuleResult(
        "volume_confirmation",
        ok,
        f"volume {ctx.volume_ratio:.2f}x the {cfg.volume_ma_period}-bar average "
        f"(need {cfg.volume_multiple:g}x)",
        value=round(ctx.volume_ratio, 2),
    )


def candle_direction(ctx: MarketContext, side: Side, cfg: StrategyConfig) -> RuleResult:
    """The trigger bar itself has to agree with the direction."""
    ok = ctx.candle.bullish if side is Side.LONG else not ctx.candle.bullish
    return RuleResult(
        "candle_direction",
        ok,
        f"trigger candle is {'bullish' if ctx.candle.bullish else 'bearish'}",
        weight=0.5,
        value=ctx.candle.bullish,
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
}
