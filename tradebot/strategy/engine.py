"""The deterministic half of the bot.

The engine never talks to a model and never places an order. It answers one
question - "do my rules say LONG, SHORT, or nothing?" - and then shapes that
answer into concrete entry/stop/target levels. Everything downstream (the AI
confirmation layer, the risk engine) can only *veto* what comes out of here.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config
from ..models import RuleResult, Side, StrategyVerdict, TradeSignal
from .context import MarketContext
from .presets import rules_for
from .rules import RULES


class StrategyEngine:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.rule_names = rules_for(config.strategy.name)
        missing = [n for n in self.rule_names if n not in RULES]
        if missing:
            raise KeyError(f"preset references unknown rules: {missing}")

    # ---------------------------------------------------------------- rules

    def evaluate_side(self, ctx: MarketContext, side: Side) -> StrategyVerdict:
        s = self.cfg.strategy
        checks: list[RuleResult] = []
        for name in self.rule_names:
            result = RULES[name](ctx, side, s)
            result.required = name not in s.optional_rules
            checks.append(result)

        total_weight = sum(c.weight for c in checks) or 1.0
        score = sum(c.weight for c in checks if c.passed) / total_weight
        required_ok = all(c.passed for c in checks if c.required)
        passed = required_ok and score >= s.min_score

        notes: list[str] = []
        if not ctx.warm:
            passed = False
            notes.append(f"warming up - only {ctx.bars} bars of history so far")
        if not ctx.in_session:
            passed = False
            notes.append("outside the configured trading session")
        if not required_ok:
            notes.append("failed required rules: " + ", ".join(
                c.name for c in checks if c.required and not c.passed))
        elif score < s.min_score:
            notes.append(f"score {score:.2f} below minimum {s.min_score:.2f}")

        return StrategyVerdict(side=side, passed=passed, score=score, checks=checks, notes=notes)

    def evaluate(self, ctx: MarketContext) -> StrategyVerdict:
        """Evaluate both directions and return the winner (or a FLAT verdict)."""
        long_v = self.evaluate_side(ctx, Side.LONG)
        short_v = self.evaluate_side(ctx, Side.SHORT)
        if long_v.passed and not short_v.passed:
            return long_v
        if short_v.passed and not long_v.passed:
            return short_v
        if long_v.passed and short_v.passed:
            # Contradictory: the data is telling us nothing useful.
            flat = StrategyVerdict(Side.FLAT, False, 0.0, [], ["both directions qualified - ambiguous"])
            return flat
        # Nothing passed; surface the closer of the two so the UI can show why.
        best = long_v if long_v.score >= short_v.score else short_v
        best.passed = False
        return best

    # ------------------------------------------------------------- levels

    def build_signal(self, ctx: MarketContext, verdict: StrategyVerdict) -> Optional[TradeSignal]:
        """Turn a passing verdict into entry / stop / TP1 / TP2."""
        if verdict.side is Side.FLAT:
            return None
        s, m = self.cfg.strategy, self.cfg.market
        entry = ctx.price
        stop = self._stop_for(ctx, verdict.side)
        if stop is None:
            return None

        distance = abs(entry - stop)
        min_distance = s.min_stop_ticks * m.tick_size
        if distance < min_distance:
            stop = entry - min_distance * verdict.side.sign
            distance = min_distance

        sign = verdict.side.sign
        tp1 = entry + distance * s.tp1_r * sign
        tp2 = entry + distance * s.tp2_r * sign

        signal = TradeSignal(
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            side=verdict.side,
            entry=_round_to(entry, m.tick_size),
            stop=_round_to(stop, m.tick_size),
            tp1=_round_to(tp1, m.tick_size),
            tp2=_round_to(tp2, m.tick_size),
            ts=ctx.candle.ts,
            strategy=s.name,
            strategy_verdict=verdict,
            rr1=s.tp1_r,
            rr2=s.tp2_r,
            reasons=[c.detail for c in verdict.checks if c.passed and c.detail],
        )
        return signal

    def _stop_for(self, ctx: MarketContext, side: Side) -> Optional[float]:
        """Behind the protecting swing, padded by ATR; ATR-only as a fallback."""
        s = self.cfg.strategy
        buffer = (ctx.atr or 0.0) * s.stop_buffer_atr
        if side is Side.LONG:
            swing = ctx.swing_low
            if swing is not None and swing < ctx.price:
                return swing - buffer
            if ctx.atr:
                return ctx.price - ctx.atr * s.stop_atr_multiple
        else:
            swing = ctx.swing_high
            if swing is not None and swing > ctx.price:
                return swing + buffer
            if ctx.atr:
                return ctx.price + ctx.atr * s.stop_atr_multiple
        return None


def _round_to(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, 10)
