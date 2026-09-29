"""The deterministic half of the bot.

The engine never calls a model and never places an order. It answers one
question - "do my rules say LONG, SHORT, or nothing?" - and shapes a passing
answer into concrete entry/stop/target levels.

It also owns duplicate-signal protection. The conditions behind a setup stay
true for several bars after they first line up, so without a gate one setup
emits a signal on every bar until it decays. A side that has fired stays
disarmed until the setup stops qualifying (or a configured number of bars
pass), which means one setup produces exactly one signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import RuleResult, Side, StrategyVerdict, TradeSignal
from .context import MarketContext
from .presets import PRESETS
from .rules import RULES
from .spec import StrategySpec


@dataclass
class GateState:
    """Per-side arming state for duplicate-signal protection."""

    armed: dict[Side, bool] = field(default_factory=lambda: {Side.LONG: True, Side.SHORT: True})
    fired_at_bar: dict[Side, Optional[int]] = field(
        default_factory=lambda: {Side.LONG: None, Side.SHORT: None})
    suppressed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "armed": {side.value: value for side, value in self.armed.items()},
            "fired_at_bar": {s.value: v for s, v in self.fired_at_bar.items()},
            "suppressed": self.suppressed,
        }


class StrategyEngine:
    def __init__(self, config) -> None:
        self.cfg = config
        self.spec = StrategySpec.from_config(_strategy_dict(config), PRESETS)
        self.gate = GateState()
        self._bar_index = 0
        self.qualified_sides = 0     # side-evaluations that passed, pre-arbitration
        self.ambiguous = 0           # bars where both directions qualified

    # ---------------------------------------------------------------- rules

    def evaluate_side(self, ctx: MarketContext, side: Side) -> StrategyVerdict:
        checks: list[RuleResult] = []
        for rule_spec in self.spec.active():
            result = RULES[rule_spec.rule](ctx, side, rule_spec.params)
            result.required = rule_spec.required
            result.weight = rule_spec.weight
            checks.append(result)

        total_weight = sum(c.weight for c in checks) or 1.0
        score = sum(c.weight for c in checks if c.passed) / total_weight
        required_ok = all(c.passed for c in checks if c.required)
        passed = required_ok and score >= self.spec.min_score

        notes: list[str] = []
        need = self.spec.min_bars()
        if not ctx.warm(need):
            passed = False
            notes.append(f"warming up - {ctx.bars}/{need} bars")
        if not ctx.in_session:
            passed = False
            notes.append("outside the configured trading session")
        if ctx.feed.stale:
            passed = False
            notes.append(f"stale market data: {ctx.feed.reason}")
        if not required_ok:
            notes.append("failed required rules: " + ", ".join(
                c.name for c in checks if c.required and not c.passed))
        elif score < self.spec.min_score:
            notes.append(f"score {score:.2f} below the {self.spec.min_score:.2f} minimum")

        return StrategyVerdict(side=side, passed=passed, score=score, checks=checks, notes=notes)

    def evaluate(self, ctx: MarketContext) -> StrategyVerdict:
        """Evaluate both directions and return the winner (or a FLAT verdict)."""
        long_v = self.evaluate_side(ctx, Side.LONG)
        short_v = self.evaluate_side(ctx, Side.SHORT)

        # Counted before arbitration: this is the clean measure of how strict
        # the rules are. Relaxing a rule can only raise it, whereas the
        # post-arbitration outcome can fall - a rule set loose enough for both
        # directions to qualify at once produces *fewer* usable setups, not
        # more, and that is worth being able to see.
        self.qualified_sides += int(long_v.passed) + int(short_v.passed)

        if long_v.passed and short_v.passed:
            self.ambiguous += 1
            return StrategyVerdict(
                Side.FLAT, False, 0.0, long_v.checks,
                ["both directions qualified - the data is ambiguous"])
        if long_v.passed:
            return long_v
        if short_v.passed:
            return short_v
        best = long_v if long_v.score >= short_v.score else short_v
        best.passed = False
        return best

    # ----------------------------------------------------------- dedupe

    def observe(self, ctx: MarketContext, verdict: StrategyVerdict) -> None:
        """Advance the gate for this bar. Call once per closed bar, before `admit`."""
        self._bar_index += 1
        sig = self.spec.signal
        for side in (Side.LONG, Side.SHORT):
            if self.gate.armed[side]:
                continue
            qualifies = verdict.passed and verdict.side is side
            if sig.rearm_on_invalidation and not qualifies:
                self._rearm(side, "setup no longer qualifies")
            elif sig.rearm_bars:
                fired = self.gate.fired_at_bar[side]
                if fired is not None and self._bar_index - fired >= sig.rearm_bars:
                    self._rearm(side, f"{sig.rearm_bars} bars elapsed")

    def admit(self, verdict: StrategyVerdict, position_side: Optional[Side]) -> tuple[bool, str]:
        """Should this qualifying setup become a signal? (duplicate + position gate)"""
        side = verdict.side
        sig = self.spec.signal
        if side is Side.FLAT:
            return False, "no direction"
        if not self.gate.armed[side]:
            self.gate.suppressed += 1
            return False, f"duplicate: a {side.value} signal already fired for this setup"
        if position_side is not None and sig.require_flat:
            if position_side is side:
                self.gate.suppressed += 1
                return False, f"already {side.value} - not stacking into the same direction"
            if not sig.allow_reversal:
                self.gate.suppressed += 1
                return False, f"holding a {position_side.value} position - reversals are disabled"
        return True, "admitted"

    def mark_fired(self, side: Side) -> None:
        self.gate.armed[side] = False
        self.gate.fired_at_bar[side] = self._bar_index

    def _rearm(self, side: Side, _reason: str) -> None:
        self.gate.armed[side] = True
        self.gate.fired_at_bar[side] = None

    def reset_gate(self) -> None:
        self.gate = GateState()
        self._bar_index = 0
        self.qualified_sides = 0
        self.ambiguous = 0

    # ------------------------------------------------------------- levels

    def build_signal(self, ctx: MarketContext, verdict: StrategyVerdict) -> Optional[TradeSignal]:
        """Turn a passing verdict into entry / stop / TP1 / TP2."""
        if verdict.side is Side.FLAT:
            return None
        entry_spec, m = self.spec.entry, self.cfg.market
        entry = ctx.price
        stop = self._stop_for(ctx, verdict.side)
        if stop is None:
            return None

        distance = abs(entry - stop)
        min_distance = entry_spec.stop.min_ticks * m.tick_size
        if distance < min_distance:
            stop = entry - min_distance * verdict.side.sign
            distance = min_distance

        sign = verdict.side.sign
        return TradeSignal(
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            side=verdict.side,
            entry=_round_to(entry, m.tick_size),
            stop=_round_to(stop, m.tick_size),
            tp1=_round_to(entry + distance * entry_spec.tp1_r * sign, m.tick_size),
            tp2=_round_to(entry + distance * entry_spec.tp2_r * sign, m.tick_size),
            ts=ctx.candle.ts,
            decision_ts=ctx.decision_ts,
            strategy=self.spec.name,
            trigger_candle=ctx.candle.to_dict(),
            strategy_verdict=verdict,
            rr1=entry_spec.tp1_r,
            rr2=entry_spec.tp2_r,
            reasons=[c.detail for c in verdict.checks if c.passed and c.detail],
        )

    def _stop_for(self, ctx: MarketContext, side: Side) -> Optional[float]:
        s = self.spec.entry.stop
        atr_value = ctx.atr(s.atr_period)
        if s.method == "fixed_ticks":
            return ctx.price - s.fixed_ticks * self.cfg.market.tick_size * side.sign
        if s.method == "swing_or_atr":
            swing = ctx.swing("low" if side is Side.LONG else "high", s.swing_width)
            buffer = (atr_value or 0.0) * s.swing_buffer_atr
            if swing is not None:
                if side is Side.LONG and swing < ctx.price:
                    return swing - buffer
                if side is Side.SHORT and swing > ctx.price:
                    return swing + buffer
        if atr_value:
            return ctx.price - atr_value * s.atr_multiple * side.sign
        return None

    # -------------------------------------------------------------- display

    def display(self, ctx: MarketContext) -> list[dict[str, Any]]:
        """Headline numbers for the dashboard, derived from the configured rules."""
        out: list[dict[str, Any]] = []
        for rule_spec in self.spec.active():
            p = rule_spec.params
            if rule_spec.rule == "ema_stack":
                for key, default in (("fast", 9), ("slow", 21)):
                    period = int(p.get(key, default))
                    out.append({"label": f"EMA{period}", "value": ctx.ema(period)})
            elif rule_spec.rule == "vwap_position":
                out.append({"label": "VWAP", "value": ctx.vwap()})
            elif rule_spec.rule == "rsi_window":
                period = int(p.get("period", 14))
                out.append({"label": f"RSI{period}", "value": ctx.rsi(period), "digits": 1})
            elif rule_spec.rule == "volume_confirmation":
                ratio = ctx.volume_ratio(int(p.get("period", 20)))
                out.append({"label": "Vol", "value": ratio, "digits": 2, "suffix": "x"})
            elif rule_spec.rule == "market_structure":
                out.append({"label": "Structure",
                            "value": ctx.trend(int(p.get("width", 2)), int(p.get("lookback", 60)))})
        atr_value = ctx.atr(self.spec.entry.stop.atr_period)
        out.append({"label": "ATR", "value": atr_value})
        seen, unique = set(), []
        for item in out:
            if item["label"] in seen:
                continue
            seen.add(item["label"])
            unique.append(item)
        return unique


def _strategy_dict(config) -> dict[str, Any]:
    raw = getattr(config, "strategy", None)
    if isinstance(raw, dict):
        return raw
    return dict(getattr(raw, "raw", {}) or {})


def _round_to(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, 10)
