from __future__ import annotations

import pytest
from conftest import make_candles

from tradebot.models import Series, Side
from tradebot.strategy.context import ContextBuilder
from tradebot.strategy.engine import StrategyEngine
from tradebot.strategy.presets import rules_for


def build(config, candles):
    series = Series("TEST", "5m")
    series.extend(candles)
    return ContextBuilder(config).build(series), series


def test_context_computes_every_field(config, uptrend):
    ctx, _ = build(config, uptrend)
    assert ctx is not None and ctx.warm
    assert ctx.ema_fast is not None and ctx.ema_slow is not None
    assert ctx.rsi is not None and ctx.atr is not None and ctx.vwap is not None
    assert ctx.ema_stack == "bullish"
    assert ctx.volume_ratio is not None


def test_uptrend_is_not_read_as_a_short(config, uptrend):
    ctx, _ = build(config, uptrend)
    engine = StrategyEngine(config)
    assert not engine.evaluate_side(ctx, Side.SHORT).passed


def test_rules_all_report_a_result(config, uptrend):
    ctx, _ = build(config, uptrend)
    verdict = StrategyEngine(config).evaluate_side(ctx, Side.LONG)
    assert {c.name for c in verdict.checks} == set(rules_for(config.strategy.name))
    assert 0.0 <= verdict.score <= 1.0


def test_optional_rules_do_not_block(config, uptrend):
    ctx, _ = build(config, uptrend)
    verdict = StrategyEngine(config).evaluate_side(ctx, Side.LONG)
    optional = [c for c in verdict.checks if c.name in config.strategy.optional_rules]
    assert optional and all(not c.required for c in optional)


def test_cold_series_never_passes(config):
    ctx, _ = build(config, make_candles([100, 101, 102]))
    verdict = StrategyEngine(config).evaluate(ctx)
    assert not verdict.passed


def test_out_of_session_blocks_the_setup(config, uptrend):
    config.market.session_start = "03:00"
    config.market.session_end = "03:30"
    ctx, _ = build(config, uptrend)
    assert not ctx.in_session
    assert not StrategyEngine(config).evaluate_side(ctx, Side.LONG).passed


def test_signal_levels_are_coherent_for_a_long(config, uptrend):
    ctx, _ = build(config, uptrend)
    engine = StrategyEngine(config)
    verdict = engine.evaluate_side(ctx, Side.LONG)
    verdict.passed = True                       # force construction for the geometry test
    signal = engine.build_signal(ctx, verdict)
    assert signal is not None
    assert signal.stop < signal.entry < signal.tp1 < signal.tp2
    r = signal.stop_distance
    assert abs(signal.tp1 - signal.entry) == pytest.approx(r * config.strategy.tp1_r, rel=0.05)


def test_signal_levels_mirror_for_a_short(config, uptrend):
    ctx, _ = build(config, uptrend)
    engine = StrategyEngine(config)
    verdict = engine.evaluate_side(ctx, Side.SHORT)
    verdict.passed = True
    signal = engine.build_signal(ctx, verdict)
    assert signal is not None
    assert signal.stop > signal.entry > signal.tp1 > signal.tp2


def test_unknown_preset_is_rejected(config):
    config.strategy.name = "does-not-exist"
    with pytest.raises(KeyError):
        StrategyEngine(config)
