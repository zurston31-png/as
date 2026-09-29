from __future__ import annotations

import pytest
from conftest import build_context, make_candles

from tradebot.models import Side
from tradebot.strategy.engine import StrategyEngine


def test_context_exposes_config_driven_features(config, uptrend):
    ctx = build_context(config, uptrend)
    assert ctx is not None
    assert ctx.ema(9) is not None and ctx.ema(21) is not None
    assert ctx.ema(9) != ctx.ema(50)          # arbitrary periods, not a fixed pair
    assert ctx.rsi(14) is not None and ctx.atr(14) is not None and ctx.vwap() is not None


def test_features_are_computed_once_and_recorded(config, uptrend):
    ctx = build_context(config, uptrend)
    assert ctx.used_features() == {}           # nothing asked for yet
    ctx.ema(9); ctx.ema(9); ctx.rsi(14)
    used = ctx.used_features()
    assert "ema_9" in used and "rsi_14" in used


def test_uptrend_is_not_read_as_a_short(config, uptrend):
    ctx = build_context(config, uptrend)
    assert not StrategyEngine(config).evaluate_side(ctx, Side.SHORT).passed


def test_every_configured_rule_reports(loose_config, uptrend):
    ctx = build_context(loose_config, uptrend)
    verdict = StrategyEngine(loose_config).evaluate_side(ctx, Side.LONG)
    assert {c.name for c in verdict.checks} == {
        "ema_stack", "rsi_window", "volume_confirmation"}
    assert 0.0 <= verdict.score <= 1.0


def test_advisory_rules_do_not_block(loose_config, uptrend):
    ctx = build_context(loose_config, uptrend)
    verdict = StrategyEngine(loose_config).evaluate_side(ctx, Side.LONG)
    advisory = [c for c in verdict.checks if c.name == "volume_confirmation"]
    assert advisory and not advisory[0].required


def test_changing_the_rsi_window_in_config_changes_the_outcome(config, uptrend):
    """The whole point of config-driven rules: no code change required."""
    ctx = build_context(config, uptrend)
    rsi_now = ctx.rsi(14)

    def verdict_with(lo, hi):
        config.strategy = {"name": "t", "min_score": 0.0, "structure_window": 20, "rules": [
            {"rule": "rsi_window", "mode": "required",
             "params": {"period": 14, "long_min": lo, "long_max": hi}}]}
        return StrategyEngine(config).evaluate_side(ctx, Side.LONG)

    assert verdict_with(rsi_now - 5, rsi_now + 5).passed
    assert not verdict_with(rsi_now + 10, rsi_now + 20).passed


def test_cold_series_never_passes(config):
    ctx = build_context(config, make_candles([100, 101, 102]))
    assert not StrategyEngine(config).evaluate(ctx).passed


def test_out_of_session_blocks_the_setup(loose_config, uptrend):
    loose_config.market.trade_session_only = True
    loose_config.market.session_start = "03:00"
    loose_config.market.session_end = "03:30"
    ctx = build_context(loose_config, uptrend)
    assert not ctx.in_session
    assert not StrategyEngine(loose_config).evaluate_side(ctx, Side.LONG).passed


def test_a_stale_feed_blocks_the_setup(loose_config, uptrend):
    from tradebot.strategy.context import FeedHealth

    ctx = build_context(loose_config, uptrend,
                        feed=FeedHealth(stale=True, reason="nothing for 900s"))
    verdict = StrategyEngine(loose_config).evaluate_side(ctx, Side.LONG)
    assert not verdict.passed
    assert any("stale" in note for note in verdict.notes)


# ------------------------------------------------------------------ levels

def forced(engine, ctx, side):
    verdict = engine.evaluate_side(ctx, side)
    verdict.passed = True                      # force construction for geometry tests
    return engine.build_signal(ctx, verdict)


def test_long_levels_are_coherent(config, uptrend):
    ctx = build_context(config, uptrend)
    engine = StrategyEngine(config)
    signal = forced(engine, ctx, Side.LONG)
    assert signal.stop < signal.entry < signal.tp1 < signal.tp2
    r = signal.stop_distance
    assert abs(signal.tp1 - signal.entry) == pytest.approx(r * engine.spec.entry.tp1_r, rel=0.05)
    assert abs(signal.tp2 - signal.entry) == pytest.approx(r * engine.spec.entry.tp2_r, rel=0.05)


def test_short_levels_mirror(config, uptrend):
    ctx = build_context(config, uptrend)
    signal = forced(StrategyEngine(config), ctx, Side.SHORT)
    assert signal.stop > signal.entry > signal.tp1 > signal.tp2


def test_target_multiples_come_from_config(config, uptrend):
    config.strategy = {"preset": "ema_vwap_rsi", "entry": {"tp1_r": 1.0, "tp2_r": 10.0}}
    ctx = build_context(config, uptrend)
    signal = forced(StrategyEngine(config), ctx, Side.LONG)
    r = signal.stop_distance
    assert abs(signal.tp1 - signal.entry) == pytest.approx(r, rel=0.05)
    assert abs(signal.tp2 - signal.entry) == pytest.approx(r * 10, rel=0.05)


def test_fixed_tick_stops(config, uptrend):
    config.strategy = {"preset": "ema_vwap_rsi",
                       "entry": {"stop": {"method": "fixed_ticks", "fixed_ticks": 20}}}
    ctx = build_context(config, uptrend)
    signal = forced(StrategyEngine(config), ctx, Side.LONG)
    assert signal.stop_distance == pytest.approx(20 * config.market.tick_size, abs=0.26)


def test_signal_carries_its_decision_timestamp_and_trigger_bar(config, uptrend):
    ctx = build_context(config, uptrend)
    signal = forced(StrategyEngine(config), ctx, Side.LONG)
    assert signal.decision_ts == ctx.decision_ts
    assert signal.trigger_candle["close"] == ctx.candle.close


# -------------------------------------------------------- duplicate gate

def test_one_setup_produces_one_signal(loose_config, uptrend):
    """The conditions stay true for bars; only the first should signal."""
    engine = StrategyEngine(loose_config)
    ctx = build_context(loose_config, uptrend)
    verdict = engine.evaluate_side(ctx, Side.LONG)
    verdict.passed = True

    engine.observe(ctx, verdict)
    assert engine.admit(verdict, None)[0]
    engine.mark_fired(Side.LONG)

    for _ in range(5):
        engine.observe(ctx, verdict)           # setup still qualifying
        admitted, reason = engine.admit(verdict, None)
        assert not admitted and "duplicate" in reason
    assert engine.gate.suppressed == 5


def test_the_gate_rearms_once_the_setup_invalidates(loose_config, uptrend):
    engine = StrategyEngine(loose_config)
    ctx = build_context(loose_config, uptrend)
    passing = engine.evaluate_side(ctx, Side.LONG)
    passing.passed = True
    failing = engine.evaluate_side(ctx, Side.LONG)
    failing.passed = False

    engine.observe(ctx, passing)
    engine.mark_fired(Side.LONG)
    assert not engine.admit(passing, None)[0]

    engine.observe(ctx, failing)               # setup stops qualifying -> re-arm
    assert engine.admit(passing, None)[0]


def test_rearm_after_a_bar_count(loose_config, uptrend):
    loose_config.strategy = {**loose_config.strategy,
                             "signal": {"rearm_on_invalidation": False, "rearm_bars": 3}}
    engine = StrategyEngine(loose_config)
    ctx = build_context(loose_config, uptrend)
    verdict = engine.evaluate_side(ctx, Side.LONG)
    verdict.passed = True

    engine.observe(ctx, verdict)
    engine.mark_fired(Side.LONG)
    for _ in range(2):
        engine.observe(ctx, verdict)
        assert not engine.admit(verdict, None)[0]
    engine.observe(ctx, verdict)
    assert engine.admit(verdict, None)[0]


def test_an_open_position_blocks_a_new_signal(loose_config, uptrend):
    engine = StrategyEngine(loose_config)
    ctx = build_context(loose_config, uptrend)
    verdict = engine.evaluate_side(ctx, Side.LONG)
    verdict.passed = True
    assert not engine.admit(verdict, Side.LONG)[0]        # same direction
    assert not engine.admit(verdict, Side.SHORT)[0]       # reversals off by default


def test_reversals_can_be_enabled(loose_config, uptrend):
    loose_config.strategy = {**loose_config.strategy,
                             "signal": {"allow_reversal": True}}
    engine = StrategyEngine(loose_config)
    ctx = build_context(loose_config, uptrend)
    verdict = engine.evaluate_side(ctx, Side.LONG)
    verdict.passed = True
    assert engine.admit(verdict, Side.SHORT)[0]
    assert not engine.admit(verdict, Side.LONG)[0]        # still no stacking
