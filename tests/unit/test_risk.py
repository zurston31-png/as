"""Adversarial tests for the risk layer.

The tests are organized around the three ways this layer can be quietly
wrong, rather than around its public surface.

**Risk that is not the risk you configured.** Everything downstream divides by
`risk_dollars` to get R, so a sizing routine that overstates or understates it
corrupts every performance number in the project at once. The tests therefore
check the invariant `risk_dollars == size * |entry - stop| * point_value`
directly, on the ROUNDED prices, and check it again through
`TradeIntent._check_geometry`, which is the independent enforcement.

**Rounding that flatters.** Snapping prices to the tick grid can move a
trade's economics either way. Nearest-tick rounding would sometimes narrow a
stop and widen a target, which manufactures a sliver of edge on every trade
and compounds over a ten-year backtest. The tests pin the direction: stops
only ever get wider, targets only ever get smaller, and R after rounding is
never above R before it.

**Limits that report the wrong cause.** Section 12's rejection analysis counts
which gate fired, so a blocked bar attributed to `max_open_positions` when the
kill switch had already tripped misstates which constraint is shaping the
results. `test_the_check_order_is_the_documented_one` constructs a state where
SEVERAL limits are breached at once and requires the most fundamental one to
be the reported cause.

NQ is used throughout because its 0.25 tick and $20 point value make the
arithmetic exact: a 10-point stop is 40 ticks and costs $200 per contract, so
a $500 budget buys exactly 2 and leaves $100 unused. Every expected value
below is derived by hand from those two numbers.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from flow_model.config.schema import RiskConfig
from flow_model.core.contracts import TradeIntent
from flow_model.core.enums import InstrumentType, Regime, Side, SetupType, WaitReason
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel
from flow_model.risk.limits import LimitBreach, RiskLimitManager
from flow_model.risk.sizing import (
    SizingRejection,
    raw_size,
    risk_per_unit,
    round_to_tick,
    size_position,
    stop_distance_of,
)

UTC = timezone.utc
TS = datetime(2020, 1, 2, 15, 0, tzinfo=UTC)
D = date(2020, 1, 2)

#: Built locally rather than from `load_config()`, deliberately. The risk layer
#: is pure arithmetic and should be testable without the YAML: a test that
#: depends on defaults.yaml fails whenever an unrelated config section is being
#: edited, and its expected values silently change when a default moves.
#: Section 8's formulas are what is under test, not the shipped parameters.
TICK = 0.25
POINT_VALUE = 20.0
NQ = InstrumentSpec(
    symbol="NQ",
    instrument_type=InstrumentType.FUTURE,
    tick_size=TICK,
    tick_value=5.0,            # 5.0 / 0.25 = $20 per point
    typical_spread_ticks=1.0,
    min_spread_ticks=1.0,
)

#: The shipped defaults, written out so the tests below are reading a fixed
#: target. Mirrors RiskConfig's own defaults as of this commit.
BASE_RISK = dict(
    starting_equity=100_000.0,
    risk_per_trade_pct=0.005,
    max_risk_per_trade_pct=0.01,
    daily_loss_limit_pct=0.02,
    weekly_loss_limit_pct=0.05,
    max_consecutive_losses=4,
    cooldown_minutes_after_loss=0.0,
    cooldown_minutes_after_limit=1440.0,
    max_open_positions=1,
    max_portfolio_heat_pct=0.02,
    max_positions_per_symbol=1,
    kill_switch_drawdown_pct=0.15,
    kill_switch_is_terminal=True,
    compound_equity=True,
    allow_fractional_contracts=False,
)


def nq_size(**overrides):
    """A sized NQ long with a 10-point stop. 40 ticks, $200/contract."""
    fields = dict(
        spec=NQ, side=Side.LONG, entry_price=18000.0, stop_price=17990.0,
        reward_risk=2.0, equity=100_000.0, risk_pct=0.005, max_risk_pct=0.01,
        min_stop_ticks=6,
    )
    fields.update(overrides)
    return size_position(**fields)


#: The schema caps every percentage at 0.25 to catch the 50-vs-0.5 error, so a
#: test cannot "switch off" a limit with 0.99. These are the largest legal
#: values; the tests that use them keep their loss magnitudes well inside.
#: Constraints the schema enforces and these values respect: every percentage
#: <= 0.25, weekly >= daily, and kill_switch strictly > daily (otherwise the
#: daily limit could never bind). The tests using RELAXED keep their losses in
#: the hundreds or low thousands, far inside a -20,000 daily bound.
RELAXED = dict(
    kill_switch_drawdown_pct=0.25,
    daily_loss_limit_pct=0.20,
    weekly_loss_limit_pct=0.24,
)


def risk_config(**overrides) -> RiskConfig:
    fields = dict(BASE_RISK)
    fields.update(overrides)
    return RiskConfig(**fields)


def limits(**overrides):
    return RiskLimitManager(risk_config(**overrides))


# ---------------------------------------------------------------------------
# the instrument's own arithmetic, asserted before anything relies on it
# ---------------------------------------------------------------------------


def test_the_nq_premise_the_rest_of_this_file_rests_on():
    assert NQ.tick_size == TICK
    assert NQ.point_value == POINT_VALUE
    assert risk_per_unit(10.0, POINT_VALUE) == 200.0
    assert 10.0 / TICK == 40.0


# ---------------------------------------------------------------------------
# tick rounding
# ---------------------------------------------------------------------------


def test_round_to_tick_lands_exactly_on_the_grid():
    """Not approximately. `ticks * tick_size` can sit a few ULPs off a grid
    point for a tick size that is not binary-representable, and a downstream
    equality check on the grid then fails for no visible reason."""
    for raw in (18000.1, 18000.12, 17999.99, 0.26, 123.456789):
        for mode in ("nearest", "up", "down"):
            snapped = round_to_tick(raw, 0.01, mode=mode)
            assert abs(snapped / 0.01 - round(snapped / 0.01)) < 1e-6, (raw, mode, snapped)


def test_round_to_tick_directions_are_what_they_say():
    assert round_to_tick(18000.10, TICK, mode="down") == 18000.0
    assert round_to_tick(18000.10, TICK, mode="up") == 18000.25
    assert round_to_tick(18000.10, TICK, mode="nearest") == 18000.0
    assert round_to_tick(18000.20, TICK, mode="nearest") == 18000.25
    # exactly on the grid: every mode is a no-op
    for mode in ("nearest", "up", "down"):
        assert round_to_tick(18000.25, TICK, mode=mode) == 18000.25


def test_round_to_tick_handles_negatives_without_flipping_direction():
    """floor/ceil on a negative ratio is where naive implementations invert."""
    assert round_to_tick(-10.10, TICK, mode="down") == -10.25
    assert round_to_tick(-10.10, TICK, mode="up") == -10.0


def test_round_to_tick_rejects_a_nonsense_grid():
    for bad in (0.0, -0.25, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            round_to_tick(100.0, bad)
    with pytest.raises(ValueError):
        round_to_tick(float("nan"), TICK)
    with pytest.raises(ValueError, match="unknown rounding mode"):
        round_to_tick(100.0, TICK, mode="sideways")


# ---------------------------------------------------------------------------
# sizing: the hand-computed base case
# ---------------------------------------------------------------------------


def test_the_base_case_is_exact():
    """$100k at 0.5% = $500. A 10-point NQ stop costs $200/contract. floor(2.5)
    = 2 contracts, $400 of actual risk, $100 of the budget deliberately
    unused -- rounding UP to 3 would risk $600, which is over budget."""
    d = nq_size().require()
    assert d.size == 2.0
    assert d.risk_dollars == 400.0
    assert d.stop_distance == 10.0
    assert d.stop_ticks == 40.0
    assert d.target_price == 18020.0
    assert d.planned_r_multiple == 2.0
    assert d.requested_risk_dollars == 500.0
    assert d.risk_pct_of_equity == 0.004


def test_risk_dollars_is_recomputed_from_the_rounded_prices():
    """The invariant every R in the project divides by. Computed from the
    prices that will actually be traded, not the ones that were requested."""
    d = nq_size(entry_price=18000.07, stop_price=17990.03).require()
    assert d.risk_dollars == pytest.approx(d.size * d.stop_distance * POINT_VALUE)
    assert round_to_tick(d.entry_price, TICK) == d.entry_price
    assert round_to_tick(d.stop_price, TICK) == d.stop_price
    assert round_to_tick(d.target_price, TICK) == d.target_price


def test_the_sizing_output_builds_a_valid_trade_intent():
    """TradeIntent._check_geometry independently re-derives the risk identity
    and the side geometry, so a sizing bug that slipped past the assertions
    above still cannot reach the journal."""
    d = nq_size().require()
    intent = TradeIntent(
        symbol="NQ", signal_ts=TS, side=Side.LONG, setup=SetupType.SETUP_2R,
        entry_price=d.entry_price, stop_price=d.stop_price, target_price=d.target_price,
        size=d.size, risk_dollars=d.risk_dollars, point_value=POINT_VALUE,
        equity_at_signal=100_000.0, flow_score=80.0, regime=Regime.TRENDING_UP,
    )
    assert intent.planned_r_multiple == 2.0
    assert intent.risk_dollars == 400.0


# ---------------------------------------------------------------------------
# sizing: rounding may only ever make a trade worse
# ---------------------------------------------------------------------------


def test_the_stop_only_ever_moves_away_from_entry():
    """A long stop rounds DOWN and a short stop rounds UP, so the stop is never
    tightened by the grid. Nearest-tick rounding would narrow it roughly half
    the time, understating risk per unit and overstating size."""
    long_d = nq_size(stop_price=17990.10)
    assert long_d.stop_price == 17990.0
    assert long_d.stop_distance == 10.0 > abs(18000.0 - 17990.10) - 1e-9

    short_d = nq_size(side=Side.SHORT, entry_price=18000.0, stop_price=18009.90)
    assert short_d.stop_price == 18010.0
    assert short_d.stop_distance == 10.0


def test_the_target_only_ever_moves_toward_entry():
    """So reward is never overstated. With a 10.05 requested stop the long
    target lands below the un-rounded ideal, not above it."""
    d = nq_size(stop_price=17989.95, reward_risk=2.0).require()
    ideal = d.entry_price + d.stop_distance * 2.0
    assert d.target_price <= ideal
    assert d.planned_r_multiple <= 2.0

    s = nq_size(side=Side.SHORT, entry_price=18000.0, stop_price=18010.05,
                reward_risk=2.0, min_stop_ticks=6).require()
    ideal_short = s.entry_price - s.stop_distance * 2.0
    assert s.target_price >= ideal_short
    assert s.planned_r_multiple <= 2.0


@pytest.mark.parametrize("stop_offset", [9.9, 10.03, 10.07, 10.11, 12.37, 7.13])
@pytest.mark.parametrize("rr", [1.0, 2.0, 3.0])
def test_rounding_never_inflates_r_for_either_side(stop_offset, rr):
    """The property, swept. If any (stop, rr, side) combination produced a
    planned R above the requested one, the backtest would book a sliver of
    free edge on those trades."""
    for side, sign in ((Side.LONG, -1.0), (Side.SHORT, +1.0)):
        d = size_position(
            spec=NQ, side=side, entry_price=18000.0,
            stop_price=18000.0 + sign * stop_offset, reward_risk=rr,
            equity=1_000_000.0, risk_pct=0.005, max_risk_pct=0.01, min_stop_ticks=1,
        )
        if d.accepted:
            assert d.planned_r_multiple <= rr + 1e-9, (side, stop_offset, rr, d.planned_r_multiple)


def test_target_rounding_is_a_no_op_for_an_integer_reward_risk():
    """Worth pinning, because it bounds how much the conservative target
    rounding can ever cost.

    Entry and stop are both snapped to the grid, so `stop_distance` is always
    an exact integer number of ticks. For an integer `reward_risk` the ideal
    target is therefore also an exact number of ticks away and already on the
    grid, and rounding it inward changes nothing. All three configured setups
    use 1.0 / 2.0 / 3.0, so on the shipped configuration this rounding never
    reduces R at all.

    My first version of this test tried to force an unreachable target with an
    integer ratio on a coarse grid and could not -- the arithmetic above is why,
    and my expectation was wrong rather than the implementation.
    """
    for rr in (1.0, 2.0, 3.0):
        for stop_ticks in (4, 7, 13, 41):
            d = size_position(
                spec=NQ, side=Side.LONG, entry_price=18000.0,
                stop_price=18000.0 - stop_ticks * TICK, reward_risk=rr,
                equity=5_000_000.0, risk_pct=0.005, max_risk_pct=0.01, min_stop_ticks=1,
            ).require()
            assert d.planned_r_multiple == rr, (rr, stop_ticks, d.planned_r_multiple)
            assert not any("target snapped" in n for n in d.notes)


def test_a_fractional_reward_risk_is_where_the_grid_actually_bites():
    """3 ticks of stop at 1.5R wants 4.5 ticks of target, which does not exist.
    Rounded inward it becomes 4 ticks, so R = 4/3 = 1.333, not 1.5. The trade
    is REJECTED rather than relabelled: the failure mode
    TradeRecord.r_label_is_honest exists to catch is a setup whose realizable
    target does not match its name."""
    d = size_position(
        spec=NQ, side=Side.LONG, entry_price=18000.0, stop_price=18000.0 - 3 * TICK,
        reward_risk=1.5, equity=5_000_000.0, risk_pct=0.005, max_risk_pct=0.01,
        min_stop_ticks=1, min_reward_risk=1.4,
    )
    assert not d.accepted
    assert d.reason == SizingRejection.REWARD_RISK_BELOW_MINIMUM
    assert d.planned_r_multiple == pytest.approx(4.0 / 3.0, abs=1e-6)
    assert "not that setup" in " ".join(d.notes)

    # Same geometry, an honest minimum: accepted, and R reports the realizable
    # 1.333 rather than the requested 1.5.
    ok = size_position(
        spec=NQ, side=Side.LONG, entry_price=18000.0, stop_price=18000.0 - 3 * TICK,
        reward_risk=1.5, equity=5_000_000.0, risk_pct=0.005, max_risk_pct=0.01,
        min_stop_ticks=1, min_reward_risk=1.3,
    ).require()
    assert ok.planned_r_multiple == pytest.approx(4.0 / 3.0, abs=1e-6)
    assert any("target snapped" in n for n in ok.notes)


def test_the_configured_setups_all_use_integer_ratios():
    """So the no-op result above applies to the shipped configuration. If a
    future setup uses a fractional ratio this fails, which is the reminder
    that target rounding starts to matter."""
    from flow_model.config.loader import load_config as _load

    for name, setup in _load().setups.items():
        assert float(setup.reward_risk).is_integer(), (name, setup.reward_risk)


# ---------------------------------------------------------------------------
# sizing: every rejection path
# ---------------------------------------------------------------------------


def test_a_stop_inside_the_noise_floor_is_rejected():
    d = nq_size(stop_price=17999.50, min_stop_ticks=6)   # 2 ticks
    assert not d.accepted
    assert d.reason == SizingRejection.STOP_TOO_TIGHT
    assert d.stop_ticks == 2.0


def test_a_stop_wider_than_the_setup_cap_is_rejected():
    d = nq_size(max_stop_distance=8.0)
    assert not d.accepted
    assert d.reason == SizingRejection.STOP_TOO_WIDE


def test_size_zero_is_a_refusal_with_the_arithmetic_in_the_note():
    """A $40 budget cannot buy one contract at $200 of risk. The note has to
    say why, because 'rejected' alone sends someone hunting a bug."""
    d = nq_size(equity=8_000.0, risk_pct=0.005)
    assert not d.accepted
    assert d.reason == SizingRejection.SIZE_ZERO
    assert d.size == 0.0
    assert "not even one unit fits" in " ".join(d.notes)


def test_risk_pct_above_the_hard_cap_is_capped_and_recorded():
    """Section 8: the manager cannot be configured to exceed
    max_risk_per_trade_pct. Capping rather than raising keeps a ten-year
    backtest alive, but trading a size other than the configured one without
    saying so would be its own dishonesty."""
    d = nq_size(risk_pct=0.05, max_risk_pct=0.01).require()
    assert d.requested_risk_dollars == 1000.0        # 1% cap, not the 5% asked
    assert d.size == 5.0                             # floor(1000/200)
    assert any("capped" in n for n in d.notes)


def test_actual_risk_never_exceeds_the_cap_across_a_sweep():
    """The invariant the whole risk layer promises. Swept rather than argued,
    including the fractional path where floored integer sizing no longer
    guarantees it for free."""
    for equity in (5_000.0, 25_000.0, 100_000.0, 1_000_000.0):
        for stop in (1.0, 3.25, 10.0, 47.5):
            for fractional in (False, True):
                d = size_position(
                    spec=NQ, side=Side.LONG, entry_price=18000.0,
                    stop_price=18000.0 - stop, reward_risk=1.0, equity=equity,
                    risk_pct=0.01, max_risk_pct=0.01, min_stop_ticks=1,
                    allow_fractional=fractional,
                )
                if d.accepted:
                    assert d.risk_dollars <= equity * 0.01 * (1 + 1e-9), (equity, stop, fractional)


def test_degenerate_and_non_finite_inputs_are_refused_not_raised():
    """A backtest must not die on bar 40,000 of 100,000 because one bar
    produced a NaN; it must record a refusal and continue."""
    assert nq_size(stop_price=18000.0).reason == SizingRejection.DEGENERATE_GEOMETRY
    assert nq_size(stop_price=18010.0).reason == SizingRejection.DEGENERATE_GEOMETRY  # long, stop above
    assert nq_size(side=Side.SHORT, stop_price=17990.0).reason == SizingRejection.DEGENERATE_GEOMETRY
    assert nq_size(equity=0.0).reason == SizingRejection.EQUITY_NON_POSITIVE
    assert nq_size(equity=-1.0).reason == SizingRejection.EQUITY_NON_POSITIVE
    assert nq_size(reward_risk=0.0).reason == SizingRejection.DEGENERATE_GEOMETRY
    for bad in (float("nan"), float("inf")):
        assert nq_size(entry_price=bad).reason == SizingRejection.NON_FINITE_INPUT
        assert nq_size(stop_price=bad).reason == SizingRejection.NON_FINITE_INPUT
        assert nq_size(equity=bad).reason == SizingRejection.NON_FINITE_INPUT


def test_require_raises_only_on_a_refusal():
    nq_size().require()
    with pytest.raises(ValueError, match="sizing was rejected"):
        nq_size(equity=100.0).require()


def test_fractional_sizing_is_opt_in():
    integer = nq_size(equity=100_000.0).require()
    frac = nq_size(equity=100_000.0, allow_fractional=True).require()
    assert integer.size == 2.0
    assert frac.size == 2.5
    assert frac.risk_dollars == 500.0


def test_raw_size_floors_and_guards_a_zero_divisor():
    assert raw_size(500.0, 200.0, allow_fractional=False) == 2.0
    assert raw_size(500.0, 200.0, allow_fractional=True) == 2.5
    assert raw_size(500.0, 0.0, allow_fractional=False) == 0.0
    assert raw_size(500.0, -5.0, allow_fractional=True) == 0.0


def test_stop_distance_is_symmetric():
    assert stop_distance_of(100.0, 90.0) == stop_distance_of(90.0, 100.0) == 10.0


def test_sizing_is_deterministic_and_the_decision_is_frozen():
    a, b = nq_size(), nq_size()
    assert a.model_dump() == b.model_dump()
    assert isinstance(a, FrozenModel)
    with pytest.raises(Exception):
        a.size = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# limits: the order of the checks
# ---------------------------------------------------------------------------


def test_the_check_order_is_the_documented_one():
    """Section 8 fixes the order, and section 12 counts which gate fired, so a
    bar blocked by several limits at once must report the most fundamental.
    Here the kill switch, the daily limit, the streak limit and the position
    limit are ALL breached; the reason must be the kill switch."""
    m = limits()
    for _ in range(6):
        m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-5_000.0)
    m.on_position_opened("NQ", 500.0)
    d = m.check(ts=TS, session_date=D, equity=70_000.0, symbol="NQ", prospective_risk=500.0)
    assert d.breach == LimitBreach.KILL_SWITCH
    assert d.wait_reason is WaitReason.RISK_LIMIT
    assert d.terminal is True


def test_without_the_kill_switch_the_daily_limit_outranks_the_streak_limit():
    """Peeling the order one layer at a time, so the ordering test above is not
    just asserting that the first branch happens to be first."""
    m = limits(kill_switch_drawdown_pct=0.25)
    for _ in range(6):
        m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-1_000.0)
    # -6000 realized is past the 2% (-2000) daily limit but only 6% of equity,
    # so the 25% kill switch is genuinely clear rather than switched off.
    d = m.check(ts=TS, session_date=D, equity=94_000.0, symbol="NQ")
    assert d.breach == LimitBreach.DAILY_LOSS_LIMIT


def test_with_both_loss_limits_clear_the_streak_limit_reports():
    m = limits(**RELAXED)
    for _ in range(4):
        m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-10.0)
    d = m.check(ts=TS, session_date=D, equity=99_960.0, symbol="NQ")
    assert d.breach == LimitBreach.MAX_CONSECUTIVE_LOSSES


# ---------------------------------------------------------------------------
# limits: each limit on its own
# ---------------------------------------------------------------------------


def test_the_kill_switch_measures_drawdown_from_the_peak_not_the_start():
    """An account up 40% and then down 15% from that peak has lost real money
    even though it is still ahead of where it began."""
    m = limits()
    m.mark_equity(140_000.0)
    assert m.high_water_mark == 140_000.0
    d = m.check(ts=TS, session_date=D, equity=119_000.0, symbol="NQ")   # -15% from peak
    assert d.breach == LimitBreach.KILL_SWITCH
    assert d.allowed is False


def test_the_kill_switch_is_terminal_and_stays_tripped():
    m = limits()
    m.check(ts=TS, session_date=D, equity=84_000.0, symbol="NQ")
    assert m.killed
    # recovery does not revive it
    later = m.check(ts=TS + timedelta(days=30), session_date=date(2020, 2, 1),
                    equity=200_000.0, symbol="NQ")
    assert later.allowed is False
    assert later.breach == LimitBreach.KILL_SWITCH
    assert later.terminal is True


def test_a_non_terminal_kill_switch_releases_when_equity_recovers():
    m = limits(kill_switch_is_terminal=False)
    assert m.check(ts=TS, session_date=D, equity=84_000.0, symbol="NQ").allowed is False
    assert m.killed is False
    assert m.check(ts=TS, session_date=D, equity=99_000.0, symbol="NQ").allowed is True


def test_the_daily_limit_is_measured_against_starting_equity():
    """A limit that shrinks with the account lets a bad run continue at
    ever-smaller size instead of stopping, which is the opposite of a circuit
    breaker."""
    m = limits(kill_switch_drawdown_pct=0.25)
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-2_000.0)   # exactly 2%
    d = m.check(ts=TS, session_date=D, equity=98_000.0, symbol="NQ")
    assert d.breach == LimitBreach.DAILY_LOSS_LIMIT


def test_the_daily_limit_resets_on_the_next_session_not_the_next_utc_day():
    """Keyed on the trading session the caller supplies. A UTC-date key would
    reset the limit mid-session for any instrument whose session spans
    midnight UTC -- it would stop limiting on exactly the overnight sessions
    where it matters most."""
    m = limits(kill_switch_drawdown_pct=0.25)
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-2_000.0)
    assert m.check(ts=TS, session_date=D, equity=98_000.0,
                   symbol="NQ").breach == LimitBreach.DAILY_LOSS_LIMIT

    # Same wall-clock instant, next SESSION. The daily bucket is empty again,
    # and the block is now the post-limit cooldown rather than the daily limit
    # -- which is how we know the daily limit itself reset. My first version of
    # this test expected the next session to be tradable outright and was
    # wrong: cooldown_minutes_after_limit defaults to 1440, so the lockout
    # deliberately outlives the session boundary.
    nxt = date(2020, 1, 3)
    assert m.realized_for_session(nxt) == 0.0
    assert m.check(ts=TS, session_date=nxt, equity=98_000.0,
                   symbol="NQ").breach == LimitBreach.COOLDOWN_ACTIVE


def test_with_no_post_limit_lockout_the_next_session_is_tradable():
    """The daily limit on its own is per-session; it is the separate
    cooldown_minutes_after_limit that carries a lockout across sessions."""
    m = limits(kill_switch_drawdown_pct=0.25, cooldown_minutes_after_limit=0.0)
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-2_000.0)
    assert m.check(ts=TS, session_date=D, equity=98_000.0, symbol="NQ").allowed is False
    assert m.check(ts=TS, session_date=date(2020, 1, 3), equity=98_000.0,
                   symbol="NQ").allowed is True


def test_the_weekly_limit_spans_sessions_within_one_iso_week():
    m = limits(kill_switch_drawdown_pct=0.25, daily_loss_limit_pct=0.03,
               weekly_loss_limit_pct=0.05)
    for day in (date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)):
        m.on_trade_closed(ts=TS, session_date=day, symbol="NQ", pnl=-2_000.0)
    assert m.realized_for_week(date(2020, 1, 9)) == -6_000.0
    # -6000 is past the 5% (-5000) weekly limit, inside the relaxed daily one.
    d = m.check(ts=TS, session_date=date(2020, 1, 9), equity=94_000.0, symbol="NQ")
    assert d.breach == LimitBreach.WEEKLY_LOSS_LIMIT
    # the following ISO week is a fresh bucket
    assert m.realized_for_week(date(2020, 1, 13)) == 0.0


def test_a_win_resets_the_loss_streak_and_a_scratch_does_not_extend_it():
    """A run of break-even trades is not a losing streak. Counting a zero-PnL
    scratch as a loss would trip the limit on trades that lost nothing."""
    m = limits(**RELAXED)
    for _ in range(3):
        m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-10.0)
    assert m.consecutive_losses == 3
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=0.0)
    assert m.consecutive_losses == 3
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=5.0)
    assert m.consecutive_losses == 0


def test_cooldown_after_a_loss_blocks_then_expires():
    m = limits(cooldown_minutes_after_loss=30.0, **RELAXED)
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-10.0)
    assert m.check(ts=TS + timedelta(minutes=29), session_date=D,
                   equity=99_990.0, symbol="NQ").breach == LimitBreach.COOLDOWN_ACTIVE
    assert m.check(ts=TS + timedelta(minutes=30), session_date=D,
                   equity=99_990.0, symbol="NQ").allowed is True


def test_a_longer_cooldown_is_never_shortened_by_a_later_shorter_one():
    """Otherwise a small loss during the post-limit lockout would release it."""
    m = limits(cooldown_minutes_after_loss=5.0, **RELAXED)
    m._arm_cooldown(TS, 600.0)
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=-10.0)
    assert m.check(ts=TS + timedelta(minutes=10), session_date=D,
                   equity=99_990.0, symbol="NQ").breach == LimitBreach.COOLDOWN_ACTIVE


def test_position_count_limits_fire_and_clear():
    m = limits(max_open_positions=2, max_positions_per_symbol=1, **RELAXED)
    m.on_position_opened("NQ", 100.0)
    assert m.check(ts=TS, session_date=D, equity=100_000.0,
                   symbol="NQ").breach == LimitBreach.MAX_POSITIONS_PER_SYMBOL
    assert m.check(ts=TS, session_date=D, equity=100_000.0, symbol="ES").allowed is True
    m.on_position_opened("ES", 100.0)
    assert m.check(ts=TS, session_date=D, equity=100_000.0,
                   symbol="GC").breach == LimitBreach.MAX_OPEN_POSITIONS
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=1.0, risk_dollars=100.0)
    assert m.open_positions() == 1
    assert m.check(ts=TS, session_date=D, equity=100_001.0, symbol="NQ").allowed is True


def test_portfolio_heat_counts_the_trade_being_proposed():
    """Checking heat without the prospective risk would admit a position that
    breaches the cap the instant it opens."""
    m = limits(max_open_positions=9, max_positions_per_symbol=9,
               max_portfolio_heat_pct=0.02, **RELAXED)
    m.on_position_opened("NQ", 1_500.0)
    assert m.portfolio_heat() == 1_500.0
    assert m.check(ts=TS, session_date=D, equity=100_000.0, symbol="ES",
                   prospective_risk=400.0).allowed is True       # 1900 <= 2000
    d = m.check(ts=TS, session_date=D, equity=100_000.0, symbol="ES", prospective_risk=600.0)
    assert d.breach == LimitBreach.MAX_PORTFOLIO_HEAT           # 2100 > 2000


def test_heat_returns_to_zero_after_the_last_exit():
    """Float residue that never clears slowly blocks all trading, which looks
    exactly like a strategy that stopped finding setups."""
    m = limits(**RELAXED)
    m.on_position_opened("NQ", 333.3333333)
    m.on_trade_closed(ts=TS, session_date=D, symbol="NQ", pnl=1.0, risk_dollars=333.3333333)
    assert m.portfolio_heat() == 0.0
    assert m.open_positions() == 0


# ---------------------------------------------------------------------------
# limits: construction, determinism, diagnostics
# ---------------------------------------------------------------------------


def test_a_config_that_exceeds_its_own_hard_cap_is_refused_at_construction():
    """Section 8: all limits are hard. A config stating an impossible
    intention should fail loudly once, not be silently clamped every bar."""
    # RiskConfig's own validator refuses this pair outright, which is the
    # first line of defence; build it past that with model_construct so the
    # manager's duplicate guard is the thing actually under test.
    bad = RiskConfig.model_construct(**{**BASE_RISK, "risk_per_trade_pct": 0.02,
                                        "max_risk_per_trade_pct": 0.01})
    with pytest.raises(ValueError, match="exceeds"):
        RiskLimitManager(bad)


def test_non_positive_starting_equity_is_refused():
    with pytest.raises(ValueError, match="positive"):
        RiskLimitManager(risk_config(), starting_equity=0.0)


def test_the_manager_never_reads_the_wall_clock():
    """It is driven by the backtest clock, so a replay must give identical
    decisions; a manager that consults datetime.now() cannot be backtested at
    all.

    Matched on the AST rather than on the source text. The text version of
    this check failed on the module's own docstring, which names
    datetime.now() while explaining that it is never called -- the same false
    positive that the identifier-vs-prose leakage guard hit. A guard that
    punishes accurate documentation teaches people to delete it.
    """
    import ast
    import inspect

    from flow_model.risk import limits as module

    FORBIDDEN = {"now", "utcnow", "today", "time", "monotonic", "perf_counter"}
    offenders = []
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN:
                offenders.append(f"line {node.lineno}: .{node.func.attr}()")
    assert not offenders, (
        "the limit manager must take the clock as an argument, not read it: "
        + "; ".join(offenders)
    )


def test_the_wall_clock_guard_would_catch_a_real_call():
    """Without this the AST check could be vacuous."""
    import ast

    tree = ast.parse("import datetime\nx = datetime.datetime.now()\n")
    hits = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "now"
    ]
    assert len(hits) == 1


def test_trip_counts_feed_rejection_analysis():
    m = limits(max_open_positions=1, **RELAXED)
    m.on_position_opened("NQ", 10.0)
    for _ in range(3):
        m.check(ts=TS, session_date=D, equity=100_000.0, symbol="ES")
    assert m.trip_counts()[LimitBreach.MAX_OPEN_POSITIONS] == 3


def test_headroom_is_reported_on_both_outcomes():
    """So a blocked bar says how far past the line it was, and an allowed bar
    says how much slack is left."""
    m = limits()
    ok = m.check(ts=TS, session_date=D, equity=100_000.0, symbol="NQ", prospective_risk=100.0)
    assert ok.allowed and ok.headroom["portfolio_heat_limit"] == 2_000.0
    blocked = m.check(ts=TS, session_date=D, equity=50_000.0, symbol="NQ")
    assert not blocked.allowed
    assert blocked.headroom["drawdown_pct"] == pytest.approx(0.5)


def test_two_identically_driven_managers_agree():
    """No hidden state and no clock reading, so the same event sequence must
    produce the same decisions."""
    def drive():
        m = limits(**RELAXED)
        out = []
        for i in range(10):
            d = date(2020, 1, 2 + i)
            m.on_trade_closed(ts=TS + timedelta(days=i), session_date=d,
                              symbol="NQ", pnl=-50.0 if i % 3 else 120.0)
            out.append(m.check(ts=TS + timedelta(days=i), session_date=d,
                               equity=100_000.0 - 20 * i, symbol="NQ").model_dump())
        return out

    assert drive() == drive()
