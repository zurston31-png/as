"""Smoke check for `backtest/costs.py` and `backtest/execution.py`.

Deliberately NOT the exhaustive suite -- a dedicated test agent follows.
What this file establishes is that the modules run, that their declared
contracts hold, and that each of section 9's directional rules actually
resolves against the trade rather than for it.

Everything here is a CONSTRUCTED `TradeIntent` and hand-built bars. It is not
a backtest with its fills inspected, and that is the point rather than a
shortcut: the committed Phase 6 finding (commit f952d3e) is that the system
takes ZERO trades over 2500 bars, because the highest Flow Score observed is
65.23 against a lowest setup threshold of 70.0. A cost model validated on
whatever trades happen to occur is validated on an arbitrary sample, and here
there is no sample at all. Lowering a threshold to manufacture one is exactly
the move the brief forbids, so the gap and same-bar cases are built directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flow_model.backtest.costs import (
    CostModel,
    CostModelError,
    CostRefusal,
    SpreadSource,
    VolatilityContext,
    combine_costs,
    vol_slippage_ticks,
)
from flow_model.backtest.execution import ExecutionModel
from flow_model.backtest.fill_timing import (
    ExecutionError,
    FillTimingError,
    FillWindow,
    min_fill_offset,
)
from flow_model.backtest.fills import (
    EntryFill,
    ExitPlan,
    FillStatus,
    MissReason,
    resolve_bar_exit,
)
from flow_model.config.loader import default_config
from flow_model.core.contracts import Bar, QuoteSnapshot, TradeIntent
from flow_model.core.enums import ExitReason, Regime, SetupType, Side

INTERVAL = 300
T0 = datetime(2021, 3, 2, 15, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def bar(
    *,
    offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 10_000.0,
    symbol: str = "NQ",
    interval: int = INTERVAL,
) -> Bar:
    """One hand-built bar, `offset` intervals after T0."""
    return Bar(
        symbol=symbol,
        close_ts=T0 + timedelta(seconds=interval * offset),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        interval_seconds=interval,
    )


def intent(
    *,
    side: Side = Side.LONG,
    entry: float = 13000.0,
    stop: float = 12980.0,
    target: float = 13020.0,
    size: float = 2.0,
    point_value: float = 20.0,
    signal_ts: datetime = T0,
) -> TradeIntent:
    return TradeIntent(
        symbol="NQ",
        signal_ts=signal_ts,
        side=side,
        setup=SetupType.SETUP_2R,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        size=size,
        risk_dollars=abs(entry - stop) * point_value * size,
        point_value=point_value,
        equity_at_signal=100_000.0,
        flow_score=71.0,
        regime=Regime.TRENDING_UP,
    )


@pytest.fixture
def execution(config):
    return config.execution


@pytest.fixture
def model(execution, nq) -> ExecutionModel:
    return ExecutionModel(execution, nq, master_seed=20240101)


@pytest.fixture
def measured() -> VolatilityContext:
    """A measured volatility ratio: ATR 25% above its own median."""
    return VolatilityContext(atr=12.5, median_atr=10.0)


# ---------------------------------------------------------------------------
# The eligibility floor: fills resolve on later bars only
# ---------------------------------------------------------------------------


def test_the_floor_is_one_bar_even_at_zero_latency():
    """Section 9's loop invariant, as arithmetic rather than a convention."""
    assert min_fill_offset(latency_seconds=0.0, interval_seconds=INTERVAL) == 1
    assert min_fill_offset(latency_seconds=0.25, interval_seconds=INTERVAL) == 1


def test_latency_longer_than_a_bar_pushes_the_fill_out():
    """"latency_ms shifts eligible fill time" -- 400s of latency skips a bar."""
    assert min_fill_offset(latency_seconds=400.0, interval_seconds=INTERVAL) == 2
    assert min_fill_offset(latency_seconds=700.0, interval_seconds=INTERVAL) == 3


def test_entry_delay_bars_adds_on_top():
    assert min_fill_offset(latency_seconds=0.25, interval_seconds=INTERVAL, entry_delay_bars=2) == 3


def test_negative_latency_is_refused():
    with pytest.raises(FillTimingError):
        min_fill_offset(latency_seconds=-1.0, interval_seconds=INTERVAL)


def test_the_signal_bar_is_absent_from_the_window_not_merely_unrequested(model):
    """The `MarketView` pattern: a careless caller cannot reach the signal bar."""
    order = model.submit(intent(), T0)
    signal_bar = bar(offset=0, open_=12995.0, high=13005.0, low=12990.0, close=13000.0)
    next_bar = bar(offset=1, open_=13000.0, high=13010.0, low=12998.0, close=13008.0)
    window = FillWindow(order, [signal_bar, next_bar])

    assert len(window) == 1
    assert window.bars == (next_bar,)
    assert signal_bar not in window.bars
    # No attribute anywhere on the object still holds it.
    held = []
    for slot in FillWindow.__slots__:
        value = getattr(window, slot)
        if isinstance(value, tuple):
            held.extend(v for v in value if isinstance(v, Bar))
    assert signal_bar not in held
    window.assert_resolves_after_signal()


#: Copied verbatim from `test_market_view.py`, which enforces the same rule on
#: `MarketView`. The reason is identical: a caller that can name an instant can
#: name one it should not reach.
TIMESTAMP_FLAVOURED = {
    "ts", "timestamp", "when", "at", "start", "end", "until", "before", "after", "date", "time",
}


def _public_window_methods() -> list[tuple[str, object]]:
    import inspect

    return [
        (name, member)
        for name, member in inspect.getmembers(FillWindow, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]


def test_no_window_accessor_takes_a_timestamp():
    """A future accessor that breaks the rule fails the build, as on `MarketView`."""
    import inspect

    methods = _public_window_methods()
    assert methods, "the introspection found no methods, so it proves nothing"
    offenders = {}
    for name, member in methods:
        bad = (set(inspect.signature(member).parameters) - {"self"}) & TIMESTAMP_FLAVOURED
        if bad:
            offenders[name] = sorted(bad)
    assert offenders == {}, (
        f"fill-window accessors taking a timestamp-flavoured parameter: {offenders}. "
        "The window only permits 'the candidates, in order'."
    )


def test_no_window_accessor_is_named_like_a_timestamp_query():
    offenders = [
        name
        for name, _ in _public_window_methods()
        if "between" in name or name.endswith("_at") or name.startswith("value_")
    ]
    assert offenders == []


def test_a_fill_on_the_signal_bar_is_rejected_by_the_result_contract(model):
    """Even if a window were bypassed, the fill record itself refuses."""
    with pytest.raises(FillTimingError):
        EntryFill(
            status=FillStatus.FILLED,
            symbol="NQ",
            side=Side.LONG,
            signal_ts=T0,
            submitted_ts=T0,
            eligible_ts=T0,
            order_type="market",
            fill_ts=T0,
            fill_bar_offset=0,
            reference_price=13000.0,
            planned_price=13000.0,
            requested_size=1.0,
            filled_size=1.0,
        )


def test_submitting_before_the_signal_raises(model):
    with pytest.raises(FillTimingError):
        model.submit(intent(), T0 - timedelta(seconds=1))


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def test_a_market_entry_fills_at_the_next_bars_open(model, measured):
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)

    assert fill.status is FillStatus.FILLED
    assert fill.reference_price == 13002.0
    assert fill.fill_bar_offset == 1
    assert fill.fill_ts == T0 + timedelta(seconds=INTERVAL)
    # The recorded price is off the price path; the cost is charged in dollars.
    assert fill.effective_price > fill.reference_price
    assert fill.costs.total > 0.0


def test_the_eligibility_lag_is_recorded_rather_than_asserted_negligible(model, measured):
    """The one residual bar data cannot remove, reported per fill."""
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)
    assert fill.eligibility_lag_seconds == pytest.approx(0.25)


def test_a_fill_that_gapped_past_the_stop_is_declined_not_filled(model, measured):
    """The inverted-geometry trap: it would 'stop out' at a PROFIT.

    A LONG planned 13000/12980/13020 whose fill lands at 12970 has its stop
    ABOVE its entry. Exiting at the stop books +10 points. That is a fictional
    win manufactured by a gap, so the entry is refused.
    """
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=12970.0, high=12975.0, low=12960.0, close=12968.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.FILL_BEYOND_STOP
    assert fill.filled_size == 0.0
    assert fill.costs.total == 0.0


def test_a_fill_that_gapped_past_the_target_is_declined(model, measured):
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13030.0, high=13040.0, low=13025.0, close=13035.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)
    assert fill.miss_reason == MissReason.FILL_BEYOND_TARGET


def test_a_thin_bar_partially_fills_and_is_not_rounded_up(model, measured):
    """`min(size, participation_rate * volume)` with a 5% default cap."""
    order = model.submit(intent(size=4.0), T0)
    # 5% of 60 = 3 contracts against a request of 4.
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0, volume=60.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)

    assert fill.status is FillStatus.PARTIAL
    assert fill.filled_size == 3.0
    assert fill.fill_fraction == pytest.approx(0.75)
    # Costs are charged on what filled, not on what was asked for.
    commission, fees = model.costs.commission_and_fees(3.0, sides=1)
    assert fill.costs.commission == commission
    assert fill.costs.exchange_fees == fees


def test_a_fill_below_min_partial_fraction_abandons_the_entry(model, measured):
    order = model.submit(intent(size=10.0), T0)
    # 5% of 60 = 3 of 10 = 0.3, below min_partial_fraction 0.5.
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0, volume=60.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)
    assert fill.miss_reason == MissReason.PARTICIPATION_BELOW_MINIMUM


def test_a_bar_with_no_measured_volume_declines_the_fill(model, measured):
    """No volume means no participation limit. The fill is declined, not freed."""
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0, volume=0.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)
    assert fill.miss_reason == MissReason.NO_VOLUME_MEASUREMENT


def test_an_unmeasured_volatility_ratio_declines_the_fill(model):
    """The `signals/engine.py` mistake, not repeated: no cheap substitute."""
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)]
    )
    fill = model.resolve_entry(order, window, vol=VolatilityContext.unmeasured("no median_atr"))

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.COST_UNMEASURABLE
    assert any(CostRefusal.VOL_MULTIPLE_UNMEASURABLE in note for note in fill.notes)
    assert fill.costs.total == 0.0


def test_an_empty_window_misses_rather_than_filling_somewhere_else(model, measured):
    order = model.submit(intent(), T0)
    fill = model.resolve_entry(order, FillWindow(order, []), vol=measured)
    assert fill.miss_reason == MissReason.NO_ELIGIBLE_BAR


# ---------------------------------------------------------------------------
# Exits: the four asymmetries
# ---------------------------------------------------------------------------


@pytest.fixture
def plan() -> ExitPlan:
    return ExitPlan.from_intent(
        intent(),
        entry_ts=T0 + timedelta(seconds=INTERVAL),
        entry_price=13000.0,
        size=2.0,
    )


def test_a_bar_that_gaps_through_the_stop_fills_at_the_open(plan):
    """Section 9's "single most common backtest overstatement"."""
    gap = bar(offset=2, open_=12950.0, high=12960.0, low=12940.0, close=12955.0)
    resolution = resolve_bar_exit(plan, gap)

    assert resolution is not None
    assert resolution.reason is ExitReason.STOP
    assert resolution.price == 12950.0  # the open, 30 points worse than the stop
    assert resolution.gapped is True
    assert resolution.ambiguous is False  # the open is chronologically first


def test_a_bar_that_gaps_through_the_target_fills_at_the_target_not_the_open(plan):
    """The same rule in the trade's favour, deliberately NOT taken."""
    gap = bar(offset=2, open_=13060.0, high=13080.0, low=13055.0, close=13070.0)
    resolution = resolve_bar_exit(plan, gap)

    assert resolution is not None
    assert resolution.reason is ExitReason.TARGET
    assert resolution.price == 13020.0  # the target, not the 13060 open
    assert resolution.gapped is True


def test_a_range_holding_both_levels_resolves_to_the_stop_and_is_tagged(plan):
    both = bar(offset=2, open_=13000.0, high=13025.0, low=12975.0, close=13010.0)
    resolution = resolve_bar_exit(plan, both)

    assert resolution.reason is ExitReason.STOP
    assert resolution.price == 12980.0
    assert resolution.ambiguous is True


def test_the_optimistic_same_bar_setting_is_still_tagged_ambiguous(plan):
    """The tag records the missing observation, not the setting.

    Untagging it under `pessimistic_same_bar=False` would make the
    sensitivity of a whole run to that assumption unmeasurable, which is the
    opposite of what section 9 asks the tag for.
    """
    both = bar(offset=2, open_=13000.0, high=13025.0, low=12975.0, close=13010.0)
    resolution = resolve_bar_exit(plan, both, pessimistic_same_bar=False)
    assert resolution.reason is ExitReason.TARGET
    assert resolution.ambiguous is True


def test_a_stop_triggers_on_a_touch_and_a_target_needs_a_trade_through(plan):
    """Inclusive for the stop, strict for the target: queue position is unseen."""
    touch_stop = bar(offset=2, open_=13000.0, high=13005.0, low=12980.0, close=12990.0)
    assert resolve_bar_exit(plan, touch_stop).reason is ExitReason.STOP

    touch_target = bar(offset=2, open_=13000.0, high=13020.0, low=12995.0, close=13015.0)
    assert resolve_bar_exit(plan, touch_target) is None


def test_a_bar_reaching_neither_level_leaves_the_position_open(model, plan, measured):
    quiet = bar(offset=2, open_=13000.0, high=13010.0, low=12995.0, close=13005.0)
    fill = model.resolve_exit(plan, quiet, vol=measured)

    assert fill.resolved is False
    assert fill.exit_reason is None
    assert fill.costs.total == 0.0


def test_finer_bars_resolve_the_ambiguity_by_observation(model, plan, measured):
    """Section 9: "Intrabar resolution uses finer-granularity bars when available"."""
    parent = bar(offset=2, open_=13000.0, high=13025.0, low=12975.0, close=13010.0)
    base = T0 + timedelta(seconds=INTERVAL)
    finer = [
        Bar(
            symbol="NQ",
            close_ts=base + timedelta(seconds=60 * i),
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=2_000.0,
            interval_seconds=60,
        )
        for i, (o, h, lo, c) in enumerate(
            [
                (13000.0, 13025.0, 12999.0, 13022.0),  # target first
                (13022.0, 13024.0, 13000.0, 13005.0),
                (13005.0, 13010.0, 12975.0, 12980.0),  # stop later
                (12980.0, 12985.0, 12976.0, 12982.0),
                (12982.0, 13000.0, 12980.0, 12995.0),
            ],
            start=1,
        )
    ]
    fill = model.resolve_exit(plan, parent, vol=measured, intrabar=finer)

    assert fill.resolved is True
    assert fill.exit_reason is ExitReason.TARGET
    assert fill.ambiguous_fill is False
    assert fill.intrabar_bars_used == 1


def test_finer_bars_that_do_not_cover_the_parent_fall_back_to_pessimism(model, plan, measured):
    """Partial intrabar coverage would decide the ambiguity by what was supplied."""
    parent = bar(offset=2, open_=13000.0, high=13025.0, low=12975.0, close=13010.0)
    base = T0 + timedelta(seconds=INTERVAL)
    # Covers only the calm middle: never reaches the parent's 12975 low.
    finer = [
        Bar(
            symbol="NQ",
            close_ts=base + timedelta(seconds=60 * i),
            open=13000.0,
            high=13005.0,
            low=12998.0,
            close=13002.0,
            volume=2_000.0,
            interval_seconds=60,
        )
        for i in (1, 2)
    ]
    fill = model.resolve_exit(plan, parent, vol=measured, intrabar=finer)

    assert fill.exit_reason is ExitReason.STOP
    assert fill.ambiguous_fill is True
    assert any("does not cover" in note for note in fill.notes)


def test_intrabar_bars_from_outside_the_parent_raise(model, plan, measured):
    parent = bar(offset=2, open_=13000.0, high=13025.0, low=12975.0, close=13010.0)
    outside = Bar(
        symbol="NQ",
        close_ts=T0 + timedelta(seconds=INTERVAL * 4),
        open=13000.0,
        high=13030.0,
        low=12970.0,
        close=13010.0,
        volume=2_000.0,
        interval_seconds=60,
    )
    with pytest.raises(FillTimingError):
        model.resolve_exit(plan, parent, vol=measured, intrabar=[outside])


def test_stop_and_target_cannot_be_priced_at_a_bar_close(model, plan, measured):
    """Routing them through the flatten path would skip the gap rule."""
    gap = bar(offset=2, open_=12950.0, high=12960.0, low=12940.0, close=12955.0)
    with pytest.raises(ExecutionError):
        model.exit_at_close(plan, gap, ExitReason.STOP, vol=measured)


def test_a_forced_flatten_prices_at_the_close_as_a_market_order(model, plan, measured):
    quiet = bar(offset=2, open_=13000.0, high=13010.0, low=12995.0, close=13005.0)
    fill = model.exit_at_close(plan, quiet, ExitReason.SESSION_CLOSE, vol=measured)

    assert fill.resolved is True
    assert fill.reference_price == 13005.0
    assert fill.exit_reason is ExitReason.SESSION_CLOSE
    assert fill.effective_price < fill.reference_price  # a LONG sells out lower
    assert fill.costs.exit_slippage > 0.0


def test_a_target_exit_pays_no_slippage_and_no_spread(model, plan, measured):
    """A limit cannot fill worse than its price, gap or no gap."""
    hit = bar(offset=2, open_=13000.0, high=13025.0, low=12999.0, close=13022.0)
    fill = model.resolve_exit(plan, hit, vol=measured)

    assert fill.exit_reason is ExitReason.TARGET
    assert fill.costs.exit_slippage == 0.0
    assert fill.costs.spread_cost == 0.0
    assert fill.costs.commission > 0.0  # it still pays to trade
    assert fill.effective_price == fill.reference_price


def test_a_stop_exit_pays_the_stop_order_surcharge(model, plan, measured):
    stopped = bar(offset=2, open_=13000.0, high=13005.0, low=12975.0, close=12985.0)
    fill = model.resolve_exit(plan, stopped, vol=measured)

    assert fill.exit_reason is ExitReason.STOP
    assert fill.slippage.stop_ticks == model.execution.stop_order_extra_slippage_ticks
    assert fill.slippage.stop_ticks > 0.0


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_the_vol_term_is_clamped_so_quiet_markets_are_not_paid_to_trade():
    assert vol_slippage_ticks(0.5, 1.4) == pytest.approx(0.2)
    assert vol_slippage_ticks(0.5, 0.6) == 0.0


def test_an_unmeasured_ratio_only_matters_when_the_coefficient_uses_it(nq, config):
    """The exception is exact: at coefficient 0.0 the ratio is not an input."""
    off = config.execution.replace(vol_slippage_coefficient=0.0)
    estimate = CostModel(off, nq).slippage(size=1.0, vol=VolatilityContext.unmeasured())
    assert estimate.measurable is True
    assert estimate.vol_ticks == 0.0


def test_an_unmeasurable_estimate_carries_no_ticks(nq, config):
    estimate = CostModel(config.execution, nq).slippage(
        size=1.0, vol=VolatilityContext.unmeasured()
    )
    assert estimate.measurable is False
    assert estimate.ticks == 0.0
    assert estimate.refusal == CostRefusal.VOL_MULTIPLE_UNMEASURABLE


def test_costs_cannot_be_assembled_from_an_unmeasurable_estimate(nq, config):
    costs = CostModel(config.execution, nq)
    estimate = costs.slippage(size=1.0, vol=VolatilityContext.unmeasured())
    with pytest.raises(CostModelError):
        costs.entry_costs(size=1.0, spread=costs.spread(), slippage=estimate)


def test_the_spread_comes_from_a_quote_when_one_is_present(nq, config, measured):
    costs = CostModel(config.execution, nq)
    quote = QuoteSnapshot(
        symbol="NQ", ts=T0, bid=12999.75, ask=13000.25, bid_size=10.0, ask_size=12.0
    )
    estimate = costs.spread(quote, fill_ts=T0 + timedelta(seconds=INTERVAL))

    assert estimate.source is SpreadSource.QUOTE
    assert estimate.ticks == pytest.approx(2.0)
    assert estimate.half_ticks == pytest.approx(1.0)


def test_a_crossed_quote_falls_back_instead_of_paying_the_trade(nq, config):
    """A non-positive spread as a 'cost' would be a rebate."""
    costs = CostModel(config.execution, nq)
    crossed = QuoteSnapshot(
        symbol="NQ", ts=T0, bid=13000.25, ask=12999.75, bid_size=10.0, ask_size=12.0
    )
    estimate = costs.spread(crossed, fill_ts=T0)

    assert estimate.source is SpreadSource.TYPICAL
    assert estimate.ticks == nq.typical_spread_ticks
    assert any("rebate" in note for note in estimate.notes)


def test_a_quote_from_the_future_raises(nq, config):
    costs = CostModel(config.execution, nq)
    future = QuoteSnapshot(
        symbol="NQ",
        ts=T0 + timedelta(seconds=INTERVAL * 2),
        bid=12999.75,
        ask=13000.25,
        bid_size=10.0,
        ask_size=12.0,
    )
    with pytest.raises(CostModelError):
        costs.spread(future, fill_ts=T0)


def test_no_quote_falls_back_to_the_spec_typical(nq, config):
    estimate = CostModel(config.execution, nq).spread()
    assert estimate.source is SpreadSource.TYPICAL
    assert estimate.ticks == nq.typical_spread_ticks


def test_a_sub_minimum_override_is_floored_and_says_so(nq, config):
    stressed = config.execution.replace(spread_ticks_override=0.25)
    estimate = CostModel(stressed, nq).spread()

    assert estimate.source is SpreadSource.OVERRIDE
    assert estimate.ticks == nq.min_spread_ticks
    assert any("min_spread_ticks" in note for note in estimate.notes)


def test_commission_and_fees_agree_with_the_instrument_spec(nq, qqq, config):
    """`InstrumentSpec` promises no two modules disagree about what NQ is worth.

    This module computes the two terms separately -- section 9 lists them
    separately and `CostBreakdown` itemizes them -- so the sum is checked
    against the spec's own round-turn figure rather than trusted.
    """
    for spec, size in ((nq, 3.0), (qqq, 400.0)):
        costs = CostModel(config.execution, spec)
        commission, fees = costs.commission_and_fees(size, sides=2)
        assert commission + fees == pytest.approx(spec.commission_for(size, sides=2))


def test_costs_are_itemized_and_never_netted(model, measured):
    """Section 12 needs cost drag attributable to a term."""
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)]
    )
    entry = model.resolve_entry(order, window, vol=measured)
    plan_ = ExitPlan.from_intent(
        intent(), entry_ts=entry.fill_ts, entry_price=entry.reference_price, size=entry.filled_size
    )
    stopped = bar(offset=2, open_=13000.0, high=13005.0, low=12975.0, close=12985.0)
    exit_fill = model.resolve_exit(plan_, stopped, vol=measured)
    total = model.round_turn_costs(entry, exit_fill)

    assert total.commission > 0.0
    assert total.exchange_fees > 0.0
    assert total.entry_slippage > 0.0
    assert total.exit_slippage > 0.0
    assert total.spread_cost > 0.0
    # The itemization sums to the total and nothing was pre-netted.
    assert total.total == pytest.approx(
        total.commission
        + total.exchange_fees
        + total.entry_slippage
        + total.exit_slippage
        + total.spread_cost
    )
    assert total == combine_costs(entry.costs, exit_fill.costs)


def test_the_round_turn_pays_both_sides_commission(model, measured, nq):
    order = model.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)]
    )
    entry = model.resolve_entry(order, window, vol=measured)
    plan_ = ExitPlan.from_intent(
        intent(), entry_ts=entry.fill_ts, entry_price=entry.reference_price, size=entry.filled_size
    )
    stopped = bar(offset=2, open_=13000.0, high=13005.0, low=12975.0, close=12985.0)
    exit_fill = model.resolve_exit(plan_, stopped, vol=measured)
    total = model.round_turn_costs(entry, exit_fill)

    expected = nq.commission_for(entry.filled_size, sides=2)
    assert total.commission + total.exchange_fees == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_a_stochastic_miss_is_reproducible_for_the_same_order(nq, config, measured):
    stressed = config.execution.replace(random_miss_probability=1.0)
    first = ExecutionModel(stressed, nq, master_seed=7)
    second = ExecutionModel(stressed, nq, master_seed=7)
    order = first.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)]
    )

    a = first.resolve_entry(order, window, vol=measured)
    b = second.resolve_entry(second.submit(intent(), T0), window, vol=measured)
    assert a.miss_reason == MissReason.RANDOM_MISS
    assert a.to_init_dict() == b.to_init_dict()


def test_adding_latency_jitter_does_not_shift_the_miss_draw(nq, config, measured):
    """Named per-order streams: a new stochastic term cannot move an old one.

    `core/determinism.py` exists for this -- "adding a new stochastic
    component therefore cannot change the draws of an existing one". Checked
    at the probability where the draw's value decides the outcome.
    """
    probability = 0.5
    without = config.execution.replace(random_miss_probability=probability)
    with_jitter = without.replace(random_latency_ms=500.0)
    window_bar = bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)

    outcomes = set()
    for execution in (without, with_jitter):
        model_ = ExecutionModel(execution, nq, master_seed=11)
        order = model_.submit(intent(), T0)
        fill = model_.resolve_entry(order, FillWindow(order, [window_bar]), vol=measured)
        outcomes.add(fill.status)
    assert len(outcomes) == 1, "the latency stream perturbed the miss stream"


def test_the_miss_draw_does_not_depend_on_how_many_orders_preceded_it(nq, config, measured):
    """Keyed by the order, not by a counter, so a partial re-run reproduces."""
    stressed = config.execution.replace(random_miss_probability=0.5)
    window_bar = bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)
    later = intent(signal_ts=T0 + timedelta(seconds=INTERVAL * 50))

    alone = ExecutionModel(stressed, nq, master_seed=3)
    order = alone.submit(later, T0 + timedelta(seconds=INTERVAL * 50))
    solo = alone.resolve_entry(
        order,
        FillWindow(
            order,
            [bar(offset=51, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)],
        ),
        vol=measured,
    )

    after = ExecutionModel(stressed, nq, master_seed=3)
    for _ in range(5):
        warm = after.submit(intent(), T0)
        after.resolve_entry(warm, FillWindow(warm, [window_bar]), vol=measured)
    order2 = after.submit(later, T0 + timedelta(seconds=INTERVAL * 50))
    sequenced = after.resolve_entry(
        order2,
        FillWindow(
            order2,
            [bar(offset=51, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)],
        ),
        vol=measured,
    )
    assert solo.to_init_dict() == sequenced.to_init_dict()


def test_a_different_seed_can_change_the_outcome(nq, config, measured):
    """A determinism test that passes for a constant generator proves nothing."""
    stressed = config.execution.replace(random_miss_probability=0.5)
    window_bar = bar(offset=1, open_=13002.0, high=13010.0, low=12999.0, close=13006.0)
    draws = set()
    for seed in range(40):
        model_ = ExecutionModel(stressed, nq, master_seed=seed)
        order = model_.submit(intent(), T0)
        fill = model_.resolve_entry(order, FillWindow(order, [window_bar]), vol=measured)
        draws.add(fill.status)
    assert draws == {FillStatus.FILLED, FillStatus.MISSED}


# ---------------------------------------------------------------------------
# The plan is never changed
# ---------------------------------------------------------------------------


def test_nothing_in_execution_moves_the_stop_or_the_target(model, measured):
    """`TradeIntent`: the execution model "may not change the plan"."""
    original = intent()
    order = model.submit(original, T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13007.5, high=13012.0, low=13004.0, close=13010.0)]
    )
    fill = model.resolve_entry(order, window, vol=measured)
    plan_ = ExitPlan.from_intent(
        original, entry_ts=fill.fill_ts, entry_price=fill.reference_price, size=fill.filled_size
    )

    assert fill.reference_price != original.entry_price  # it filled worse
    assert plan_.stop_price == original.stop_price
    assert plan_.target_price == original.target_price
    assert order.intent.to_init_dict() == original.to_init_dict()


def test_an_inverted_exit_plan_is_refused(model):
    with pytest.raises(ValueError):
        ExitPlan(
            symbol="NQ",
            side=Side.LONG,
            entry_ts=T0,
            entry_price=13000.0,
            stop_price=13010.0,
            target_price=13020.0,
            size=1.0,
        )


def test_the_model_refuses_an_intent_for_another_symbol(model):
    other = intent().replace(symbol="ES")
    with pytest.raises(ExecutionError):
        model.submit(other, T0)


def test_a_stop_entry_that_can_never_fill_is_refused_at_construction(nq, config):
    """`ExecutionConfig` validates this for `limit` but not for `stop`.

    Both are resting entries with the same property, so the refusal is made
    here rather than letting a ten-year run report `no_eligible_bar` on every
    bar.
    """
    impossible = config.execution.replace(entry_order_type="stop", max_entry_wait_bars=0)
    with pytest.raises(ExecutionError):
        ExecutionModel(impossible, nq, master_seed=1)


def test_a_limit_entry_fills_at_its_limit_and_pays_no_slippage(nq, config, measured):
    """A limit cannot fill worse than its price, so no slippage is charged.

    Its real cost is miss risk, which the trade-through requirement and
    `max_entry_wait_bars` model instead.
    """
    resting = config.execution.replace(entry_order_type="limit")
    model_ = ExecutionModel(resting, nq, master_seed=5)
    order = model_.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13005.0, high=13008.0, low=12999.0, close=13004.0)]
    )
    fill = model_.resolve_entry(order, window, vol=measured)

    assert fill.status is FillStatus.FILLED
    assert fill.reference_price == 13000.0
    assert fill.effective_price == fill.reference_price
    assert fill.costs.entry_slippage == 0.0
    assert fill.costs.spread_cost == 0.0
    assert fill.costs.commission > 0.0


def test_a_limit_merely_touched_is_not_treated_as_filled(nq, config, measured):
    """Queue position is unobservable from bar data, so a touch is not evidence."""
    resting = config.execution.replace(entry_order_type="limit")
    model_ = ExecutionModel(resting, nq, master_seed=5)
    order = model_.submit(intent(), T0)
    window = FillWindow(
        order, [bar(offset=1, open_=13005.0, high=13008.0, low=13000.0, close=13004.0)]
    )
    fill = model_.resolve_entry(order, window, vol=measured)
    assert fill.miss_reason == MissReason.LIMIT_NOT_TRADED_THROUGH


def test_the_lookahead_surface_imports_on_its_own():
    """`fill_timing.py` holds the one invariant worth reading in full.

    It is split out so it can be read without the cost model or the fill
    rules, which only holds while it does not import them. The dependency
    runs one way: fill_timing <- fills <- execution, with costs beside them.
    """
    import ast
    from pathlib import Path

    import flow_model.backtest.fill_timing as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    backtest_deps = {name for name in imported if name.startswith("flow_model.backtest")}
    assert backtest_deps == set(), (
        f"fill_timing.py must not depend on the rest of backtest/: {backtest_deps}"
    )


def test_execution_re_exports_the_whole_fill_model():
    """One import for the event loop, which is a separate task's caller."""
    import flow_model.backtest.execution as module

    for name in ("EntryFill", "ExitFill", "ExitPlan", "FillWindow", "PendingOrder",
                 "FillStatus", "MissReason", "resolve_bar_exit", "min_fill_offset",
                 "ExecutionError", "FillTimingError", "ExecutionModel"):
        assert name in module.__all__, name
        assert getattr(module, name, None) is not None, name


def test_from_config_builds_a_model_for_a_configured_symbol():
    config = default_config()
    model_ = ExecutionModel.from_config(config, "NQ")
    assert model_.spec.symbol == "NQ"
    assert model_.master_seed == config.backtest.seed
    assert model_.execution is config.execution
