"""Adversarial tests for the execution and cost model (ARCHITECTURE.md section 9).

Organized around the four ways this layer can be quietly wrong, rather than
around its public surface. Every clause of section 9 is a place where an error
has a DIRECTION, and in every case the flattering direction is the one that
survives review, because it produces a backtest that looks better rather than
one that looks broken.

**A price the market never offered.** A stop filled at its limit instead of
the gap open, a resting limit filled on a touch, an entry chased at a price the
signal never saw. Section 9 calls the stop case "the single most common
backtest overstatement", so `test_a_gap_through_the_stop_costs_the_gap_*`
measures the overstatement being refused rather than merely asserting that a
branch was taken: the fill is compared against the naive stop-price fill and
the difference is required to be a LOSS of the exact size of the gap.

**An assumption reported as an observation.** When a bar's range holds both
the stop and the target, the touch order was never seen. The answer is to
resolve against the trade AND tag it, so the sensitivity is measurable;
untagging the optimistic setting, or clearing the tag because a partial finer
series was supplied, converts a missing observation into an invisible one. The
same-bar and intrabar tests pin the tag as load-bearing and measure the swing
it covers ($2426.25 on one constructed bar).

**A quantity that was never measured, priced at whatever is cheapest.** No
ATR, no volume, no quote, a zero, a NaN. `signals/engine.py` carries the scar
this rule comes from -- a missing ATR there did not degrade the stop cap, it
REMOVED it -- and the same mistake is available here as a free fill in
unmeasured volatility, on exactly the bars where volatility is most likely to
be extreme. These tests require a refusal or a declared degradation, never a
silent default.

**A fill that used the bar it was signalled from.** "Fills resolve on later
bars only" is the one invariant whose violation does not produce a wrong
number in one place but invalidates every number the system will ever produce.
`test_*_same_bar_*` and `test_*_eligibility_*` attack it from five directions
(feeding the signal bar, a mid-bar signal timestamp, a hand-built result
record, a latency equal to one bar, and a window assembled by hand) and
require every one to be refused.

Five bugs were found and fixed while writing this file; each has a named
regression test below:

* HIGH -- a bar whose OPEN was exactly AT the target booked a full target win
  (`test_a_bar_opening_exactly_at_the_target_is_not_a_fill`). Targets are
  resting limits and the module's own rule for them is strict, but the gap
  branch used `>=`. Bar opens and target prices both sit on the tick grid, so
  exact equality is ordinary. A bar that opened at the target and then traded
  down through the stop booked +$1520 instead of -$880.
* HIGH -- a finer-granularity series covering only part of the parent bar's
  TIME span was accepted as "available" whenever it happened to touch both of
  the parent's price extremes, and resolved the ordering with
  `ambiguous_fill=False` (`test_finer_bars_with_a_hole_*`). The module already
  rejected partial PRICE coverage for exactly this reason.
* MEDIUM -- `min_fill_offset` used `max(1, ceil(latency/interval))`, which at a
  latency of exactly one bar returned the bar whose OPEN is the signal instant
  itself, pricing a full bar of latency as a zero-latency fill
  (`test_a_latency_of_exactly_one_bar_*`).
* MEDIUM -- `exit_at_close` did not check `bar.is_valid`, so a NaN close became
  a NaN exit price carrying real dollar costs, and neither result record
  rejected a non-finite price (`test_exit_at_close_refuses_*`,
  `test_the_result_records_refuse_a_non_finite_price`).
* MEDIUM -- nothing checked that a `FillWindow` belonged to the order being
  resolved against it, so the natural event-loop shape (one window per bar,
  every pending order resolved against it) filled a slow order at a fast
  order's bar, a bar early and $400 better
  (`test_a_window_built_for_one_order_cannot_resolve_another`).

Three things are deliberately NOT tested here, because an honest coverage gap
is a finding and a vacuous test is a lie:

* `resolve_exit` permits an exit on the ENTRY bar, justified by the entry
  having been at that bar's open. That justification holds for a market entry
  and fails for a limit or stop entry, which fills mid-bar -- so the part of
  the bar BEFORE the fill can resolve a target the position did not yet hold.
  `ExitPlan` does not record the order type, so the fix is an API change that
  belongs with the ledger that will make the call, and a flag defaulting to
  the permissive value would be the cap-removing mistake again.
* `resolve_entry` dates its spread guard from the fill BAR's close while a
  market fill prints at that bar's open, so a quote up to one interval after
  the fill print can price it while carrying `SpreadSource.QUOTE`. The honest
  instant for a mid-bar limit fill is not knowable from bar data, and the
  choice couples to the ledger's quote-selection policy, so only the coarser
  guard (a quote postdating the whole bar raises) is pinned here.
* Section 9's partial-fill clause is implemented single-bar, with no
  accumulation across bars. That is the builder's declared omission -- section
  9 specifies no fill-price-averaging policy -- and testing an unspecified
  policy would be inventing one.

Phase 6 measured this system over 2500 consecutive bars and it takes ZERO
trades (the highest Flow Score observed is 65.23 against a lowest threshold of
70.0, commit f952d3e). So every scenario here is a hand-built `TradeIntent`
against hand-built bars. That is not a workaround: a cost model validated on
whatever trades happen to occur is validated on an arbitrary sample, and here
the sample would be empty.

NQ is used throughout because its arithmetic is exact. Tick 0.25 and
$5.00/tick means $20.00 per point, one tick of slippage is $5 per contract,
and the 20-point stop below is 80 ticks worth $400 per contract -- so every
expected value in this file is a number written down by hand rather than read
back from the implementation. `ExecutionConfig` and `InstrumentSpec` are built
locally for the reason `test_risk.py` gives: a test coupled to
`config/defaults.yaml` fails whenever an unrelated section is edited, and its
expected values change silently when a default moves.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from flow_model.backtest.costs import (
    CostModel,
    CostModelError,
    CostRefusal,
    SlippageEstimate,
    SpreadSource,
    VolatilityContext,
    adverse_price,
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
    ExitFill,
    ExitPlan,
    FillStatus,
    MissReason,
    resolve_bar_exit,
    validated_intrabar,
)
from flow_model.config.schema import ExecutionConfig
from flow_model.core.contracts import (
    Bar,
    CostBreakdown,
    QuoteSnapshot,
    TradeIntent,
    TradeRecord,
)
from flow_model.core.enums import (
    ExitReason,
    InstrumentType,
    Regime,
    Side,
    SetupType,
    SplitPhase,
)
from flow_model.core.instruments import InstrumentSpec

UTC = timezone.utc

#: A bar close. Every signal in this file fires here, so "the signal bar" is
#: the bar closing at T0 and every eligible bar closes strictly after it.
T0 = datetime(2021, 3, 2, 15, 0, tzinfo=UTC)
INTERVAL = 300

# --- the instrument, built locally -----------------------------------------
#
# tick_value / tick_size = 5.00 / 0.25 = $20.00 per point.
#   1 tick of slippage            = $5.00 per contract
#   the 20-point stop below       = 80 ticks = $400 per contract
#   at size 2                     = $800 of risk, which is `risk_dollars`
TICK = 0.25
TICK_VALUE = 5.0
POINT_VALUE = 20.0
COMMISSION = 2.25
EXCHANGE_FEE = 0.37

NQ = InstrumentSpec(
    symbol="NQ",
    instrument_type=InstrumentType.FUTURE,
    tick_size=TICK,
    tick_value=TICK_VALUE,
    commission_per_side=COMMISSION,
    exchange_fee_per_side=EXCHANGE_FEE,
    commission_per_unit=False,  # per order, so size does not scale it
    typical_spread_ticks=1.0,
    min_spread_ticks=1.0,
)

#: A per-SHARE instrument, so the `commission_per_unit` branch is exercised
#: against a second set of numbers: 600 shares = $3.00 commission, $0.12 fees.
QQQ = InstrumentSpec(
    symbol="QQQ",
    instrument_type=InstrumentType.ETF,
    tick_size=0.01,
    tick_value=0.01,  # $1.00 per point
    commission_per_side=0.005,
    exchange_fee_per_side=0.0002,
    commission_per_unit=True,
    typical_spread_ticks=1.0,
    min_spread_ticks=1.0,
    min_size=1.0,
    max_size=100_000.0,
)

#: Fractional size allowed, so the size-grid floor can be shown to be the
#: thing that turns a 1.5-contract cap into a 1-contract fill.
QQQ_FRACTIONAL = QQQ.replace(allow_fractional_size=True, min_size=0.01)

# --- the measured volatility ratio -----------------------------------------
#
# ATR 12.5 against a median of 10.0 is a ratio of 1.25, so the configured
# vol_slippage_coefficient of 0.5 adds 0.5 * (1.25 - 1) = 0.125 ticks.
MEASURED = VolatilityContext(atr=12.5, median_atr=10.0)

# --- the per-side cost of the trade used throughout ------------------------
#
# Default ExecutionConfig, NQ, size 2, vol ratio 1.25:
#   market entry slippage = (1.0 base + 0.125 vol + 0.0 size) ticks
#                         = 1.125 * $5.00 * 2 = $11.25
#   spread                = typical 1.0 tick, half per side
#                         = 0.5 * $5.00 * 2  = $5.00
#   commission + fees                        = $2.25 + $0.37
#   ------------------------------------------------------------
#   ENTRY                                    = $18.87
#
#   stop exit slippage    = (1.125 + 1.0 stop surcharge) ticks
#                         = 2.125 * $5.00 * 2 = $21.25
#   + spread $5.00 + commission $2.62        = $28.87
#
#   target exit: a resting limit pays neither slippage nor spread
#                         = commission + fees = $2.62
ENTRY_COST = 18.87
STOP_EXIT_COST = 28.87
TARGET_EXIT_COST = 2.62
ROUND_TURN_VIA_STOP = ENTRY_COST + STOP_EXIT_COST  # 47.74
ROUND_TURN_VIA_TARGET = ENTRY_COST + TARGET_EXIT_COST  # 21.49

RISK_DOLLARS = 800.0  # 20 points * $20.00 * 2 contracts


# ---------------------------------------------------------------------------
# Local builders. Deliberately functions rather than fixtures: each test says
# in one line exactly which bar it is talking about, and no test can be
# changed by an edit to a shared fixture it never mentions.
# ---------------------------------------------------------------------------


def config(**overrides) -> ExecutionConfig:
    """`ExecutionConfig` with the shipped defaults, overridden explicitly."""
    return ExecutionConfig(**overrides)


def model(spec: InstrumentSpec = NQ, *, seed: int = 20240101, **overrides) -> ExecutionModel:
    return ExecutionModel(config(**overrides), spec, master_seed=seed)


def bar(
    offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    volume: float = 100_000.0,
    symbol: str = "NQ",
    interval: int = INTERVAL,
    base: datetime = T0,
) -> Bar:
    """The bar closing `offset` intervals after `base`. Offset 0 is the signal bar."""
    return Bar(
        symbol=symbol,
        close_ts=base + timedelta(seconds=interval * offset),
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
    entry: float = 13_000.0,
    stop: float = 12_980.0,
    target: float = 13_040.0,
    size: float = 2.0,
    symbol: str = "NQ",
    point_value: float = POINT_VALUE,
    signal_ts: datetime = T0,
) -> TradeIntent:
    """A 2R NQ long: 20-point stop, 40-point target, 2 contracts, $800 of risk."""
    return TradeIntent(
        symbol=symbol,
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


def entry_bar(open_: float = 13_002.0, *, volume: float = 100_000.0) -> Bar:
    """The first eligible bar, opening two points above the planned entry."""
    return bar(1, open_, open_ + 8.0, open_ - 3.0, open_ + 4.0, volume=volume)


def filled_entry(
    exec_model: ExecutionModel | None = None,
    *,
    order_intent: TradeIntent | None = None,
    bars=None,
    vol: VolatilityContext = MEASURED,
    quote: QuoteSnapshot | None = None,
) -> tuple[TradeIntent, EntryFill]:
    """Submit, open a window, resolve. The three calls every test starts with."""
    exec_model = exec_model or model()
    order_intent = order_intent or intent()
    order = exec_model.submit(order_intent, order_intent.signal_ts)
    window = exec_model.open_window(order, bars if bars is not None else [entry_bar()])
    return order_intent, exec_model.resolve_entry(order, window, vol=vol, quote=quote)


def plan_from(order_intent: TradeIntent, fill: EntryFill) -> ExitPlan:
    return ExitPlan.from_intent(
        order_intent,
        entry_ts=fill.fill_ts,
        entry_price=fill.reference_price,
        size=fill.filled_size,
    )


def record_from(
    order_intent: TradeIntent,
    fill: EntryFill,
    exit_fill: ExitFill,
    costs: CostBreakdown,
) -> TradeRecord:
    """The journal row, with gross_pnl computed off the PRICE PATH by hand.

    `TradeRecord` re-derives `gross_pnl` from the prices and requires
    `pnl == gross_pnl - costs.total`, so a cost or fill bug that slipped past
    the assertions above still cannot reach the journal.
    """
    gross = (
        order_intent.side.sign
        * (exit_fill.reference_price - fill.reference_price)
        * order_intent.point_value
        * fill.filled_size
    )
    return TradeRecord(
        trade_id="t-1",
        run_id="r-1",
        symbol=order_intent.symbol,
        config_hash="cfg",
        split_label=SplitPhase.TRAIN,
        signal_ts=order_intent.signal_ts,
        entry_ts=fill.fill_ts,
        exit_ts=exit_fill.fill_ts,
        bars_held=1,
        side=order_intent.side,
        setup=order_intent.setup,
        planned_entry=order_intent.entry_price,
        stop_price=order_intent.stop_price,
        target_price=order_intent.target_price,
        entry_price=fill.reference_price,
        exit_price=exit_fill.reference_price,
        size=order_intent.size,
        filled_size=fill.filled_size,
        point_value=order_intent.point_value,
        exit_reason=exit_fill.exit_reason,
        ambiguous_fill=exit_fill.ambiguous_fill,
        regime=order_intent.regime,
        risk_dollars=order_intent.risk_dollars,
        gross_pnl=gross,
        costs=costs,
        pnl=gross - costs.total,
        flow_score=order_intent.flow_score,
    )


def long_plan(entry: float = 13_002.0, *, size: float = 2.0) -> ExitPlan:
    """An open LONG: filled at 13002, stop 12980, target 13040."""
    return ExitPlan(
        symbol="NQ",
        side=Side.LONG,
        entry_ts=T0 + timedelta(seconds=INTERVAL),
        entry_price=entry,
        stop_price=12_980.0,
        target_price=13_040.0,
        size=size,
    )


def short_plan() -> ExitPlan:
    """The mirror: a SHORT filled at 12998, stop 13020, target 12960."""
    return ExitPlan(
        symbol="NQ",
        side=Side.SHORT,
        entry_ts=T0 + timedelta(seconds=INTERVAL),
        entry_price=12_998.0,
        stop_price=13_020.0,
        target_price=12_960.0,
        size=2.0,
    )


# ===========================================================================
# 1. Every cost term, separately, then summed, then against CostBreakdown
# ===========================================================================


def test_commission_and_fees_are_two_terms_not_one():
    """Section 9 lists "commissions + exchange fees"; `CostBreakdown` itemizes them.

    A single `fees` figure cannot answer how much of the hurdle is the broker
    and how much is the exchange, and the two scale differently across
    instruments.
    """
    commission, fees = CostModel(config(), NQ).commission_and_fees(2.0, sides=1)

    # NQ is per ORDER, so size does not scale either term.
    assert commission == pytest.approx(2.25)
    assert fees == pytest.approx(0.37)


def test_commission_scales_with_size_only_on_a_per_unit_instrument():
    """`commission_per_unit` is the ETF case; getting it backwards is a 600x error."""
    commission, fees = CostModel(config(), QQQ).commission_and_fees(600.0, sides=1)

    # 0.005 * 600 = 3.00, and 0.0002 * 600 = 0.12.
    assert commission == pytest.approx(3.0)
    assert fees == pytest.approx(0.12)
    # NQ at the same size is unchanged, because it charges per order.
    assert CostModel(config(), NQ).commission_and_fees(600.0, sides=1)[0] == pytest.approx(2.25)


def test_the_two_terms_sum_to_the_instrument_specs_own_figure():
    """`InstrumentSpec` promises no two modules disagree about what NQ is worth.

    This module reimplements the `commission_per_unit` rule in order to split
    the two terms apart, and a second implementation of the same arithmetic is
    exactly how that promise gets broken. Checked on both instruments and on
    both side counts.
    """
    for spec, size in ((NQ, 3.0), (QQQ, 400.0)):
        costs = CostModel(config(), spec)
        for sides in (1, 2):
            commission, fees = costs.commission_and_fees(size, sides=sides)
            assert commission + fees == pytest.approx(spec.commission_for(size, sides=sides))


def test_the_spread_cost_is_half_a_spread_per_side():
    """A market order crosses half the book; a round turn pays one full spread."""
    costs = CostModel(config(), NQ)
    estimate = costs.spread()

    assert estimate.source is SpreadSource.TYPICAL
    assert estimate.ticks == pytest.approx(1.0)
    assert estimate.half_ticks == pytest.approx(0.5)
    # 0.5 ticks * $5.00 * 2 contracts = $5.00 per side, $10.00 the round turn.
    assert costs.ticks_to_dollars(estimate.half_ticks, 2.0) == pytest.approx(5.0)


def test_a_quoted_spread_is_measured_in_ticks_not_points():
    """A points/ticks confusion here is a 4x cost error on NQ and silent."""
    quote = QuoteSnapshot(
        symbol="NQ", ts=T0, bid=12_999.0, ask=13_001.0, bid_size=5.0, ask_size=5.0
    )
    estimate = CostModel(config(), NQ).spread(quote, fill_ts=T0 + timedelta(seconds=INTERVAL))

    # 2.00 points / 0.25 = 8 ticks, so 4 ticks per side = $20/contract.
    assert estimate.source is SpreadSource.QUOTE
    assert estimate.ticks == pytest.approx(8.0)
    assert CostModel(config(), NQ).ticks_to_dollars(estimate.half_ticks, 2.0) == pytest.approx(40.0)


def test_the_base_slippage_term_stands_alone():
    """`base_slippage_ticks` is the floor every non-passive fill pays."""
    estimate = CostModel(config(vol_slippage_coefficient=0.0), NQ).slippage(
        size=2.0, vol=VolatilityContext.unmeasured()
    )

    assert estimate.base_ticks == pytest.approx(1.0)
    assert estimate.vol_ticks == 0.0
    assert estimate.size_ticks == 0.0
    assert estimate.stop_ticks == 0.0
    assert estimate.ticks == pytest.approx(1.0)


def test_the_volatility_term_is_the_configured_coefficient_times_the_excess():
    """The field's own units: "extra slippage ticks per 1.0 of (ATR/median ATR - 1)".

    Section 9's prose writes this term as `vol_multiplier * ATR_pct`, which is a
    different quantity -- at ATR 12.5 on a 13000 price, `atr/close` gives
    0.00048 ticks against the field's 0.125, about 260x smaller and effectively
    vacuous. The committed field carries its own semantics, so the field is
    what is tested; the divergence from section 9's literal text is recorded in
    the module's own notes.
    """
    estimate = CostModel(config(), NQ).slippage(size=2.0, vol=MEASURED)

    # 0.5 * (12.5/10.0 - 1) = 0.5 * 0.25 = 0.125 ticks.
    assert estimate.vol_ticks == pytest.approx(0.125)
    assert vol_slippage_ticks(0.5, 1.25) == pytest.approx(0.125)


def test_the_size_impact_term_is_per_unit_and_charged_on_every_unit():
    """Impact per unit times size gives the ticks; the cost is ticks times size.

    The term is therefore quadratic in size, which is what a linear impact
    model means. A linear cost would understate every oversized fill.
    """
    costs = CostModel(config(size_impact_ticks_per_unit=0.1), NQ)
    estimate = costs.slippage(size=2.0, vol=MEASURED)

    # 0.1 ticks/unit * 2 units = 0.2 ticks of impact.
    assert estimate.size_ticks == pytest.approx(0.2)
    # 0.2 ticks * $5.00 * 2 units = $2.00.
    assert costs.ticks_to_dollars(estimate.size_ticks, 2.0) == pytest.approx(2.0)


def test_a_stop_order_pays_the_configured_surcharge_on_top():
    """Section 9: "stop-market orders additionally pay stop_slippage_ticks"."""
    costs = CostModel(config(), NQ)
    market = costs.slippage(size=2.0, vol=MEASURED, is_stop_order=False)
    stopped = costs.slippage(size=2.0, vol=MEASURED, is_stop_order=True)

    assert market.stop_ticks == 0.0
    assert stopped.stop_ticks == pytest.approx(1.0)
    # 1.125 ticks market, 2.125 stop: exactly one extra tick, $10 at size 2.
    assert stopped.ticks - market.ticks == pytest.approx(1.0)


def test_the_four_slippage_terms_sum_to_the_estimates_total():
    """Kept term-by-term so a sweep can attribute a cost change to its knob."""
    estimate = CostModel(config(size_impact_ticks_per_unit=0.1), NQ).slippage(
        size=2.0, vol=MEASURED, is_stop_order=True
    )

    # 1.0 + 0.125 + 0.2 + 1.0 = 2.325 ticks.
    assert estimate.ticks == pytest.approx(2.325)
    assert estimate.ticks == pytest.approx(
        estimate.base_ticks + estimate.vol_ticks + estimate.size_ticks + estimate.stop_ticks
    )


def test_every_term_lands_in_its_own_breakdown_field_and_sums_to_the_total():
    """Section 12 needs cost drag attributable to a term, not one `costs` number.

    The whole hand computation in one place: a stop exit on 2 NQ with the
    size-impact term switched on.
    """
    costs = CostModel(config(size_impact_ticks_per_unit=0.1), NQ)
    slippage = costs.slippage(size=2.0, vol=MEASURED, is_stop_order=True)
    breakdown = costs.exit_costs(size=2.0, spread=costs.spread(), slippage=slippage)

    assert breakdown.commission == pytest.approx(2.25)
    assert breakdown.exchange_fees == pytest.approx(0.37)
    assert breakdown.entry_slippage == 0.0  # an exit does not fill the entry field
    assert breakdown.exit_slippage == pytest.approx(23.25)  # 2.325 ticks * $5 * 2
    assert breakdown.spread_cost == pytest.approx(5.0)  # 0.5 ticks * $5 * 2
    # 2.25 + 0.37 + 23.25 + 5.00 = 30.87
    assert breakdown.total == pytest.approx(30.87)
    assert breakdown.total == pytest.approx(
        breakdown.commission
        + breakdown.exchange_fees
        + breakdown.entry_slippage
        + breakdown.exit_slippage
        + breakdown.spread_cost
    )


def test_a_round_turn_is_the_field_wise_sum_of_its_two_sides():
    """Itemization must survive the addition, or the breakdown is decorative."""
    order_intent, fill = filled_entry()
    exit_fill = model().resolve_exit(
        plan_from(order_intent, fill),
        bar(2, 13_000.0, 13_005.0, 12_975.0, 12_985.0),
        vol=MEASURED,
    )
    total = model().round_turn_costs(fill, exit_fill)

    assert fill.costs.total == pytest.approx(ENTRY_COST)
    assert exit_fill.costs.total == pytest.approx(STOP_EXIT_COST)
    assert total.commission == pytest.approx(4.5)  # $2.25 twice
    assert total.exchange_fees == pytest.approx(0.74)
    assert total.entry_slippage == pytest.approx(11.25)
    assert total.exit_slippage == pytest.approx(21.25)
    assert total.spread_cost == pytest.approx(10.0)  # one full spread, round turn
    assert total.total == pytest.approx(ROUND_TURN_VIA_STOP)
    assert total == combine_costs(fill.costs, exit_fill.costs)
    # Cross-checked against the spec's own round-turn arithmetic.
    assert total.commission + total.exchange_fees == pytest.approx(
        NQ.commission_for(2.0, sides=2)
    )


def test_combine_costs_never_nets_one_term_against_another():
    """A field-wise sum, so a large spread cannot be hidden behind a small one."""
    left = CostBreakdown(commission=1.0, spread_cost=5.0, entry_slippage=2.0)
    right = CostBreakdown(commission=1.0, spread_cost=7.0, exit_slippage=3.0)
    total = combine_costs(left, right)

    assert total.commission == pytest.approx(2.0)
    assert total.spread_cost == pytest.approx(12.0)
    assert total.entry_slippage == pytest.approx(2.0)
    assert total.exit_slippage == pytest.approx(3.0)
    assert total.total == pytest.approx(19.0)


def test_a_resting_limit_pays_neither_slippage_nor_spread():
    """A limit cannot fill worse than its price, so charging it books a cost
    the market cannot impose. Its real cost is miss risk, which is modelled by
    the trade-through requirement and `max_entry_wait_bars`, not here.
    """
    costs = CostModel(config(), NQ)
    passive = costs.slippage(size=2.0, vol=MEASURED, passive=True)
    breakdown = costs.entry_costs(size=2.0, spread=costs.spread(), slippage=passive)

    assert passive.ticks == 0.0
    assert breakdown.entry_slippage == 0.0
    assert breakdown.spread_cost == 0.0
    # Commission and fees are still paid: the exchange does not care how we filled.
    assert breakdown.total == pytest.approx(2.62)


def test_a_passive_estimate_cannot_also_be_a_stop_order():
    """A stop is a market order by construction, so "passive stop" is incoherent
    and would collect the surcharge while paying no spread."""
    with pytest.raises(CostModelError):
        CostModel(config(), NQ).slippage(
            size=2.0, vol=MEASURED, is_stop_order=True, passive=True
        )


def test_the_effective_price_moves_against_the_trade_on_both_sides():
    """`effective_price` is the human sanity-check figure, and its SIGN is the
    only thing it has to get right: a diagnostic that reported a better price
    than the fill would make every manual review of a fill agree with a bug.
    """
    # Buying: a higher price is worse. 1 tick of 0.25 on 13000.
    assert adverse_price(13_000.0, 1.0, TICK, against_sign=1) == pytest.approx(13_000.25)
    # Selling: a lower price is worse.
    assert adverse_price(13_000.0, 1.0, TICK, against_sign=-1) == pytest.approx(12_999.75)
    with pytest.raises(ValueError):
        adverse_price(13_000.0, -1.0, TICK, against_sign=1)


def test_the_recorded_price_is_off_the_price_path_and_costs_are_dollars():
    """The one convention the whole layer depends on, and the validator CANNOT
    catch its violation.

    `TradeRecord` checks `gross_pnl` against the prices and `pnl == gross -
    costs.total`. Folding slippage into `entry_price` AND itemizing it charges
    every trade twice with both equations still holding; folding it in and
    leaving `CostBreakdown` empty satisfies the validator too and reports zero
    cost drag. So the test asserts the convention directly, and measures what
    the undetectable double-charge would have cost.
    """
    _, fill = filled_entry()

    assert fill.reference_price == pytest.approx(13_002.0)  # the bar's open, unadjusted
    assert fill.effective_price > fill.reference_price  # the diagnostic, worse
    assert fill.costs.entry_slippage > 0.0 and fill.costs.spread_cost > 0.0

    # 1.125 slip + 0.5 half-spread = 1.625 ticks = 0.40625 points.
    assert fill.effective_price - fill.reference_price == pytest.approx(0.40625)
    # Charging that in the price as well as in CostBreakdown would double it:
    # 1.625 ticks * $5 * 2 = $16.25 of phantom cost per entry.
    double_charged = (fill.effective_price - fill.reference_price) * POINT_VALUE * 2.0
    assert double_charged == pytest.approx(16.25)


# ===========================================================================
# 2. Gap handling -- "the single most common backtest overstatement"
# ===========================================================================


def test_a_gap_through_the_stop_costs_the_gap_rather_than_saving_it():
    """The overstatement being refused, measured against the naive fill.

    A LONG filled at 13002 with its stop at 12980 meets a bar that OPENS at
    12950. The honest fill is the open; the flattering one is the stop. The
    test requires the honest fill to be the LOSS and pins the difference.
    """
    order_intent, fill = filled_entry()
    plan = plan_from(order_intent, fill)
    gap = bar(2, 12_950.0, 12_960.0, 12_940.0, 12_955.0)

    honest = model().resolve_exit(plan, gap, vol=MEASURED)
    flattering = model(fill_gaps_at_open=False).resolve_exit(plan, gap, vol=MEASURED)

    assert honest.exit_reason is ExitReason.STOP
    assert honest.reference_price == pytest.approx(12_950.0)  # the OPEN
    assert flattering.reference_price == pytest.approx(12_980.0)  # the stop

    # (12950 - 13002) * $20 * 2 = -$2080 against (12980 - 13002) * $20 * 2 = -$880.
    honest_gross = (honest.reference_price - 13_002.0) * POINT_VALUE * 2.0
    naive_gross = (flattering.reference_price - 13_002.0) * POINT_VALUE * 2.0
    assert honest_gross == pytest.approx(-2080.0)
    assert naive_gross == pytest.approx(-880.0)
    # 30 points of gap * $20 * 2 = $1200, and it is a LOSS, not a saving.
    assert honest_gross - naive_gross == pytest.approx(-1200.0)
    assert honest_gross < naive_gross


def test_a_gap_is_tagged_gapped_and_never_ambiguous():
    """The open is chronologically first, so a gap fill is CERTAIN, not unobserved.

    Collapsing the two tags would make the ambiguous count -- the one section 9
    asks to be measurable -- include fills that are not ambiguous at all.
    """
    resolution = resolve_bar_exit(long_plan(), bar(2, 12_950.0, 12_960.0, 12_940.0, 12_955.0))

    assert resolution.gapped is True
    assert resolution.ambiguous is False


def test_disabling_the_gap_rule_says_in_line_that_it_overstates_results():
    """A setting that flatters results must say so where it is used, not only
    in the schema, because the note is what lands in the journal row."""
    resolution = resolve_bar_exit(
        long_plan(),
        bar(2, 12_950.0, 12_960.0, 12_940.0, 12_955.0),
        fill_gaps_at_open=False,
    )

    assert resolution.price == pytest.approx(12_980.0)
    assert any("overstates" in note for note in resolution.notes)


def test_a_favourable_gap_through_the_target_does_not_collect_the_gap():
    """The same rule run in the trade's favour, deliberately not taken.

    A resting limit receives its limit price; a bar opening 20 points past the
    target does not hand the trade 20 free points. `fill_gaps_at_open` is NOT
    applied here, and that asymmetry is the content of the rule.
    """
    resolution = resolve_bar_exit(long_plan(), bar(2, 13_060.0, 13_080.0, 13_055.0, 13_070.0))

    assert resolution.reason is ExitReason.TARGET
    assert resolution.price == pytest.approx(13_040.0)  # the target, not the 13060 open
    assert resolution.gapped is True
    # Collecting the open would have added 20 points * $20 * 2 = $800 of fiction.
    assert (13_060.0 - 13_040.0) * POINT_VALUE * 2.0 == pytest.approx(800.0)


def test_the_favourable_gap_is_not_collected_even_with_the_gap_rule_off():
    """`fill_gaps_at_open=False` is the flattering setting for stops; it must not
    become the flattering setting for targets as well."""
    resolution = resolve_bar_exit(
        long_plan(),
        bar(2, 13_060.0, 13_080.0, 13_055.0, 13_070.0),
        fill_gaps_at_open=False,
    )
    assert resolution.price == pytest.approx(13_040.0)


def test_the_gap_asymmetry_holds_for_a_short_as_well():
    """A sign error here would invert both rules at once and look symmetric."""
    plan = short_plan()  # stop 13020 above, target 12960 below

    stop_gap = resolve_bar_exit(plan, bar(2, 13_050.0, 13_060.0, 13_045.0, 13_055.0))
    assert stop_gap.reason is ExitReason.STOP
    assert stop_gap.price == pytest.approx(13_050.0)  # the open, 30 points worse

    target_gap = resolve_bar_exit(plan, bar(2, 12_940.0, 12_945.0, 12_930.0, 12_935.0))
    assert target_gap.reason is ExitReason.TARGET
    assert target_gap.price == pytest.approx(12_960.0)  # the target, not the better open


def test_an_entry_that_gaps_past_its_own_stop_is_declined_not_filled():
    """The fictional win: a LONG filled below its stop has stop > entry, so
    "stopping out" books a PROFIT the market never offered.

    Declined rather than filled-and-adjusted, because adjusting means moving
    the stop and `TradeIntent` forbids this module from changing the plan.
    """
    _, fill = filled_entry(bars=[bar(1, 12_950.0, 12_960.0, 12_940.0, 12_955.0)])

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.FILL_BEYOND_STOP
    assert fill.filled_size == 0.0
    assert fill.costs.total == 0.0
    # Had it filled at 12950 with the stop left at 12980, the "stop" would have
    # paid (12980 - 12950) * $20 * 2 = +$1200.
    assert (12_980.0 - 12_950.0) * POINT_VALUE * 2.0 == pytest.approx(1200.0)


def test_an_entry_that_gaps_past_its_own_target_is_declined_and_could_not_be_planned():
    """Past the target the planned reward is already consumed, and the bracket
    inverts the other way: an exit AT the target would be a LOSS wearing a
    TARGET label. `ExitPlan` independently refuses the same geometry.
    """
    _, fill = filled_entry(bars=[bar(1, 13_045.0, 13_050.0, 13_042.0, 13_048.0)])

    assert fill.miss_reason == MissReason.FILL_BEYOND_TARGET
    with pytest.raises(ValueError):
        ExitPlan.from_intent(
            intent(), entry_ts=T0 + timedelta(seconds=INTERVAL), entry_price=13_045.0, size=2.0
        )


def test_a_stop_entry_that_gaps_takes_the_open_not_the_trigger():
    """The entry side of the same rule. A buy stop triggered by a gap fills at
    the open, and the gap belongs in the price path rather than hidden behind
    the trigger price.
    """
    _, fill = filled_entry(
        exec_model=model(entry_order_type="stop"),
        bars=[bar(1, 13_005.0, 13_010.0, 13_002.0, 13_008.0)],
    )

    assert fill.status is FillStatus.FILLED
    assert fill.reference_price == pytest.approx(13_005.0)  # the open, not the 13000 trigger
    # 5 points worse than the trigger: 5 * $20.00 * 2 contracts = $200, charged
    # to the price path rather than hidden behind the trigger.
    assert (fill.reference_price - 13_000.0) * POINT_VALUE * 2.0 == pytest.approx(200.0)
    # It also pays the stop-order surcharge: (1.0 + 0.125 + 1.0) ticks * $5 * 2.
    assert fill.costs.entry_slippage == pytest.approx(21.25)


# ===========================================================================
# 3. Same-bar ambiguity -- the tag has to be load-bearing
# ===========================================================================


def test_a_range_holding_both_levels_resolves_to_the_stop_and_is_tagged():
    """The touch order was never observed, so the resolution goes against the
    trade and the missing observation is recorded."""
    resolution = resolve_bar_exit(long_plan(), bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0))

    assert resolution.reason is ExitReason.STOP
    assert resolution.price == pytest.approx(12_980.0)
    assert resolution.ambiguous is True
    assert resolution.gapped is False  # nothing gapped; the open was inside both levels


def test_the_optimistic_setting_flips_the_outcome_and_keeps_the_tag():
    """The tag records the missing observation, not the configuration.

    Untagging the optimistic setting is the one change that would make a run's
    sensitivity to this assumption invisible, which is the opposite of what
    section 9 asks the tag for.
    """
    both = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)
    optimistic = resolve_bar_exit(long_plan(), both, pessimistic_same_bar=False)

    assert optimistic.reason is ExitReason.TARGET
    assert optimistic.price == pytest.approx(13_040.0)
    assert optimistic.ambiguous is True


def test_the_same_bar_assumption_is_worth_2426_dollars_on_one_bar():
    """The sensitivity section 9 asks to be measurable, measured.

    One constructed bar, one setting, both outcomes priced end to end. If this
    number were small the assumption would not matter; it is three times the
    trade's own risk.
    """
    order_intent, fill = filled_entry()
    plan = plan_from(order_intent, fill)
    both = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)

    pessimistic = model().resolve_exit(plan, both, vol=MEASURED)
    optimistic = model(pessimistic_same_bar=False).resolve_exit(plan, both, vol=MEASURED)

    # Stop: (12980 - 13002) * $20 * 2 = -$880, less $47.74 of costs = -$927.74.
    stop_net = (12_980.0 - 13_002.0) * POINT_VALUE * 2.0 - ROUND_TURN_VIA_STOP
    # Target: (13040 - 13002) * $20 * 2 = +$1520, less $21.49 = +$1498.51.
    target_net = (13_040.0 - 13_002.0) * POINT_VALUE * 2.0 - ROUND_TURN_VIA_TARGET
    assert stop_net == pytest.approx(-927.74)
    assert target_net == pytest.approx(1498.51)
    assert target_net - stop_net == pytest.approx(2426.25)

    assert pessimistic.exit_reason is ExitReason.STOP
    assert optimistic.exit_reason is ExitReason.TARGET
    assert pessimistic.ambiguous_fill and optimistic.ambiguous_fill
    assert model().round_turn_costs(fill, pessimistic).total == pytest.approx(
        ROUND_TURN_VIA_STOP
    )
    assert model().round_turn_costs(fill, optimistic).total == pytest.approx(
        ROUND_TURN_VIA_TARGET
    )
    # In R, against $800 of risk: -1.1597R versus +1.8731R.
    assert stop_net / RISK_DOLLARS == pytest.approx(-1.159675)
    assert target_net / RISK_DOLLARS == pytest.approx(1.873138, abs=1e-6)


def test_a_stop_triggers_on_a_touch_but_a_target_needs_a_trade_through():
    """Inclusive for the stop, strict for the target, and both are asymmetric
    on purpose: a print at our stop fires a market order, while a print at our
    resting limit does not prove our order filled -- queue position is
    unobservable from bar data.
    """
    plan = long_plan()
    # Exactly AT the stop: fires.
    touched = resolve_bar_exit(plan, bar(2, 13_000.0, 13_010.0, 12_980.0, 12_990.0))
    assert touched is not None and touched.reason is ExitReason.STOP
    assert touched.ambiguous is False

    # Exactly AT the target: not a fill.
    reached = resolve_bar_exit(plan, bar(2, 13_000.0, 13_040.0, 12_995.0, 13_030.0))
    assert reached is None


def test_a_bar_opening_exactly_at_the_target_is_not_a_fill():
    """REGRESSION (HIGH). The gap branch used `>=`, so a bar whose OPEN was
    exactly at the target booked a full target win -- tagged `gapped` rather
    than `ambiguous`, and bypassing the strict rule the module applies to the
    same level two lines later. Bar opens and target prices both sit on the
    tick grid, so exact equality is ordinary rather than a corner case.

    Here the bar opens at the target, never trades above it, and then falls
    through the stop. The correct resolution is the STOP.
    """
    plan = long_plan()
    opened_at_target = bar(2, 13_040.0, 13_040.0, 12_970.0, 12_975.0)
    resolution = resolve_bar_exit(plan, opened_at_target)

    assert resolution.reason is ExitReason.STOP
    assert resolution.price == pytest.approx(12_980.0)
    assert resolution.gapped is False
    # The bug booked (13040 - 13002) * $20 * 2 = +$1520 where the honest answer
    # is (12980 - 13002) * $20 * 2 = -$880: a $2400 swing on one bar.
    assert (13_040.0 - 13_002.0) * POINT_VALUE * 2.0 - (
        12_980.0 - 13_002.0
    ) * POINT_VALUE * 2.0 == pytest.approx(2400.0)

    # The SHORT mirror, which used `<=` for the same reason.
    short = short_plan()
    short_resolution = resolve_bar_exit(
        short, bar(2, 12_960.0, 13_030.0, 12_960.0, 13_025.0)
    )
    assert short_resolution.reason is ExitReason.STOP


def test_a_bar_opening_above_the_target_still_fills_at_the_target():
    """The fix must not break the real gap: strictly through is still through."""
    resolution = resolve_bar_exit(long_plan(), bar(2, 13_040.25, 13_050.0, 13_035.0, 13_045.0))

    assert resolution.reason is ExitReason.TARGET
    assert resolution.price == pytest.approx(13_040.0)
    assert resolution.gapped is True


def test_a_bar_reaching_neither_level_leaves_the_position_open_and_costs_nothing():
    """An unresolved bar must not quietly book a flat exit."""
    quiet = model().resolve_exit(
        long_plan(), bar(2, 13_000.0, 13_010.0, 12_995.0, 13_005.0), vol=MEASURED
    )

    assert quiet.resolved is False
    assert quiet.exit_reason is None
    assert quiet.reference_price is None
    assert quiet.costs.total == 0.0
    with pytest.raises(ExecutionError):
        model().round_turn_costs(filled_entry()[1], quiet)


# ===========================================================================
# 4. Intrabar resolution -- "available" has to mean usable
# ===========================================================================


def _finer(rows, *, base: datetime, interval: int = 60, symbol: str = "NQ"):
    """Finer bars, one per row, closing `interval` apart from `base`."""
    return [
        Bar(
            symbol=symbol,
            close_ts=base + timedelta(seconds=interval * i),
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=2_000.0,
            interval_seconds=interval,
        )
        for i, (o, h, lo, c) in enumerate(rows, start=1)
    ]


def test_finer_bars_covering_the_parent_resolve_the_ordering_by_observation():
    """When the ordering IS observed, the trade is not tagged ambiguous -- the
    tag would then be a count of coarse data rather than of unknowns."""
    parent = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)
    five = _finer(
        [
            (13_000.0, 13_045.0, 12_999.0, 13_042.0),  # target first
            (13_042.0, 13_044.0, 13_000.0, 13_005.0),
            (13_005.0, 13_010.0, 12_975.0, 12_980.0),  # stop later
            (12_980.0, 12_985.0, 12_976.0, 12_982.0),
            (12_982.0, 13_000.0, 12_980.0, 12_995.0),
        ],
        base=parent.open_ts,
    )
    fill = model().resolve_exit(long_plan(), parent, vol=MEASURED, intrabar=five)

    assert fill.exit_reason is ExitReason.TARGET
    assert fill.ambiguous_fill is False
    assert fill.intrabar_bars_used == 1  # the first resolution wins


def test_finer_bars_with_a_hole_in_the_parents_span_are_not_an_observation():
    """REGRESSION (HIGH). A finer series was accepted whenever it touched both
    of the parent's price extremes, with no check that it covered the parent's
    TIME. Three 60s bars covering the LAST THREE MINUTES of a five-minute
    parent reached its high and its low, were trusted, resolved to the TARGET
    -- and cleared `ambiguous_fill`, so the sensitivity became invisible rather
    than merely unresolved.

    The first two minutes are unobserved, and the stop may have been touched
    there. Touching the parent's extremes is not the same as observing the
    order in which they were touched, which is the identical defect the module
    already rejects for partial PRICE coverage.
    """
    parent = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)
    last_three_minutes = _finer(
        [
            (13_000.0, 13_045.0, 13_000.0, 13_042.0),
            (13_042.0, 13_045.0, 12_975.0, 12_980.0),
            (12_980.0, 13_000.0, 12_975.0, 12_995.0),
        ],
        base=parent.open_ts + timedelta(seconds=120),
    )
    usable, notes = validated_intrabar(long_plan(), parent, last_three_minutes)

    assert usable == ()
    assert any("time span" in note for note in notes)

    fill = model().resolve_exit(
        long_plan(), parent, vol=MEASURED, intrabar=last_three_minutes
    )
    assert fill.exit_reason is ExitReason.STOP  # the pessimistic rule applies
    assert fill.ambiguous_fill is True
    assert fill.intrabar_bars_used == 0


def test_a_gap_in_the_middle_of_a_finer_series_is_also_a_hole():
    """A missing minute cannot be told apart from a minute with no trades, and
    only one of those two readings is safe to resolve an ambiguity on."""
    parent = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)
    rows = [
        (13_000.0, 13_045.0, 12_999.0, 13_042.0),
        (13_042.0, 13_044.0, 13_000.0, 13_005.0),
        (13_005.0, 13_010.0, 12_975.0, 12_980.0),
        (12_980.0, 12_985.0, 12_976.0, 12_982.0),
        (12_982.0, 13_000.0, 12_980.0, 12_995.0),
    ]
    complete = _finer(rows, base=parent.open_ts)
    # Drop the FOURTH minute, which carries neither the parent's high nor its
    # low -- so the price-coverage check still passes and the hole in time is
    # the only defect left for the series to be rejected on.
    punctured = [b for b in complete if b is not complete[3]]

    assert validated_intrabar(long_plan(), parent, complete)[0] != ()
    usable, notes = validated_intrabar(long_plan(), parent, punctured)
    assert usable == ()
    assert any("discontinuity" in note for note in notes)


def test_finer_bars_that_miss_the_parents_extremes_fall_back_to_pessimism():
    """Partial price coverage decides the ambiguity by which half of the bar
    happened to be supplied."""
    parent = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)
    calm_middle = _finer(
        [(13_000.0, 13_005.0, 12_998.0, 13_002.0)] * 5, base=parent.open_ts
    )
    fill = model().resolve_exit(long_plan(), parent, vol=MEASURED, intrabar=calm_middle)

    assert fill.exit_reason is ExitReason.STOP
    assert fill.ambiguous_fill is True
    assert any("does not cover" in note for note in fill.notes)


def test_finer_bars_reaching_further_than_the_parent_are_used_and_reported():
    """A larger excursion is evidence, not noise -- but the two series disagree
    and that is a data-quality finding rather than something to smooth over."""
    parent = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)
    deeper = _finer(
        [
            (13_000.0, 13_010.0, 12_970.0, 12_980.0),  # 5 points below the parent's low
            (12_980.0, 13_045.0, 12_975.0, 13_042.0),
            (13_042.0, 13_045.0, 13_000.0, 13_005.0),
            (13_005.0, 13_010.0, 12_999.0, 13_002.0),
            (13_002.0, 13_010.0, 13_000.0, 13_010.0),
        ],
        base=parent.open_ts,
    )
    fill = model().resolve_exit(long_plan(), parent, vol=MEASURED, intrabar=deeper)

    assert fill.exit_reason is ExitReason.STOP
    assert fill.ambiguous_fill is False
    assert any("reach further" in note for note in fill.notes)


def test_intrabar_bars_that_are_not_finer_or_not_inside_the_parent_raise():
    """A "finer" series at the parent's own interval observes nothing extra, and
    a bar from outside the parent is either a duplicate or data from the future.
    """
    plan = long_plan()
    parent = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)

    same_interval = _finer(
        [(13_000.0, 13_045.0, 12_975.0, 13_010.0)], base=parent.open_ts, interval=INTERVAL
    )
    with pytest.raises(ExecutionError):
        validated_intrabar(plan, parent, same_interval)

    outside = _finer([(13_000.0, 13_045.0, 12_975.0, 13_010.0)], base=parent.close_ts)
    with pytest.raises(FillTimingError):
        validated_intrabar(plan, parent, outside)

    wrong_symbol = _finer(
        [(13_000.0, 13_045.0, 12_975.0, 13_010.0)], base=parent.open_ts, symbol="ES"
    )
    with pytest.raises(ExecutionError):
        validated_intrabar(plan, parent, wrong_symbol)


def test_an_ambiguous_finest_bar_is_tagged_at_that_granularity():
    """Finer data narrows the ambiguity; it does not necessarily remove it, and
    a resolution that still had to guess must still say so."""
    parent = bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0)
    both_in_one_minute = _finer(
        [
            (13_000.0, 13_045.0, 12_975.0, 13_010.0),  # both levels inside one 60s bar
            (13_010.0, 13_012.0, 13_005.0, 13_008.0),
            (13_008.0, 13_010.0, 13_000.0, 13_002.0),
            (13_002.0, 13_005.0, 13_000.0, 13_004.0),
            (13_004.0, 13_010.0, 13_002.0, 13_010.0),
        ],
        base=parent.open_ts,
    )
    fill = model().resolve_exit(long_plan(), parent, vol=MEASURED, intrabar=both_in_one_minute)

    assert fill.exit_reason is ExitReason.STOP
    assert fill.ambiguous_fill is True


# ===========================================================================
# 5. Partial fills -- a partial is not a whole one
# ===========================================================================


def test_the_fill_is_capped_at_the_participation_rate_times_bar_volume():
    """Section 9: `size filled ~ min(size, participation_rate * bar_volume)`.

    A 30-contract bar at 5% participation yields 1.5 contracts, and NQ does not
    trade fractional contracts, so the fill is 1 -- not 1.5, and not 2.
    """
    _, fill = filled_entry(bars=[entry_bar(volume=30.0)])

    assert fill.status is FillStatus.PARTIAL
    assert fill.participation_cap == pytest.approx(1.5)  # 0.05 * 30
    assert fill.filled_size == pytest.approx(1.0)  # floored to the size grid
    assert fill.fill_fraction == pytest.approx(0.5)
    assert any("floored" in note for note in fill.notes)


def test_a_partial_fill_is_charged_on_the_filled_size_not_the_requested_one():
    """Charging the requested size overstates costs; charging the full fill's
    costs against a half position understates the hurdle per contract. Both are
    the same bug and this is the number that catches it.
    """
    _, partial = filled_entry(bars=[entry_bar(volume=30.0)])

    # One contract: 1.125 ticks * $5 * 1 = $5.625 slippage, 0.5 * $5 * 1 = $2.50
    # spread, plus $2.25 + $0.37 charged per ORDER regardless of size.
    assert partial.costs.entry_slippage == pytest.approx(5.625)
    assert partial.costs.spread_cost == pytest.approx(2.5)
    assert partial.costs.commission == pytest.approx(2.25)
    assert partial.costs.total == pytest.approx(10.745)
    # Not the two-contract figure.
    assert partial.costs.total != pytest.approx(ENTRY_COST)


def test_a_fractional_instrument_keeps_the_cap_unrounded():
    """The size grid, not the participation rule, is what turns 1.5 into 1 --
    so where fractional size is legal the cap must survive intact."""
    order_intent = intent(symbol="QQQ", entry=400.0, stop=399.0, target=402.0, size=2.0,
                          point_value=1.0)
    exec_model = ExecutionModel(config(), QQQ_FRACTIONAL, master_seed=1)
    order = exec_model.submit(order_intent, T0)
    window = exec_model.open_window(
        order, [bar(1, 400.2, 401.0, 399.5, 400.5, volume=30.0, symbol="QQQ")]
    )
    fill = exec_model.resolve_entry(order, window, vol=MEASURED)

    assert fill.filled_size == pytest.approx(1.5)  # 0.05 * 30, unfloored
    assert fill.fill_fraction == pytest.approx(0.75)
    assert fill.status is FillStatus.PARTIAL


def test_a_fill_below_min_partial_fraction_abandons_the_entry():
    """The field's own description prescribes this: "below this fill fraction,
    abandon the entry". A quarter position is not the setup that was tested.
    """
    # 4 contracts requested, 1.5 available, floored to 1 -> fraction 0.25 < 0.50.
    _, fill = filled_entry(order_intent=intent(size=4.0), bars=[entry_bar(volume=30.0)])

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.PARTICIPATION_BELOW_MINIMUM
    assert fill.filled_size == 0.0
    assert fill.costs.total == 0.0


def test_a_fraction_exactly_at_the_minimum_is_allowed():
    """The boundary belongs on the permissive side of its own description
    ("below this ... abandon"), and a half-position at exactly the configured
    floor is the case most likely to be mis-specified by an off-by-one.
    """
    _, fill = filled_entry(bars=[entry_bar(volume=30.0)])  # fraction exactly 0.50

    assert fill.status is FillStatus.PARTIAL
    assert fill.fill_fraction == pytest.approx(config().min_partial_fraction)


def test_partial_fills_disabled_abandons_rather_than_filling_whole():
    """Filling whole takes more of the bar's volume than the cap says we can,
    which is the participation limit deleted rather than respected."""
    _, fill = filled_entry(
        exec_model=model(allow_partial_fills=False), bars=[entry_bar(volume=30.0)]
    )

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.PARTIAL_FILLS_DISABLED
    assert fill.filled_size == 0.0


def test_a_cap_that_rounds_to_zero_is_a_miss_with_its_own_reason():
    """Section 12's rejection analysis asks how often each cause fires, so a
    sub-minimum size must not report the same code as a thin partial."""
    # 0.05 * 10 = 0.5 contracts, floored to 0, below NQ's min_size of 1.
    _, fill = filled_entry(bars=[entry_bar(volume=10.0)])

    assert fill.miss_reason == MissReason.SIZE_ROUNDS_TO_ZERO


def test_the_instruments_max_size_binds_and_is_visible_in_the_notes():
    """`risk/sizing.py` never consults `max_size`, so this clamp is the only
    thing between an oversized intent and a fill the instrument cannot take.
    It must bind, charge costs on the clamped size, and say that it bound.
    """
    exec_model = ExecutionModel(config(), NQ.replace(max_size=1.0), master_seed=1)
    order_intent, fill = filled_entry(exec_model=exec_model)

    assert fill.status is FillStatus.PARTIAL
    assert fill.filled_size == pytest.approx(1.0)
    assert any("max_size" in note for note in fill.notes)
    assert fill.costs.entry_slippage == pytest.approx(5.625)  # one contract, not two
    assert order_intent.size == 2.0  # the plan itself is untouched


def test_a_partial_fill_does_not_rescale_the_stop_to_restore_r():
    """Section 0.2: R is fixed at entry and never revised. Rescaling the stop to
    make a half position realize a full R holds R constant by moving the
    thesis, and the note has to say so where the journal can see it.
    """
    order_intent, fill = filled_entry(bars=[entry_bar(volume=30.0)])
    plan = plan_from(order_intent, fill)

    assert plan.stop_price == order_intent.stop_price
    assert plan.target_price == order_intent.target_price
    assert plan.size == pytest.approx(1.0)
    assert any("rescales" in note for note in fill.notes)


# ===========================================================================
# 6. Missed entries -- a miss is a different journal row, not a zero
# ===========================================================================


def test_a_limit_never_traded_through_is_missed_not_filled():
    """Section 9: "a limit entry is missed if price never trades through by
    max_entry_wait_bars". The second bar TOUCHES the limit exactly, which is
    not evidence of a fill: queue position is unobservable from bar data.
    """
    exec_model = model(entry_order_type="limit", max_entry_wait_bars=2)
    _, fill = filled_entry(
        exec_model=exec_model,
        bars=[
            bar(1, 13_002.0, 13_010.0, 13_000.5, 13_006.0),  # never reaches 13000
            bar(2, 13_006.0, 13_012.0, 13_000.0, 13_008.0),  # touches it exactly
        ],
    )

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.LIMIT_NOT_TRADED_THROUGH


def test_a_missed_entry_is_not_a_taken_trade_with_zero_pnl():
    """Different journal rows. A miss has no price, no timestamp, no size and
    no costs, and cannot be turned into a round turn -- a zero-PnL trade would
    dilute every per-trade statistic in the system with trades that never
    happened.
    """
    exec_model = model(entry_order_type="limit", max_entry_wait_bars=1)
    _, fill = filled_entry(
        exec_model=exec_model, bars=[bar(1, 13_002.0, 13_010.0, 13_000.5, 13_006.0)]
    )

    assert fill.filled is False
    assert fill.filled_size == 0.0
    assert fill.fill_ts is None
    assert fill.reference_price is None
    assert fill.effective_price is None
    assert fill.costs.total == 0.0
    assert fill.miss_reason  # always states a cause
    with pytest.raises(ExecutionError):
        exec_model.round_turn_costs(
            fill, ExitFill(resolved=False, symbol="NQ", side=Side.LONG, size=2.0)
        )


def test_a_missed_fill_record_cannot_carry_a_price_or_a_cost():
    """The contract, not the code path: a MISSED row with costs on it would be
    a trade that paid commission without existing."""
    base = dict(
        status=FillStatus.MISSED,
        symbol="NQ",
        side=Side.LONG,
        signal_ts=T0,
        submitted_ts=T0,
        eligible_ts=T0,
        order_type="market",
        planned_price=13_000.0,
        requested_size=2.0,
        miss_reason=MissReason.RANDOM_MISS,
    )
    EntryFill(**base)  # the honest row
    with pytest.raises(ValueError):
        EntryFill(**base, costs=CostBreakdown(commission=2.25))
    with pytest.raises(ValueError):
        EntryFill(**{**base, "miss_reason": ""})


def test_a_limit_that_trades_through_fills_at_its_limit_and_pays_no_slippage():
    """The other side of miss risk: when a resting limit does fill, it fills at
    its price. Charging it slippage would book a cost the market cannot impose.
    """
    exec_model = model(entry_order_type="limit", max_entry_wait_bars=2)
    _, fill = filled_entry(
        exec_model=exec_model,
        bars=[
            bar(1, 13_002.0, 13_010.0, 13_000.5, 13_006.0),
            bar(2, 13_002.0, 13_004.0, 12_999.75, 13_001.0),  # through, by one tick
        ],
    )

    assert fill.status is FillStatus.FILLED
    assert fill.reference_price == pytest.approx(13_000.0)  # the limit
    assert fill.fill_bar_offset == 2
    assert fill.costs.entry_slippage == 0.0
    assert fill.costs.spread_cost == 0.0
    assert fill.costs.total == pytest.approx(2.62)  # commission and fees only
    assert fill.effective_price == pytest.approx(fill.reference_price)


def test_a_market_order_is_not_re_offered_on_later_bars():
    """`max_entry_wait_bars` is scoped to RESTING entries. Re-offering a market
    order would let a backtest chase an entry it missed, at prices the signal
    never saw.
    """
    exec_model = model(max_entry_wait_bars=5)
    order = exec_model.submit(intent(), T0)
    window = exec_model.open_window(
        order, [entry_bar(), bar(2, 13_010.0, 13_020.0, 13_005.0, 13_015.0)]
    )

    assert len(window) == 1
    assert [offset for offset, _ in window.enumerate_offsets()] == [1]


def test_a_stop_entry_never_touched_is_missed_with_its_own_reason():
    """A stop entry and a limit entry fail for different reasons, and the
    rejection analysis counts causes rather than refusals."""
    exec_model = model(entry_order_type="stop", max_entry_wait_bars=1)
    _, fill = filled_entry(
        exec_model=exec_model, bars=[bar(1, 12_995.0, 12_999.0, 12_990.0, 12_996.0)]
    )

    assert fill.miss_reason == MissReason.STOP_NOT_TRIGGERED


def test_a_resting_entry_that_can_never_fill_is_refused_at_construction():
    """`ExecutionConfig` rejects `limit` with `max_entry_wait_bars=0` but not
    `stop`, which has the identical property. A config stating an impossible
    intention fails loudly rather than returning `no_eligible_bar` on every bar
    of a ten-year run.
    """
    with pytest.raises(ExecutionError):
        ExecutionModel(
            config(entry_order_type="stop", max_entry_wait_bars=0), NQ, master_seed=1
        )
    # The market case is legal: a market order ignores the wait horizon.
    ExecutionModel(config(entry_order_type="market", max_entry_wait_bars=0), NQ, master_seed=1)


# ===========================================================================
# 7. Fill timing -- the lookahead surface
# ===========================================================================


def test_the_eligibility_floor_is_one_bar_at_every_latency():
    """Section 9's loop invariant as arithmetic. There is no configuration,
    `latency_ms=0` included, that returns 0."""
    for latency in (0.0, 0.25, 1.0, 150.0, 299.999, 300.0, 1_000_000.0):
        assert min_fill_offset(latency_seconds=latency, interval_seconds=INTERVAL) >= 1


def test_the_floor_matches_the_latency_in_bars_by_hand():
    """250ms against 300s bars is the next bar; 400s skips one; 700s skips two.

    The table is hand-derived from `floor(latency/interval) + 1`, which is the
    first bar whose OPEN is at or after the eligibility instant.
    """
    assert min_fill_offset(latency_seconds=0.25, interval_seconds=INTERVAL) == 1
    assert min_fill_offset(latency_seconds=299.999, interval_seconds=INTERVAL) == 1
    assert min_fill_offset(latency_seconds=300.0, interval_seconds=INTERVAL) == 2
    assert min_fill_offset(latency_seconds=400.0, interval_seconds=INTERVAL) == 2
    assert min_fill_offset(latency_seconds=600.0, interval_seconds=INTERVAL) == 3
    assert min_fill_offset(latency_seconds=700.0, interval_seconds=INTERVAL) == 3
    # entry_delay_bars is additive on top, for robustness testing.
    assert min_fill_offset(latency_seconds=0.25, interval_seconds=INTERVAL, entry_delay_bars=2) == 3


def test_the_residual_lag_is_always_under_one_bar():
    """The invariant the `ceil` form broke, stated as the thing it actually is.

    The fill reference is the chosen bar's OPEN, which precedes eligibility by
    `latency - (offset - 1) * interval`. `fill_timing.py` claims that residual
    is "250ms at the default latency, and `latency mod interval` whenever the
    latency exceeds a bar", and the claim has to hold at EVERY latency: with
    `max(1, ceil(...))` it was a full interval at every exact multiple of the
    bar, which is the one place the model paid nothing for latency.

    Monotonicity is asserted alongside it because the `ceil` form satisfied
    monotonicity -- it extended a plateau rather than going backwards -- so
    monotonicity alone would not have found the bug.
    """
    latencies = [seconds / 4.0 for seconds in range(0, 6_000)]
    offsets = [
        min_fill_offset(latency_seconds=latency, interval_seconds=INTERVAL)
        for latency in latencies
    ]
    assert offsets == sorted(offsets)
    for latency, offset in zip(latencies, offsets):
        lag = latency - (offset - 1) * INTERVAL
        assert 0.0 <= lag < INTERVAL, f"latency {latency}s left a {lag}s residual"


def test_a_latency_of_exactly_one_bar_does_not_fill_at_the_signal_price():
    """REGRESSION (MEDIUM). `max(1, ceil(latency/interval))` returned offset 1
    for a latency of exactly one bar, and offset 1's OPEN print IS the signal
    instant -- so 300 seconds of configured latency bought a fill at the price
    the signal itself saw, and `eligibility_lag_seconds` recorded a full bar.
    Configuring more latency produced a zero-latency fill.

    The honest answer is the first bar whose open is at or after eligibility.
    """
    exec_model = model(latency_ms=300_000.0)
    order = exec_model.submit(intent(), T0)
    window = exec_model.open_window(
        order,
        [
            bar(1, 13_002.0, 13_010.0, 12_999.0, 13_006.0),
            bar(2, 13_012.0, 13_020.0, 13_009.0, 13_016.0),
        ],
    )
    fill = exec_model.resolve_entry(order, window, vol=MEASURED)

    assert order.eligible_ts == T0 + timedelta(seconds=300)
    assert fill.fill_bar_offset == 2
    assert fill.reference_price == pytest.approx(13_012.0)  # not the 13002 signal-instant open
    # The fill bar opens exactly when the order became eligible, so no residual.
    assert fill.eligibility_lag_seconds == pytest.approx(0.0)


def test_the_default_latency_is_eligible_on_the_next_bar_with_a_recorded_residual():
    """250ms against 300s bars: the next bar, and the 0.25s by which that bar's
    OPEN print precedes eligibility is recorded per fill rather than asserted
    negligible. Bar data holds no finer price, so this is the one residual that
    cannot be removed at this granularity -- it is measured instead.
    """
    _, fill = filled_entry()

    assert fill.fill_bar_offset == 1
    assert fill.fill_ts == T0 + timedelta(seconds=INTERVAL)
    assert fill.eligibility_lag_seconds == pytest.approx(0.25)
    assert any("predates eligibility" in note for note in fill.notes)


def test_the_signal_bar_is_dropped_at_construction_not_merely_unrequested():
    """`FillWindow` copies the `MarketView` pattern: the signal bar is ABSENT.
    Code that reaches into a private attribute still cannot find it.
    """
    order = model().submit(intent(), T0)
    signal_bar = bar(0, 12_990.0, 13_005.0, 12_985.0, 13_000.0)
    window = model().open_window(order, [signal_bar, entry_bar()])

    assert len(window) == 1
    assert all(candidate.close_ts > T0 for candidate in window.bars)
    held = [
        value
        for name in FillWindow.__slots__
        for value in (getattr(window, name),)
    ]
    flattened = [
        item
        for value in held
        for item in (value if isinstance(value, tuple) else (value,))
    ]
    assert signal_bar not in flattened
    assert any("dropped at construction" in note for note in window.notes)


def test_a_window_holding_only_the_signal_bar_misses_rather_than_filling_there():
    """The "data ran out" path has to be an explicit miss. A silent default here
    would be indistinguishable from a fill that was never attempted."""
    order = model().submit(intent(), T0)
    window = model().open_window(order, [bar(0, 12_990.0, 13_005.0, 12_985.0, 13_000.0)])
    fill = model().resolve_entry(order, window, vol=MEASURED)

    assert len(window) == 0
    assert fill.miss_reason == MissReason.NO_ELIGIBLE_BAR


def test_a_bar_straddling_the_signal_instant_is_refused():
    """A mid-bar signal timestamp means part of the candidate bar's range
    PREDATES the signal, so a fill in it could use a price the signal was
    derived from. That is a clock bug in the caller and it raises rather than
    being silently dropped, because dropping it hides the bug.
    """
    mid_bar_signal = intent(signal_ts=T0 - timedelta(seconds=100))
    order = model().submit(mid_bar_signal, mid_bar_signal.signal_ts)

    with pytest.raises(FillTimingError):
        model().open_window(order, [bar(0, 12_990.0, 13_005.0, 12_985.0, 13_000.0)])


def test_the_window_audit_inspects_every_retained_bar_not_just_the_first():
    """`MarketView.assert_no_future_leak` once inspected one row and so could not
    see a leak in another. This is a white-box poke at the same shape: a
    pre-signal bar is forced into the window's private tuple in the SECOND
    position, where only a loop over every candidate can find it.

    Reached by reaching in, deliberately. With a uniform interval and an
    ascending list, construction cannot produce a bad non-first bar -- if the
    first candidate passes, every later one does -- so the loop is defence in
    depth, and this test is what keeps it that way rather than letting it decay
    into a check of row one.
    """
    exec_model = model(entry_order_type="limit", max_entry_wait_bars=3)
    order = exec_model.submit(intent(), T0)
    window = exec_model.open_window(
        order, [entry_bar(), bar(2, 13_006.0, 13_012.0, 13_000.0, 13_008.0)]
    )
    window.assert_resolves_after_signal()  # clean, and idempotent
    window.assert_resolves_after_signal()

    window._bars = window._bars + (bar(0, 12_990.0, 13_005.0, 12_985.0, 13_000.0),)
    with pytest.raises(FillTimingError):
        window.assert_resolves_after_signal()


def test_a_same_bar_fill_cannot_be_recorded_even_by_hand():
    """The last line of defence: `EntryFill`'s own validator. A caller that
    bypasses the window entirely still cannot produce a same-bar fill record.
    """
    base = dict(
        status=FillStatus.FILLED,
        symbol="NQ",
        side=Side.LONG,
        signal_ts=T0,
        submitted_ts=T0,
        eligible_ts=T0,
        order_type="market",
        reference_price=13_000.0,
        planned_price=13_000.0,
        requested_size=2.0,
        filled_size=2.0,
    )
    with pytest.raises(FillTimingError):
        EntryFill(**base, fill_ts=T0, fill_bar_offset=1)  # fill at the signal instant
    with pytest.raises(FillTimingError):
        EntryFill(**base, fill_ts=T0 - timedelta(seconds=1), fill_bar_offset=1)
    with pytest.raises(FillTimingError):
        EntryFill(**base, fill_ts=T0 + timedelta(seconds=INTERVAL), fill_bar_offset=0)


def test_a_window_built_for_one_order_cannot_resolve_another():
    """REGRESSION (MEDIUM). `open_window` and `resolve_entry` are separate public
    calls and nothing checked that the window belonged to the order. A
    `FillWindow` carries ONE order's eligibility floor, so the natural event-loop
    shape -- build one window per bar, resolve every pending order against it --
    filled a slow order at the fast order's bar.

    Here a 400-second order resolves against a window built at the default
    250ms latency: it fills a bar early, 10 points better, and the record it
    produces is internally consistent, so nothing downstream can see it.
    """
    bars = [
        bar(1, 13_002.0, 13_010.0, 12_999.0, 13_006.0),
        bar(2, 13_012.0, 13_020.0, 13_009.0, 13_016.0),
    ]
    quick = model()
    slow = model(latency_ms=400_000.0)
    quick_order = quick.submit(intent(), T0)
    slow_order = slow.submit(intent(), T0)
    cached = quick.open_window(quick_order, bars)

    with pytest.raises(FillTimingError):
        slow.resolve_entry(slow_order, cached, vol=MEASURED)

    # What the crossed pair would have taken: offset 1 at 13002 instead of
    # offset 2 at 13012, which is 10 points * $20 * 2 = $400 on one trade.
    honest = slow.resolve_entry(slow_order, slow.open_window(slow_order, bars), vol=MEASURED)
    assert honest.fill_bar_offset == 2
    assert honest.reference_price == pytest.approx(13_012.0)
    assert (13_012.0 - 13_002.0) * POINT_VALUE * 2.0 == pytest.approx(400.0)


def test_a_fill_before_its_own_eligibility_instant_cannot_be_recorded():
    """REGRESSION (MEDIUM). The structural half: the result record checked that
    the fill came after the SIGNAL but not that it came after ELIGIBILITY, so
    latency an order was never held to was latency that was not modelled.
    """
    with pytest.raises(FillTimingError):
        EntryFill(
            status=FillStatus.FILLED,
            symbol="NQ",
            side=Side.LONG,
            signal_ts=T0,
            submitted_ts=T0,
            eligible_ts=T0 + timedelta(seconds=400),
            order_type="market",
            fill_ts=T0 + timedelta(seconds=INTERVAL),  # before eligibility
            fill_bar_offset=1,
            reference_price=13_002.0,
            planned_price=13_000.0,
            requested_size=2.0,
            filled_size=2.0,
        )


def test_submitting_before_the_signal_or_for_another_symbol_raises():
    """Both are caller contract violations that would otherwise produce a
    perfectly plausible fill."""
    with pytest.raises(FillTimingError):
        model().submit(intent(), T0 - timedelta(seconds=1))
    with pytest.raises(ExecutionError):
        model().submit(intent(symbol="ES"), T0)


def test_an_unordered_or_mixed_interval_bar_list_raises():
    """"The next bar" is meaningless in an unordered list, and the eligibility
    floor is counted in bars, which cannot be counted in two units at once."""
    order = model(entry_order_type="limit", max_entry_wait_bars=3).submit(intent(), T0)
    exec_model = model(entry_order_type="limit", max_entry_wait_bars=3)

    with pytest.raises(ExecutionError):
        exec_model.open_window(order, [bar(2, 13_006.0, 13_012.0, 13_000.0, 13_008.0), entry_bar()])
    with pytest.raises(ExecutionError):
        exec_model.open_window(
            order,
            [entry_bar(), bar(2, 13_006.0, 13_012.0, 13_000.0, 13_008.0, interval=60)],
        )


def test_an_exit_before_the_entry_raises_on_both_exit_paths():
    """An exit priced before its own entry is a negative holding time, and
    `TradeRecord` would reject the row -- but only after the ledger had already
    booked it."""
    plan = long_plan()
    early = bar(0, 13_000.0, 13_010.0, 12_970.0, 12_990.0)

    with pytest.raises(FillTimingError):
        model().resolve_exit(plan, early, vol=MEASURED)
    with pytest.raises(FillTimingError):
        model().exit_at_close(plan, early, ExitReason.SESSION_CLOSE, vol=MEASURED)


# ===========================================================================
# 8. Unmeasurable inputs -- no guess may be a cheap guess
# ===========================================================================


def test_an_unmeasured_volatility_ratio_declines_the_fill():
    """`signals/engine.py` carries the scar: a missing ATR there "does not
    degrade the cap, it REMOVES it", and a 20x-ATR stop was accepted by a setup
    capping at 2x. The analogous mistake here is a free fill in unmeasured
    volatility, available on exactly the bars where volatility is most likely
    to be extreme. The entry is declined by name instead.
    """
    _, fill = filled_entry(vol=VolatilityContext.unmeasured("ATR unavailable"))

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.COST_UNMEASURABLE
    assert fill.costs.total == 0.0


def test_an_unmeasurable_estimate_carries_no_ticks_at_all():
    """A partial slippage figure would be charged as if it were the whole cost,
    so the refusal has to be total rather than a smaller number."""
    estimate = CostModel(config(), NQ).slippage(size=2.0, vol=VolatilityContext.unmeasured())

    assert estimate.measurable is False
    assert estimate.refusal == CostRefusal.VOL_MULTIPLE_UNMEASURABLE
    assert estimate.ticks == 0.0
    # And the contract forbids the half-measure outright.
    with pytest.raises(ValueError):
        SlippageEstimate(measurable=False, refusal="x", base_ticks=1.0)
    with pytest.raises(ValueError):
        SlippageEstimate(measurable=False)  # a refusal must state its reason


def test_costs_cannot_be_assembled_from_an_unmeasurable_estimate():
    """Pricing an unmeasured fill at zero slippage is the flattering reading of
    a measurement that was never taken, so the assembly itself refuses."""
    costs = CostModel(config(), NQ)
    estimate = costs.slippage(size=2.0, vol=VolatilityContext.unmeasured())

    with pytest.raises(CostModelError):
        costs.entry_costs(size=2.0, spread=costs.spread(), slippage=estimate)


def test_an_absent_ratio_is_not_a_ratio_of_one():
    """"Not measured" and "exactly at its own median" are different statements,
    and the collapse of the first into the second is the whole bug class: 1.0 is
    the value that makes the volatility term vanish.
    """
    assert VolatilityContext.unmeasured().is_measured is False
    assert VolatilityContext.unmeasured().multiple is None
    assert VolatilityContext(vol_multiple=1.0).is_measured is True
    assert VolatilityContext(vol_multiple=1.0).multiple == pytest.approx(1.0)

    at_median = CostModel(config(), NQ).slippage(size=2.0, vol=VolatilityContext(vol_multiple=1.0))
    assert at_median.measurable is True
    assert at_median.vol_ticks == 0.0


def test_a_nan_or_non_positive_atr_is_refused_rather_than_used():
    """A feature layer that emits NaN must not be able to launder it into a
    measured ratio. NaN compares False against every threshold, so an
    unchecked NaN ATR produces a term that silently vanishes."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            VolatilityContext(atr=bad, median_atr=10.0)
        with pytest.raises(ValueError):
            VolatilityContext(vol_multiple=bad)

    # A non-positive ATR is an unset field or a flat window, not a calm market.
    assert VolatilityContext(atr=0.0, median_atr=10.0).multiple is None
    assert VolatilityContext(atr=12.5, median_atr=0.0).multiple is None
    assert VolatilityContext(vol_multiple=-1.0).multiple is None


def test_the_ratio_only_matters_when_the_coefficient_uses_it():
    """The one declared exception: at `vol_slippage_coefficient = 0.0` the ratio
    is not an input to anything, so its absence cannot flatter a result. The
    degradation is still declared in the notes rather than silent.
    """
    estimate = CostModel(config(vol_slippage_coefficient=0.0), NQ).slippage(
        size=2.0, vol=VolatilityContext.unmeasured()
    )

    assert estimate.measurable is True
    assert estimate.ticks == pytest.approx(1.0)  # base_slippage_ticks alone
    assert any("not an input" in note for note in estimate.notes)

    _, fill = filled_entry(
        exec_model=model(vol_slippage_coefficient=0.0),
        vol=VolatilityContext.unmeasured(),
    )
    assert fill.status is FillStatus.FILLED
    # 1.0 tick * $5 * 2 = $10.00 of slippage, against $11.25 with the vol term.
    assert fill.costs.entry_slippage == pytest.approx(10.0)


def test_below_median_volatility_is_cheap_but_never_a_rebate():
    """`coefficient * (multiple - 1)` is NEGATIVE in a quiet market, and a
    negative slippage pays the trade to trade. Quiet markets are cheaper than
    volatile ones because the term is zero, not because the term is negative.
    """
    assert vol_slippage_ticks(0.5, 0.6) == 0.0  # unclamped: 0.5 * -0.4 = -0.2
    quiet = CostModel(config(), NQ).slippage(size=2.0, vol=VolatilityContext(vol_multiple=0.6))

    assert quiet.vol_ticks == 0.0
    assert quiet.ticks == pytest.approx(1.0)  # the base, not less than it
    assert quiet.ticks >= config().base_slippage_ticks


def test_a_bar_with_zero_volume_declines_the_fill():
    """The participation limit cannot be evaluated and there is no configured
    volume fallback. Assuming unconstrained liquidity is the cheap guess, and
    it is cheapest exactly on the thinnest bars.
    """
    _, fill = filled_entry(bars=[entry_bar(volume=0.0)])

    assert fill.status is FillStatus.MISSED
    assert fill.miss_reason == MissReason.NO_VOLUME_MEASUREMENT
    assert fill.filled_size == 0.0


def test_a_bar_with_no_volume_measurement_never_produces_a_fill():
    """A NaN volume is structurally invalid data rather than a market
    condition, so it raises at window construction instead of becoming a miss.
    Either answer satisfies the invariant that matters -- that nothing fills --
    and the assertion is written against that invariant rather than against the
    branch that currently implements it.
    """
    order = model().submit(intent(), T0)
    nan_volume = entry_bar(volume=float("nan"))

    assert nan_volume.is_valid is False
    try:
        window = model().open_window(order, [nan_volume])
        fill = model().resolve_entry(order, window, vol=MEASURED)
    except ExecutionError:
        return
    assert fill.filled is False, "a bar with no volume measurement must never fill"


def test_a_structurally_invalid_bar_never_prices_anything():
    """A NaN price is not a price, and it survives every downstream comparison
    silently -- `math.isclose(nan, nan)` is False, so the row is rejected only
    long after the ledger has marked a position out at it.
    """
    nan_close = Bar(
        symbol="NQ",
        close_ts=T0 + timedelta(seconds=INTERVAL * 2),
        open=13_000.0,
        high=13_010.0,
        low=12_990.0,
        close=float("nan"),
        volume=1_000.0,
        interval_seconds=INTERVAL,
    )
    assert nan_close.is_valid is False

    with pytest.raises(ExecutionError):
        model().resolve_exit(long_plan(), nan_close, vol=MEASURED)


def test_exit_at_close_refuses_a_structurally_invalid_bar():
    """REGRESSION (MEDIUM). `resolve_exit` checked `bar.is_valid`; `exit_at_close`
    did not -- and it is the path that uses `bar.close` AS the exit price. A NaN
    close produced `reference_price = nan` on a fill carrying $18.87 of real
    costs, and a session flatten is exactly where a broken final bar appears.
    """
    for broken in (
        Bar(
            symbol="NQ",
            close_ts=T0 + timedelta(seconds=INTERVAL * 2),
            open=13_000.0,
            high=13_010.0,
            low=12_990.0,
            close=float("nan"),
            volume=1_000.0,
            interval_seconds=INTERVAL,
        ),
        Bar(
            symbol="NQ",
            close_ts=T0 + timedelta(seconds=INTERVAL * 2),
            open=13_000.0,
            high=13_010.0,
            low=12_990.0,
            close=13_000.0,
            volume=-5.0,
            interval_seconds=INTERVAL,
        ),
    ):
        with pytest.raises(ExecutionError):
            model().exit_at_close(long_plan(), broken, ExitReason.SESSION_CLOSE, vol=MEASURED)


def test_the_result_records_refuse_a_non_finite_price():
    """REGRESSION (MEDIUM). The structural half of the same bug: both result
    records accepted NaN and infinity in a price field, so any caller -- not
    just `exit_at_close` -- could mint one. A rule a careless caller cannot
    violate is worth more than a documented one.
    """
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            EntryFill(
                status=FillStatus.FILLED,
                symbol="NQ",
                side=Side.LONG,
                signal_ts=T0,
                submitted_ts=T0,
                eligible_ts=T0,
                order_type="market",
                fill_ts=T0 + timedelta(seconds=INTERVAL),
                fill_bar_offset=1,
                reference_price=bad,
                planned_price=13_000.0,
                requested_size=2.0,
                filled_size=2.0,
            )
        with pytest.raises(ValueError):
            ExitFill(
                resolved=True,
                symbol="NQ",
                side=Side.LONG,
                exit_reason=ExitReason.STOP,
                fill_ts=T0 + timedelta(seconds=INTERVAL),
                reference_price=bad,
                size=2.0,
            )


def test_no_quote_falls_back_to_the_specs_typical_spread_and_says_which():
    """Section 9: "bid/ask from quotes if present, else typical_spread_ticks".
    A run's spread cost is a blend of a measurement and an assumption, and the
    proportion decides how much of the cost total is evidence -- so the source
    is recorded per fill rather than inferred.
    """
    estimate = CostModel(config(), NQ).spread()

    assert estimate.source is SpreadSource.TYPICAL
    assert estimate.ticks == pytest.approx(NQ.typical_spread_ticks)


def test_a_crossed_or_zero_width_quote_is_rejected_rather_than_paid_out():
    """A non-positive spread charged as a cost is a REBATE: the trade would be
    paid to trade, on exactly the ticks where the book is broken."""
    for bid, ask in ((13_000.25, 12_999.75), (13_000.0, 13_000.0)):
        quote = QuoteSnapshot(
            symbol="NQ", ts=T0, bid=bid, ask=ask, bid_size=10.0, ask_size=10.0
        )
        estimate = CostModel(config(), NQ).spread(quote, fill_ts=T0)

        assert estimate.source is SpreadSource.TYPICAL
        assert estimate.ticks > 0.0
        assert estimate.half_ticks > 0.0
        assert any("rebate" in note for note in estimate.notes)


def test_a_non_finite_quote_falls_back_instead_of_propagating_a_nan():
    """A NaN spread multiplies into a NaN cost and then into a NaN equity
    curve, which is worse than an assumed spread because nothing flags it."""
    quote = QuoteSnapshot(
        symbol="NQ", ts=T0, bid=float("nan"), ask=13_001.0, bid_size=1.0, ask_size=1.0
    )
    estimate = CostModel(config(), NQ).spread(quote, fill_ts=T0)

    assert estimate.source is SpreadSource.TYPICAL
    assert math.isfinite(estimate.ticks)


def test_a_quote_tighter_than_the_instrument_can_trade_is_floored_both_ways():
    """A spread the instrument cannot quote is a discount on a price that does
    not exist, and both the override and a real quote can carry one."""
    tight_quote = QuoteSnapshot(
        symbol="NQ", ts=T0, bid=13_000.0, ask=13_000.05, bid_size=1.0, ask_size=1.0
    )
    quoted = CostModel(config(), NQ).spread(tight_quote, fill_ts=T0)
    assert quoted.ticks == pytest.approx(NQ.min_spread_ticks)
    assert any("min_spread_ticks" in note for note in quoted.notes)

    overridden = CostModel(config(spread_ticks_override=0.25), NQ).spread()
    assert overridden.source is SpreadSource.OVERRIDE
    assert overridden.ticks == pytest.approx(NQ.min_spread_ticks)
    assert any("min_spread_ticks" in note for note in overridden.notes)


def test_the_override_outranks_a_real_quote_and_the_switch_outranks_both():
    """A stress run priced at a chosen spread must not be silently beaten by a
    quote, or the stress run reports costs nobody asked for."""
    quote = QuoteSnapshot(
        symbol="NQ", ts=T0, bid=12_999.0, ask=13_001.0, bid_size=5.0, ask_size=5.0
    )
    fill_ts = T0 + timedelta(seconds=INTERVAL)

    overridden = CostModel(config(spread_ticks_override=4.0), NQ).spread(quote, fill_ts=fill_ts)
    assert overridden.source is SpreadSource.OVERRIDE
    assert overridden.ticks == pytest.approx(4.0)

    ignored = CostModel(config(use_quotes_when_available=False), NQ).spread(
        quote, fill_ts=fill_ts
    )
    assert ignored.source is SpreadSource.TYPICAL
    assert any("use_quotes_when_available is False" in note for note in ignored.notes)


def test_a_quote_that_postdates_the_fill_bar_raises():
    """Pricing a fill off a quote that had not happened yet is lookahead in the
    cost model, which is the one place nobody audits for it."""
    future = QuoteSnapshot(
        symbol="NQ",
        ts=T0 + timedelta(seconds=INTERVAL * 2),
        bid=12_999.0,
        ask=13_001.0,
        bid_size=1.0,
        ask_size=1.0,
    )
    with pytest.raises(CostModelError):
        CostModel(config(), NQ).spread(future, fill_ts=T0)
    with pytest.raises(CostModelError):
        filled_entry(quote=future)


def test_a_quote_for_another_instrument_raises():
    """An ES quote pricing an NQ fill is a $4-per-tick error wearing the
    `SpreadSource.QUOTE` label, which reads as a measurement."""
    other = QuoteSnapshot(
        symbol="ES", ts=T0, bid=3_999.0, ask=4_001.0, bid_size=1.0, ask_size=1.0
    )
    with pytest.raises(CostModelError):
        CostModel(config(), NQ).spread(other, fill_ts=T0)


def test_an_unmeasurable_cost_at_the_exit_raises_rather_than_booking_it_free():
    """The one asymmetry that cannot be made symmetric: the position is already
    open and the level is already reached, so refusing the exit would hold a
    position the market closed. A backtest that cannot price its exits stops
    rather than booking them at zero.
    """
    with pytest.raises(ExecutionError):
        model().resolve_exit(
            long_plan(),
            bar(2, 12_990.0, 13_000.0, 12_975.0, 12_985.0),
            vol=VolatilityContext.unmeasured(),
        )
    with pytest.raises(ExecutionError):
        model().exit_at_close(
            long_plan(),
            bar(2, 12_990.0, 13_000.0, 12_985.0, 12_990.0),
            ExitReason.TIME_STOP,
            vol=VolatilityContext.unmeasured(),
        )


def test_a_stop_or_target_cannot_be_priced_at_a_bar_close():
    """Routing a STOP through the flatten path would skip the gap rule and the
    same-bar rule, which is precisely how a stop comes to be filled at its
    limit price."""
    for reason in (ExitReason.STOP, ExitReason.TARGET):
        with pytest.raises(ExecutionError):
            model().exit_at_close(
                long_plan(), bar(2, 13_000.0, 13_010.0, 12_995.0, 13_005.0), reason, vol=MEASURED
            )


def test_a_forced_flatten_is_priced_at_the_close_as_a_market_order():
    """The ledger's exits -- time stop, session flatten, kill switch -- are
    market orders and pay a market order's costs. Pricing them passively would
    make a time-stopped book look cheaper than it is."""
    flatten = model().exit_at_close(
        long_plan(),
        bar(2, 13_000.0, 13_010.0, 12_995.0, 13_005.0),
        ExitReason.SESSION_CLOSE,
        vol=MEASURED,
    )

    assert flatten.resolved is True
    assert flatten.reference_price == pytest.approx(13_005.0)  # the close
    assert flatten.exit_reason is ExitReason.SESSION_CLOSE
    assert flatten.gapped is False and flatten.ambiguous_fill is False
    # A market exit: 1.125 ticks (no stop surcharge) * $5 * 2 = $11.25, plus the
    # half spread $5.00 and $2.62 of commission and fees.
    assert flatten.costs.exit_slippage == pytest.approx(11.25)
    assert flatten.costs.spread_cost == pytest.approx(5.0)
    assert flatten.costs.total == pytest.approx(18.87)


# ===========================================================================
# 9. Determinism -- a cost model that is not reproducible is unfalsifiable
# ===========================================================================


def test_the_same_intent_against_the_same_bars_fills_identically_twice():
    """No accumulated state anywhere, so a replay and a single call at one
    timestamp agree. `regime/detector.py` records why that matters: a stateful
    component there made results depend on evaluation order and was caught as a
    HIGH finding.
    """
    first_intent, first = filled_entry(exec_model=model(seed=7))
    second_intent, second = filled_entry(exec_model=model(seed=7))

    assert first.to_init_dict() == second.to_init_dict()
    exit_bar = bar(2, 13_000.0, 13_005.0, 12_975.0, 12_985.0)
    exit_a = model(seed=7).resolve_exit(plan_from(first_intent, first), exit_bar, vol=MEASURED)
    exit_b = model(seed=7).resolve_exit(plan_from(second_intent, second), exit_bar, vol=MEASURED)
    assert exit_a.to_init_dict() == exit_b.to_init_dict()
    assert exit_a.costs.total == exit_b.costs.total


def test_the_default_configuration_draws_from_no_generator_at_all():
    """`random_latency_ms=0` and `random_miss_probability=0` are the defaults, so
    the default backtest must be reproducible without reference to any
    generator. The observable is that no jitter and no draw note appear.
    """
    order = model().submit(intent(), T0)

    assert order.jitter_seconds == 0.0
    assert order.notes == ()
    assert order.eligible_ts == T0 + timedelta(milliseconds=250)


def test_stochastic_settings_stay_reproducible_from_the_seed():
    """Stress testing that cannot be re-run is not a measurement."""
    stressed = dict(random_latency_ms=500.0, random_miss_probability=0.3)
    runs = []
    for _ in range(3):
        exec_model = model(seed=11, **stressed)
        order = exec_model.submit(intent(), T0)
        window = exec_model.open_window(order, [entry_bar()])
        runs.append(
            (order.jitter_seconds, exec_model.resolve_entry(order, window, vol=MEASURED))
        )

    assert len({jitter for jitter, _ in runs}) == 1
    assert runs[0][0] > 0.0  # the jitter stream was actually used
    assert len({fill.to_init_json() for _, fill in runs}) == 1


def test_a_draw_does_not_depend_on_how_many_orders_preceded_it():
    """Keyed by the ORDER's identity rather than by a running counter, so a
    re-run, a resumed run and a single-order unit test all agree. A counter
    would make every fill depend on the whole history before it.
    """
    stressed = dict(random_miss_probability=0.4, random_latency_ms=250.0)
    alone = model(seed=5, **stressed)
    order = alone.submit(intent(), T0)
    solo = alone.resolve_entry(order, alone.open_window(order, [entry_bar()]), vol=MEASURED)

    warmed = model(seed=5, **stressed)
    for minute in range(1, 6):
        other = intent(signal_ts=T0 - timedelta(seconds=INTERVAL * minute))
        warm_order = warmed.submit(other, other.signal_ts)
        warmed.resolve_entry(
            warm_order,
            warmed.open_window(
                warm_order,
                [bar(1, 13_002.0, 13_010.0, 12_999.0, 13_006.0, base=other.signal_ts)],
            ),
            vol=MEASURED,
        )
    after = warmed.submit(intent(), T0)
    later = warmed.resolve_entry(after, warmed.open_window(after, [entry_bar()]), vol=MEASURED)

    assert solo.to_init_dict() == later.to_init_dict()


def test_adding_a_stochastic_term_does_not_shift_another_streams_draws():
    """Named streams exist so a new stochastic component cannot change the draws
    of an existing one. Checked at a miss probability where the draw's value
    decides the outcome, across several seeds.
    """
    for seed in range(12):
        without = model(seed=seed, random_miss_probability=0.5)
        with_jitter = model(seed=seed, random_miss_probability=0.5, random_latency_ms=400.0)
        plain_order = without.submit(intent(), T0)
        jitter_order = with_jitter.submit(intent(), T0)
        bars = [entry_bar(), bar(2, 13_004.0, 13_012.0, 13_001.0, 13_008.0)]

        plain = without.resolve_entry(
            plain_order, without.open_window(plain_order, bars), vol=MEASURED
        )
        jittered = with_jitter.resolve_entry(
            jitter_order, with_jitter.open_window(jitter_order, bars), vol=MEASURED
        )

        assert plain.status is jittered.status
        assert plain.miss_reason == jittered.miss_reason


def test_the_miss_stream_is_not_a_constant():
    """A determinism test that would pass for a generator returning one value
    proves nothing, so both outcomes are required to occur across seeds."""
    outcomes = set()
    for seed in range(40):
        exec_model = model(seed=seed, random_miss_probability=0.5)
        order = exec_model.submit(intent(), T0)
        outcomes.add(
            exec_model.resolve_entry(
                order, exec_model.open_window(order, [entry_bar()]), vol=MEASURED
            ).status
        )

    assert outcomes == {FillStatus.FILLED, FillStatus.MISSED}


def test_the_probabilistic_miss_is_drawn_before_any_price_is_inspected():
    """Applying it afterwards would make the stress-test miss rate conditional
    on the outcome, since the orders that survive the price checks are not a
    random sample. The observable: a miss happens even on a bar that could not
    have filled for other reasons, and the reason reported is the random one.
    """
    exec_model = model(seed=3, random_miss_probability=1.0)
    order = exec_model.submit(intent(), T0)
    # A bar that would independently decline the fill for lack of volume.
    window = exec_model.open_window(order, [entry_bar(volume=0.0)])
    fill = exec_model.resolve_entry(order, window, vol=MEASURED)

    assert fill.miss_reason == MissReason.RANDOM_MISS


# ===========================================================================
# 10. The journal row -- the validators are the last line, not the first
# ===========================================================================


def test_a_stopped_out_trade_produces_a_validating_record_and_does_not_realize_minus_one_r():
    """`TradeRecord` re-derives `gross_pnl` from the price path and requires
    `pnl == gross_pnl - costs.total`, so a cost or fill bug that slipped past
    every assertion above still cannot reach the journal.

    And the number that will look like a bug later: a stop does NOT realize
    -1.00R. The entry filled 2 points away from `planned_entry` while
    `risk_dollars` stays defined by the PLANNED stop distance (section 0.2: R is
    fixed at entry and never revised), so a backtest reporting exactly -1.00R
    on every stop has quietly re-derived R from the realized fill.
    """
    order_intent, fill = filled_entry()
    exit_fill = model().resolve_exit(
        plan_from(order_intent, fill),
        bar(2, 13_000.0, 13_005.0, 12_975.0, 12_985.0),
        vol=MEASURED,
    )
    record = record_from(
        order_intent, fill, exit_fill, model().round_turn_costs(fill, exit_fill)
    )

    # (12980 - 13002) * $20 * 2 = -$880 gross, less $47.74 = -$927.74 net.
    assert record.gross_pnl == pytest.approx(-880.0)
    assert record.costs.total == pytest.approx(ROUND_TURN_VIA_STOP)
    assert record.pnl == pytest.approx(-927.74)
    # -927.74 / 800 = -1.159675R: -1.10R of entry slippage, -0.0597R of costs.
    assert record.r_multiple == pytest.approx(-1.159675)
    assert record.r_multiple != pytest.approx(-1.0, abs=1e-3)
    assert record.cost_drag_r == pytest.approx(0.059675)
    assert record.planned_r_multiple == pytest.approx(2.0)


def test_a_target_trade_produces_a_validating_record_and_does_not_realize_two_r():
    """The symmetric surprise: the 2R target realizes +1.8731R, not +2R, and the
    cost hurdle through a target is less than half the hurdle through a stop
    because a resting limit pays neither slippage nor spread.
    """
    order_intent, fill = filled_entry()
    exit_fill = model().resolve_exit(
        plan_from(order_intent, fill),
        bar(2, 13_010.0, 13_045.0, 13_005.0, 13_040.0),
        vol=MEASURED,
    )
    costs = model().round_turn_costs(fill, exit_fill)
    record = record_from(order_intent, fill, exit_fill, costs)

    assert exit_fill.exit_reason is ExitReason.TARGET
    # (13040 - 13002) * $20 * 2 = +$1520 gross, less $21.49 = +$1498.51.
    assert record.gross_pnl == pytest.approx(1520.0)
    assert costs.total == pytest.approx(ROUND_TURN_VIA_TARGET)
    assert record.pnl == pytest.approx(1498.51)
    assert record.r_multiple == pytest.approx(1.873138, abs=1e-6)
    assert record.is_win is True
    # The hurdle depends on the exit mix: 0.0269R through a target, 0.0597R
    # through a stop. An analytics layer quoting one figure quotes neither.
    assert record.cost_drag_r == pytest.approx(0.026862, abs=1e-6)


def test_a_gapped_stop_record_validates_at_the_gap_price():
    """The overstatement refused, carried all the way into the journal row."""
    order_intent, fill = filled_entry()
    exit_fill = model().resolve_exit(
        plan_from(order_intent, fill),
        bar(2, 12_950.0, 12_960.0, 12_940.0, 12_955.0),
        vol=MEASURED,
    )
    record = record_from(
        order_intent, fill, exit_fill, model().round_turn_costs(fill, exit_fill)
    )

    assert record.exit_price == pytest.approx(12_950.0)
    assert record.gross_pnl == pytest.approx(-2080.0)
    assert record.pnl == pytest.approx(-2127.74)
    assert record.r_multiple == pytest.approx(-2.659675)
    assert record.ambiguous_fill is False  # a gap is certain, not unobserved


def test_an_ambiguous_record_carries_the_tag_into_the_journal():
    """The tag has to survive the trip, or section 9's "so the sensitivity of
    results to this assumption is measurable" has no row to measure."""
    order_intent, fill = filled_entry()
    exit_fill = model().resolve_exit(
        plan_from(order_intent, fill),
        bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0),
        vol=MEASURED,
    )
    record = record_from(
        order_intent, fill, exit_fill, model().round_turn_costs(fill, exit_fill)
    )

    assert record.ambiguous_fill is True
    assert record.exit_reason is ExitReason.STOP


def test_a_partial_fill_record_is_sized_and_costed_as_a_partial():
    """A partial accounted as a whole one overstates both the PnL and the R of
    every thin-bar trade. The record's own `filled_size` check is independent
    of this module's arithmetic.
    """
    order_intent, fill = filled_entry(bars=[entry_bar(volume=30.0)])
    plan = plan_from(order_intent, fill)
    exit_fill = model().resolve_exit(
        plan, bar(2, 13_000.0, 13_005.0, 12_975.0, 12_985.0), vol=MEASURED
    )
    costs = model().round_turn_costs(fill, exit_fill)
    record = record_from(order_intent, fill, exit_fill, costs)

    assert record.filled_size == pytest.approx(1.0)
    assert record.size == pytest.approx(2.0)  # the plan is recorded unchanged
    # One contract: (12980 - 13002) * $20 * 1 = -$440 gross.
    assert record.gross_pnl == pytest.approx(-440.0)
    # Costs on one contract: entry $10.745 + exit ($2.125 ticks * $5 + $2.50 +
    # $2.62) = $10.745 + $15.745 = $26.49.
    assert costs.total == pytest.approx(26.49)
    assert record.pnl == pytest.approx(-466.49)
    # R is unchanged at $800 because the PLAN was sized for two contracts, so a
    # half fill realizes about half the loss. Nothing rescaled it.
    assert record.risk_dollars == pytest.approx(RISK_DOLLARS)
    assert record.r_multiple == pytest.approx(-0.583113, abs=1e-6)


def test_folding_costs_into_the_price_also_validates_which_is_why_the_convention_matters():
    """The record's validators are the last line of defence, not the first, and
    this is the gap in them. A record built from `effective_price` while ALSO
    itemizing the costs passes every check and double-charges the trade.

    Stated here as an assertion so the limitation is recorded rather than
    assumed: the convention in `backtest/costs.py` is load-bearing precisely
    because `TradeRecord` cannot enforce it.
    """
    order_intent, fill = filled_entry()
    exit_fill = model().resolve_exit(
        plan_from(order_intent, fill),
        bar(2, 13_000.0, 13_005.0, 12_975.0, 12_985.0),
        vol=MEASURED,
    )
    costs = model().round_turn_costs(fill, exit_fill)
    honest = record_from(order_intent, fill, exit_fill, costs)

    double_charged = honest.replace(
        entry_price=fill.effective_price,
        exit_price=exit_fill.effective_price,
        gross_pnl=(fill.effective_price * -1 + exit_fill.effective_price) * POINT_VALUE * 2.0,
        pnl=(fill.effective_price * -1 + exit_fill.effective_price) * POINT_VALUE * 2.0
        - costs.total,
    )

    # Both rows satisfy the validator; they differ by the costs charged twice.
    assert double_charged.pnl < honest.pnl
    # Entry 1.625 ticks + exit 2.625 ticks = 4.25 ticks * $5 * 2 = $42.50.
    assert honest.pnl - double_charged.pnl == pytest.approx(42.5)


def test_nothing_in_the_execution_model_moves_the_stop_the_target_or_the_size():
    """`TradeIntent`: the execution model "may fill it worse, partially, or not
    at all, but it may not change the plan". Section 0.2 is what that protects.
    """
    order_intent = intent()
    before = order_intent.to_init_dict()
    _, fill = filled_entry(order_intent=order_intent, bars=[entry_bar(volume=30.0)])
    plan = plan_from(order_intent, fill)
    model().resolve_exit(plan, bar(2, 13_000.0, 13_045.0, 12_975.0, 13_010.0), vol=MEASURED)

    assert order_intent.to_init_dict() == before
    assert plan.stop_price == pytest.approx(order_intent.stop_price)
    assert plan.target_price == pytest.approx(order_intent.target_price)
    assert fill.planned_price == pytest.approx(order_intent.entry_price)
    assert fill.requested_size == pytest.approx(order_intent.size)


def test_an_exit_plan_cannot_be_built_with_inverted_geometry():
    """The independent enforcement: even a caller that skipped
    `_plan_invalidated` cannot assemble a bracket whose stop pays a profit."""
    with pytest.raises(ValueError):
        ExitPlan(
            symbol="NQ",
            side=Side.LONG,
            entry_ts=T0 + timedelta(seconds=INTERVAL),
            entry_price=12_970.0,  # below its own stop
            stop_price=12_980.0,
            target_price=13_040.0,
            size=2.0,
        )
    with pytest.raises(ValueError):
        ExitPlan(
            symbol="NQ",
            side=Side.SHORT,
            entry_ts=T0 + timedelta(seconds=INTERVAL),
            entry_price=13_030.0,  # above its own stop
            stop_price=13_020.0,
            target_price=12_960.0,
            size=2.0,
        )


def test_the_model_refuses_bars_and_intents_for_another_instrument():
    """An ES bar priced with NQ's $20 point value is a 4x PnL error that no
    downstream validator can see, because every equation still balances."""
    order = model().submit(intent(), T0)
    with pytest.raises(ExecutionError):
        model().open_window(
            order, [bar(1, 4_000.0, 4_005.0, 3_995.0, 4_002.0, symbol="ES")]
        )
    with pytest.raises(ExecutionError):
        model().resolve_exit(
            long_plan(),
            bar(2, 4_000.0, 4_005.0, 3_995.0, 4_002.0, symbol="ES"),
            vol=MEASURED,
        )
