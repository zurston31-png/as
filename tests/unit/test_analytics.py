"""Adversarial tests for analytics/metrics.py and analytics/breakdowns.py.

These are the numbers every conclusion in the final report will rest on, so
the tests are organized around the four ways this layer can produce a
*flattering* number rather than around its public surface.

**A zero that reads as a measurement.** The system as committed takes NO
trades: the highest Flow Score measured over 2500 consecutive bars is 65.23
against a lowest setup threshold of 70.0 (commit f952d3e). So the empty trade
set is the live case, not an edge case, and `win_rate = 0.0` over zero trades
would be read as "we measured, and it lost". The first group of tests pins
which quantities are measurements over an empty set (counts and sums, and the
flat curve's zero drawdown) and which must be an explicit refusal (every mean,
rate and ratio). `test_the_rendered_summary_of_the_live_case_shows_no_rate`
checks the same thing in the human-readable rendering, because that is what a
reader actually sees.

**A number produced from garbage, or a crash instead of a refusal.** A NaN
equity mark, a loss too small to register in R, a growth rate that overflows a
float: each of these used to produce either a confusing `ValidationError`
raised far from its cause, or -- worse -- a number. Three bugs found here and
fixed in the module are regression-tested in the second group, each named in
its test's docstring.

**Phases merged into one number.** Section 0.8: mixing in-sample and
out-of-sample trades is the easiest accidental way to manufacture a result.
The third group proves the refusal is STRUCTURAL -- every registered metric,
every aggregate, every breakdown entry point raises -- rather than documented,
and that it cannot be switched off by config. The one sanctioned cross-phase
view is checked to compute each cell from its own phase and to emit no
combined row.

**A rate projected from three trades.** Section 12 requires a cell below
`min_sample_n` to be marked INSUFFICIENT rather than shown as a win rate. The
fourth group proves the suppressed number is not merely hidden from the
reader but absent from the object, and that the cell is still reported (with
its counts) rather than dropped -- "do not hide bad periods" cuts both ways.

The arithmetic group hand-computes every metric on a five-trade book whose
realized R values are chosen, not read back: +2R, +1R, +0.5R, -0.5R, -1R. Each
expected value is derived in a comment from those five numbers. A separate
two-trade book reproduces the Phase 7a cost hurdle ($21.49 through a resting
limit, $47.74 through a market stop on a $400 R) and shows the 54.19%
breakeven win rate falling out of the records rather than being quoted.

Every config object here is built LOCALLY. A test coupled to
`config/defaults.yaml` fails whenever an unrelated section is edited, and its
expected values change silently when a default moves; `min_sample_n = 5` below
is chosen so the five-trade book sits exactly on the boundary, which the
shipped default (30) could not demonstrate.

NQ arithmetic, used throughout: point value $20, size 2, a 10-point stop. So
R = 2 x 10 x $20 = $400 and one point = $40. Every exit price below is derived
from a chosen net R: exit = entry + (net_R x 400 + costs) / 40, which lands on
the 0.25 tick grid for each value chosen.
"""

from __future__ import annotations

import ast
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flow_model.analytics import breakdowns as B
from flow_model.analytics import metrics as M
from flow_model.config.schema import AnalyticsConfig, ResearchTargets
from flow_model.core.contracts import CostBreakdown, EquityPoint, TradeRecord
from flow_model.core.enums import (
    ExitReason,
    Regime,
    Session,
    SetupType,
    Side,
    SplitPhase,
)

UTC = timezone.utc
T0 = datetime(2021, 3, 1, 15, 0, tzinfo=UTC)

#: NQ, size 2, 10-point stop. R = 2 * 10 * $20 = $400; one point = $40.
POINT_VALUE = 20.0
SIZE = 2.0
ENTRY = 20_000.0
STOP = 19_990.0
TARGET = 20_010.0
R_DOLLARS = 400.0
DOLLARS_PER_POINT = POINT_VALUE * SIZE  # 40.0

#: Phase 7a measured one NQ round turn at vol multiple 1.25: a target is a
#: resting limit (no spread, no slippage), a stop is a market order that also
#: pays stop_order_extra_slippage_ticks. On a $400 R that is 0.0537R and
#: 0.1194R -- not one constant, which is why drag is reported per exit reason.
TARGET_COST = 21.49
STOP_COST = 47.74

#: Built locally, not loaded. min_sample_n=5 puts the five-trade book exactly
#: on the suppression boundary; trading_days_per_year=250 is deliberately NOT
#: the schema default of 252, so a test can prove the policy value is the one
#: actually used in the annualization rather than a constant in the module.
LOCAL_ANALYTICS = AnalyticsConfig(
    risk_free_rate_annual=0.0,
    trading_days_per_year=250,
    min_sample_n=5,
    flow_score_buckets=(70.0, 80.0, 100.0),
)
POLICY = M.MetricPolicy.from_analytics(LOCAL_ANALYTICS)

#: Same knobs, with the risk-free hurdle switched on, so one test can show the
#: hurdle is actually subtracted rather than quietly dropped.
POLICY_RF = M.MetricPolicy.from_analytics(
    LOCAL_ANALYTICS.replace(risk_free_rate_annual=0.04)
)

TRAIN = SplitPhase.TRAIN


# ---------------------------------------------------------------------------
# Local builders
# ---------------------------------------------------------------------------


def trade(
    trade_id: str,
    *,
    exit_price: float,
    commission: float = 0.0,
    exit_reason: ExitReason = ExitReason.TARGET,
    minutes: float = 0.0,
    hold_minutes: float = 10.0,
    bars_held: int = 2,
    mae_r: float = -0.25,
    mfe_r: float = 1.0,
    split_label: SplitPhase | None = TRAIN,
    regime: Regime = Regime.LOW_VOL,
    setup: SetupType = SetupType.SCALP_1R,
    session: Session = Session.RTH_MID,
    flow_score: float = 75.0,
    symbol: str = "NQ",
    side: Side = Side.LONG,
    target_price: float = TARGET,
    ambiguous_fill: bool = False,
) -> TradeRecord:
    """One self-consistent NQ TradeRecord.

    `gross_pnl` and `pnl` are DERIVED from the price path and the costs here,
    which is also what `TradeRecord`'s own validators enforce -- so a test
    cannot smuggle in a record whose P&L disagrees with its prices.
    """
    signal_ts = T0 + timedelta(minutes=minutes)
    costs = CostBreakdown(commission=commission)
    gross = side.sign * (exit_price - ENTRY) * POINT_VALUE * SIZE
    return TradeRecord(
        trade_id=trade_id,
        run_id="adversarial",
        symbol=symbol,
        config_hash="cfg",
        signal_ts=signal_ts,
        entry_ts=signal_ts + timedelta(minutes=1),
        exit_ts=signal_ts + timedelta(minutes=1 + hold_minutes),
        bars_held=bars_held,
        session=session,
        side=side,
        setup=setup,
        planned_entry=ENTRY,
        stop_price=STOP,
        target_price=target_price,
        entry_price=ENTRY,
        exit_price=exit_price,
        size=SIZE,
        filled_size=SIZE,
        point_value=POINT_VALUE,
        exit_reason=exit_reason,
        ambiguous_fill=ambiguous_fill,
        risk_dollars=R_DOLLARS,
        gross_pnl=gross,
        costs=costs,
        pnl=gross - costs.total,
        mae_r=mae_r,
        mfe_r=mfe_r,
        flow_score=flow_score,
        regime=regime,
        split_label=split_label,
    )


def exit_for(net_r: float, commission: float = 0.0) -> float:
    """The long exit price that realizes `net_r` after `commission`.

    net $ = net_r * 400; gross $ = net $ + commission; one point = $40.
    """
    return ENTRY + (net_r * R_DOLLARS + commission) / DOLLARS_PER_POINT


def five_trade_book() -> list[TradeRecord]:
    """Realized R of exactly +2, +1, +0.5, -0.5, -1, each paying $20 of costs.

    | id | net R | net $ | gross $ | exit price                |
    | a  | +2.0  |  +800 |   +820  | 20000 + 820/40 = 20020.50 |
    | b  | +1.0  |  +400 |   +420  |                  20010.50 |
    | c  | +0.5  |  +200 |   +220  |                  20005.50 |
    | d  | -0.5  |  -200 |   -180  |                  19995.50 |
    | e  | -1.0  |  -400 |   -380  |                  19990.50 |

    Trade `e` closes at 19990.50, half a point ABOVE the 19990 stop: a
    session-close exit, not a stopped-out one. The $20 of costs is what turns
    a -9.5 point gross into a -1R net, which is the point of the book.
    """
    spec = [
        ("a", +2.0, ExitReason.TARGET, 2, -0.25, 2.25),
        ("b", +1.0, ExitReason.TARGET, 4, -0.25, 1.25),
        ("c", +0.5, ExitReason.TIME_STOP, 6, -0.25, 0.75),
        ("d", -0.5, ExitReason.TIME_STOP, 8, -0.50, 0.50),
        ("e", -1.0, ExitReason.SESSION_CLOSE, 10, -1.00, 0.25),
    ]
    return [
        trade(
            tid,
            exit_price=exit_for(net_r, 20.0),
            commission=20.0,
            exit_reason=reason,
            minutes=60 * i,
            hold_minutes=10 * (i + 1),
            bars_held=bars,
            mae_r=mae,
            mfe_r=mfe,
        )
        for i, (tid, net_r, reason, bars, mae, mfe) in enumerate(spec)
    ]


def hurdle_book() -> list[TradeRecord]:
    """One target exit and one stop exit at the Phase 7a measured costs.

    target: gross +$400 (exit 20010), costs $21.49 -> net $378.51 -> +0.946275R
    stop:   gross -$400 (exit 19990), costs $47.74 -> net -$447.74 -> -1.11935R
    """
    return [
        trade("t", exit_price=TARGET, commission=TARGET_COST,
              exit_reason=ExitReason.TARGET, minutes=0),
        trade("s", exit_price=STOP, commission=STOP_COST,
              exit_reason=ExitReason.STOP, minutes=60),
    ]


def equity(
    values: list[float],
    *,
    step: timedelta = timedelta(hours=24),
    start: datetime = T0,
    high_water_mark: float = 0.0,
) -> list[EquityPoint]:
    """An equity series. `high_water_mark` defaults to 0.0 exactly as
    `EquityPoint` does, so the drawdown code is exercised against the unset
    default rather than a helpfully pre-filled peak."""
    return [
        EquityPoint(
            ts=start + step * i,
            equity=value,
            realized_equity=value,
            high_water_mark=high_water_mark,
        )
        for i, value in enumerate(values)
    ]


def flat_curve(n: int = 2500) -> list[EquityPoint]:
    """What a run today actually produces: 2500 five-minute marks, unchanged."""
    return equity([100_000.0] * n, step=timedelta(minutes=5), high_water_mark=100_000.0)


#: Over an empty trade set these five are the ONLY registry metrics allowed to
#: carry a number, because a count and a sum over nothing are their identity
#: element -- no trades were taken, no money moved -- and the flat equity curve
#: says the same thing independently. Everything else is a mean, a rate or a
#: ratio, and the average of nothing is not zero.
DEFINED_AT_ZERO_TRADES = frozenset(
    {"trade_count", "total_pnl", "gross_profit", "gross_loss", "total_costs"}
)


# ---------------------------------------------------------------------------
# 1. n = 0, which is the live case
# ---------------------------------------------------------------------------


def test_the_live_case_completes_and_says_undefined_rather_than_zero():
    """The first thing that will actually run is a backtest that took no
    trades. It must produce a valid, explicitly empty result."""
    summary = M.performance_summary(
        [], flat_curve(), split_label=SplitPhase.TEST, policy=POLICY
    )

    assert summary.n_trades == 0
    assert (summary.n_wins, summary.n_losses, summary.n_scratch) == (0, 0, 0)
    assert summary.sample is M.SampleStatus.EMPTY
    assert any("n=0" in note for note in summary.notes)
    assert any("65.23" in note for note in summary.notes), (
        "the note should cite the measured finding, not just assert emptiness"
    )


def test_exactly_the_counts_and_sums_are_defined_over_an_empty_trade_set():
    """Stronger than "no metric crashes": it pins WHICH quantities are
    measurements at n=0. A new mean or rate that forgets the n=0 branch, or a
    count that starts refusing, fails here."""
    defined = set()
    for name, function in M.TRADE_METRICS.items():
        value = function([], split_label=None, policy=POLICY)
        assert isinstance(value, M.MetricValue), name
        assert value.n == 0, name
        if value.is_defined:
            defined.add(name)
            assert value.require() == 0.0, name
        else:
            assert value.undefined is M.Undefined.NO_TRADES, name

    assert defined == set(DEFINED_AT_ZERO_TRADES)


def test_a_win_rate_over_zero_trades_is_undefined_not_zero_percent():
    """0.0 in a win-rate column reads as "we measured, and it lost". The
    difference between that and "undefined" is the whole point of the module."""
    value = M.win_rate([], split_label=None, policy=POLICY)
    assert value.value is None
    assert value.undefined is M.Undefined.NO_TRADES
    assert value.or_none() is None
    with pytest.raises(M.UndefinedMetricError):
        value.require()
    assert "undefined" in str(value)


def test_profit_factor_over_zero_trades_is_undefined_not_infinity_and_not_zero():
    """Both wrong answers are tempting: 0/0 as infinity (no losses!) or as
    0.0 (no profit!). Neither is a measurement."""
    value = M.profit_factor([], split_label=None, policy=POLICY)
    assert not value.is_defined
    assert value.undefined is M.Undefined.NO_TRADES


def test_the_flat_curve_measures_zero_drawdown_but_refuses_a_sharpe():
    """A flat curve genuinely has zero drawdown depth at every point -- that
    is data. It has no dispersion at all, so the Sharpe ratio is 0/0, and the
    mean of an empty set of drawdown episodes is not zero either."""
    summary = M.performance_summary(
        [], flat_curve(), split_label=SplitPhase.TEST, policy=POLICY
    )

    assert summary.drawdown.max_drawdown_pct.require() == 0.0
    assert summary.drawdown.max_drawdown_dollars.require() == 0.0
    assert summary.drawdown.n_episodes == 0
    assert summary.drawdown.average_drawdown_pct.undefined is (
        M.Undefined.NO_DRAWDOWN_EPISODE
    )
    assert summary.drawdown.longest_recovery_seconds.undefined is (
        M.Undefined.NO_DRAWDOWN_EPISODE
    )
    assert summary.sharpe.undefined is M.Undefined.ZERO_DISPERSION
    assert summary.sortino.undefined is M.Undefined.ZERO_DISPERSION
    # Measured zeros: the curve started and ended at the same equity, and zero
    # trades over a known window really is zero trades per year.
    assert summary.total_return.require() == 0.0
    assert summary.cagr.require() == 0.0
    assert summary.trades_per_year.require() == 0.0


def test_the_rendered_summary_of_the_live_case_shows_no_rate():
    """Whatever the objects say, a reader sees `describe()`. A line reading
    `win_rate=0` there would undo every refusal above."""
    lines = M.performance_summary(
        [], flat_curve(), split_label=SplitPhase.TEST, policy=POLICY
    ).describe()
    rendered = "\n".join(lines)

    assert "win_rate=undefined[no_trades] (n=0)" in rendered
    assert "win_rate=0" not in rendered
    assert "profit_factor=0" not in rendered
    assert "expectancy_r=0" not in rendered


def test_an_empty_breakdown_keeps_its_shape_and_holds_no_numbers():
    """Every declared-domain dimension still emits its full grid, so nothing
    downstream needs an "if there are no trades" branch -- and every cell of
    that grid is EMPTY rather than a zero."""
    tables = B.breakdown_all([], split_label=None, policy=POLICY)

    assert set(tables) == {d.name for d in B.default_dimensions(POLICY)}
    expected_cells = {
        "regime": len(tuple(Regime)),
        "session": len(tuple(Session)),
        "setup": len(tuple(SetupType)),
        "exit_reason": len(tuple(ExitReason)),
        "day_of_week": 7,
        "month_of_year": 12,
        "ambiguous_fill": 2,
        "flow_score_bucket": len(B.flow_score_bucket_domain(POLICY.flow_score_buckets)),
        # No declared domain: nothing observed, so no cells at all.
        "market": 0,
        "year": 0,
        "month": 0,
        "r_target": 0,
    }
    assert {k: len(v.cells) for k, v in tables.items()} == expected_cells

    for table in tables.values():
        assert table.n_trades == 0
        for cell in table.cells:
            assert cell.status is M.SampleStatus.EMPTY
            assert (cell.n, cell.n_wins, cell.n_losses, cell.n_scratch) == (0, 0, 0, 0)
            assert all(not v.is_defined for v in cell.metrics.values())
            assert all(
                v.undefined is M.Undefined.NO_TRADES for v in cell.metrics.values()
            )


def test_trimming_empty_cells_on_an_empty_set_is_an_empty_table_not_an_error():
    """`include_empty=False` is a presentation choice, and the degenerate case
    of it is the live one."""
    tables = B.breakdown_all([], split_label=None, policy=POLICY, include_empty=False)
    assert all(table.cells == () for table in tables.values())
    assert all(table.n_trades == 0 for table in tables.values())


def test_zero_trades_over_a_known_window_is_measured_but_needs_the_window():
    """"Zero trades per year" is a real measurement and currently the system's
    only performance number. It needs a window, and the span between the first
    and last of zero trades does not exist -- so with no window supplied the
    answer is undefined for want of an input, which is a different thing from
    zero."""
    one_year = 365.25 * 24 * 3600.0
    measured = M.trade_frequency([], split_label=None, policy=POLICY, span_seconds=one_year)
    assert measured.require() == 0.0
    assert measured.unit == "trades/year"
    assert measured.n == 0

    assert M.trade_frequency([], split_label=None, policy=POLICY).undefined is (
        M.Undefined.NO_EQUITY_SERIES
    )


# ---------------------------------------------------------------------------
# 2. a number produced from garbage, or a crash instead of a refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_equity_mark_is_refused_at_the_door(bad):
    """BUG FOUND (fixed): a NaN equity mark produced BOTH a crash and a number.

    Every comparison against NaN is false, so the NaN walked through the
    drawdown scan as "not below the running peak" and surfaced as a pydantic
    `ValidationError` from inside `MetricValue` -- raised far from its cause
    and naming the metric rather than the input. Meanwhile `total_return` and
    `cagr`, which read only the endpoints, cheerfully reported numbers
    computed across a corrupted curve (0.039 and 0.43 on a 40-point series
    with one NaN in the middle), and `sortino` reported `no_downside` because
    NaN is not less than anything.

    A non-finite mark is an upstream bug, exactly like a series that runs
    backwards, so it is refused the same way: once, at the door, naming the
    timestamp.
    """
    points = equity([100_000.0, 101_000.0, 102_000.0])
    broken = list(points)
    broken[1] = points[1].replace(equity=bad)

    for function in (
        M.total_return,
        M.cagr,
        M.max_drawdown,
        M.average_drawdown,
        M.recovery_time,
        M.sharpe_ratio,
        M.sortino_ratio,
        M.drawdown_profile,
    ):
        with pytest.raises(M.NonFiniteEquityError):
            function(broken, split_label=None, policy=POLICY)

    with pytest.raises(M.NonFiniteEquityError):
        M.simple_returns(broken)
    with pytest.raises(M.NonFiniteEquityError):
        M.performance_summary([], broken, split_label=None, policy=POLICY)

    # It is an AnalyticsError, so a caller catching the layer's refusals
    # catches this one too.
    assert issubclass(M.NonFiniteEquityError, M.AnalyticsError)


def test_a_non_finite_high_water_mark_is_refused_too():
    """`drawdown_episodes` reads the first point's stored high water mark as a
    lower bound on the starting peak, so a NaN there would poison the peak and
    every depth measured against it."""
    points = equity([100_000.0, 99_000.0])
    broken = [points[0].replace(high_water_mark=float("nan")), points[1]]
    with pytest.raises(M.NonFiniteEquityError):
        M.drawdown_profile(broken, split_label=None, policy=POLICY)


def test_a_loss_too_small_to_register_in_r_does_not_produce_a_zero_breakeven():
    """BUG FOUND (fixed): the most flattering number this module could print.

    `TradeRecord.r_multiple` rounds to 6 decimals, so a loss under 5e-7 R
    (here -$0.0001 against a $400 R) is a real loss -- `is_loss` is True, it
    counts in `loss_rate` -- whose mean R is exactly -0.0. Dividing by that
    gave:

      * `win_loss_ratio`  -> ZeroDivisionError, a bare crash;
      * `breakeven_win_rate` -> 0.0, i.e. "this system breaks even at a 0%
        win rate", published as a measurement.

    Both are now `ZERO_AVERAGE_LOSS`. `profit_factor` is unaffected because it
    divides dollars, not rounded R, and 400/0.0001 = 4e6 is arithmetically
    what happened.
    """
    micro = -0.0001 / R_DOLLARS  # -2.5e-7 R, which rounds to -0.0
    book = [
        trade("w", exit_price=exit_for(1.0), minutes=0),
        trade("l", exit_price=exit_for(micro), minutes=30,
              exit_reason=ExitReason.TIME_STOP),
    ]
    assert book[1].is_loss and book[1].r_multiple == 0.0

    for function in (M.win_loss_ratio, M.breakeven_win_rate):
        value = function(book, split_label=TRAIN, policy=POLICY)
        assert not value.is_defined, function.__name__
        assert value.undefined is M.Undefined.ZERO_AVERAGE_LOSS
        assert value.n == 2
    # The loss is still counted: suppressing the ratio must not suppress data.
    assert M.loss_rate(book, split_label=TRAIN, policy=POLICY).require() == 0.5
    assert M.profit_factor(book, split_label=TRAIN, policy=POLICY).require() > 0.0


def test_a_growth_rate_that_overflows_a_float_is_undefined_not_an_exception():
    """BUG FOUND (fixed): `cagr` raised OverflowError on a short span.

    Compounding a +1% move made over five minutes into an annual rate needs
    1.01 ** 105192, which leaves the float range: `(end/start) ** (1/years)`
    raised `OverflowError: (34, 'Numerical result out of range')` straight out
    of a report. The arithmetic is not wrong, the quantity is simply not
    representable, so it is now reported undefined with the span in the note.
    A span long enough to be representable still reports its (large) number --
    this test does not quietly suppress inconvenient values, only
    unrepresentable ones.
    """
    five_minutes = equity([100_000.0, 101_000.0], step=timedelta(minutes=5))
    value = M.cagr(five_minutes, split_label=None, policy=POLICY)
    assert not value.is_defined
    assert value.undefined is M.Undefined.NOT_REPRESENTABLE
    assert "too short" in value.note
    # The honest number for that window is still available and is not absurd.
    assert M.total_return(five_minutes, split_label=None, policy=POLICY).require() == (
        pytest.approx(0.01)
    )

    one_day = equity([100_000.0, 101_000.0], step=timedelta(days=1))
    assert M.cagr(one_day, split_label=None, policy=POLICY).is_defined


def test_no_metric_value_can_exist_as_a_sentinel_or_a_non_finite_number():
    """One representation of undefined, enforced by the type: no -1.0, no
    0.0-means-missing, no NaN that a plot would silently drop."""
    with pytest.raises(ValueError):
        M.MetricValue(name="x", value=1.0, undefined=M.Undefined.NO_TRADES)
    with pytest.raises(ValueError):
        M.MetricValue(name="x")
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            M.MetricValue(name="x", value=bad)
    with pytest.raises(ValueError):
        M.MetricValue(name="x", value=1.0, n=-1)


def test_profit_factor_with_winners_and_no_losers_is_undefined_not_infinite():
    """An "infinite profit factor" is the signature of a sample too small to
    contain a loss, and it is the number most likely to be printed as a
    headline."""
    wins = [trade(f"w{i}", exit_price=exit_for(1.0), minutes=10 * i) for i in range(3)]
    value = M.profit_factor(wins, split_label=TRAIN, policy=POLICY)
    assert not value.is_defined
    assert value.undefined is M.Undefined.NO_LOSSES
    assert "no denominator" in value.note


def test_profit_factor_with_losers_and_no_winners_is_zero_which_is_measured():
    """The mirror case is NOT symmetric: gross profit really was zero, so 0.0
    here is a measurement and refusing it would hide a loss."""
    losses = [
        trade(f"l{i}", exit_price=exit_for(-1.0), minutes=10 * i,
              exit_reason=ExitReason.STOP)
        for i in range(3)
    ]
    assert M.profit_factor(losses, split_label=TRAIN, policy=POLICY).require() == 0.0


def test_a_backwards_equity_series_is_refused_but_repeated_timestamps_are_not():
    """Sorting a backwards series would hide an upstream ordering bug behind a
    plausible drawdown. Equal timestamps are legitimate -- several marks at one
    bar close -- and must not be rejected with it."""
    forward = equity([100_000.0, 101_000.0, 102_000.0])
    with pytest.raises(M.NonChronologicalEquityError):
        M.total_return(list(reversed(forward)), split_label=None, policy=POLICY)

    same_ts = [
        EquityPoint(ts=T0, equity=100_000.0, realized_equity=100_000.0),
        EquityPoint(ts=T0, equity=100_500.0, realized_equity=100_500.0),
    ]
    assert M.total_return(same_ts, split_label=None, policy=POLICY).require() == (
        pytest.approx(0.005)
    )
    # ...but a per-year rate over a zero-length window is 0/0, not a number.
    assert M.cagr(same_ts, split_label=None, policy=POLICY).undefined is (
        M.Undefined.ZERO_ELAPSED_TIME
    )
    assert M.trade_frequency(
        [], same_ts, split_label=None, policy=POLICY
    ).undefined is M.Undefined.ZERO_ELAPSED_TIME


def test_sharpe_below_the_declared_minimum_observations_is_undefined():
    """The minimum is a declared constant, not a silent 2: below it the
    standard error of the statistic is the size of the statistic, so the
    number carries no information while looking exactly like one that does."""
    assert M.MIN_RETURN_OBSERVATIONS >= 10
    assert POLICY.min_return_observations == M.MIN_RETURN_OBSERVATIONS

    alternating = [100_000.0 * (2.0 if i % 2 else 1.0) for i in range(31)]
    # 31 points -> 30 returns, exactly the minimum.
    enough = M.sharpe_ratio(
        equity(alternating), split_label=None, policy=POLICY, periods_per_year=1.0
    )
    assert enough.is_defined
    one_short = M.sharpe_ratio(
        equity(alternating[:-1]), split_label=None, policy=POLICY, periods_per_year=1.0
    )
    assert one_short.undefined is M.Undefined.INSUFFICIENT_OBSERVATIONS
    assert str(M.MIN_RETURN_OBSERVATIONS) in one_short.note
    assert one_short.n == 29


def test_sortino_refuses_a_flat_curve_rather_than_reporting_the_hurdle():
    """A zero-dispersion series has no downside risk to divide by, only the
    risk-free shortfall. Reporting that ratio would print a large negative
    number that measures the hurdle rather than the strategy -- and a flat
    curve is what zero trades produces."""
    value = M.sortino_ratio(flat_curve(50), split_label=None, policy=POLICY_RF)
    assert value.undefined is M.Undefined.ZERO_DISPERSION
    assert "total_return" in value.note  # points the reader somewhere real


def test_an_all_positive_return_series_has_no_sortino_denominator():
    """No observation below the target means downside deviation is zero: the
    ratio is undefined, never infinity.

    MY FIRST EXPECTATION WAS WRONG: I first used a CONSTANT positive series,
    which is refused earlier as `ZERO_DISPERSION`. Both statements are true of
    that series and the module checks the more fundamental one first, so the
    series here is dispersed but never negative, which isolates the downside
    denominator.
    """
    returns = [0.01] * 20 + [0.02] * 15
    value = M.sortino_from_returns(
        returns, split_label=None, policy=POLICY, periods_per_year=1.0
    )
    assert value.undefined is M.Undefined.NO_DOWNSIDE
    assert M.sharpe_from_returns(
        returns, split_label=None, policy=POLICY, periods_per_year=1.0
    ).is_defined, "the same series has a perfectly good Sharpe ratio"


@pytest.mark.parametrize("periods", [0.0, -1.0])
def test_a_non_positive_annualization_factor_is_rejected(periods):
    """sqrt of a non-positive factor is not a Sharpe ratio. Raising beats
    returning something."""
    with pytest.raises(ValueError):
        M.sharpe_from_returns([0.01, -0.01] * 20, split_label=None, periods_per_year=periods)
    with pytest.raises(ValueError):
        M.sortino_from_returns([0.01, -0.01] * 20, split_label=None, periods_per_year=periods)


def test_a_ruined_account_has_no_return_or_growth_ratio():
    """Dividing by a zero starting equity, or taking a root of a negative
    terminal equity, produces a number that means nothing."""
    assert M.total_return(
        equity([0.0, 50_000.0]), split_label=None, policy=POLICY
    ).undefined is M.Undefined.NON_POSITIVE_EQUITY
    assert M.cagr(
        equity([100_000.0, 0.0]), split_label=None, policy=POLICY
    ).undefined is M.Undefined.NON_POSITIVE_EQUITY
    # A drawdown through zero is still measurable in dollars, and -105% is the
    # honest depth: the account lost more than its peak.
    through_zero = equity([100_000.0, -5_000.0, 20_000.0])
    assert M.max_drawdown(
        through_zero, split_label=None, policy=POLICY
    ).require() == pytest.approx(-1.05)
    assert M.sharpe_ratio(
        through_zero, split_label=None, policy=POLICY
    ).undefined is M.Undefined.NON_POSITIVE_EQUITY


# ---------------------------------------------------------------------------
# 3. phases merged into one number (section 0.8)
# ---------------------------------------------------------------------------


def two_phase_book() -> list[TradeRecord]:
    """Three TRAIN winners and one TEST loser.

    Merged, that is a 75% win rate and +0.5R expectancy; honestly reported it
    is 100% on three in-sample trades and 0% on one out-of-sample trade. The
    merged number is the one these tests make impossible.
    """
    return [
        trade(f"tr{i}", exit_price=exit_for(1.0), minutes=10 * i, split_label=TRAIN)
        for i in range(3)
    ] + [
        trade("te", exit_price=exit_for(-1.0), minutes=100,
              exit_reason=ExitReason.STOP, split_label=SplitPhase.TEST)
    ]


def test_every_registered_metric_refuses_a_two_phase_trade_set():
    """The refusal is a property of the registry, not of the metrics that
    happened to be written carefully. A new metric cannot be added without
    it, because the registry's signature obliges `split_label`."""
    mixed = two_phase_book()
    for name, function in M.TRADE_METRICS.items():
        for label in (TRAIN, SplitPhase.TEST, None):
            try:
                function(mixed, split_label=label, policy=POLICY)
            except M.MixedSplitError:
                continue
            pytest.fail(f"{name} accepted a trade set spanning TRAIN and TEST")


def test_every_aggregate_entry_point_refuses_a_two_phase_trade_set():
    """The aggregates are where a flattering merged number would actually be
    produced, and they are not in the registry sweep above. `names=()` is in
    this list because it USED to slip through: the refusal lived in the
    metrics `compute_trade_metrics` called, so calling it with no metrics
    returned an empty dict from a mixed set without raising (BUG, fixed)."""
    mixed = two_phase_book()
    calls = {
        "streaks": lambda: M.streaks(mixed, split_label=TRAIN, policy=POLICY),
        "r_distribution": lambda: M.r_distribution(mixed, split_label=TRAIN, policy=POLICY),
        "cost_drag": lambda: M.cost_drag_by_exit_reason(
            mixed, split_label=TRAIN, policy=POLICY
        ),
        "trade_frequency": lambda: M.trade_frequency(
            mixed, flat_curve(10), split_label=TRAIN, policy=POLICY
        ),
        "performance_summary": lambda: M.performance_summary(
            mixed, flat_curve(10), split_label=TRAIN, policy=POLICY
        ),
        "compute_trade_metrics": lambda: M.compute_trade_metrics(
            mixed, split_label=TRAIN, policy=POLICY
        ),
        "compute_trade_metrics(names=())": lambda: M.compute_trade_metrics(
            mixed, split_label=TRAIN, policy=POLICY, names=()
        ),
        "breakdown_all": lambda: B.breakdown_all(mixed, split_label=TRAIN, policy=POLICY),
        "breakdown(split_phase)": lambda: B.breakdown(
            mixed, "split_phase", split_label=None, policy=POLICY
        ),
        "assert_single_phase": lambda: M.assert_single_phase(mixed, TRAIN),
    }
    for label, call in calls.items():
        try:
            call()
        except M.MixedSplitError:
            continue
        pytest.fail(f"{label} accepted a trade set spanning TRAIN and TEST")


def test_every_breakdown_dimension_refuses_a_two_phase_trade_set():
    """A dimension that happens not to be the split phase is not a licence to
    merge: the cells of a `regime` table built from TRAIN and TEST trades
    would each be an unlabelled number."""
    mixed = two_phase_book()
    for dimension in B.default_dimensions(POLICY):
        for handle in (dimension, dimension.name):
            try:
                B.breakdown(mixed, handle, split_label=TRAIN, policy=POLICY)
            except M.MixedSplitError:
                continue
            pytest.fail(
                f"the {dimension.name} breakdown accepted TRAIN and TEST trades"
            )


def test_an_unlabelled_trade_counts_as_its_own_phase():
    """An unlabelled number mixed into a labelled one is exactly the
    unattributable claim section 0.8 forbids, so `None` is a phase and not a
    wildcard."""
    mixed = [
        trade("u", exit_price=exit_for(1.0), split_label=None),
        trade("t", exit_price=exit_for(1.0), minutes=10, split_label=TRAIN),
    ]
    assert M.phases_present(mixed) == (None, TRAIN)
    for label in (None, TRAIN):
        with pytest.raises(M.MixedSplitError):
            M.win_rate(mixed, split_label=label, policy=POLICY)


def test_a_single_phase_set_declared_as_another_phase_is_a_different_refusal():
    """Two different mistakes deserve two different errors: one is merging
    phases, the other is mislabelling a set that is internally consistent --
    which would attribute an in-sample result to the holdout."""
    train_only = [trade("t", exit_price=exit_for(1.0), split_label=TRAIN)]
    with pytest.raises(M.SplitLabelMismatchError):
        M.expectancy_r(train_only, split_label=SplitPhase.TEST, policy=POLICY)
    with pytest.raises(M.SplitLabelMismatchError):
        M.expectancy_r(train_only, split_label=None, policy=POLICY)
    with pytest.raises(M.SplitLabelMismatchError):
        B.breakdown(train_only, "regime", split_label=SplitPhase.SEALED_OOS, policy=POLICY)
    # Both are AnalyticsErrors, so a caller can catch the policy as a whole.
    assert issubclass(M.SplitLabelMismatchError, M.AnalyticsError)
    assert issubclass(M.MixedSplitError, M.AnalyticsError)


def test_the_refusal_cannot_be_switched_off_by_config():
    """`AnalyticsConfig.allow_mixed_split_aggregation` exists and defaults to
    False. The module deliberately does not read it: a knob that can turn the
    refusal off is not a refusal. This test pins that deviation, so flipping
    the field later cannot silently reopen the hole."""
    permissive = AnalyticsConfig(allow_mixed_split_aggregation=True, min_sample_n=5)
    policy = M.MetricPolicy.from_analytics(permissive)
    assert not hasattr(policy, "allow_mixed_split_aggregation")
    with pytest.raises(M.MixedSplitError):
        M.win_rate(two_phase_book(), split_label=TRAIN, policy=policy)


def test_the_one_cross_phase_view_computes_each_cell_from_its_own_phase():
    """`breakdown_by_split_phase` is the sanctioned way to look at TRAIN and
    TEST together. Each cell must be computed under its own label -- not under
    one shared label, and not from the pooled set -- and there must be no
    combined row for a reader to quote."""
    mixed = two_phase_book() + [
        trade("un", exit_price=exit_for(1.0), minutes=200, split_label=None)
    ]
    policy = M.MetricPolicy.from_analytics(LOCAL_ANALYTICS.replace(min_sample_n=1))
    table = B.breakdown_by_split_phase(mixed, policy=policy, include_empty=False)

    assert table.spans_phases is True
    assert [cell.key for cell in table.cells] == ["TRAIN", "TEST", "UNLABELLED"]
    assert [cell.n for cell in table.cells] == [3, 1, 1]
    assert table.n_trades == 5
    # The numbers are per phase: 3/3 in sample, 0/1 out of sample. The merged
    # 4/5 = 80% that pooling would give appears nowhere.
    assert table.cell("TRAIN").metrics["win_rate"].require() == 1.0
    assert table.cell("TEST").metrics["win_rate"].require() == 0.0
    assert table.cell("UNLABELLED").metrics["win_rate"].require() == 1.0
    assert all(cell.n < table.n_trades for cell in table.cells)
    assert all(
        cell.metrics["win_rate"].or_none() != pytest.approx(0.8) for cell in table.cells
    )


def test_the_cross_phase_view_declares_every_phase_including_the_sealed_one():
    """A phase with no trades is a fact about the sample, and the sealed
    holdout showing an empty cell is the reassuring case: it says nothing was
    computed from it."""
    table = B.breakdown_by_split_phase(two_phase_book(), policy=POLICY)
    keys = [cell.key for cell in table.cells]
    assert keys == [*(str(p) for p in SplitPhase), "UNLABELLED"]
    sealed = table.cell(str(SplitPhase.SEALED_OOS))
    assert sealed.n == 0 and sealed.status is M.SampleStatus.EMPTY
    assert all(not v.is_defined for v in sealed.metrics.values())


def test_an_equity_based_metric_records_its_label_without_being_able_to_check_it():
    """Stated plainly rather than implied: `EquityPoint` carries no phase, so
    the equity-curve metrics take the caller's label on trust. The structural
    guard lives on the trade-based functions, which can actually see a phase.
    This is a KNOWN LIMITATION, pinned here so nobody mistakes the absence of
    a raise for a check."""
    curve = equity([100_000.0, 101_000.0])
    for label in (TRAIN, SplitPhase.SEALED_OOS, None):
        assert M.total_return(curve, split_label=label, policy=POLICY).is_defined
    assert M.drawdown_profile(
        curve, split_label=SplitPhase.TEST, policy=POLICY
    ).split_label is SplitPhase.TEST


# ---------------------------------------------------------------------------
# 4. a rate projected from three trades (section 12's suppression rule)
# ---------------------------------------------------------------------------


def chop_cell(n: int, *, policy: M.MetricPolicy = POLICY) -> B.BreakdownCell:
    """A regime table over `n` identical CHOP winners, returning that cell."""
    trades = [
        trade(f"c{i}", exit_price=exit_for(1.0), minutes=10 * i, regime=Regime.CHOP)
        for i in range(n)
    ]
    table = B.breakdown(trades, "regime", split_label=TRAIN, policy=policy)
    cell = table.cell(str(Regime.CHOP))
    assert cell is not None
    return cell


def test_a_cell_below_min_sample_n_reports_insufficient_not_a_win_rate():
    """Section 12: "any cell with N below min_sample_n is marked INSUFFICIENT
    rather than shown as a win rate (small-sample win rates are the primary
    way people fool themselves)". Three winners out of three is 100%, and 100%
    of three is the most misleading number in the report."""
    cell = chop_cell(3)

    assert cell.n == 3
    assert (cell.n_wins, cell.n_losses, cell.n_scratch) == (3, 0, 0)
    assert cell.status is M.SampleStatus.INSUFFICIENT
    assert cell.is_reportable is False
    for name, value in cell.metrics.items():
        assert not value.is_defined, name
        assert value.undefined is M.Undefined.INSUFFICIENT_SAMPLE, name
        assert value.n == 3, name  # N travels WITH the refusal
        assert "min_sample_n" in value.note


def test_the_counts_survive_suppression_because_they_are_data():
    """The refusal is about the rate, which projects three wins onto a
    population -- not about the three wins. Hiding those would be hiding
    data."""
    cell = chop_cell(2)
    assert (cell.n, cell.n_wins) == (2, 2)
    assert cell.metrics["trade_count"].is_defined is False
    assert cell.metrics["trade_count"].n == 2  # still readable, via n


def test_the_suppressed_number_is_absent_from_the_object_not_merely_hidden():
    """Suppression that leaves the value in a field is a filter, not a
    refusal: the next person to write a dashboard reads the field. The cell is
    serialized here and searched for the number it must not contain.

    (The cell is 3 wins / 0 losses / 0 scratch, so no count is 1 and a stray
    1.0 can only be the suppressed win rate or loss-rate complement.)
    """
    cell = chop_cell(3)
    payload = cell.to_init_dict()

    floats: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, float):
            floats.append(node)

    walk(payload)
    assert floats == [], f"a suppressed cell holds no numbers, found {floats}"
    assert all(value["value"] is None for value in payload["metrics"].values())
    assert "1.0" not in cell.to_init_json()


def test_the_suppression_boundary_is_the_configured_n_exactly():
    """One trade either side of `min_sample_n`. A boundary written as `>` when
    it should be `>=` would suppress a legitimate cell or publish an
    illegitimate one, and nothing else in the suite would notice."""
    assert POLICY.min_sample_n == 5
    assert chop_cell(4).status is M.SampleStatus.INSUFFICIENT
    assert chop_cell(5).status is M.SampleStatus.ADEQUATE
    assert chop_cell(5).metrics["win_rate"].require() == 1.0
    assert chop_cell(0 + 1).status is M.SampleStatus.INSUFFICIENT

    # And the threshold is read from the policy, not from a constant here.
    lenient = M.MetricPolicy.from_analytics(LOCAL_ANALYTICS.replace(min_sample_n=2))
    assert chop_cell(2, policy=lenient).status is M.SampleStatus.ADEQUATE
    assert chop_cell(2, policy=lenient).metrics["win_rate"].require() == 1.0


def test_an_insufficient_cell_stays_in_the_table_rather_than_being_dropped():
    """Dropping thin cells would be the mirror sin: the reader would see only
    the cells that happened to have enough trades, which is a filtered sample.
    The cell stays, with its counts and its refusal, and `series()` keeps it as
    a gap."""
    trades = [
        trade(f"t{i}", exit_price=exit_for(1.0), minutes=10 * i,
              regime=Regime.CHOP if i < 2 else Regime.LOW_VOL)
        for i in range(7)
    ]
    table = B.breakdown(trades, "regime", split_label=TRAIN, policy=POLICY)

    assert table.n_trades == 7
    assert table.cell(str(Regime.CHOP)).status is M.SampleStatus.INSUFFICIENT
    assert table.cell(str(Regime.LOW_VOL)).status is M.SampleStatus.ADEQUATE
    assert table.n_insufficient == 1
    assert table.n_empty == len(tuple(Regime)) - 2
    assert len(table.reportable_cells) == 1
    series = dict(table.series("win_rate"))
    assert series[str(Regime.CHOP)].undefined is M.Undefined.INSUFFICIENT_SAMPLE
    assert series[str(Regime.LOW_VOL)].require() == 1.0
    assert len(series) == len(table.cells), "a plot gets every cell, gaps included"


def test_a_cell_cannot_be_constructed_holding_a_number_it_should_have_hidden():
    """The validator is the backstop for the suppression rule: if a future
    edit forgets to blank a metric, the object refuses to exist rather than
    quietly publishing it."""
    with pytest.raises(ValueError, match="suppressed cell must hold no numbers"):
        B.BreakdownCell(
            dimension="regime",
            key="CHOP",
            n=3,
            n_wins=3,
            n_losses=0,
            n_scratch=0,
            status=M.SampleStatus.INSUFFICIENT,
            metrics={"win_rate": M.MetricValue.measured("win_rate", 1.0, n=3)},
        )
    with pytest.raises(ValueError, match="suppressed cell must hold no numbers"):
        B.BreakdownCell(
            dimension="regime",
            key="CHOP",
            n=0,
            n_wins=0,
            n_losses=0,
            n_scratch=0,
            status=M.SampleStatus.EMPTY,
            metrics={"trade_count": M.MetricValue.measured("trade_count", 0.0, n=0)},
        )


def test_a_cell_whose_counts_do_not_sum_to_its_n_is_rejected():
    """Wins + losses + scratches is the whole cell. A mismatch means a trade
    was classified twice or not at all."""
    with pytest.raises(ValueError, match="do not sum"):
        B.BreakdownCell(
            dimension="regime", key="CHOP", n=3, n_wins=1, n_losses=1, n_scratch=0,
            status=M.SampleStatus.ADEQUATE, metrics={},
        )


def test_a_breakdown_that_loses_a_trade_is_rejected():
    """BUG FOUND (fixed): the guard was self-referential.

    `Breakdown.n_trades` was built as `sum(c.n for c in cells)` and then the
    validator compared it with `sum(c.n for c in cells)`, so the documented
    invariant -- "A dimension may not drop a trade" -- could never fire from
    `_build`, whatever a key function did. `n_trades` is now the INPUT count,
    which is the only thing the cells can be checked against.
    """
    cell = B.BreakdownCell(
        dimension="regime", key="CHOP", n=3, n_wins=3, n_losses=0, n_scratch=0,
        status=M.SampleStatus.INSUFFICIENT,
        metrics={"win_rate": M.MetricValue.undefined_because(
            "win_rate", M.Undefined.INSUFFICIENT_SAMPLE, n=3
        )},
    )
    with pytest.raises(ValueError, match="may not drop a trade"):
        B.Breakdown(
            dimension="regime", description="regime at entry", split_label=TRAIN,
            min_sample_n=5, metric_names=("win_rate",), n_trades=4, cells=(cell,),
        )


def test_every_dimension_conserves_the_trade_count_it_was_given():
    """Every trade must land in exactly one cell of every dimension, including
    the dimensions whose keys are computed (Flow Score bucket, R target,
    year-month) and including the trimmed `include_empty=False` presentation.

    STATED PLAINLY: the grouping in `_build` is currently lossless by
    construction, so this test cannot distinguish the fixed `n_trades` from
    the self-referential one it replaced -- it is a guard against a future
    key function that drops a trade, which the anchoring fix makes able to
    fire. The test that proves the validator itself works is the one above.
    """
    book = five_trade_book() + [
        trade("f", exit_price=exit_for(1.0), minutes=400, symbol="ES",
              regime=Regime.HIGH_VOL, setup=SetupType.SETUP_2R,
              target_price=20_020.0, flow_score=100.0, session=Session.RTH_OPEN,
              ambiguous_fill=True),
    ]
    for dimension in (*B.default_dimensions(POLICY), B.SPLIT_PHASE_DIMENSION):
        for include_empty in (True, False):
            table = B.breakdown(
                book, dimension, split_label=TRAIN, policy=POLICY,
                include_empty=include_empty,
            )
            assert table.n_trades == len(book), dimension.name
            assert sum(c.n for c in table.cells) == len(book), dimension.name
            assert sum(c.n_wins + c.n_losses + c.n_scratch for c in table.cells) == (
                len(book)
            ), dimension.name


def test_the_top_level_summary_is_not_suppressed_but_states_its_sample():
    """A documented asymmetry, pinned so it stays deliberate: section 12's
    suppression rule is about breakdown CELLS, and suppressing the headline
    too would hide the only honest summary of a thin run. The summary must
    therefore carry the verdict explicitly."""
    thin = [trade(f"t{i}", exit_price=exit_for(1.0), minutes=10 * i) for i in range(3)]
    summary = M.performance_summary(thin, split_label=TRAIN, policy=POLICY)

    assert summary.sample is M.SampleStatus.INSUFFICIENT
    assert summary.min_sample_n == 5
    assert summary.n_trades == 3
    assert summary.trade_metrics["win_rate"].require() == 1.0  # reported, with N
    assert summary.trade_metrics["win_rate"].n == 3
    assert any("below min_sample_n" in note for note in summary.notes)
    assert any("INSUFFICIENT" in note for note in summary.notes)


# ---------------------------------------------------------------------------
# 5. the arithmetic, hand-computed on five chosen R values
# ---------------------------------------------------------------------------


def test_the_book_under_test_is_the_book_the_arithmetic_assumes():
    """Guards every expected value below. If the builder drifts -- a cost, a
    point value, a tick -- this fails first and names the cause, instead of
    twenty expectations failing with mysterious numbers."""
    book = five_trade_book()
    assert [t.r_multiple for t in book] == [2.0, 1.0, 0.5, -0.5, -1.0]
    assert [t.pnl for t in book] == [800.0, 400.0, 200.0, -200.0, -400.0]
    assert [t.gross_pnl for t in book] == [820.0, 420.0, 220.0, -180.0, -380.0]
    assert [t.costs.total for t in book] == [20.0] * 5
    assert {t.risk_dollars for t in book} == {400.0}
    # R is read from the record, never re-derived: pnl / risk_dollars.
    assert all(
        t.r_multiple == pytest.approx(t.pnl / t.risk_dollars) for t in book
    )


def test_win_loss_and_scratch_rates_share_one_denominator():
    """Three winners of five is 60%. The denominator is EVERY trade: an
    "excluding scratches" convention would quietly lift the rate, so the
    scratch mass is reported instead of removed."""
    book = five_trade_book()
    assert M.win_rate(book, split_label=TRAIN, policy=POLICY).require() == 0.6
    assert M.loss_rate(book, split_label=TRAIN, policy=POLICY).require() == 0.4
    assert M.scratch_rate(book, split_label=TRAIN, policy=POLICY).require() == 0.0

    # 3 wins, 1 loss, 1 scratch: 3/5, 1/5, 1/5 -- not 3/4 = 75%.
    with_scratch = [
        trade("w1", exit_price=exit_for(1.0), minutes=0),
        trade("w2", exit_price=exit_for(1.0), minutes=10),
        trade("w3", exit_price=exit_for(1.0), minutes=20),
        trade("l1", exit_price=exit_for(-1.0), minutes=30, exit_reason=ExitReason.STOP),
        trade("s1", exit_price=ENTRY, minutes=40, exit_reason=ExitReason.TIME_STOP),
    ]
    assert with_scratch[-1].pnl == 0.0 and with_scratch[-1].is_scratch
    rates = {
        name: M.TRADE_METRICS[name](
            with_scratch, split_label=TRAIN, policy=POLICY
        ).require()
        for name in ("win_rate", "loss_rate", "scratch_rate")
    }
    assert rates == {"win_rate": 0.6, "loss_rate": 0.2, "scratch_rate": 0.2}
    assert sum(rates.values()) == 1.0
    assert rates["win_rate"] != pytest.approx(0.75)


def test_expectancy_is_the_mean_net_r_and_matches_the_textbook_form():
    """sum(R)/n = (2 + 1 + 0.5 - 0.5 - 1) / 5 = 2.0 / 5 = 0.4R.

    Cross-checked against the textbook expansion, which is arithmetic this
    module never performs: p_win * avg_win - p_loss * |avg_loss|
    = 0.6 * (7/6) - 0.4 * 0.75 = 0.7 - 0.3 = 0.4R. The two agreeing is the
    evidence that expectancy is net of costs and read from realized R.
    """
    book = five_trade_book()
    assert M.expectancy_r(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(0.4)
    )
    assert M.mean_r(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(0.4)
    )
    assert 0.6 * (7 / 6) - 0.4 * 0.75 == pytest.approx(0.4)
    # In dollars: +$800 over five trades on a $400 R = $160 = 0.4R.
    assert M.expectancy_dollars(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(160.0)
    )
    assert 160.0 / R_DOLLARS == pytest.approx(0.4)


def test_average_win_against_average_loss_and_their_ratio():
    """avg win  = (2 + 1 + 0.5) / 3 = 3.5 / 3 = 7/6 = 1.1666...R
    avg loss = (-0.5 - 1) / 2 = -0.75R, reported NEGATIVE as experienced
    ratio    = (7/6) / 0.75 = 14/9 = 1.5555...
    """
    book = five_trade_book()
    assert M.average_win_r(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(7 / 6)
    )
    assert M.average_win_r(book, split_label=TRAIN, policy=POLICY).n == 3
    loss = M.average_loss_r(book, split_label=TRAIN, policy=POLICY)
    assert loss.require() == pytest.approx(-0.75)
    assert loss.require() < 0.0, "a loss is not a positive number"
    assert loss.n == 2
    assert M.win_loss_ratio(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(14 / 9)
    )


def test_profit_factor_is_the_net_quotient_and_not_the_gross_one():
    """net: gross_profit 800 + 400 + 200 = $1400; gross_loss -200 - 400 = -$600
    -> 1400 / 600 = 7/3 = 2.3333...

    Before costs it would be 1460 / 540 = 2.7037, and that difference -- 16%
    of the headline -- is the whole reason the metric must be net.
    """
    book = five_trade_book()
    assert M.gross_profit(book, split_label=TRAIN, policy=POLICY).require() == 1400.0
    assert M.gross_loss(book, split_label=TRAIN, policy=POLICY).require() == -600.0
    assert M.total_pnl(book, split_label=TRAIN, policy=POLICY).require() == 800.0
    assert M.total_costs(book, split_label=TRAIN, policy=POLICY).require() == 100.0

    measured = M.profit_factor(book, split_label=TRAIN, policy=POLICY).require()
    assert measured == pytest.approx(7 / 3)
    assert measured != pytest.approx(1460.0 / 540.0)


def test_median_and_dispersion_of_realized_r():
    """sorted R: [-1, -0.5, 0.5, 1, 2] -> median 0.5 (not the mean, 0.4).
    sample stdev (ddof=1): deviations from 0.4 are 1.6, 0.6, 0.1, -0.9, -1.4;
    squares 2.56 + 0.36 + 0.01 + 0.81 + 1.96 = 5.70; 5.70/4 = 1.425;
    sqrt(1.425) = 1.1937336...
    """
    book = five_trade_book()
    assert M.median_r(book, split_label=TRAIN, policy=POLICY).require() == 0.5
    assert M.stdev_r(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(math.sqrt(1.425))
    )


def test_breakeven_win_rate_comes_from_this_sample_not_from_a_constant():
    """|avg_loss| / (avg_win + |avg_loss|) = 0.75 / (7/6 + 0.75)
    = 0.75 / (23/12) = 9/23 = 0.3913043...

    A zero-cost symmetric 1R book gives exactly 0.5, which is the control: any
    deviation from 50% has to come from the measured win and loss sizes.
    """
    assert M.breakeven_win_rate(
        five_trade_book(), split_label=TRAIN, policy=POLICY
    ).require() == pytest.approx(9 / 23)

    costless = [
        trade("w", exit_price=exit_for(1.0), minutes=0),
        trade("l", exit_price=exit_for(-1.0), minutes=10, exit_reason=ExitReason.STOP),
    ]
    assert M.breakeven_win_rate(
        costless, split_label=TRAIN, policy=POLICY
    ).require() == 0.5


def test_the_measured_cost_hurdle_puts_breakeven_above_fifty_percent():
    """The brief's figure, re-derived from records rather than quoted.

    target: +$400 gross - $21.49 = +$378.51 -> +0.946275R
    stop:   -$400 gross - $47.74 = -$447.74 -> -1.119350R
    breakeven = 1.11935 / (0.946275 + 1.11935) = 1.11935 / 2.065625 = 0.5418941

    So a symmetric 1R system needs 54.19%, not 50%, and the 4.19 points are
    the round-turn costs. Cross-checked by showing expectancy at exactly that
    win rate is zero, which is what "breakeven" has to mean.
    """
    book = hurdle_book()
    assert [t.r_multiple for t in book] == [0.946275, -1.11935]

    hurdle = M.breakeven_win_rate(book, split_label=TRAIN, policy=POLICY).require()
    assert hurdle == pytest.approx(0.5418941, abs=1e-7)
    assert hurdle > 0.5
    assert round(hurdle * 100, 2) == 54.19

    expectancy_at_hurdle = hurdle * 0.946275 - (1.0 - hurdle) * 1.11935
    assert expectancy_at_hurdle == pytest.approx(0.0, abs=1e-12)


def test_cost_drag_is_reported_per_exit_reason_because_one_figure_is_wrong():
    """A target is a resting limit; a stop is a market order that also pays
    stop_order_extra_slippage_ticks. On a $400 R:

      target: $21.49 / $400 = 0.053725R
      stop:   $47.74 / $400 = 0.119350R   -> 2.22x the target

    A single "cost drag" number describes neither, so it is only published
    beside the exit mix, and its note says so.
    """
    report = M.cost_drag_by_exit_reason(hurdle_book(), split_label=TRAIN, policy=POLICY)

    assert report.by_reason_r[ExitReason.TARGET].require() == (
        pytest.approx(TARGET_COST / R_DOLLARS)
    )
    assert report.by_reason_r[ExitReason.STOP].require() == (
        pytest.approx(STOP_COST / R_DOLLARS)
    )
    assert report.by_reason_r[ExitReason.TARGET].require() == pytest.approx(0.053725)
    assert report.by_reason_r[ExitReason.STOP].require() == pytest.approx(0.11935)
    assert report.by_reason_r[ExitReason.STOP].require() > (
        2.0 * report.by_reason_r[ExitReason.TARGET].require()
    )
    assert report.by_reason_dollars[ExitReason.STOP].require() == STOP_COST
    assert report.exit_counts == {ExitReason.STOP: 1, ExitReason.TARGET: 1}
    assert report.mix_weighted_r.require() == pytest.approx(
        (TARGET_COST + STOP_COST) / 2 / R_DOLLARS
    )
    assert "exit_counts" in report.mix_weighted_r.note
    assert "mix" in M.mean_cost_drag_r(
        hurdle_book(), split_label=TRAIN, policy=POLICY
    ).note


def test_the_exit_mix_of_the_five_trade_book_is_reported_in_declared_order():
    """The report must not inherit the order trades arrived in, or two runs
    over the same book would produce different column orders. ExitReason
    declares STOP before TARGET before TIME_STOP before SESSION_CLOSE."""
    report = M.cost_drag_by_exit_reason(
        list(reversed(five_trade_book())), split_label=TRAIN, policy=POLICY
    )
    assert tuple(report.by_reason_r) == (
        ExitReason.TARGET, ExitReason.TIME_STOP, ExitReason.SESSION_CLOSE
    )
    assert report.exit_counts == {
        ExitReason.TARGET: 2, ExitReason.TIME_STOP: 2, ExitReason.SESSION_CLOSE: 1
    }
    # Every trade in this book paid $20 on a $400 R.
    assert all(v.require() == pytest.approx(0.05) for v in report.by_reason_r.values())


def test_the_r_distribution_percentiles_and_histogram_are_hand_computed():
    """sorted R: [-1, -0.5, 0.5, 1, 2], linear interpolation on index (n-1)q:

      p5  -> index 0.2 -> -1 + 0.2 * 0.5 = -0.9
      p25 -> index 1.0 -> -0.5
      p50 -> index 2.0 ->  0.5
      p75 -> index 3.0 ->  1.0
      p95 -> index 3.8 ->  1 + 0.8 * 1 = 1.8

    The histogram bins are half-open [lower, upper), so an R landing exactly
    on an edge goes to the bin above it.
    """
    dist = M.r_distribution(five_trade_book(), split_label=TRAIN, policy=POLICY)

    assert dist.n_trades == 5
    assert {k: v.require() for k, v in dist.percentiles.items()} == {
        "p5": pytest.approx(-0.9),
        "p25": pytest.approx(-0.5),
        "p50": pytest.approx(0.5),
        "p75": pytest.approx(1.0),
        "p95": pytest.approx(1.8),
    }
    assert dist.minimum.require() == -1.0
    assert dist.maximum.require() == 2.0
    assert dist.mean.require() == pytest.approx(0.4)
    populated = {b.label: b.count for b in dist.histogram if b.count}
    assert populated == {
        "[-1,-0.5)": 1, "[-0.5,0)": 1, "[0.5,1)": 1, "[1,1.5)": 1, "[2,3)": 1
    }
    assert sum(b.count for b in dist.histogram) == 5


def test_an_r_outside_the_histogram_range_is_counted_in_an_open_tail():
    """Clipping an outlier into the end bucket would understate the tail,
    which is the part of the distribution a risk reader cares about most."""
    outlier = [
        trade("big", exit_price=exit_for(5.0, 20.0), commission=20.0, minutes=0),
        trade("bad", exit_price=exit_for(-4.0, 20.0), commission=20.0, minutes=60,
              exit_reason=ExitReason.STOP),
    ]
    assert [t.r_multiple for t in outlier] == [5.0, -4.0]
    dist = M.r_distribution(outlier, split_label=TRAIN, policy=POLICY)
    tails = {b.label: b.count for b in dist.histogram if b.count}
    assert tails == {"[-inf,-3)": 1, "[3,+inf)": 1}
    assert dist.maximum.require() == 5.0
    assert dist.minimum.require() == -4.0


def test_excursions_and_holding_times_are_read_from_the_record():
    """MAE/MFE: mae_r = [-0.25, -0.25, -0.25, -0.5, -1.0] -> mean -2.25/5 =
    -0.45, worst -1.0; mfe_r = [2.25, 1.25, 0.75, 0.5, 0.25] -> mean 5.0/5 =
    1.0, best 2.25. Holding: 10, 20, 30, 40, 50 minutes -> mean and median
    1800 s; bars 2, 4, 6, 8, 10 -> mean 6.
    """
    book = five_trade_book()
    assert M.mean_mae_r(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(-0.45)
    )
    assert M.worst_mae_r(book, split_label=TRAIN, policy=POLICY).require() == -1.0
    assert M.mean_mfe_r(book, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(1.0)
    )
    assert M.best_mfe_r(book, split_label=TRAIN, policy=POLICY).require() == 2.25
    assert M.mean_holding_seconds(book, split_label=TRAIN, policy=POLICY).require() == (
        1800.0
    )
    assert M.median_holding_seconds(
        book, split_label=TRAIN, policy=POLICY
    ).require() == 1800.0
    assert M.mean_bars_held(book, split_label=TRAIN, policy=POLICY).require() == 6.0
    # The excursion note warns that 0.0 is also the unmeasured default.
    assert "indistinguishable" in M.mean_mae_r(
        book, split_label=TRAIN, policy=POLICY
    ).note


def test_streaks_are_measured_in_exit_order_and_a_scratch_breaks_both():
    """The five-trade book realizes W W W L L in exit order: 3 and 2, and the
    run in progress at the end is two losses, reported as -2.

    A scratch is neither a win nor a loss, so it interrupts a run rather than
    extending it -- otherwise a flat trade in the middle of three winners
    would be reported as a four-trade winning streak.
    """
    book = five_trade_book()
    profile = M.streaks(book, split_label=TRAIN, policy=POLICY)
    assert profile.max_consecutive_wins.require() == 3.0
    assert profile.max_consecutive_losses.require() == 2.0
    assert profile.current_streak.require() == -2.0
    assert profile.n_trades == 5

    # Argument order is not meaningful; exit order is.
    shuffled = [book[3], book[0], book[4], book[2], book[1]]
    assert M.streaks(shuffled, split_label=TRAIN, policy=POLICY).to_init_dict() == (
        profile.to_init_dict()
    )

    broken = [
        trade("w1", exit_price=exit_for(1.0), minutes=0),
        trade("w2", exit_price=exit_for(1.0), minutes=10),
        trade("s", exit_price=ENTRY, minutes=20, exit_reason=ExitReason.TIME_STOP),
        trade("w3", exit_price=exit_for(1.0), minutes=30),
    ]
    assert M.streaks(
        broken, split_label=TRAIN, policy=POLICY
    ).max_consecutive_wins.require() == 2.0


def test_drawdown_depth_recovery_and_the_episode_still_open():
    """Daily curve 100k -> 110k -> 88k -> 110k -> 99k.

    episode 1: peak 110,000 at day 1, trough 88,000 at day 2, recovered day 3
               depth -22,000 = -20.0% of the peak; recovery 2 days = 172,800 s
    episode 2: peak 110,000 at day 3, trough 99,000 at day 4, NOT recovered
               depth -11,000 = -10.0%
    average depth = (-0.20 + -0.10) / 2 = -0.15
    still underwater at the end: day 4 - day 3 = 86,400 s

    The unrecovered episode is kept and counted: "do not hide bad periods"
    starts with the one still open at the end of the series.
    """
    profile = M.drawdown_profile(
        equity([100_000.0, 110_000.0, 88_000.0, 110_000.0, 99_000.0]),
        split_label=TRAIN,
        policy=POLICY,
    )
    assert profile.n_episodes == 2
    assert profile.n_unrecovered == 1
    assert profile.max_drawdown_pct.require() == pytest.approx(-0.2)
    assert profile.max_drawdown_dollars.require() == pytest.approx(-22_000.0)
    assert profile.average_drawdown_pct.require() == pytest.approx(-0.15)
    assert profile.longest_recovery_seconds.require() == pytest.approx(172_800.0)
    assert profile.mean_recovery_seconds.require() == pytest.approx(172_800.0)
    assert profile.seconds_underwater_at_end.require() == pytest.approx(86_400.0)
    assert profile.trough_ts == T0 + timedelta(hours=48)
    # Depths are negative everywhere, matching EquityPoint.drawdown_pct and
    # TradeRecord.mae_r, so a drawdown never changes sign across the system.
    assert all(e.depth_dollars < 0.0 and e.depth_pct < 0.0 for e in profile.episodes)
    assert M.recovery_time(
        equity([100_000.0, 90_000.0, 95_000.0]), split_label=TRAIN, policy=POLICY
    ).undefined is M.Undefined.NOT_RECOVERED


def test_a_series_that_starts_mid_drawdown_does_not_understate_it():
    """`high_water_mark` defaults to 0.0 on EquityPoint, so the running peak
    is computed from the series -- but the FIRST point's stored value is
    honoured as a lower bound, or a fold that begins underwater would report
    no drawdown at all."""
    mid_run = [
        EquityPoint(ts=T0 + timedelta(hours=i), equity=value,
                    realized_equity=value, high_water_mark=100_000.0)
        for i, value in enumerate([90_000.0, 95_000.0, 100_000.0])
    ]
    profile = M.drawdown_profile(mid_run, split_label=TRAIN, policy=POLICY)
    assert profile.n_episodes == 1
    assert profile.max_drawdown_pct.require() == pytest.approx(-0.1)
    assert profile.max_drawdown_dollars.require() == pytest.approx(-10_000.0)

    # Without the stored peak the same shape looks like a clean rising curve.
    assert M.drawdown_profile(
        equity([90_000.0, 95_000.0, 100_000.0]), split_label=TRAIN, policy=POLICY
    ).max_drawdown_pct.require() == 0.0


def test_sharpe_and_sortino_are_hand_computed_including_the_risk_free_hurdle():
    """A 31-point curve alternating 100k and 200k gives 30 returns: fifteen of
    +1.0 and fifteen of -0.5.

      mean   = (15 * 1.0 + 15 * -0.5) / 30 = 7.5 / 30 = 0.25
      stdev  = 0.75 * sqrt(30/29)   (deviations are all +/-0.75, ddof=1)
      sharpe = 0.25 / (0.75 * sqrt(30/29)) = (1/3) * sqrt(29/30) = 0.3277307

    With a 4% annual risk-free rate and periods_per_year = 1 the hurdle is
    exactly 0.04, so the ratio becomes (0.25 - 0.04) / stdev -- a test that
    the hurdle is subtracted rather than quietly dropped.

    Sortino uses downside deviation over ALL observations (ddof 0):
      downside = sqrt(15 * 0.5^2 / 30) = sqrt(0.125) = 0.3535534
      sortino  = 0.25 / 0.3535534 = 0.7071068

    Annualization is sqrt(periods_per_year) exactly, which the last assertion
    pins at periods_per_year = 4.
    """
    curve = equity([100_000.0 * (2.0 if i % 2 else 1.0) for i in range(31)])
    stdev = 0.75 * math.sqrt(30 / 29)

    no_hurdle = M.sharpe_ratio(curve, split_label=None, policy=POLICY, periods_per_year=1.0)
    assert no_hurdle.require() == pytest.approx((1 / 3) * math.sqrt(29 / 30))
    assert no_hurdle.unit == "ratio"
    assert no_hurdle.n == 30

    with_hurdle = M.sharpe_ratio(
        curve, split_label=None, policy=POLICY_RF, periods_per_year=1.0
    )
    assert with_hurdle.require() == pytest.approx((0.25 - 0.04) / stdev)
    assert with_hurdle.require() < no_hurdle.require()

    sortino = M.sortino_ratio(curve, split_label=None, policy=POLICY, periods_per_year=1.0)
    assert sortino.require() == pytest.approx(0.25 / math.sqrt(0.125))

    annualized = M.sharpe_ratio(curve, split_label=None, policy=POLICY, periods_per_year=4.0)
    assert annualized.require() == pytest.approx(2.0 * no_hurdle.require())
    assert "4 periods/year" in annualized.note

    # Per-period, with no annualization supplied, the unit says so and no
    # annual rate is applied to a period of unknown length.
    per_period = M.sharpe_from_returns(
        [1.0] * 15 + [-0.5] * 15, split_label=None, policy=POLICY_RF
    )
    assert per_period.unit == "ratio/period"
    assert per_period.require() == pytest.approx(0.25 / stdev)


def test_the_annualization_factor_is_measured_from_the_series_and_the_policy():
    """13 marks six hours apart span four calendar dates, so the series
    carries 12 / 4 = 3 observations per trading day, and with
    trading_days_per_year = 250 that is 750 periods per year.

    `trading_days_per_year` is 250 here on purpose -- it is NOT the schema
    default of 252 -- so this test fails if the factor is ever taken from a
    constant in the module instead of from the policy.
    """
    series = equity([100_000.0 + 10.0 * i for i in range(13)], step=timedelta(hours=6))
    assert len({p.ts.date() for p in series}) == 4

    measured = M.infer_periods_per_year(series, policy=POLICY)
    assert measured.require() == pytest.approx(750.0)
    assert measured.n == 12

    other = M.MetricPolicy.from_analytics(
        LOCAL_ANALYTICS.replace(trading_days_per_year=252)
    )
    assert M.infer_periods_per_year(series, policy=other).require() == (
        pytest.approx(756.0)
    )
    # A one-point series has no sampling density to measure.
    assert M.infer_periods_per_year(
        equity([100_000.0]), policy=POLICY
    ).undefined is M.Undefined.NO_EQUITY_SERIES


def test_total_return_and_cagr_over_an_exact_two_year_span():
    """100,000 -> 121,000 over exactly two Julian years (730.5 days):
      total return = 121/100 - 1 = 0.21
      CAGR         = 1.21 ** (1/2) - 1 = 1.1 - 1 = 0.10 exactly
    """
    two_years = equity([100_000.0, 121_000.0], step=timedelta(days=730.5))
    assert M.total_return(two_years, split_label=TRAIN, policy=POLICY).require() == (
        pytest.approx(0.21)
    )
    growth = M.cagr(two_years, split_label=TRAIN, policy=POLICY)
    assert growth.require() == pytest.approx(0.1)
    assert "2 calendar years" in growth.note

    # Five trades over that window is 2.5 trades per year.
    assert M.trade_frequency(
        five_trade_book(), two_years, split_label=TRAIN, policy=POLICY
    ).require() == pytest.approx(2.5)


def test_the_r_target_dimension_reads_the_planned_multiple_off_the_prices():
    """A setup called 1R that actually targets 0.4R against a full stop would
    show up as a 0.4R cell here, which is the point: the label comes from the
    record's own prices, not from the setup's name."""
    book = five_trade_book() + [
        trade("two", exit_price=exit_for(2.0, 20.0), commission=20.0, minutes=400,
              target_price=20_020.0, setup=SetupType.SETUP_2R),
    ]
    table = B.breakdown(book, "r_target", split_label=TRAIN, policy=POLICY)
    assert [(c.key, c.n) for c in table.cells] == [("1R", 5), ("2R", 1)]


# ---------------------------------------------------------------------------
# 6. determinism
# ---------------------------------------------------------------------------


def test_two_runs_over_the_same_data_are_byte_identical():
    """Section 0.5. The books are built twice from scratch rather than reused,
    so this also catches a metric that mutates its input or caches state
    between calls."""
    curve = [100_000.0, 110_000.0, 88_000.0, 110_000.0, 99_000.0]
    first = M.performance_summary(
        five_trade_book(), equity(curve), split_label=TRAIN, policy=POLICY
    )
    second = M.performance_summary(
        five_trade_book(), equity(curve), split_label=TRAIN, policy=POLICY
    )
    assert first.to_init_json() == second.to_init_json()

    tables_a = B.breakdown_all(five_trade_book(), split_label=TRAIN, policy=POLICY)
    tables_b = B.breakdown_all(five_trade_book(), split_label=TRAIN, policy=POLICY)
    assert [t.to_init_json() for t in tables_a.values()] == (
        [t.to_init_json() for t in tables_b.values()]
    )
    assert list(tables_a) == list(tables_b)


def test_the_order_of_a_trade_list_does_not_change_the_numbers():
    """A trade list is a SET -- its argument order carries no information --
    while a streak is not, which is why the sequence-dependent metrics sort on
    (exit_ts, entry_ts, trade_id) first.

    KNOWN LIMITATION, pinned rather than hidden: `stdev_r` is the one metric
    whose result can move in the last bits under reordering, because numpy's
    summation order follows the array. The tolerance below is 1e-12 -- a
    thousand times smaller than any difference a reader could act on -- and
    every other metric is bit-identical.
    """
    book = five_trade_book()
    orders = [
        [book[i] for i in perm]
        for perm in ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3), (1, 4, 0, 3, 2))
    ]
    baseline = {
        name: function(book, split_label=TRAIN, policy=POLICY)
        for name, function in M.TRADE_METRICS.items()
    }
    for ordering in orders:
        for name, function in M.TRADE_METRICS.items():
            value = function(ordering, split_label=TRAIN, policy=POLICY)
            if name == "stdev_r":
                assert value.require() == pytest.approx(
                    baseline[name].require(), abs=1e-12
                ), name
            else:
                assert value.to_init_dict() == baseline[name].to_init_dict(), name


def test_cell_order_is_declared_rather_than_data_dependent():
    """Two runs whose trades arrive in a different order must produce the same
    columns in the same places, or a report diff is unreadable. Declared-domain
    keys keep their declared order; observed keys follow an explicit sort."""
    book = [
        trade("z", exit_price=exit_for(1.0), minutes=0, symbol="NQ",
              regime=Regime.CHOP, session=Session.RTH_CLOSE,
              split_label=TRAIN),
        trade("y", exit_price=exit_for(1.0), minutes=10, symbol="ES",
              regime=Regime.LOW_VOL, session=Session.ASIA, split_label=TRAIN),
    ]
    forward = B.breakdown_all(book, split_label=TRAIN, policy=POLICY)
    reverse = B.breakdown_all(list(reversed(book)), split_label=TRAIN, policy=POLICY)
    for name, table in forward.items():
        assert [c.key for c in table.cells] == [c.key for c in reverse[name].cells], name

    # Declared domains keep enum declaration order...
    assert [c.key for c in forward["regime"].cells] == [str(r) for r in Regime]
    assert [c.key for c in forward["session"].cells] == [str(s) for s in Session]
    assert [c.key for c in forward["day_of_week"].cells] == list(B.DAY_OF_WEEK_LABELS)
    # ...and observed keys are sorted explicitly, not by arrival.
    assert [c.key for c in forward["market"].cells] == ["ES", "NQ"]


def test_chronological_dimensions_sort_chronologically_not_lexically():
    """A year-month column ordered by string would put 2021-01 before
    2020-12 only by luck; the dimension declares an explicit sort key."""
    late_2020 = datetime(2020, 12, 30, 15, 0, tzinfo=UTC)
    book = [
        trade("a", exit_price=exit_for(1.0), minutes=0),
        trade("b", exit_price=exit_for(1.0),
              minutes=(late_2020 - T0).total_seconds() / 60.0),
    ]
    years = B.breakdown(book, "year", split_label=TRAIN, policy=POLICY)
    months = B.breakdown(book, "month", split_label=TRAIN, policy=POLICY)
    assert [c.key for c in years.cells] == ["2020", "2021"]
    assert [c.key for c in months.cells] == ["2020-12", "2021-03"]
    assert [c.n for c in months.cells] == [1, 1]


def test_metric_columns_follow_the_registry_order():
    """Column order comes from the source-ordered registry, so it does not
    depend on dict iteration luck or on which metrics happened to be
    requested first."""
    computed = M.compute_trade_metrics(
        five_trade_book(), split_label=TRAIN, policy=POLICY,
        names=tuple(M.TRADE_METRICS),
    )
    assert tuple(computed) == tuple(M.TRADE_METRICS)
    cell = B.breakdown(
        five_trade_book(), "setup", split_label=TRAIN, policy=POLICY,
        metrics=M.DEFAULT_CELL_METRICS,
    ).cell(str(SetupType.SCALP_1R))
    assert tuple(cell.metrics) == tuple(M.DEFAULT_CELL_METRICS)


def test_nothing_in_analytics_reads_a_clock_a_file_or_a_random_number():
    """Pure functions, no I/O, no state, no clock (section 0.5). A metric that
    read `datetime.now()` -- say to annualize "up to today" -- would make a
    report un-reproducible, and the drift would be invisible in any single
    run."""
    forbidden = {
        "now", "utcnow", "today", "fromtimestamp", "time", "monotonic", "perf_counter",
        "open", "read_text", "read_bytes", "write_text", "connect", "default_rng",
        "random", "shuffle", "choice", "seed", "uniform", "normal",
    }
    for module in (M, B):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            assert called not in forbidden, (
                f"{module.__name__} calls {called}() at line {node.lineno}"
            )


def test_analytics_depends_only_on_numpy_pydantic_and_the_standard_library():
    """"Avoid unnecessary dependencies": numpy / pandas / pydantic / stdlib
    sqlite3 only, and no pandas is needed to take a mean."""
    allowed_roots = {
        "__future__", "math", "collections", "datetime", "dataclasses", "typing",
        "numpy", "pydantic", "flow_model",
    }
    for module in (M, B):
        tree = ast.parse(Path(module.__file__).read_text())
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert roots <= allowed_roots, f"{module.__name__} imports {roots - allowed_roots}"
        assert "pandas" not in roots and "scipy" not in roots


# ---------------------------------------------------------------------------
# 7. degenerate inputs
# ---------------------------------------------------------------------------


def test_a_single_trade_reports_its_counts_and_refuses_everything_else():
    """n=1 is the first sample a live system produces, and the sample where a
    confident number is most dangerous. Rates are reported WITH their n of 1;
    dispersion and two-sided ratios are refused."""
    one = [trade("only", exit_price=exit_for(1.0))]
    assert M.trade_count(one, split_label=TRAIN, policy=POLICY).require() == 1.0
    win = M.win_rate(one, split_label=TRAIN, policy=POLICY)
    assert (win.require(), win.n) == (1.0, 1)
    assert M.mean_r(one, split_label=TRAIN, policy=POLICY).require() == 1.0
    assert M.median_r(one, split_label=TRAIN, policy=POLICY).require() == 1.0
    assert M.stdev_r(one, split_label=TRAIN, policy=POLICY).undefined is (
        M.Undefined.INSUFFICIENT_OBSERVATIONS
    )
    assert M.profit_factor(one, split_label=TRAIN, policy=POLICY).undefined is (
        M.Undefined.NO_LOSSES
    )
    assert M.win_loss_ratio(one, split_label=TRAIN, policy=POLICY).undefined is (
        M.Undefined.NO_LOSSES
    )
    dist = M.r_distribution(one, split_label=TRAIN, policy=POLICY)
    assert dist.stdev.undefined is M.Undefined.INSUFFICIENT_OBSERVATIONS
    assert all(v.require() == 1.0 for v in dist.percentiles.values())
    streak = M.streaks(one, split_label=TRAIN, policy=POLICY)
    assert (streak.max_consecutive_wins.require(), streak.current_streak.require()) == (
        1.0, 1.0
    )


def test_a_single_equity_point_is_not_a_series():
    """One mark has no return, no span and no growth rate. Its drawdown is
    0.0 because zero depth was observed at the only point there is -- stated
    here so the asymmetry with the ratios is deliberate rather than
    accidental."""
    one_point = equity([100_000.0])
    for function in (M.total_return, M.cagr, M.sharpe_ratio, M.sortino_ratio):
        assert function(
            one_point, split_label=TRAIN, policy=POLICY
        ).undefined is M.Undefined.NO_EQUITY_SERIES
    profile = M.drawdown_profile(one_point, split_label=TRAIN, policy=POLICY)
    assert profile.max_drawdown_pct.require() == 0.0
    assert profile.n_episodes == 0
    assert profile.average_drawdown_pct.undefined is M.Undefined.NO_DRAWDOWN_EPISODE
    assert M.simple_returns(one_point) == []

    empty_profile = M.drawdown_profile([], split_label=TRAIN, policy=POLICY)
    assert empty_profile.max_drawdown_pct.undefined is M.Undefined.NO_EQUITY_SERIES
    assert empty_profile.n_points == 0


def test_an_all_winning_book_refuses_the_loss_side_rather_than_zeroing_it():
    """Three winners: the win rate is 1.0 and the loss rate is 0.0 -- both
    measurements, because trades happened. The average loss is not 0.0, it
    does not exist."""
    wins = [trade(f"w{i}", exit_price=exit_for(1.0), minutes=10 * i) for i in range(3)]
    assert M.win_rate(wins, split_label=TRAIN, policy=POLICY).require() == 1.0
    assert M.loss_rate(wins, split_label=TRAIN, policy=POLICY).require() == 0.0
    assert M.gross_loss(wins, split_label=TRAIN, policy=POLICY).require() == 0.0
    for function in (M.average_loss_r, M.win_loss_ratio, M.profit_factor,
                     M.breakeven_win_rate):
        assert function(
            wins, split_label=TRAIN, policy=POLICY
        ).undefined is M.Undefined.NO_LOSSES, function.__name__
    assert M.streaks(
        wins, split_label=TRAIN, policy=POLICY
    ).max_consecutive_losses.require() == 0.0


def test_an_all_losing_book_refuses_the_win_side_rather_than_zeroing_it():
    """The mirror image, and the one that matters for a report nobody wants to
    write: expectancy and total P&L are real negative measurements, while the
    average win is undefined."""
    losses = [
        trade(f"l{i}", exit_price=exit_for(-1.0), minutes=10 * i,
              exit_reason=ExitReason.STOP)
        for i in range(3)
    ]
    assert M.win_rate(losses, split_label=TRAIN, policy=POLICY).require() == 0.0
    assert M.expectancy_r(losses, split_label=TRAIN, policy=POLICY).require() == -1.0
    assert M.total_pnl(losses, split_label=TRAIN, policy=POLICY).require() == -1200.0
    assert M.profit_factor(losses, split_label=TRAIN, policy=POLICY).require() == 0.0
    for function in (M.average_win_r, M.win_loss_ratio, M.breakeven_win_rate):
        assert function(
            losses, split_label=TRAIN, policy=POLICY
        ).undefined is M.Undefined.NO_WINS, function.__name__


def test_a_zero_risk_trade_cannot_exist_which_is_why_r_may_be_read():
    """Every R in this module is `pnl / risk_dollars` read off the record, so
    the division by zero has to be impossible at the contract. It is:
    `risk_dollars` is `Field(gt=0)`. This test is the reason `mean_r` needs no
    zero-risk branch."""
    fields = dict(
        trade_id="z", run_id="r", symbol="NQ", config_hash="c",
        signal_ts=T0, entry_ts=T0, exit_ts=T0 + timedelta(minutes=1),
        side=Side.LONG, setup=SetupType.SCALP_1R, planned_entry=ENTRY,
        stop_price=STOP, target_price=TARGET, entry_price=ENTRY, exit_price=ENTRY,
        size=SIZE, filled_size=SIZE, point_value=POINT_VALUE,
        exit_reason=ExitReason.TIME_STOP, gross_pnl=0.0, pnl=0.0,
        flow_score=75.0, regime=Regime.LOW_VOL,
    )
    for risk in (0.0, -400.0):
        with pytest.raises(ValueError):
            TradeRecord(risk_dollars=risk, **fields)
    assert TradeRecord(risk_dollars=R_DOLLARS, **fields).r_multiple == 0.0


def test_a_nan_cannot_reach_a_metric_through_a_trade_record():
    """The other half of the NaN story: the equity series is refused at the
    analytics door, and a NaN in a trade is refused at the contract. Between
    them, no mean, sum or percentile in this module can be handed one."""
    base = dict(
        trade_id="n", run_id="r", symbol="NQ", config_hash="c",
        signal_ts=T0, entry_ts=T0, exit_ts=T0 + timedelta(minutes=1),
        side=Side.LONG, setup=SetupType.SCALP_1R, planned_entry=ENTRY,
        stop_price=STOP, target_price=TARGET, entry_price=ENTRY, exit_price=TARGET,
        size=SIZE, filled_size=SIZE, point_value=POINT_VALUE,
        exit_reason=ExitReason.TARGET, risk_dollars=R_DOLLARS,
        gross_pnl=400.0, pnl=400.0, flow_score=75.0, regime=Regime.LOW_VOL,
    )
    nan = float("nan")
    for field in ("pnl", "gross_pnl", "mae_r", "mfe_r", "flow_score", "risk_dollars",
                  "entry_price", "exit_price", "size"):
        with pytest.raises(ValueError):
            TradeRecord(**{**base, field: nan})
    # A NaN cost is refused one layer earlier still, by CostBreakdown itself,
    # so it cannot even be built to hand to a record.
    with pytest.raises(ValueError):
        CostBreakdown(commission=nan)


def test_a_trade_iterable_is_materialized_before_it_is_walked_twice():
    """Most of these functions walk their input more than once (counts, then
    sums, then excursions). A generator handed straight through would be
    exhausted on the first pass and silently report an n of 0 for everything
    after it."""
    book = five_trade_book()
    assert M.win_rate(iter(book), split_label=TRAIN, policy=POLICY).require() == 0.6
    summary = M.performance_summary(
        iter(book), iter(equity([100_000.0, 101_000.0])),
        split_label=TRAIN, policy=POLICY,
    )
    assert summary.n_trades == 5
    assert summary.trade_metrics["win_rate"].require() == 0.6
    assert summary.r_distribution.n_trades == 5
    assert B.breakdown(
        iter(book), "setup", split_label=TRAIN, policy=POLICY
    ).n_trades == 5


@pytest.mark.parametrize("score", [70.0, 70.1, 79.999, 80.0, 99.9, 100.0])
def test_every_in_range_flow_score_lands_in_exactly_one_declared_bucket(score):
    """A score labelled with a bucket the domain does not declare produces TWO
    cells for one bucket -- a declared-but-empty one and an observed one -- and
    splits the sample. Boundaries and the closed top edge are where that
    happens; the local edges here are (70, 80, 100)."""
    edges = POLICY.flow_score_buckets
    label = B.flow_score_bucket_label(score, edges)
    domain = B.flow_score_bucket_domain(edges)
    assert label in domain
    assert domain.count(label) == 1
    assert domain == ("[70,80)", "[80,100]")


def test_a_flow_score_outside_the_configured_buckets_gets_a_visible_cell():
    """A score outside the configured edges means the buckets are
    mis-configured. Folding it into an end bucket would hide that; it gets its
    own labelled cell outside the declared domain instead, and no trade is
    lost."""
    edges = POLICY.flow_score_buckets
    assert B.flow_score_bucket_label(65.23, edges) == "<70"
    assert B.flow_score_bucket_label(100.1, edges) == ">100"
    assert "<70" not in B.flow_score_bucket_domain(edges)

    book = [
        trade("below", exit_price=exit_for(1.0), minutes=0, flow_score=65.23),
        trade("inside", exit_price=exit_for(1.0), minutes=10, flow_score=75.0),
    ]
    table = B.breakdown(book, "flow_score_bucket", split_label=TRAIN, policy=POLICY)
    assert [(c.key, c.n) for c in table.cells] == [
        ("[70,80)", 1), ("[80,100]", 0), ("<70", 1)
    ]
    assert table.n_trades == 2
    with pytest.raises(ValueError):
        B.flow_score_bucket_label(75.0, (70.0,))


def test_a_new_dimension_needs_no_metric_and_a_new_metric_needs_no_dimension():
    """The two sides meet only through `TRADE_METRICS`. This is the property
    that keeps section 12's dimension list extensible: a custom dimension gets
    every registered metric without touching metrics.py, and a metric name
    added to the request appears in every cell without touching breakdowns.py.
    """
    by_side = B.Dimension(
        name="side",
        description="direction of the position",
        key_of=lambda t: t.side,
        label_of=str,
        domain=tuple(Side),
    )
    book = five_trade_book()
    table = B.breakdown(
        book, by_side, split_label=TRAIN, policy=POLICY,
        metrics=("win_rate", "expectancy", "max_consecutive_losses"),
    )
    assert [(c.key, c.n, str(c.status)) for c in table.cells] == [
        ("LONG", 5, "ADEQUATE"), ("SHORT", 0, "EMPTY")
    ]
    assert tuple(table.cell("LONG").metrics) == (
        "win_rate", "expectancy", "max_consecutive_losses"
    )
    assert table.cell("LONG").metrics["win_rate"].require() == 0.6
    assert table.metric_names == ("win_rate", "expectancy", "max_consecutive_losses")

    with pytest.raises(KeyError):
        B.breakdown(book, by_side, split_label=TRAIN, policy=POLICY, metrics=("sharpe",))
    with pytest.raises(KeyError):
        B.dimension_by_name("moon_phase", policy=POLICY)


def test_the_registry_covers_the_metrics_section_12_names():
    """Section 12 names the metric set, and `validation/hypotheses.py` refers
    to breakdown keys by name ("win_rate", "expectancy"). A rename that broke
    either link would be caught here rather than in Phase 8."""
    required = {
        "trade_count", "win_rate", "loss_rate", "expectancy", "mean_r", "median_r",
        "profit_factor", "average_win_r", "average_loss_r", "win_loss_ratio",
        "breakeven_win_rate", "mean_cost_drag_r", "mean_mae_r", "mean_mfe_r",
        "mean_holding_seconds", "mean_bars_held", "max_consecutive_wins",
        "max_consecutive_losses", "total_pnl",
    }
    assert required <= set(M.TRADE_METRICS)
    assert set(M.DEFAULT_CELL_METRICS) <= set(M.TRADE_METRICS)

    # Every registered function reports itself under its registry key, or a
    # mislabelled column appears in every report that uses it.
    book = five_trade_book()
    for name, function in M.TRADE_METRICS.items():
        value = function(book, split_label=TRAIN, policy=POLICY)
        assert value.name in {name, f"{name}_r"}, f"{name} reports {value.name}"

    # Section 12's equity-series metrics exist as functions, not as registry
    # entries, because they take an equity curve rather than a trade list.
    for attribute in ("sharpe_ratio", "sortino_ratio", "max_drawdown",
                      "average_drawdown", "recovery_time", "total_return", "cagr",
                      "trade_frequency", "r_distribution", "streaks",
                      "cost_drag_by_exit_reason"):
        assert callable(getattr(M, attribute))


def test_the_policy_is_built_from_a_config_and_nothing_else():
    """The knobs live in config/schema.py in exactly one place. Building the
    policy from a LOCAL AnalyticsConfig is what lets these tests pin values
    without depending on defaults.yaml."""
    import types

    assert POLICY.min_sample_n == 5
    assert POLICY.trading_days_per_year == 250
    assert POLICY.flow_score_buckets == (70.0, 80.0, 100.0)
    assert POLICY.risk_free_rate_annual == 0.0

    root_like = types.SimpleNamespace(analytics=LOCAL_ANALYTICS)
    assert M.MetricPolicy.from_config(root_like) == POLICY
    assert M.MetricPolicy.from_config(LOCAL_ANALYTICS) == POLICY
    with pytest.raises(TypeError):
        M.MetricPolicy.from_config(object())

    # default() is the schema's own defaults, constructed in process.
    assert M.MetricPolicy.default() == M.MetricPolicy.from_analytics(AnalyticsConfig())
    assert M.MetricPolicy.default().min_sample_n == AnalyticsConfig().min_sample_n


def test_a_target_is_reported_as_a_comparison_with_the_measurement_first():
    """analytics may read research_targets ONLY to report whether a measured
    value met one. The measured value is the primary field, the verdict is
    secondary, and the note says it never selects a parameter."""
    targets = ResearchTargets(
        combined_10y_win_rate=0.72,
        scalp_1r_win_rate_favourable_regimes=(0.88, 0.92),
    )
    summary = M.performance_summary(
        five_trade_book(), split_label=TRAIN, policy=POLICY
    )
    comparisons = M.compare_to_research_targets(summary, targets)

    combined = comparisons[0]
    assert combined.measured.name == "win_rate"
    assert combined.measured.require() == 0.6      # measured, stated first
    assert combined.target == 0.72
    assert combined.met is False                   # 60% did not meet 72%
    assert "never selects a parameter" in combined.note
    assert "measured" in combined.describe()
    # The SCALP_1R-in-favourable-regimes cell was not supplied, so the
    # comparison says there is no measurement rather than reusing the overall
    # rate as a stand-in.
    assert comparisons[1].measured.value is None
    assert comparisons[1].met is None


def test_an_undefined_measurement_meets_no_target_and_fails_none():
    """The live case: with no trades there is nothing to compare, and "no
    data" must not resolve to False (refuted) or True (met)."""
    summary = M.performance_summary([], flat_curve(10), split_label=TRAIN, policy=POLICY)
    for comparison in M.compare_to_research_targets(summary, ResearchTargets()):
        assert comparison.met is None
        assert comparison.measured.value is None
        assert "no measurement" in comparison.describe()

    measured = M.MetricValue.measured("win_rate", 0.9, n=100, unit="fraction")
    assert M.compare_to_target(measured, 0.72).met is True
    assert M.compare_to_target(
        measured, 0.72, comparison=M.Comparison.AT_MOST
    ).met is False
    assert M.compare_to_target(
        measured, 0.88, comparison=M.Comparison.WITHIN_RANGE, target_upper=0.92
    ).met is True
    assert M.compare_to_target(
        M.MetricValue.measured("win_rate", 0.95, n=100), 0.88,
        comparison=M.Comparison.WITHIN_RANGE, target_upper=0.92,
    ).met is False
    with pytest.raises(ValueError):
        M.TargetComparison(
            label="x", measured=measured, comparison=M.Comparison.WITHIN_RANGE,
            target=0.5,
        )
    with pytest.raises(ValueError):
        M.TargetComparison(
            label="x", measured=measured, comparison=M.Comparison.AT_LEAST,
            target=0.5, target_upper=0.9,
        )


def test_a_summary_with_no_equity_series_separates_missing_input_from_zero():
    """Trades but no equity curve: the trade metrics are real measurements and
    the curve-based ones are undefined for want of an input. Reporting 0.0 for
    "max drawdown" when no curve was supplied would be the most dangerous
    confusion in the module -- it reads as "this strategy never lost"."""
    summary = M.performance_summary(
        five_trade_book(), split_label=TRAIN, policy=POLICY
    )
    assert summary.trade_metrics["expectancy"].require() == pytest.approx(0.4)
    assert summary.total_return.undefined is M.Undefined.NO_EQUITY_SERIES
    assert summary.cagr.undefined is M.Undefined.NO_EQUITY_SERIES
    assert summary.sharpe.undefined is M.Undefined.NO_EQUITY_SERIES
    assert summary.sortino.undefined is M.Undefined.NO_EQUITY_SERIES
    assert summary.drawdown.max_drawdown_pct.undefined is M.Undefined.NO_EQUITY_SERIES
    assert summary.trades_per_year.undefined is M.Undefined.NO_EQUITY_SERIES
    assert summary.periods_per_year.undefined is M.Undefined.NO_EQUITY_SERIES
    assert any("for want of inputs" in note for note in summary.notes)


def test_every_function_that_reports_performance_takes_a_split_label():
    """Section 12: "Every function takes a split_label and refuses to mix
    phases silently." Checked on the signatures, so a new metric cannot be
    added without the obligation -- and the three documented exceptions are
    listed explicitly rather than left as an accident."""
    import inspect

    describes_shape_only = {"simple_returns", "drawdown_episodes", "infer_periods_per_year"}
    reporting = [
        *M.TRADE_METRICS.values(),
        M.streaks, M.r_distribution, M.cost_drag_by_exit_reason, M.trade_frequency,
        M.performance_summary, M.compute_trade_metrics, M.sharpe_ratio, M.sortino_ratio,
        M.sharpe_from_returns, M.sortino_from_returns, M.max_drawdown,
        M.average_drawdown, M.recovery_time, M.drawdown_profile, M.total_return, M.cagr,
        B.breakdown, B.breakdown_all,
    ]
    for function in reporting:
        parameters = inspect.signature(function).parameters
        assert "split_label" in parameters, function.__name__
        assert parameters["split_label"].kind is inspect.Parameter.KEYWORD_ONLY, (
            function.__name__
        )

    for name in describes_shape_only:
        parameters = inspect.signature(getattr(M, name)).parameters
        assert "split_label" not in parameters, name

    # breakdown_by_split_phase is the single sanctioned cross-phase view, so
    # it takes NO label: there is no one phase it could be given.
    assert "split_label" not in inspect.signature(B.breakdown_by_split_phase).parameters


def test_the_default_dimensions_cover_the_section_12_list():
    """market, year, month, day-of-week, session, regime, setup, Flow Score
    bucket, R target -- plus the three the rest of the architecture asks for:
    exit_reason (cost drag by exit), ambiguous_fill (section 9 says the
    sensitivity of results to the same-bar assumption must be measurable,
    which means comparing the two cells) and month_of_year."""
    names = [d.name for d in B.default_dimensions(POLICY)]
    assert set(names) >= {
        "market", "year", "month", "day_of_week", "session", "regime", "setup",
        "flow_score_bucket", "r_target",
    }
    assert {"exit_reason", "ambiguous_fill", "month_of_year"} <= set(names)
    assert len(names) == len(set(names))

    mixed_fills = [
        trade("clean", exit_price=exit_for(1.0), minutes=0, ambiguous_fill=False),
        trade("amb", exit_price=exit_for(-1.0), minutes=10, ambiguous_fill=True,
              exit_reason=ExitReason.STOP),
    ]
    table = B.breakdown(mixed_fills, "ambiguous_fill", split_label=TRAIN, policy=POLICY)
    assert [(c.key, c.n) for c in table.cells] == [("clean_fill", 1), ("ambiguous_fill", 1)]
