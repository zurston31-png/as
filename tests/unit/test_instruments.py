"""Instrument economics -- the numbers every later phase multiplies by."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flow_model.core.enums import InstrumentType
from flow_model.core.instruments import InstrumentSpec, SessionWindow


def test_point_value_is_derived_not_declared(nq):
    # Declaring point_value separately from tick_size/tick_value invites the
    # two to disagree; it is derived instead.
    assert nq.point_value == pytest.approx(20.0)
    assert "point_value" not in InstrumentSpec.input_keys()


@pytest.mark.parametrize(
    "tick_size,tick_value,expected",
    [(0.25, 5.0, 20.0), (0.25, 12.5, 50.0), (0.10, 10.0, 100.0), (0.01, 0.01, 1.0)],
)
def test_real_contract_point_values(tick_size, tick_value, expected):
    spec = InstrumentSpec(
        symbol="X", instrument_type=InstrumentType.FUTURE,
        tick_size=tick_size, tick_value=tick_value,
    )
    assert spec.point_value == pytest.approx(expected)


def test_risk_per_unit(nq):
    assert nq.risk_per_unit(20.0) == pytest.approx(400.0)
    assert nq.risk_per_unit(0.25) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        nq.risk_per_unit(0.0)
    with pytest.raises(ValueError):
        nq.risk_per_unit(-1.0)


def test_round_to_tick(nq):
    assert nq.round_to_tick(18000.13) == pytest.approx(18000.25)
    assert nq.round_to_tick(18000.12) == pytest.approx(18000.0)
    assert nq.round_to_tick(18000.0) == pytest.approx(18000.0)
    # idempotent
    once = nq.round_to_tick(17999.37)
    assert nq.round_to_tick(once) == pytest.approx(once)


def test_tick_price_conversions_roundtrip(nq):
    assert nq.ticks_to_price(8) == pytest.approx(2.0)
    assert nq.price_to_ticks(2.0) == pytest.approx(8.0)
    assert nq.price_to_ticks(nq.ticks_to_price(13)) == pytest.approx(13)


def test_commission_per_order_vs_per_unit(nq, qqq):
    # Futures: per order, size-independent.
    assert nq.commission_for(1, sides=2) == pytest.approx(2 * (2.25 + 0.37))
    assert nq.commission_for(10, sides=2) == pytest.approx(2 * (2.25 + 0.37))
    # ETF: per share, scales with size.
    assert qqq.commission_for(100, sides=2) == pytest.approx(2 * 0.005 * 100)
    assert qqq.commission_for(1, sides=1) == pytest.approx(0.005)
    with pytest.raises(ValueError):
        nq.commission_for(1, sides=0)


def test_session_window_half_open():
    rth = SessionWindow(name="RTH", start="09:30", end="16:00")
    assert rth.contains_minutes(9 * 60 + 30)       # inclusive start
    assert not rth.contains_minutes(16 * 60)        # exclusive end
    assert rth.contains_minutes(16 * 60 - 1)
    assert not rth.contains_minutes(9 * 60 + 29)
    assert not rth.wraps_midnight


def test_session_window_wrapping_midnight():
    eth = SessionWindow(name="ETH", start="18:00", end="17:00")
    assert eth.wraps_midnight
    assert eth.contains_minutes(18 * 60)
    assert eth.contains_minutes(2 * 60)      # after midnight
    assert eth.contains_minutes(16 * 60 + 59)
    assert not eth.contains_minutes(17 * 60)     # the maintenance break
    assert not eth.contains_minutes(17 * 60 + 30)


def test_session_window_rejects_bad_time_format():
    with pytest.raises(ValidationError):
        SessionWindow(name="bad", start="9:30", end="16:00")
    with pytest.raises(ValidationError):
        SessionWindow(name="bad", start="25:00", end="16:00")


def test_tradable_index_requires_execution_proxy():
    """An index cannot be traded directly; modelling fills on it is fiction."""
    with pytest.raises(ValidationError, match="execution_proxy"):
        InstrumentSpec(
            symbol="SPX", instrument_type=InstrumentType.INDEX,
            tick_size=0.01, tick_value=0.01, tradable=True,
        )
    ok = InstrumentSpec(
        symbol="SPX", instrument_type=InstrumentType.INDEX,
        tick_size=0.01, tick_value=0.01, tradable=True, execution_proxy="ES",
    )
    assert ok.execution_proxy == "ES"


def test_reference_only_index_is_allowed():
    spec = InstrumentSpec(
        symbol="SPX", instrument_type=InstrumentType.INDEX,
        tick_size=0.01, tick_value=0.01, tradable=False,
    )
    assert not spec.tradable


def test_spread_and_size_bounds_validated():
    with pytest.raises(ValidationError, match="min_spread_ticks"):
        InstrumentSpec(
            symbol="X", instrument_type=InstrumentType.FUTURE,
            tick_size=0.25, tick_value=5.0,
            typical_spread_ticks=1.0, min_spread_ticks=2.0,
        )
    with pytest.raises(ValidationError, match="min_size"):
        InstrumentSpec(
            symbol="X", instrument_type=InstrumentType.FUTURE,
            tick_size=0.25, tick_value=5.0, min_size=10, max_size=5,
        )


def test_spec_is_immutable(nq):
    with pytest.raises(ValidationError):
        nq.tick_size = 0.5
