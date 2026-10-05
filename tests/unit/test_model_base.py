"""Round-trip guarantees for the shared frozen base.

A `TradeRecord` has to survive: construct -> dump -> SQLite -> read ->
construct. These tests exist because the first implementation silently broke
on nested computed fields, which would have corrupted the trade journal.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from flow_model.core.contracts import ComponentScore, CostBreakdown, FlowScore, TradeRecord
from flow_model.core.enums import Component


def test_full_dump_includes_derived_columns(make_trade):
    dumped = make_trade().model_dump()
    assert "r_multiple" in dumped
    assert "cost_drag_r" in dumped
    assert "total" in dumped["costs"]           # nested computed field


def test_init_dict_excludes_derived_columns_recursively(make_trade):
    init = make_trade().to_init_dict()
    assert "r_multiple" not in init
    assert "total" not in init["costs"]


def test_round_trip_through_init_dict(make_trade):
    trade = make_trade()
    assert TradeRecord(**trade.to_init_dict()) == trade


def test_from_mapping_tolerates_derived_and_foreign_columns(make_trade):
    """Simulates reading a row back out of SQLite."""
    trade = make_trade()
    row = trade.model_dump() | {"rowid": 17, "inserted_at": "2020-01-01"}
    assert TradeRecord.from_mapping(row) == trade


def test_json_round_trip(make_trade):
    trade = make_trade()
    assert TradeRecord.from_mapping(json.loads(trade.model_dump_json())) == trade


def test_nested_dict_of_models_round_trips():
    """FlowScore holds dict[Component, ComponentScore], each with a computed
    `points`; a flat exclusion would miss them."""
    from datetime import datetime, timezone

    comps = {
        Component.ORDER_FLOW: ComponentScore(
            component=Component.ORDER_FLOW, magnitude=0.8, direction=1, weight=25.0
        ),
        Component.STRUCTURE: ComponentScore(
            component=Component.STRUCTURE, magnitude=0.6, direction=-1, weight=20.0
        ),
    }
    score = FlowScore(symbol="NQ", ts=datetime(2020, 1, 1, tzinfo=timezone.utc), components=comps)
    assert FlowScore(**score.to_init_dict()) == score
    assert FlowScore.from_mapping(score.model_dump()) == score
    assert FlowScore.from_mapping(json.loads(score.model_dump_json())) == score


def test_direct_construction_still_forbids_unknown_fields(make_trade):
    with pytest.raises(ValidationError, match="flowscore"):
        make_trade().replace(flowscore=10.0)


def test_caller_exclude_is_unioned_not_substituted(make_trade):
    """A caller-supplied exclude must not drop the computed-field exclusion --
    that caused an infinite recursion in the first implementation."""
    out = make_trade().to_init_dict(exclude={"notes"} & set())
    assert "r_multiple" not in out
    out2 = make_trade().to_init_dict(exclude={"symbol"})
    assert "symbol" not in out2 and "r_multiple" not in out2


def test_replace_validates_unlike_model_copy(make_trade):
    trade = make_trade()
    # model_copy would happily produce an invalid record; replace must not.
    with pytest.raises(ValidationError):
        trade.replace(pnl=123456.0)


def test_models_are_immutable(make_trade):
    with pytest.raises(ValidationError):
        make_trade().pnl = 1.0


def test_computed_and_input_key_sets_are_disjoint():
    assert not (TradeRecord.computed_keys() & TradeRecord.input_keys())
    assert not (CostBreakdown.computed_keys() & CostBreakdown.input_keys())
