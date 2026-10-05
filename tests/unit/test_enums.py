"""Enum semantics."""

from __future__ import annotations

import json

import pytest

from flow_model.core.enums import (
    Component,
    DataQuality,
    Regime,
    SetupType,
    Side,
    SignalAction,
    SplitPhase,
    WaitReason,
)


def test_side_sign_and_opposite():
    assert Side.LONG.sign == 1
    assert Side.SHORT.sign == -1
    assert Side.LONG.opposite() is Side.SHORT
    assert Side.SHORT.opposite() is Side.LONG
    assert Side.LONG.opposite().opposite() is Side.LONG


def test_enums_serialize_as_plain_strings():
    assert json.loads(json.dumps({"r": Regime.CHOP.value}))["r"] == "CHOP"
    assert str(SetupType.SCALP_1R) == "SCALP_1R"
    assert Side("LONG") is Side.LONG


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ((DataQuality.GOOD, DataQuality.GOOD), DataQuality.GOOD),
        ((DataQuality.GOOD, DataQuality.DEGRADED), DataQuality.DEGRADED),
        ((DataQuality.DEGRADED, DataQuality.STALE), DataQuality.STALE),
        ((DataQuality.GOOD, DataQuality.MISSING), DataQuality.MISSING),
    ],
)
def test_data_quality_worst_is_the_minimum(statuses, expected):
    assert DataQuality.worst(*statuses) is expected


def test_data_quality_worst_of_nothing_is_missing():
    # No data at all must never grade better than MISSING.
    assert DataQuality.worst() is DataQuality.MISSING


def test_data_quality_ordering_and_tradability():
    ranks = [q.rank for q in (DataQuality.MISSING, DataQuality.STALE, DataQuality.DEGRADED, DataQuality.GOOD)]
    assert ranks == sorted(ranks)
    assert DataQuality.GOOD.is_tradable(DataQuality.DEGRADED)
    assert DataQuality.DEGRADED.is_tradable(DataQuality.DEGRADED)
    assert not DataQuality.STALE.is_tradable(DataQuality.DEGRADED)
    assert not DataQuality.MISSING.is_tradable(DataQuality.DEGRADED)


def test_regime_unknown_exists_for_warmup():
    # Phase-1 contract: UNKNOWN means "not enough data", and is never tradable.
    assert Regime.UNKNOWN in set(Regime)


def test_five_components_exactly():
    assert len(list(Component)) == 5
    assert {c.value for c in Component} == {
        "options_flow", "order_flow", "structure", "liquidity", "vol_momentum",
    }


def test_signal_action_has_wait():
    assert {a.value for a in SignalAction} == {"LONG", "SHORT", "WAIT"}


def test_every_gate_has_a_wait_reason():
    """Each gate in the documented pipeline must be able to name itself."""
    required = {
        "data_quality", "warmup", "regime_blocked", "liquidity", "no_structure",
        "score_below_threshold", "orderflow_conflict", "options_contradiction",
        "volatility_band", "rr_too_low", "risk_limit",
    }
    assert required <= {r.value for r in WaitReason}


def test_split_phases_include_sealed():
    assert SplitPhase.SEALED_OOS.value == "SEALED_OOS"
    assert len(list(SplitPhase)) == 4
