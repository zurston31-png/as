"""Varying the EXIT, now that entry scoring has been settled.

app/shadow/exit_policy.py held exit levels fixed on purpose: "When entry
scoring has been settled by evidence, varying the exit becomes the next
experiment; running both at once would answer neither." The 97-round-trip
record settled it - a positive gross edge per position that execution
cost more than erased - so the exit is the next variable. These tests pin
that the capability arrived without disturbing anything that came before.
"""
import json

import pytest

from app.shadow.challengers import Challenger, _parse as load_challengers
from app.shadow.exit_policy import ExitPolicy


def _base() -> ExitPolicy:
    return ExitPolicy(
        stop_loss_pct=0.15, take_profit_pct=0.30,
        trailing_enabled=True, trailing_activation_pct=0.10,
        trailing_distance_pct=0.05,
        break_even_enabled=True, break_even_trigger_pct=0.08,
        break_even_buffer_pct=0.01,
        max_hold_hours=4.0,
    )


# ---------------------------------------------------------------------------
# the override must be a true no-op when nothing is named
# ---------------------------------------------------------------------------

def test_overriding_nothing_returns_an_identical_policy():
    """The whole safety of adding this: an entry-scoring challenger, and
    the champion itself, resolve exactly as they did before. If this drifts
    even slightly, every historical shadow row becomes incomparable to
    every new one and nothing in the table would say so."""
    base = _base()
    assert base.with_overrides() == base
    assert base.with_overrides().fingerprint() == base.fingerprint()


def test_only_the_named_levels_change():
    base = _base()
    out = base.with_overrides(take_profit_pct=0.60)

    assert out.take_profit_pct == 0.60
    assert out.stop_loss_pct == base.stop_loss_pct
    assert out.max_hold_hours == base.max_hold_hours
    assert out.trailing_enabled == base.trailing_enabled
    assert out.break_even_trigger_pct == base.break_even_trigger_pct


def test_the_fingerprint_changes_so_rows_are_never_silently_pooled():
    """A dataset spanning two exit policies must say so in the row."""
    base = _base()
    longer = base.with_overrides(max_hold_hours=24.0)
    assert longer.fingerprint() != base.fingerprint()


# ---------------------------------------------------------------------------
# one variable at a time
# ---------------------------------------------------------------------------

def test_a_challenger_that_varies_both_entry_and_exit_is_refused():
    """Refused, not run. A challenger beating the champion on both would
    leave no way to tell which half did it, and a number that cannot be
    interpreted is worse than no number - it still gets quoted."""
    raw = json.dumps([{
        "strategy_id": "both",
        "min_score_to_enter": 75,
        "take_profit_pct": 0.60,
    }])
    assert load_challengers(raw) == []


def test_an_exit_only_challenger_loads():
    raw = json.dumps([{
        "strategy_id": "long-hold",
        "description": "no hair-trigger exit; let the stop and target work",
        "take_profit_pct": 0.60,
        "max_hold_hours": 24.0,
    }])
    loaded = load_challengers(raw)

    assert len(loaded) == 1
    c = loaded[0]
    assert c.varies_exit is True
    assert c.varies_entry is False
    assert c.max_hold_hours == 24.0


def test_an_entry_only_challenger_still_loads_and_varies_no_exit():
    raw = json.dumps([{
        "strategy_id": "pickier",
        "min_score_to_enter": 80,
    }])
    loaded = load_challengers(raw)

    assert len(loaded) == 1
    assert loaded[0].varies_entry is True
    assert loaded[0].varies_exit is False


# ---------------------------------------------------------------------------
# the resolver picks the right policy per row
# ---------------------------------------------------------------------------

def test_only_exit_challengers_get_their_own_policy(monkeypatch):
    """Everything else falls through to the shared base instance, so the
    champion's rows cannot be resolved under a challenger's exit."""
    from app.shadow import resolver

    monkeypatch.setattr(
        "app.shadow.challengers.enabled",
        lambda: [
            Challenger(strategy_id="pickier", min_score_to_enter=80.0),
            Challenger(strategy_id="long-hold", max_hold_hours=24.0),
        ],
    )
    policies = resolver._policies_by_strategy(_base())

    assert set(policies) == {"long-hold"}
    assert policies["long-hold"].max_hold_hours == 24.0
    assert policies["long-hold"].stop_loss_pct == _base().stop_loss_pct


def test_no_exit_challengers_means_an_empty_map(monkeypatch):
    """And an empty map means every row resolves under the base policy -
    the exact behaviour that existed before exits could be varied."""
    from app.shadow import resolver

    monkeypatch.setattr("app.shadow.challengers.enabled", lambda: [])
    assert resolver._policies_by_strategy(_base()) == {}
