"""Regression tests for defects found in code review of this branch.

Each one is a crash or a silent stall that the existing suite did not
reach, so each gets a test that reproduces the exact condition rather than
a nearby one.
"""
import datetime as dt

import pytest

from app.config import settings


def _series(n: int):
    from app.data.candles import Candle, CandleSeries

    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    return CandleSeries(symbol="T", timeframe="5m", candles=[
        Candle(timestamp=base + dt.timedelta(minutes=5 * i), open=1.0, high=1.1,
               low=0.9, close=1.0 + i * 0.01, volume=100.0)
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# price_acceleration read one bar further back than it checked for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bars", [0, 1, 59, 60, 61, 120])
def test_price_acceleration_never_raises_on_any_length(bars):
    """The guard accepted exactly 60 bars and the longest lookback then read
    closes[-61], which is off the end of a 60-element list.

    60 is not an unusual length - it is the boundary the guard was written
    around, so a token sitting on it hit this every pass. `extract` calls
    this outside any handler, so the IndexError took out the whole feature
    set for that token rather than one feature.
    """
    from app.early.features import price_acceleration

    out = price_acceleration(_series(bars))
    assert len(out) == 4


def test_the_boundary_length_now_produces_features_or_says_why():
    """Either answer is fine; crashing is not. At 61 bars the longest
    lookback is finally satisfiable, so the features become available."""
    from app.early.features import price_acceleration

    at_boundary = {f.name: f for f in price_acceleration(_series(60))}
    assert not at_boundary["return_long"].available
    assert "61" in at_boundary["return_long"].detail

    above = {f.name: f for f in price_acceleration(_series(61))}
    assert above["return_long"].available


def test_extract_survives_the_boundary_length():
    """The path that actually made this fatal - no handler between here and
    the caller."""
    from app.early.features import extract

    features = extract(series=_series(60), market=None, observations=[])
    assert features, "feature extraction returned nothing at the boundary length"


# ---------------------------------------------------------------------------
# scored_event was unbound when live scoring was off
# ---------------------------------------------------------------------------

def test_scored_event_is_bound_before_any_branch_can_read_it():
    """`scored_event` was assigned only inside `if
    LIVE_SIGNAL_SCORE_ENABLED:`, but read later by the regime back-fill and
    by the early engine - and the early engine's guard is
    EARLY_SIGNAL_ENABLED, which is independent.

    With scoring off and the early engine on, that read raised
    UnboundLocalError inside a `try` that logs and continues: the early
    engine stopped working for every candidate and nothing said why. Read
    from source because reproducing it needs the whole market-data stack,
    and the initialisation is the thing that must not regress.
    """
    import inspect

    from app.services import trading_service

    source = inspect.getsource(trading_service._evaluate_and_enter)
    init = source.index("scored_event = None")
    guard = source.index("if settings.LIVE_SIGNAL_SCORE_ENABLED:")
    assert init < guard, (
        "scored_event must be initialised BEFORE the scoring branch - it is read "
        "by code paths that run whether or not scoring is enabled"
    )


def test_the_early_engine_guard_does_not_imply_scoring_ran():
    """Pins why the initialisation is needed: the two settings are
    independent, so any combination has to be safe."""
    from app.config import Settings

    assert "LIVE_SIGNAL_SCORE_ENABLED" in Settings.model_fields
    assert "EARLY_SIGNAL_ENABLED" in Settings.model_fields


# ---------------------------------------------------------------------------
# the Pine payload emitted invalid JSON for sub-1 prices
# ---------------------------------------------------------------------------

def test_the_pine_number_format_keeps_the_leading_zero():
    """Pine's "#" is an OPTIONAL digit: str.tostring(0.00042, "#.####...")
    yields ".00042", and a bare leading "." is not a valid JSON number - so
    the webhook rejected the whole payload. Memecoin prices are nearly
    always below 1, which made this the normal case."""
    import pathlib

    pine = pathlib.Path("pine/memecoin_signal_strategy.pine").read_text()
    assert 'PRICE_FMT = "0.' in pine
    assert 'IND_FMT   = "0.' in pine
    assert 'PRICE_FMT = "#.' not in pine
    assert 'IND_FMT   = "#.' not in pine


def test_a_leading_dot_really_is_rejected_as_json():
    """The premise, so the test above is not resting on an assumption about
    a language nobody here can run."""
    import json

    json.loads('{"price": 0.00042}')
    with pytest.raises(json.JSONDecodeError):
        json.loads('{"price": .00042}')


def test_the_webhook_rejects_a_payload_with_a_bare_leading_dot():
    """End to end: this is what the bot actually did with the old format -
    a 422 with no obvious cause, for every alert on a sub-1 token."""
    from app import webhook_debug

    payload, error = webhook_debug.parse_body('{"price": .00042}')
    assert payload is None or error, "a bare leading dot must not parse cleanly"
