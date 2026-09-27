"""The shadow challenger system recorded 1,385 opportunities and never ran.

Every challenger declined every opportunity for the entire life of the
system, and the comparison reported confident effect sizes, p-values and
eight-regime breakdowns about arms that had never taken a trade. Three
separate faults lined up to produce that, and these tests pin all three.
"""
import datetime as dt

import pytest

from app import models
from app.shadow import resolver
from app.shadow.compare import PairedComparison
from app.shadow.exit_policy import ExitPolicy

NOW = dt.datetime(2026, 9, 27, 12, 0, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# 1. the root cause: the series never reached the recorder
# ---------------------------------------------------------------------------

def test_signal_score_carries_the_series_it_was_built_from():
    """app/services/trading_service.py has always read
    `getattr(score, "series", None)`. The field did not exist, so that was
    always None, so app/shadow/recorder.py returned "no candle history to
    score" for every challenger on every opportunity - forever, silently."""
    from app.signals.scoring import SignalScore

    score = SignalScore(score=70.0, direction="long")
    assert hasattr(score, "series"), (
        "SignalScore must expose .series or every challenger is blind"
    )
    assert score.series is None          # defaulted, so old callers still work

    score.series = ["not really candles"]
    assert getattr(score, "series", None) == ["not really candles"]


def test_the_live_gate_populates_it(monkeypatch):
    """The one place that has the series in hand must hand it over.

    Fetching the candles again inside the recorder would be a second
    request AND could land on a different bar, making the paired
    comparison unpaired in the one way it exists to prevent.
    """
    import asyncio

    import app.signals.live_gate as gate
    from app.config import settings
    from app.signals.scoring import SignalScore

    # A real list so the gate's own len() >= SIGNAL_SCORE_MIN_CANDLES check
    # is exercised rather than stubbed around.
    series = list(range(settings.SIGNAL_SCORE_MIN_CANDLES + 1))

    async def fake_fetch(*a, **k):
        return series

    monkeypatch.setattr(gate, "fetch_candles", fake_fetch)
    monkeypatch.setattr(
        gate, "score_signal", lambda s: SignalScore(score=71.0, direction="long")
    )
    monkeypatch.setattr(gate, "classify_full", lambda s, **k: "regime")

    score = asyncio.run(gate.evaluate_live_entry_signal("solana", "mint", "SYM"))

    assert score is not None
    assert score.series is series, "the gate fetched candles and dropped them"


# ---------------------------------------------------------------------------
# 2. an arm that never traded is not a result
# ---------------------------------------------------------------------------

def _comparison(*, champion_trades, challenger_trades, paired=1385):
    c = PairedComparison(challenger_id="loose-60")
    c.paired = paired
    c.both_rejected = paired - champion_trades
    c.champion_returns = [1.0] * champion_trades
    c.challenger_returns = [0.0] * paired
    c.champion_trade_returns = [1.0] * champion_trades
    c.challenger_trade_returns = [1.0] * challenger_trades
    return c


def test_a_challenger_that_never_entered_is_reported_as_not_measured():
    """Declining everything arithmetically produces a huge negative lift.
    Printing that as an effect size dresses a wiring fault as a finding."""
    c = _comparison(champion_trades=479, challenger_trades=0)
    verdict = c.verdict()

    assert verdict.startswith("NOT_MEASURED")
    assert "entered 0 of 1385" in verdict
    assert "has not been compared" in verdict


def test_a_challenger_that_did_trade_is_still_judged_normally():
    """The guard must not swallow real comparisons."""
    c = _comparison(champion_trades=479, challenger_trades=300)
    verdict = c.verdict()
    assert not verdict.startswith("NOT_MEASURED")


# ---------------------------------------------------------------------------
# 3. an impossible return is unmeasurable, not a number
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, exit_price):
        self.exit_price = exit_price
        self.exit_at = NOW
        self.exit_reason = "take-profit"
        self.max_favorable_pct = 1.0
        self.max_adverse_pct = -1.0
        self.bars = 5


def _policy():
    return ExitPolicy(
        stop_loss_pct=0.15, take_profit_pct=0.30, trailing_enabled=False,
        trailing_activation_pct=0.1, trailing_distance_pct=0.05,
        break_even_enabled=False, break_even_trigger_pct=0.08,
        break_even_buffer_pct=0.01, max_hold_hours=4.0,
    )


def _row(entry_price):
    return models.ShadowPosition(
        strategy_id="champion", token_address="mint", symbol="SYM",
        entry_price=entry_price, opened_at=NOW - dt.timedelta(minutes=10),
        size_usd=100.0, fees_pct=0.0025, slippage_pct=0.0015,
    )


def test_an_implausible_return_is_recorded_as_unmeasurable():
    """One row at 489,000x set the champion's mean per-opportunity return
    to 10,205,008%. A NULL return is excluded from every consumer; a huge
    number is not, and it silently became the whole comparison."""
    row = _row(entry_price=0.000001)
    resolver._close_out(row, result=_Result(exit_price=1.0), policy=_policy(), now=NOW)

    assert row.return_pct is None, "an impossible return must never be a number"
    assert row.closed_at == NOW
    assert "implausible return" in row.exit_reason
    assert "unmeasurable" in row.exit_reason


def test_a_zero_entry_price_is_unmeasurable_rather_than_infinite():
    row = _row(entry_price=0.0)
    resolver._close_out(row, result=_Result(exit_price=1.0), policy=_policy(), now=NOW)

    assert row.return_pct is None
    assert "not usable" in row.exit_reason


def test_a_large_but_real_gain_is_still_recorded():
    """The bound must exclude scale faults without discarding the genuine
    outliers a memecoin book exists to catch. 5x is a trade, not a fault."""
    row = _row(entry_price=1.0)
    resolver._close_out(row, result=_Result(exit_price=5.0), policy=_policy(), now=NOW)

    assert row.return_pct is not None
    assert row.gross_return_pct == pytest.approx(400.0)


def test_a_normal_loss_is_unaffected():
    row = _row(entry_price=1.0)
    resolver._close_out(row, result=_Result(exit_price=0.9), policy=_policy(), now=NOW)

    assert row.return_pct is not None
    assert row.gross_return_pct == pytest.approx(-10.0)
