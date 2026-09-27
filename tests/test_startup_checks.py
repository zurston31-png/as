"""Tests for app/startup_checks.py and, more importantly, a regression
guard on the shipped defaults themselves.

The bug these exist for: every individual setting was reasonable, but the
COMBINATION made trading impossible, and that failure was completely
silent - the bot started fine, discovered tokens, and never traded, which
is indistinguishable from "no good setups this week". Two defaults were
wrong at once:

  * MIN_SIGNAL_SCORE_TO_ENTER=75 sat at roughly the 97th percentile of what
    the scoring engine actually produces, so ~3% of setups could clear it
    even before the rug check took its cut.
  * SIGNAL_SCORE_MIN_CANDLES=60 x 15m needs 15h of history while
    SCANNER_MIN_TOKEN_AGE_HOURS=6 admitted younger tokens, which then
    always failed the score gate.

test_shipped_defaults_can_actually_trade is the important one: it runs the
REAL scoring engine over a realistic spread of market regimes and asserts
the default threshold admits a sane fraction of them. If someone tightens a
default back into "never trades" territory, that test fails instead of the
bot going quiet in production.
"""
import pytest

from app.config import settings
from app.data.candles import Timeframe
from app.data.providers import SyntheticCandleProvider
from app.signals.scoring import score_signal
from app.startup_checks import check_config_coherence


# ---------------------------------------------------------------------------
# the shipped defaults must be capable of trading
# ---------------------------------------------------------------------------

def test_shipped_defaults_have_no_coherence_warnings():
    assert check_config_coherence() == []


def test_score_gate_and_candle_requirement_do_not_contradict():
    """The scanner must not admit tokens younger than the history the score
    gate needs, or those tokens are discovered and then rejected forever."""
    timeframe = Timeframe(settings.SIGNAL_SCORE_TIMEFRAME)
    required_hours = settings.SIGNAL_SCORE_MIN_CANDLES * timeframe.seconds / 3600
    assert settings.SCANNER_MIN_TOKEN_AGE_HOURS >= required_hours, (
        f"scanner admits {settings.SCANNER_MIN_TOKEN_AGE_HOURS}h-old tokens but the score "
        f"gate needs {required_hours}h of history - those tokens can never trade"
    )


def test_shipped_defaults_can_actually_trade():
    """Run the real scoring engine across a realistic spread of regimes and
    confirm the default threshold admits a workable fraction.

    Bounded on BOTH sides on purpose. Too low and the bot trades chop (the
    thing the original spec explicitly did not want); too high and it never
    trades at all, which is what actually shipped."""
    pool = ["bull"] * 3 + ["pump"] * 2 + ["sideways"] * 3 + ["bear"] + ["high_volatility"]
    scores = []
    for seed in range(1, 11):
        for regime in pool:
            series = SyntheticCandleProvider(regime=regime, seed=seed).fetch("T", Timeframe.M15, limit=300)
            result = score_signal(series)
            scores.append((regime, result.score, result.direction))

    qualifying = [s for s in scores if s[1] >= settings.MIN_SIGNAL_SCORE_TO_ENTER and s[2] == "long"]
    rate = len(qualifying) / len(scores)

    assert rate > 0.05, (
        f"only {rate:.1%} of setups clear MIN_SIGNAL_SCORE_TO_ENTER="
        f"{settings.MIN_SIGNAL_SCORE_TO_ENTER} - the bot would effectively never trade"
    )
    assert rate < 0.75, (
        f"{rate:.1%} of setups clear the threshold - that is not 'rejecting weak setups'"
    )


def test_qualifying_setups_are_mostly_trending_regimes():
    """The gate should be selective in the right direction: most of what it
    admits ought to be a genuine trend, not chop."""
    pool = ["bull"] * 3 + ["pump"] * 2 + ["sideways"] * 3 + ["bear"] + ["high_volatility"]
    qualifying = []
    for seed in range(1, 11):
        for regime in pool:
            series = SyntheticCandleProvider(regime=regime, seed=seed).fetch("T", Timeframe.M15, limit=300)
            result = score_signal(series)
            if result.score >= settings.MIN_SIGNAL_SCORE_TO_ENTER and result.direction == "long":
                qualifying.append(regime)

    assert qualifying, "nothing qualified at all"
    trending = sum(1 for r in qualifying if r in ("bull", "pump"))
    precision = trending / len(qualifying)
    assert precision > 0.70, f"only {precision:.1%} of admitted setups were trending regimes"


# ---------------------------------------------------------------------------
# the checks themselves
# ---------------------------------------------------------------------------

def test_impossible_age_vs_candles_combination_is_flagged(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_ENABLED", True)
    monkeypatch.setattr(settings, "SIGNAL_SCORE_MIN_CANDLES", 60)
    monkeypatch.setattr(settings, "SIGNAL_SCORE_TIMEFRAME", "15m")
    monkeypatch.setattr(settings, "SCANNER_MIN_TOKEN_AGE_HOURS", 6.0)

    warnings = check_config_coherence()
    assert any("IMPOSSIBLE COMBINATION" in w for w in warnings)


def test_an_unreachable_score_threshold_is_flagged(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_SIGNAL_SCORE_ENABLED", True)
    monkeypatch.setattr(settings, "MIN_SIGNAL_SCORE_TO_ENTER", 85.0)

    warnings = check_config_coherence()
    assert any("percentile" in w for w in warnings)


def test_disabled_rug_check_is_flagged(monkeypatch):
    monkeypatch.setattr(settings, "RUGCHECK_ENABLED", False)
    assert any("RUGCHECK_ENABLED=false" in w for w in check_config_coherence())


def test_disabled_scanner_is_flagged(monkeypatch):
    monkeypatch.setattr(settings, "SCANNER_ENABLED", False)
    assert any("SCANNER_ENABLED=false" in w for w in check_config_coherence())


def test_an_invalid_timeframe_is_flagged(monkeypatch):
    monkeypatch.setattr(settings, "SIGNAL_SCORE_TIMEFRAME", "not-a-timeframe")
    assert any("not a valid timeframe" in w for w in check_config_coherence())


@pytest.mark.parametrize(
    "field,value",
    [
        ("MAX_CONCURRENT_POSITIONS", 0),
        ("MAX_DAILY_TRADES", 0),
        ("MAX_TRADE_SIZE_USD", 0.0),
        ("MAX_TOTAL_EXPOSURE_PCT", 0.0),
    ],
)
def test_risk_limits_that_block_everything_are_flagged(monkeypatch, field, value):
    monkeypatch.setattr(settings, field, value)
    assert any(field in w for w in check_config_coherence())


def test_checks_never_mutate_settings(monkeypatch):
    """These only ever warn - a deployer may have a good reason for an odd
    combination and this module has no business overruling them."""
    monkeypatch.setattr(settings, "MIN_SIGNAL_SCORE_TO_ENTER", 99.0)
    check_config_coherence()
    assert settings.MIN_SIGNAL_SCORE_TO_ENTER == 99.0


# ---------------------------------------------------------------------------
# .env.example must stay in step with Settings
# ---------------------------------------------------------------------------

def test_every_setting_is_documented_in_env_example():
    """A setting nobody can discover is a setting nobody will configure.

    This drifts silently: adding a field to Settings works fine without
    touching .env.example, and the gap only shows up when an operator asks
    why the bot behaves in a way they cannot find a knob for. Six settings
    had already drifted when this test was written, including the paper
    fill-model knobs that change simulated P&L directly.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    declared = set(re.findall(
        r"^\s{4}([A-Z][A-Z0-9_]+)\s*:", (root / "app" / "config.py").read_text(), re.M
    ))
    documented = set(re.findall(
        r"^([A-Z][A-Z0-9_]+)=", (root / ".env.example").read_text(), re.M
    ))

    assert not (declared - documented), (
        f"settings missing from .env.example: {sorted(declared - documented)}"
    )
    assert not (documented - declared), (
        f".env.example documents keys that are not settings: {sorted(documented - declared)}"
    )
