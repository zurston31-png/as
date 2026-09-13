"""Structural guards for the two properties that fail silently when broken.

These are not behaviour tests - tests/test_token_identity.py and
tests/test_rugcheck_filters.py already cover the behaviour. These read the
SOURCE, and they exist because both properties are ones a future change
can quietly undo while every behaviour test still passes.

  Identity. `instrument_key` is only load-bearing if every dedup, exposure
  and cooldown lookup goes through it. Add one `filter_by(symbol=...)` to
  the risk path and the per-token exposure cap starts pooling two
  unrelated assets - no exception, no failing assertion, just a limit that
  permits twice what it says. See app/identity.py.

  Fail-closed screening. Every no-data path in the rug engine must reject.
  A single `passed=True` on a path meaning "we could not check" turns the
  security gate into a formality for exactly the tokens least likely to be
  indexed, which are the new ones the scanner spends all its time on.

An audit that lives in a commit message protects nothing. These fail.
"""
import inspect

import pytest

from app import identity
from app.config import settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

# Functions that decide "is this the same token?" for a purpose where being
# wrong costs money: position dedup, exposure caps, cooldowns.
IDENTITY_CRITICAL = (
    ("app.services.trading_service", "_open_position_for"),
    ("app.services.portfolio", "get_token_exposure_usd"),
    ("app.risk.manager", "RiskManager._cooldown_remaining_seconds"),
)


def _source(module_name: str, qualname: str) -> str:
    import importlib

    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return inspect.getsource(obj)


@pytest.mark.parametrize("module_name,qualname", IDENTITY_CRITICAL)
def test_identity_critical_lookups_go_through_instrument_key(module_name, qualname):
    """Each of these answers "is this the same token?" for a decision that
    costs money when it is wrong. Matching on the symbol instead would
    make an unrelated mint called PEPE block, or share a cap with, the one
    actually held."""
    source = _source(module_name, qualname)
    assert "instrument_key" in source, (
        f"{module_name}.{qualname} no longer resolves identity through "
        "app.identity.instrument_key - see that module for why symbol matching "
        "is silently wrong rather than visibly wrong"
    )


@pytest.mark.parametrize("module_name,qualname", IDENTITY_CRITICAL)
def test_identity_critical_lookups_do_not_filter_on_symbol(module_name, qualname):
    """The complementary half: calling instrument_key and ALSO narrowing
    the query by symbol would re-introduce the collision through the back
    door, and the test above would not notice."""
    source = _source(module_name, qualname)
    assert "filter_by(symbol=" not in source
    assert ".symbol ==" not in source


def test_a_missing_mint_is_reported_as_weak_rather_than_accepted_quietly():
    """The symbol fallback exists so the bot degrades instead of crashing.
    It must stay visible, or a whole class of collision becomes
    undetectable."""
    assert identity.is_weak(None)
    assert not identity.is_weak("So11111111111111111111111111111111111111112")


def test_the_cex_namespace_is_a_different_namespace_not_a_fallback(monkeypatch):
    """On a CEX the exchange ticker genuinely is canonical. That is a
    deliberate branch, and conflating it with the missing-mint fallback
    would make a real identity look like a degraded one."""
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", "cex")
    assert identity.instrument_key("BONK", None) == "BONK"
    assert not identity.is_weak(None), "a CEX ticker is not a weak identity"


# ---------------------------------------------------------------------------
# fail-closed screening
# ---------------------------------------------------------------------------

async def test_no_token_address_is_refused():
    """The scanner and the webhook both reach this. A signal with no mint
    cannot be screened at all, and cannot be traded either."""
    from app.rugcheck.filters import run_rug_checks

    report = await run_rug_checks("solana", None)
    assert not report.passed
    assert "no on-chain token address" in " ".join(report.reasons)


async def test_a_token_no_scanner_can_find_is_refused(monkeypatch):
    """The important one. A brand-new mint is exactly what this bot hunts
    and exactly what security scanners have not indexed yet, so "not
    found" is the single most common no-data outcome - and treating it as
    clean would disable screening precisely where it matters most."""
    from app.rugcheck import filters

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(filters, "fetch_goplus_solana", nothing, raising=False)
    monkeypatch.setattr(filters, "fetch_rugcheck", nothing, raising=False)

    report = await filters.run_rug_checks(
        "solana", "So11111111111111111111111111111111111111112"
    )
    assert not report.passed
    assert report.lookup_outcomes, "a refusal must record what was tried"


async def test_disabling_screening_is_loud_and_recorded(monkeypatch, caplog):
    """RUGCHECK_ENABLED=false is the one legitimate pass-open path. It has
    to be impossible to have on by accident: it warns, and it stamps the
    report so a trade taken without screening is identifiable afterwards
    rather than looking like a trade that passed."""
    import logging

    from app.rugcheck.filters import run_rug_checks

    monkeypatch.setattr(settings, "RUGCHECK_ENABLED", False)
    with caplog.at_level(logging.WARNING):
        report = await run_rug_checks("solana", "So11111111111111111111111111111111111111112")

    assert report.passed
    assert report.source == "disabled"
    assert "NOT being screened" in caplog.text
    assert report.lookup_outcomes == ["screening disabled via RUGCHECK_ENABLED=false"]


def test_screening_is_enabled_by_default():
    """The shipped default has to be the safe one - a deployer who never
    touches this setting must get screening."""
    from app.config import Settings

    assert Settings.model_fields["RUGCHECK_ENABLED"].default is True


# ---------------------------------------------------------------------------
# the hard ceilings cannot be raised from .env
# ---------------------------------------------------------------------------

def test_no_env_value_can_exceed_a_hard_ceiling(monkeypatch):
    """tests/test_risk_manager.py checks each clamp individually. This
    checks the property they collectively exist for: there is no
    combination of .env values that produces a manager past its
    ceilings."""
    from app.risk import manager as risk

    for name, value in (
        ("MAX_PORTFOLIO_PCT_PER_TRADE", 0.95),
        ("DAILY_LOSS_LIMIT_PCT", 0.99),
        ("MAX_CONCURRENT_POSITIONS", 10_000),
        ("MAX_EXPOSURE_PER_TOKEN_PCT", 5.0),
        ("MAX_TOTAL_EXPOSURE_PCT", 9.0),
        ("MAX_DAILY_TRADES", 100_000),
        ("TRADE_COOLDOWN_SECONDS", 10**9),
    ):
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "STOP_LOSS_PCT", 0.0001)

    rm = risk.RiskManager()
    assert rm.max_pct_per_trade <= risk.HARD_MAX_PORTFOLIO_PCT_PER_TRADE
    assert rm.daily_loss_limit_pct <= risk.HARD_MAX_DAILY_LOSS_PCT
    assert rm.stop_loss_pct >= risk.HARD_MIN_STOP_LOSS_PCT
    assert rm.max_concurrent_positions <= risk.HARD_MAX_CONCURRENT_POSITIONS
    assert rm.max_exposure_per_token_pct <= risk.HARD_MAX_EXPOSURE_PER_TOKEN_PCT
    assert rm.max_total_exposure_pct <= risk.HARD_MAX_TOTAL_EXPOSURE_PCT
    assert rm.max_daily_trades <= risk.HARD_MAX_DAILY_TRADES
    assert rm.cooldown_seconds <= risk.HARD_MAX_COOLDOWN_SECONDS


def test_live_trading_stays_off_by_default():
    """CLAUDE.md's first non-negotiable. Worth a test rather than a
    convention: the default is what a fresh deployment gets."""
    from app.config import Settings

    assert Settings.model_fields["LIVE_TRADING"].default is False
    assert settings.LIVE_TRADING is False


# ---------------------------------------------------------------------------
# the shared paper-only guard
# ---------------------------------------------------------------------------

def test_the_paper_guard_reads_both_flags(monkeypatch):
    """Either flag alone means the configuration is not paper-only.
    LIVE_TRADING=false with the acknowledgement left on is one restart away
    from real orders, and reading only LIVE_TRADING is exactly how the
    operator scripts came to print "PAPER mode" for a live-configured
    deployment."""
    from app.safety import paper_only

    monkeypatch.setattr(settings, "LIVE_TRADING", False)
    monkeypatch.setattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", False)
    assert paper_only.is_paper_only()
    assert paper_only.violation_reason() is None
    paper_only.require_paper_only()          # must not raise

    for flag in ("LIVE_TRADING", "LIVE_EXECUTION_ACKNOWLEDGED"):
        monkeypatch.setattr(settings, "LIVE_TRADING", False)
        monkeypatch.setattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", False)
        monkeypatch.setattr(settings, flag, True)

        assert not paper_only.is_paper_only(), f"{flag} alone must break paper-only"
        assert flag in paper_only.violation_reason()
        with pytest.raises(paper_only.LiveExecutionRefused):
            paper_only.require_paper_only()


def test_the_guard_names_both_flags_when_both_are_on(monkeypatch):
    """The message has to say what to change, not just that something is
    wrong - an operator fixing one flag and hitting the same refusal
    learns nothing."""
    from app.safety import paper_only

    monkeypatch.setattr(settings, "LIVE_TRADING", True)
    monkeypatch.setattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", True)
    reason = paper_only.violation_reason()
    assert "LIVE_TRADING" in reason
    assert "LIVE_EXECUTION_ACKNOWLEDGED" in reason


def test_the_operator_entry_points_check_before_acting():
    """The launcher and the two operator scripts are stdlib-only (they run
    before the venv exists), so they cannot import the guard and read .env
    directly instead. This pins that they still check, and that they check
    BOTH flags rather than LIVE_TRADING alone."""
    import pathlib

    for path, fn in (
        ("scripts/setup_and_run.py", "refuse_if_live_configured"),
        ("scripts/send_test_signal.py", "refuse_if_live_configured"),
    ):
        source = pathlib.Path(path).read_text()
        assert f"def {fn}" in source, f"{path} has no live-config guard"
        assert f"{fn}()" in source.replace(f"def {fn}()", ""), f"{path} defines the guard but never calls it"
        assert "LIVE_EXECUTION_ACKNOWLEDGED" in source, f"{path} only checks LIVE_TRADING"

    scan = pathlib.Path("scripts/scan_once.py").read_text()
    assert "require_paper_only()" in scan


def test_no_document_tells_an_operator_to_enable_live_trading():
    """CLAUDE.md's first non-negotiable. Describing the interlocks is fine;
    instructing someone to switch them on is not."""
    import pathlib
    import re

    instructing = re.compile(
        r"(before setting|then set|and set)\s+`?LIVE_TRADING=true", re.I
    )
    for path in ("README.md", "deploy/vps_setup.md", "COLLECTION.md",
                 "GETTING_STARTED_WINDOWS.md"):
        doc = pathlib.Path(path)
        if not doc.exists():
            continue
        assert not instructing.search(doc.read_text()), (
            f"{path} instructs the operator to enable live trading"
        )
