"""Tests for run_rug_checks orchestration: source ordering, fallback, and
the per-source reporting that makes a rejection actionable.

A rejection reading only "no security scanner had a record" hid whether a
lookup errored, was blocked, or genuinely returned nothing — the outcomes
below are what distinguishes those.
"""
import pytest

import app.rugcheck.filters as filters
from app.config import settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_market_liquidity(monkeypatch):
    """Keep DexScreener out of these tests; depth is covered elsewhere."""
    async def none(_addr):
        return None
    monkeypatch.setattr(filters.price_feed, "get_liquidity_usd", none)


def _rugcheck_ok() -> dict:
    return {
        "mintAuthority": None, "freezeAuthority": None,
        "token": {"mintAuthority": None, "freezeAuthority": None},
        "topHolders": [{"address": "A1", "owner": "O1", "pct": 2.0}],
        "markets": [{"marketType": "raydium", "lp": {}}],
        "lockers": {}, "totalMarketLiquidity": 500_000.0,
        "risks": [], "rugged": False,
    }


def _patch(monkeypatch, *, rugcheck=None, rugcheck_exc=None, goplus=None, goplus_exc=None):
    async def fake_rugcheck(_mint):
        if rugcheck_exc:
            raise rugcheck_exc
        return rugcheck or {}

    async def fake_goplus(_chain, _addr):
        if goplus_exc:
            raise goplus_exc
        return goplus or {}

    monkeypatch.setattr(filters.rugcheck_xyz, "fetch_token_report", fake_rugcheck)
    monkeypatch.setattr(filters.goplus, "fetch_token_security", fake_goplus)


async def test_no_token_address_rejected():
    report = await filters.run_rug_checks("solana", None)
    assert not report.passed
    assert "no on-chain token address" in report.reasons[0]


async def test_rugcheck_used_first_on_solana(monkeypatch):
    _patch(monkeypatch, rugcheck=_rugcheck_ok(), goplus={"should": "not be used"})
    report = await filters.run_rug_checks("solana", "SomeMint")
    assert report.passed, report.reasons
    assert "rugcheck" in report.raw


async def test_falls_back_to_goplus_when_rugcheck_empty(monkeypatch):
    goplus_data = {
        "mintable": {"status": "0"}, "freezable": {"status": "0"},
        "dex": [{"burn_percent": "100", "tvl": "400000"}],
        "holders": [{"account": "H1", "percent": "0.03", "tag": ""}],
        "lp_holders": [], "creators": [{"address": "C1"}],
    }
    _patch(monkeypatch, rugcheck={}, goplus=goplus_data)
    report = await filters.run_rug_checks("solana", "SomeMint")
    assert report.passed, report.reasons
    assert "goplus" in report.raw


async def test_falls_back_to_goplus_when_rugcheck_errors(monkeypatch):
    goplus_data = {
        "mintable": {"status": "0"}, "freezable": {"status": "0"},
        "dex": [{"burn_percent": "100", "tvl": "400000"}],
        "holders": [{"account": "H1", "percent": "0.03", "tag": ""}],
        "lp_holders": [], "creators": [{"address": "C1"}],
    }
    _patch(monkeypatch, rugcheck_exc=RuntimeError("boom"), goplus=goplus_data)
    report = await filters.run_rug_checks("solana", "SomeMint")
    assert report.passed, report.reasons


async def test_both_sources_empty_reports_each_outcome(monkeypatch):
    _patch(monkeypatch, rugcheck={}, goplus={})
    report = await filters.run_rug_checks("solana", "SomeMint")
    assert not report.passed
    reason = report.reasons[0]
    assert "rugcheck.xyz: no record" in reason
    assert "goplus: no record" in reason


async def test_lookup_error_is_named_in_the_rejection(monkeypatch):
    _patch(monkeypatch, rugcheck_exc=TimeoutError("timed out"), goplus={})
    report = await filters.run_rug_checks("solana", "SomeMint")
    assert not report.passed
    reason = report.reasons[0]
    assert "lookup failed" in reason
    assert "TimeoutError" in reason
    assert "goplus: no record" in reason


async def test_market_liquidity_overrides_scanner_depth(monkeypatch):
    _patch(monkeypatch, rugcheck=_rugcheck_ok())

    async def deep(_addr):
        return 1_234_567.0
    monkeypatch.setattr(filters.price_feed, "get_liquidity_usd", deep)

    report = await filters.run_rug_checks("solana", "SomeMint")
    assert report.liquidity_usd == pytest.approx(1_234_567.0)


async def test_disabled_filter_skips_lookups(monkeypatch):
    monkeypatch.setattr(settings, "RUGCHECK_ENABLED", False)
    report = await filters.run_rug_checks("solana", "SomeMint")
    assert report.passed
    assert report.raw.get("skipped") is True
