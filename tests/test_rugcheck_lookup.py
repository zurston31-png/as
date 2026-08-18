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


SOLANA_MINT = "RmtMAYVTTFv2iK9muMrXEoAnSSsZPPgRPbqZCKwNDYk"
EVM_ADDRESS = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"


# --- chain routing ----------------------------------------------------------

@pytest.mark.parametrize("declared", ["solana", "Solana", "SOLANA", "sol", "", "typo", "ethereum"])
def test_solana_mint_always_routes_to_solana(declared):
    """The address encoding is unambiguous; the typed label is not."""
    assert filters.resolve_chain(declared, SOLANA_MINT) == "solana"


@pytest.mark.parametrize("declared,expected", [
    ("ethereum", "ethereum"), ("bsc", "bsc"), ("base", "base"),
    ("", "ethereum"), ("solana", "ethereum"),
])
def test_evm_address_routes_to_an_evm_chain(declared, expected):
    assert filters.resolve_chain(declared, EVM_ADDRESS) == expected


def test_unrecognisable_address_falls_back_to_declared_label():
    assert filters.resolve_chain("bsc", "not-a-real-address") == "bsc"


def test_address_shape_detection():
    assert filters.looks_like_solana_address(SOLANA_MINT)
    assert not filters.looks_like_evm_address(SOLANA_MINT)
    assert filters.looks_like_evm_address(EVM_ADDRESS)
    assert not filters.looks_like_solana_address(EVM_ADDRESS)
    # base58 excludes 0/O/I/l, so an address containing them is not a mint
    assert not filters.looks_like_solana_address("0OIl" + "1" * 30)


async def test_mislabelled_solana_token_still_reaches_rugcheck(monkeypatch):
    """Regression: a Solana mint sent with any other chain label skipped the
    Solana specialist entirely and was rejected on GoPlus's thin coverage,
    reading as a verdict on the token rather than a routing mistake."""
    _patch(monkeypatch, rugcheck=_rugcheck_ok(), goplus={})
    report = await filters.run_rug_checks("ethereum", SOLANA_MINT)
    assert report.passed, report.reasons
    assert "rugcheck" in report.raw


async def test_rejection_names_the_chain_it_screened_as(monkeypatch):
    _patch(monkeypatch, rugcheck={}, goplus={})
    report = await filters.run_rug_checks("whatever", SOLANA_MINT)
    assert not report.passed
    reason = report.reasons[0]
    assert "screened as solana" in reason
    assert "rugcheck.xyz: no record" in reason
    assert "goplus: no record" in reason


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
