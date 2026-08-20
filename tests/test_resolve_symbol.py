"""Resolving a chart symbol to the mint it actually refers to.

The rule under test is app/identity.py's: the mint is the token, the symbol
is a label. These cover the ways a naive resolver picks the wrong one -
substring matches, a copycat with real money behind it, liquidity split
across pools - because every one of them ends with the bot screening one
token and trading another.
"""
import pytest

from app.analysis.resolve_symbol import (
    MIN_CREDIBLE_LIQUIDITY_USD,
    Resolution,
    _fold_pairs,
    resolve_symbol,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _pair(symbol, address, liq, *, chain="solana", vol=0.0, name=None, fdv=None):
    return {
        "chainId": chain,
        "baseToken": {"symbol": symbol, "address": address, "name": name or f"{symbol} token"},
        "liquidity": {"usd": liq},
        "volume": {"h24": vol},
        "priceUsd": "0.5",
        "fdv": fdv,
    }


def test_only_an_exact_symbol_match_counts():
    """A substring match turns a search for DOGE into every DOGE-suffixed
    derivative on the chain, and the first hit is then whichever of them is
    biggest - not DOGE."""
    pairs = [
        _pair("DOGE", "DogeMint111", 900_000),
        _pair("DOGEGF", "DogegfMint222", 5_000_000),
        _pair("BABYDOGE", "BabyMint333", 3_000_000),
    ]
    candidates = _fold_pairs(pairs, "DOGE", "solana")
    assert [c.token_address for c in candidates] == ["DogeMint111"]


def test_symbol_match_is_case_insensitive_and_ignores_padding():
    pairs = [_pair(" wif ", "WifMint111", 100_000)]
    assert [c.token_address for c in _fold_pairs(pairs, "WIF", "solana")] == ["WifMint111"]


def test_liquidity_is_summed_across_a_mints_pools_not_taken_from_the_deepest():
    """A token split over three pools is not less liquid than the same depth
    in one. Taking the max would rank a single-pool copycat above the real
    token whenever the real one is spread out."""
    pairs = [
        _pair("BONK", "RealMint111", 400_000, vol=1_000),
        _pair("BONK", "RealMint111", 400_000, vol=2_000),
        _pair("BONK", "RealMint111", 400_000, vol=3_000),
        _pair("BONK", "CopycatMint222", 900_000, vol=50),
    ]
    candidates = _fold_pairs(pairs, "BONK", "solana")
    assert candidates[0].token_address == "RealMint111"
    assert candidates[0].liquidity_usd == pytest.approx(1_200_000)
    assert candidates[0].volume_24h_usd == pytest.approx(6_000)
    assert candidates[0].pair_count == 3


def test_a_different_chain_is_excluded_when_a_chain_is_requested():
    """PEPE exists on several chains. A Solana deployment must not be handed
    the Ethereum contract address, which its rug-check and execution paths
    cannot use."""
    pairs = [
        _pair("PEPE", "0xEthPepe", 10_000_000, chain="ethereum"),
        _pair("PEPE", "SolPepeMint", 200_000, chain="solana"),
    ]
    assert [c.token_address for c in _fold_pairs(pairs, "PEPE", "solana")] == ["SolPepeMint"]
    both = _fold_pairs(pairs, "PEPE", None)
    assert len(both) == 2


def test_fdv_is_taken_from_whichever_pair_reports_it():
    """FDV is a property of the token, not the pool, but individual pairs
    omit it - so a mint whose first pair lacks it must still carry it."""
    pairs = [
        _pair("WIF", "WifMint111", 500_000, fdv=None),
        _pair("WIF", "WifMint111", 100_000, fdv=2_500_000),
    ]
    assert _fold_pairs(pairs, "WIF", "solana")[0].fdv_usd == pytest.approx(2_500_000)


# ---------------------------------------------------------------------------
# the ambiguity verdict - the part that decides whether a human is needed
# ---------------------------------------------------------------------------

def _resolution(*pairs):
    return Resolution(
        symbol="TEST", requested_chain="solana",
        candidates=_fold_pairs(list(pairs), "TEST", "solana"),
    )


def test_a_lone_credible_mint_resolves():
    res = _resolution(_pair("TEST", "OnlyMint111", 500_000))
    assert res.unambiguous
    assert res.verdict() == "RESOLVED"


def test_a_dominant_mint_resolves_despite_a_dust_squatter():
    res = _resolution(
        _pair("TEST", "RealMint111", 5_000_000),
        _pair("TEST", "SquatMint222", 1_200),
    )
    assert res.unambiguous
    assert res.best.token_address == "RealMint111"


def test_two_comparable_claimants_are_reported_ambiguous_rather_than_ranked():
    """This is the case that matters. Both are real pools with real money;
    picking the larger is a coin flip, and losing it means every trade goes
    into the copycat. The command must refuse to answer."""
    res = _resolution(
        _pair("TEST", "MintA111", 600_000),
        _pair("TEST", "MintB222", 400_000),
    )
    assert not res.unambiguous
    assert "AMBIGUOUS" in res.verdict()
    # It still reports both - the human needs the addresses to compare.
    assert len(res.candidates) == 2


def test_a_symbol_whose_best_claimant_is_dust_does_not_resolve():
    res = _resolution(_pair("TEST", "DustMint111", MIN_CREDIBLE_LIQUIDITY_USD - 1))
    assert not res.unambiguous
    assert "TOO THIN" in res.verdict()


def test_no_listing_at_all_is_reported_as_no_match():
    res = _resolution()
    assert not res.unambiguous
    assert "NO MATCH" in res.verdict()
    assert res.best is None


@pytest.mark.anyio
async def test_an_unreachable_listing_source_is_an_error_not_an_empty_answer(monkeypatch):
    """"No match" and "could not ask" must not look the same. The first means
    the symbol is untradeable; the second means try again - and silently
    conflating them is how a network blip becomes a wrong config."""
    import app.analysis.resolve_symbol as module

    async def boom(url, **kwargs):
        raise TimeoutError("upstream timed out")

    monkeypatch.setattr(module, "_get_json", boom)
    res = await resolve_symbol("WIF", "solana")
    assert res.error is not None
    assert "UNRESOLVED" in res.verdict()
    assert res.candidates == []


@pytest.mark.anyio
async def test_a_non_dict_response_is_an_error_rather_than_a_crash(monkeypatch):
    import app.analysis.resolve_symbol as module

    async def html_error_page(url, **kwargs):
        return "<html>rate limited</html>"

    monkeypatch.setattr(module, "_get_json", html_error_page)
    res = await resolve_symbol("WIF", "solana")
    assert res.error is not None
    assert not res.unambiguous
