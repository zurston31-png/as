"""Resolving a chart symbol to the mint it actually refers to.

The rule under test is app/identity.py's: the mint is the token, the symbol
is a label. These cover the ways a naive resolver picks the wrong one -
substring matches, liquidity split across pools, and above all a copycat
that simply declares a huge pool - because every one of them ends with the
bot screening one token and trading another.
"""
import pytest

from app.analysis.resolve_symbol import (
    MIN_CREDIBLE_LIQUIDITY_USD,
    MIN_CREDIBLE_VOLUME_24H_USD,
    Resolution,
    _fold_pairs,
    describe_address,
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


def _real(symbol, address, *, liq=3_000_000, vol=500_000):
    """A mint with a market: real depth and real trading against it."""
    return _pair(symbol, address, liq, vol=vol)


# ---------------------------------------------------------------------------
# the bug this module was rewritten for
# ---------------------------------------------------------------------------

def test_a_pool_claiming_a_fortune_that_nobody_trades_loses_to_a_real_market():
    """Regression, from a real run. Searching WIF returned three pools each
    claiming over a billion dollars against four dollars of daily volume,
    and the first version of this module - which ranked on liquidity - put
    all three above the genuine mint and then truncated the genuine one out
    of the results entirely.

    Reported liquidity is a number attached to a pool anyone can create.
    Volume is what someone had to actually do.
    """
    pairs = [
        _pair("WIF", "FakeMintA", 1_506_847_379, vol=4),
        _pair("WIF", "FakeMintB", 1_012_256_085, vol=4),
        _pair("WIF", "FakeMintC", 774_257_604, vol=4),
        _real("WIF", "RealWifMint", liq=3_000_000, vol=1_200_000),
    ]
    ranked = _fold_pairs(pairs, "WIF", "solana")
    assert ranked[0].token_address == "RealWifMint"
    assert [c.token_address for c in ranked if c.live] == ["RealWifMint"]
    assert all(c.dead_pool for c in ranked if c.token_address.startswith("Fake"))


def test_a_dead_pool_says_why_it_was_rejected():
    """The reason has to name both numbers. "Low liquidity" would be the
    opposite of what is wrong with a pool claiming $1.5bn."""
    res = Resolution(
        symbol="WIF", requested_chain="solana",
        candidates=_fold_pairs([_pair("WIF", "FakeMint", 1_506_847_379, vol=4)], "WIF", "solana"),
    )
    reason = res.candidates[0].why_not_live()
    assert "1,506,847,379" in reason and "$4" in reason
    assert "NO REAL MARKET" in res.verdict()


def test_a_display_cap_can_never_hide_a_live_candidate(monkeypatch):
    """The genuine mint was lost to `limit`, not just to the sort. Filling
    the list with fakes must not push a real answer off the end."""
    import app.analysis.resolve_symbol as module

    fakes = [_pair("WIF", f"FakeMint{i}", 1_000_000_000 + i, vol=4) for i in range(20)]
    payload = {"pairs": fakes + [_real("WIF", "RealWifMint")]}

    async def listing(url, **kwargs):
        return payload

    monkeypatch.setattr(module, "_get_json", listing)

    import anyio
    res = anyio.run(lambda: module.resolve_symbol("WIF", "solana", limit=3))
    assert "RealWifMint" in [c.token_address for c in res.candidates]
    assert res.unambiguous


def test_a_deep_pool_with_thin_but_plausible_trading_is_still_live():
    """The turnover floor exists to catch fabrications, not to demand that a
    quiet token be busy. It must sit far below anything genuine."""
    quiet = _fold_pairs([_pair("TEST", "QuietMint", 2_000_000, vol=50_000)], "TEST", "solana")[0]
    assert quiet.live
    assert not quiet.dead_pool


# ---------------------------------------------------------------------------
# identity: matching the right symbol, on the right chain, across pools
# ---------------------------------------------------------------------------

def test_only_an_exact_symbol_match_counts():
    """A substring match turns a search for DOGE into every DOGE-suffixed
    derivative on the chain, and the top hit is then whichever of them is
    busiest - not DOGE."""
    pairs = [
        _real("DOGE", "DogeMint111"),
        _real("DOGEGF", "DogegfMint222", vol=5_000_000),
        _real("BABYDOGE", "BabyMint333", vol=3_000_000),
    ]
    assert [c.token_address for c in _fold_pairs(pairs, "DOGE", "solana")] == ["DogeMint111"]


def test_symbol_match_is_case_insensitive_and_ignores_padding():
    pairs = [_real(" wif ", "WifMint111")]
    assert [c.token_address for c in _fold_pairs(pairs, "WIF", "solana")] == ["WifMint111"]


def test_volume_and_liquidity_are_summed_across_a_mints_pools():
    """A token split over three pools is not less liquid, or less traded,
    than the same totals in one. The real BONK mint carries seventeen."""
    pairs = [
        _pair("BONK", "RealMint111", 400_000, vol=300_000),
        _pair("BONK", "RealMint111", 400_000, vol=300_000),
        _pair("BONK", "RealMint111", 400_000, vol=300_000),
        _pair("BONK", "CopycatMint222", 900_000, vol=20_000),
    ]
    candidates = _fold_pairs(pairs, "BONK", "solana")
    assert candidates[0].token_address == "RealMint111"
    assert candidates[0].liquidity_usd == pytest.approx(1_200_000)
    assert candidates[0].volume_24h_usd == pytest.approx(900_000)
    assert candidates[0].pair_count == 3


def test_a_different_chain_is_excluded_when_a_chain_is_requested():
    """PEPE exists on several chains. A Solana deployment must not be handed
    the Ethereum contract address, which its rug-check and execution paths
    cannot use."""
    pairs = [
        _real("PEPE", "0xEthPepe", vol=10_000_000, liq=50_000_000),
        _real("PEPE", "SolPepeMint"),
    ]
    pairs[0]["chainId"] = "ethereum"
    assert [c.token_address for c in _fold_pairs(pairs, "PEPE", "solana")] == ["SolPepeMint"]
    assert len(_fold_pairs(pairs, "PEPE", None)) == 2


def test_fdv_is_taken_from_whichever_pair_reports_it():
    """FDV is a property of the token, not the pool, but individual pairs
    omit it - so a mint whose first pair lacks it must still carry it."""
    pairs = [
        _pair("WIF", "WifMint111", 500_000, vol=100_000, fdv=None),
        _pair("WIF", "WifMint111", 100_000, vol=50_000, fdv=2_500_000),
    ]
    assert _fold_pairs(pairs, "WIF", "solana")[0].fdv_usd == pytest.approx(2_500_000)


# ---------------------------------------------------------------------------
# the verdict - the part that decides whether a human is needed
# ---------------------------------------------------------------------------

def _resolution(*pairs):
    return Resolution(
        symbol="TEST", requested_chain="solana",
        candidates=_fold_pairs(list(pairs), "TEST", "solana"),
    )


def test_a_lone_traded_mint_resolves():
    res = _resolution(_real("TEST", "OnlyMint111"))
    assert res.unambiguous
    assert res.verdict() == "RESOLVED"


def test_a_dominant_mint_resolves_despite_a_dust_squatter():
    res = _resolution(
        _real("TEST", "RealMint111", vol=5_000_000),
        _pair("TEST", "SquatMint222", 1_200, vol=30),
    )
    assert res.unambiguous
    assert res.best.token_address == "RealMint111"


def test_two_comparably_traded_claimants_are_ambiguous_rather_than_ranked():
    """This is the case that matters, and PEPE on Solana really is one. Both
    have genuine markets; picking the busier is a coin flip, and losing it
    means every trade goes into the wrong token."""
    res = _resolution(
        _real("TEST", "MintA111", vol=387_940),
        _real("TEST", "MintB222", vol=312_256),
    )
    assert not res.unambiguous
    assert "AMBIGUOUS" in res.verdict()
    assert len(res.live_candidates) == 2  # both shown - the human needs both


def test_a_symbol_whose_only_claimant_is_dust_does_not_resolve():
    res = _resolution(_pair("TEST", "DustMint111", MIN_CREDIBLE_LIQUIDITY_USD - 1, vol=5))
    assert not res.unambiguous
    assert "NO REAL MARKET" in res.verdict()


def test_a_deep_pool_below_the_volume_floor_does_not_resolve():
    res = _resolution(_pair("TEST", "QuietMint", 40_000, vol=MIN_CREDIBLE_VOLUME_24H_USD - 1))
    assert not res.unambiguous
    assert res.best is None


def test_no_listing_at_all_is_reported_as_no_match():
    res = _resolution()
    assert not res.unambiguous
    assert "NO MATCH" in res.verdict()
    assert res.best is None


# ---------------------------------------------------------------------------
# address lookup - the direction that does not depend on search quality
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_an_address_lookup_reports_whatever_symbol_the_mint_carries(monkeypatch):
    """Asking about a mint must not filter on an expected symbol - the whole
    point is to find out what the address is."""
    import app.analysis.resolve_symbol as module

    async def listing(url, **kwargs):
        return {"pairs": [_real("SOMETHING", "TheMint")]}

    monkeypatch.setattr(module, "_get_json", listing)
    res = await describe_address("TheMint", "solana")
    assert res.best is not None
    assert res.best.symbol == "SOMETHING"


@pytest.mark.anyio
async def test_an_address_lookup_ignores_pairs_for_other_mints(monkeypatch):
    """The tokens endpoint returns pairs, and a pair has two sides. Only the
    mint that was asked about may come back."""
    import app.analysis.resolve_symbol as module

    async def listing(url, **kwargs):
        return {"pairs": [_real("WIF", "TheMint"), _real("USDC", "SomeOtherMint")]}

    monkeypatch.setattr(module, "_get_json", listing)
    res = await describe_address("TheMint", "solana")
    assert [c.token_address for c in res.candidates] == ["TheMint"]


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------

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
