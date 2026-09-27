"""Tests for MarketSnapshot parsing in app/services/price_feed.py.

Parsing is split out from the fetch precisely so it can be tested against a
recorded DexScreener payload with no network involved. The properties that
matter: every window the market-quality score depends on is actually
extracted, and a field the API omitted stays None instead of becoming a
fabricated zero.
"""
import datetime as dt

from app.services.price_feed import _best_pair, _snapshot_from_pair


def _pair(**overrides) -> dict:
    """A DexScreener /latest/dex/tokens pair, shaped as the API returns it."""
    created_ms = int(
        (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)).timestamp() * 1000
    )
    pair = {
        "chainId": "solana",
        "dexId": "raydium",
        "pairAddress": "PairAddr111",
        "baseToken": {"address": "MintAddr111", "name": "Test Token", "symbol": "TEST"},
        "quoteToken": {"address": "So11111111111111111111111111111111111111112", "symbol": "SOL"},
        "priceUsd": "0.0042",
        "liquidity": {"usd": 180000.5, "base": 1000.0, "quote": 500.0},
        "fdv": 4200000,
        "marketCap": 3900000,
        "volume": {"m5": 1200.0, "h1": 24000.0, "h6": 130000.0, "h24": 460000.0},
        "priceChange": {"m5": 0.4, "h1": 6.2, "h6": -3.1, "h24": 41.7},
        "txns": {
            "m5": {"buys": 4, "sells": 3},
            "h1": {"buys": 61, "sells": 44},
            "h6": {"buys": 300, "sells": 240},
            "h24": {"buys": 600, "sells": 380},
        },
        "pairCreatedAt": created_ms,
    }
    pair.update(overrides)
    return pair


def test_every_window_the_quality_score_needs_is_parsed():
    snap = _snapshot_from_pair(_pair(), "MintAddr111")

    assert snap.price_usd == 0.0042
    assert snap.liquidity_usd == 180000.5
    assert (snap.volume_5m_usd, snap.volume_1h_usd, snap.volume_6h_usd, snap.volume_24h_usd) == (
        1200.0, 24000.0, 130000.0, 460000.0,
    )
    assert (snap.price_change_5m_pct, snap.price_change_1h_pct) == (0.4, 6.2)
    assert (snap.price_change_6h_pct, snap.price_change_24h_pct) == (-3.1, 41.7)
    assert (snap.buys_1h, snap.sells_1h) == (61, 44)
    assert (snap.buys_24h, snap.sells_24h) == (600, 380)


def test_identity_and_venue_are_parsed():
    snap = _snapshot_from_pair(_pair(), "MintAddr111")
    assert snap.token_address == "MintAddr111"
    assert snap.token_symbol == "TEST"
    assert snap.token_name == "Test Token"
    assert snap.dex_id == "raydium"
    assert snap.chain_id == "solana"
    assert snap.pair_address == "PairAddr111"
    assert snap.market_cap_usd == 3900000
    assert snap.fdv_usd == 4200000


def test_the_canonical_identity_is_the_mint_not_the_symbol():
    """Symbols are not unique and are trivially spoofed - two different
    mints called BONK are two different assets."""
    snap = _snapshot_from_pair(_pair(), "MintAddr111")
    assert snap.token_address == "MintAddr111"

    # If DexScreener omits baseToken.address, fall back to the mint we asked
    # about - never to the symbol.
    snap = _snapshot_from_pair(_pair(baseToken={"symbol": "TEST"}), "MintAddr111")
    assert snap.token_address == "MintAddr111"


def test_omitted_fields_stay_none_rather_than_becoming_zero():
    """A token with unavailable volume is not a token with zero volume, and
    the scores downstream treat those two cases very differently."""
    snap = _snapshot_from_pair(_pair(volume={}, txns={}, liquidity={}), "MintAddr111")
    assert snap.volume_24h_usd is None
    assert snap.volume_1h_usd is None
    assert snap.liquidity_usd is None
    assert snap.buys_24h is None and snap.sells_24h is None
    assert snap.total_txns_24h is None


def test_unparseable_numbers_become_none_not_a_crash():
    snap = _snapshot_from_pair(_pair(priceUsd="N/A", fdv=None), "MintAddr111")
    assert snap.price_usd is None
    assert snap.fdv_usd is None


def test_a_missing_creation_time_leaves_age_unknown():
    snap = _snapshot_from_pair(_pair(pairCreatedAt=None), "MintAddr111")
    assert snap.pair_created_at is None
    assert snap.age_hours is None


def test_pool_age_is_derived_from_the_creation_timestamp():
    snap = _snapshot_from_pair(_pair(), "MintAddr111")
    assert 71 < snap.age_hours < 73  # 3 days


def test_derived_transaction_properties():
    snap = _snapshot_from_pair(_pair(), "MintAddr111")
    assert snap.total_txns_24h == 980
    assert round(snap.buy_sell_ratio_24h, 3) == round(600 / 380, 3)


def test_no_sells_reads_as_infinite_not_as_missing():
    """All buys and no sells is a real, and suspicious, reading - reporting
    it as None would hide it."""
    snap = _snapshot_from_pair(_pair(txns={"h24": {"buys": 50, "sells": 0}}), "MintAddr111")
    assert snap.buy_sell_ratio_24h == float("inf")


def test_the_observation_is_stamped_on_arrival():
    snap = _snapshot_from_pair(_pair(), "MintAddr111")
    assert snap.age_seconds() < 5
    assert snap.observed_at.tzinfo is not None


def test_the_deepest_pool_wins_when_a_token_has_several():
    pairs = [
        _pair(pairAddress="thin", liquidity={"usd": 5_000}),
        _pair(pairAddress="deep", liquidity={"usd": 900_000}),
        _pair(pairAddress="mid", liquidity={"usd": 120_000}),
    ]
    assert _best_pair(pairs)["pairAddress"] == "deep"


def test_no_pairs_means_no_snapshot():
    assert _best_pair([]) is None
