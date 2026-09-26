"""Tests for opportunity identity.

This is the join key for every paired comparison, so two failure modes
matter and they pull in opposite directions: an id that splits too easily
records a retry as a second independent sample, and an id that merges too
easily pairs a decision made at one price with an outcome measured from
another. The tests below pin both edges.
"""
import datetime as dt

from app.shadow.opportunity import ID_LENGTH, opportunity_id, quantize, time_bucket

NOW = dt.datetime(2026, 5, 1, 12, 0, 3, tzinfo=dt.timezone.utc)
MINT = "So11111111111111111111111111111111111111112"


def test_a_canonical_event_id_wins_outright():
    """When the source already decided what counts as one delivery,
    re-deriving that from market data can only be a worse guess."""
    early = opportunity_id(MINT, NOW, 0.01, event_id="scan-4471")
    later = opportunity_id(
        "a-completely-different-mint",
        NOW + dt.timedelta(hours=3), 99.0, event_id="scan-4471",
    )
    assert early == later == opportunity_id(MINT, NOW, None, event_id="scan-4471")


def test_two_duplicate_reads_seconds_apart_are_one_opportunity():
    """The bug this scheme replaced: hashing the exact reference price meant
    a duplicated scanner call, whose quote differed in the sixth decimal,
    hashed differently and was recorded as a second sample."""
    first = opportunity_id(MINT, NOW, 0.0100012)
    retry = opportunity_id(MINT, NOW + dt.timedelta(seconds=38), 0.0100034)
    assert first == retry


def test_a_real_move_inside_one_bucket_is_a_new_opportunity():
    """Thirty percent in forty seconds is normal in this market and it is
    genuinely a different trade. Merging those would pair a decision made
    at one price with an outcome measured from another."""
    calm = opportunity_id(MINT, NOW, 0.0100012)
    spiked = opportunity_id(MINT, NOW + dt.timedelta(seconds=38), 0.013)
    assert calm != spiked


def test_a_later_bucket_is_a_new_opportunity():
    assert opportunity_id(MINT, NOW, 0.01) != opportunity_id(
        MINT, NOW + dt.timedelta(minutes=5), 0.01
    )


def test_different_tokens_never_share_an_id():
    assert opportunity_id("MintA", NOW, 0.01) != opportunity_id("MintB", NOW, 0.01)


def test_the_id_never_depends_on_a_database_row():
    """Only inputs the opportunity itself carries. Anything order-dependent
    would mint a fresh identity for an observation already recorded, which
    is exactly what a restart replay must not do."""
    assert opportunity_id(MINT, NOW, 0.01) == opportunity_id(MINT, NOW, 0.01)
    assert len(opportunity_id(MINT, NOW, 0.01)) == ID_LENGTH


def test_a_naive_timestamp_is_read_as_utc():
    """Mixing an aware and a naive clock would silently split every pair
    the moment one call site forgot the timezone."""
    naive = NOW.replace(tzinfo=None)
    assert opportunity_id(MINT, naive, 0.01) == opportunity_id(MINT, NOW, 0.01)


def test_snapshot_keys_are_order_independent():
    a = opportunity_id(MINT, NOW, snapshot={"price": 0.01, "liquidity": 250_000})
    b = opportunity_id(MINT, NOW, snapshot={"liquidity": 250_000, "price": 0.01})
    assert a == b


def test_a_snapshot_axis_can_separate_two_looks_at_the_same_price():
    """Same price, drained pool: not the same chance to trade."""
    deep = opportunity_id(MINT, NOW, snapshot={"price": 0.01, "liquidity": 250_000})
    thin = opportunity_id(MINT, NOW, snapshot={"price": 0.01, "liquidity": 9_000})
    assert deep != thin


def test_quantize_is_relative_so_one_rule_covers_every_magnitude():
    """A memecoin price of 0.000000031 and a pool of 4.1 million have to go
    through the same tolerance, which an absolute rounding cannot do."""
    assert quantize(0.000000031) == quantize(0.000000031 * 1.001)
    assert quantize(4_100_000) == quantize(4_100_000 * 1.001)
    assert quantize(0.000000031) != quantize(0.000000031 * 1.5)


def test_quantize_handles_zero_and_negatives_without_reaching_log():
    assert quantize(0.0) == "0"
    assert quantize(-5.0).startswith("-")
    assert quantize(-5.0) != quantize(5.0)


def test_time_bucket_snaps_down_to_the_bucket_start():
    assert time_bucket(NOW, 60) == time_bucket(
        NOW.replace(second=59, microsecond=999999), 60
    )
    assert time_bucket(NOW, 60) % 60 == 0


def test_a_missing_price_is_not_the_same_as_a_zero_price():
    """Zero would read as a real quote of zero, which is a different -
    and impossible - market state."""
    assert opportunity_id(MINT, NOW, None) != opportunity_id(MINT, NOW, 0.0)
