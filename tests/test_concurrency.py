"""Tests for app/concurrency.py and the buy-path race it closes.

The bug: the buy path checks "no open position in this token" and then,
several network round-trips later, creates one. The scanner loop and the
webhook share an event loop, so two coroutines could interleave inside that
window, both see an empty book, and both open a position - double the
intended size, split across two rows so the exposure cap never sees it.
"""
import asyncio

import pytest

from app.concurrency import (
    AlreadyReserved,
    clear_reservations,
    reserve_entry,
    reserved_keys,
)

pytestmark = pytest.mark.anyio

MINT = "RaceMint1111111111111111111111111111111111"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean():
    clear_reservations()
    yield
    clear_reservations()


async def test_a_reservation_is_visible_while_held_and_gone_after():
    assert reserved_keys() == set()
    async with reserve_entry(MINT):
        assert MINT in reserved_keys()
    assert reserved_keys() == set()


async def test_a_second_entry_for_the_same_mint_is_refused():
    async with reserve_entry(MINT):
        with pytest.raises(AlreadyReserved):
            async with reserve_entry(MINT):
                pytest.fail("the second reservation must not be granted")


async def test_different_mints_do_not_block_each_other():
    """The guard must not serialise the scanner. Two candidates for
    different tokens have to be able to proceed at the same time."""
    async with reserve_entry("MintA"):
        async with reserve_entry("MintB"):
            assert reserved_keys() == {"MintA", "MintB"}


async def test_the_reservation_is_released_when_the_entry_raises():
    """A leaked reservation would lock the bot out of a token until
    restart - a worse failure than the double-buy it prevents."""
    with pytest.raises(ValueError):
        async with reserve_entry(MINT):
            raise ValueError("rug check blew up")
    assert reserved_keys() == set()

    # ...and the token is immediately usable again.
    async with reserve_entry(MINT):
        pass


async def test_two_interleaved_entries_produce_exactly_one_winner():
    """The actual race, reproduced.

    Both coroutines await between claiming and finishing, which is what
    makes the interleaving possible in the first place. Exactly one must
    get through.
    """
    entered = []

    async def attempt(tag):
        try:
            async with reserve_entry(MINT):
                await asyncio.sleep(0.01)      # the network round-trips
                entered.append(tag)
        except AlreadyReserved:
            return "refused"
        return "entered"

    results = await asyncio.gather(attempt("a"), attempt("b"))

    assert len(entered) == 1, f"exactly one entry should have proceeded, got {entered}"
    assert sorted(results) == ["entered", "refused"]


async def test_many_simultaneous_attempts_still_yield_one_winner():
    winners = 0

    async def attempt():
        nonlocal winners
        try:
            async with reserve_entry(MINT):
                await asyncio.sleep(0.005)
                winners += 1
        except AlreadyReserved:
            pass

    await asyncio.gather(*(attempt() for _ in range(25)))
    assert winners == 1


async def test_reservations_are_sequential_not_permanent():
    """Once the first entry completes the token is free again - the guard
    covers one attempt, it is not a blacklist."""
    async with reserve_entry(MINT):
        pass
    async with reserve_entry(MINT):
        assert MINT in reserved_keys()


def test_clearing_is_available_for_startup_recovery():
    """A fresh process cannot have entries in flight; inheriting a stale
    set from a previous run would block those tokens forever."""
    clear_reservations()
    assert reserved_keys() == set()
