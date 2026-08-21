"""Reservations: stopping the same position being opened - or closed -
twice at once.

The buy path checks "is there already an open position in this token?" and
then, several network round-trips later, creates one. Between those two
moments it awaits a rug check, a candle fetch, a market snapshot and a
simulated fill - hundreds of milliseconds, sometimes seconds.

The scanner loop and the TradingView webhook both run on the same event
loop and both call that path. Nothing stopped two coroutines interleaving
inside that window, both seeing "no open position", and both opening one.
The result is a double-size position that no risk limit authorised, split
across two rows so the exposure cap does not see it either.

The fix is a reservation, not a lock around the whole pipeline. Holding a
lock across every await would serialise the scanner's network work and
throw away most of its throughput for a guard that only needs to cover a
decision. Instead:

    acquire the lock -> is anyone holding or reserving this mint?
                     -> no: record the reservation
    release the lock -> do the slow work
                     -> release the reservation, whatever the outcome

The lock is held only for the set membership test, so contention is
negligible.

THE EXIT SIDE HAS THE SAME RACE, AND IT IS WORSE

`close_position` reads a position's qty, awaits the sell, and only then
marks the row closed. The position monitor's stop-loss and a TradingView
sell alert are two callers of that path on one event loop, so a stop
firing at the same moment a sell arrives has both of them read the same
open position, both sell it, and both credit the proceeds to cash. The
paper account is then paid twice for a position it held once, with two
sell legs recorded against one entry - and unlike the entry race there is
no downstream check to catch it, because "is there an open position?" was
true for both.

`reserve_exit` closes that, keyed on the position id rather than the mint:
what must not happen twice is the exit of one specific position.

SCOPE, stated plainly: this guards ONE PROCESS. It is the correct guard
for this bot, which runs its scanner, monitor and webhook in a single
event loop. Two bot processes against one database would share neither
the lock nor the set, and would need a database-level lock instead. That
deployment is not supported and this module cannot detect it.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_reserved: set[str] = set()
_reserved_exits: set[str] = set()


class AlreadyReserved(RuntimeError):
    """Raised when an entry or exit for this target is already in flight."""


@contextlib.asynccontextmanager
async def _reserve(pool: set[str], key: str):
    """Claim `key` in `pool` for the duration of one attempt.

    The reservation is always released, including when the work fails or
    raises - a leaked reservation would lock the bot out of a token (or,
    worse, out of exiting a position) until restart, which is a worse
    failure than the one being prevented.
    """
    async with _lock:
        if key in pool:
            raise AlreadyReserved(key)
        pool.add(key)

    try:
        yield
    finally:
        async with _lock:
            pool.discard(key)


@contextlib.asynccontextmanager
async def reserve_entry(key: str):
    """Claim a mint for the duration of one entry attempt.

    Raises AlreadyReserved if another coroutine is already partway through
    opening a position in this token.
    """
    async with _reserve(_reserved, key):
        yield


@contextlib.asynccontextmanager
async def reserve_exit(position_id: int):
    """Claim one POSITION for the duration of one exit attempt.

    Keyed on the position id, not the mint: what must not happen twice is
    the exit of one specific position, and keying on the mint would make a
    partial exit of one position block an unrelated one.

    Raises AlreadyReserved if another coroutine is already partway through
    selling this position - the stop-loss firing while a TradingView sell
    alert is in flight, most realistically.
    """
    async with _reserve(_reserved_exits, f"position:{position_id}"):
        yield


def reserved_keys() -> set[str]:
    """Snapshot of in-flight entries, for the dashboard's health panel."""
    return set(_reserved)


def reserved_exits() -> set[str]:
    """Snapshot of in-flight exits."""
    return set(_reserved_exits)


def clear_reservations() -> None:
    """Drop every reservation. Only for tests - a fresh process starts
    with empty sets anyway, since they are plain process memory."""
    _reserved.clear()
    _reserved_exits.clear()
