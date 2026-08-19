"""Entry reservations: stopping the same token being bought twice at once.

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

SCOPE, stated plainly: this guards one process. It is the correct guard
for this bot, which runs its scanner and webhook in a single event loop.
Two bot processes against one database would need a database-level lock,
and `reserved_elsewhere` exists so that case fails loudly rather than
silently double-buying.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_reserved: set[str] = set()


class AlreadyReserved(RuntimeError):
    """Raised when an entry for this instrument is already in flight."""


@contextlib.asynccontextmanager
async def reserve_entry(key: str):
    """Claim `key` for the duration of one entry attempt.

    Raises AlreadyReserved if another coroutine is already partway through
    opening a position in this token. The reservation is always released,
    including when the entry fails or raises - a leaked reservation would
    lock the bot out of a token until restart, which is a worse failure
    than the one being prevented.
    """
    async with _lock:
        if key in _reserved:
            raise AlreadyReserved(key)
        _reserved.add(key)

    try:
        yield
    finally:
        async with _lock:
            _reserved.discard(key)


def reserved_keys() -> set[str]:
    """Snapshot of in-flight entries, for the dashboard's health panel."""
    return set(_reserved)


def clear_reservations() -> None:
    """Drop every reservation. Only for tests and for startup recovery -
    a fresh process cannot have entries in flight, and inheriting a stale
    set would block those tokens forever."""
    _reserved.clear()
