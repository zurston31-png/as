"""Deterministic idempotency keys for inbound alerts.

THE FAILURE THIS PREVENTS

The webhook has no memory. Post the same TradingView alert twice - a
retry after a timeout the bot actually handled, a proxy replaying a
buffered request, someone re-firing a captured body - and it is processed
twice, as two unrelated signals. A duplicate BUY can open a second
position the risk limits never sized for; a duplicate SELL is worse,
because the close path finds the position, sells it, and a second delivery
arriving before the first commits can sell it again.

app/concurrency.py already stops two entries for one mint overlapping
inside a single process. It does not help here: a replay arriving a minute
later does not overlap anything, and the reservation set is per-process
memory that a restart empties.

WHY THE UNIQUENESS IS IN THE DATABASE

"Look for an existing row, and insert if there isn't one" is two
statements with a gap between them, and the gap is exactly the window a
duplicate delivery lands in. The SELECT is kept as a fast path because it
avoids a pointless INSERT in the common case, but it is not the guarantee.
The guarantee is a UNIQUE index: the second INSERT fails at the database,
whatever the application believed it had checked.

WHAT GOES INTO THE KEY

The instrument (canonical mint identity, not the ticker), the direction,
the alert's OWN timestamp - the `time` field, which the shipped Pine
script fills from the BAR, not from the moment the request was built - and
the price. Arrival time is what a replay changes, so keying on it would
make every replay look new and defeat the whole thing.

Price is in there for alerts this repo did not write. The shipped script
uses `alert.freq_once_per_bar_close`, so one bar produces at most one buy
and one sell and the bar time alone is already unique. A hand-written
alert firing intrabar would emit several genuine alerts sharing one bar
time, and without the price they would collide and the later ones would be
discarded as replays. A true replay resends the identical body, price
included, so adding it costs that case nothing.

WHEN THE KEY IS NULL, AND WHY THAT IS NOT A CHEAT

An alert with no `time` field cannot be deduplicated: two genuine alerts
on consecutive bars are then byte-identical, and a key built from what
remains would reject the second real signal as a replay. Suppressing a
true signal is a worse failure than processing a duplicate, so the key is
NULL and no uniqueness is enforced for that alert.

NULL is not "no duplicates found" - it is "this alert cannot be checked",
and `unprotected_reason` says so in as many words. Both SQLite and
Postgres allow many NULLs in a unique index, so the protected alerts stay
protected either way. The fix is on the sending side: put `time` in the
alert body, which the shipped Pine script already does.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

from app.identity import instrument_key

# Bumped if the key's inputs ever change. Without it, a redefinition would
# make every previously-seen alert look new - a silent replay window on the
# one deploy that changes the rule.
KEY_VERSION = 1


def alert_key(
    *,
    source: str,
    symbol: str,
    token_address: str | None,
    signal_type: str,
    event_time: dt.datetime | None,
    price: float | None = None,
) -> str | None:
    """The idempotency key for one inbound alert, or None if it has none.

    Deterministic across processes and restarts: the same alert always
    produces the same key, which is the only property that makes a unique
    index able to reject a replay that arrives after a redeploy.
    """
    if event_time is None:
        return None

    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=dt.timezone.utc)

    payload = {
        "v": KEY_VERSION,
        "source": source,
        "instrument": instrument_key(symbol, token_address),
        "signal": signal_type,
        # Normalised to UTC so the same instant expressed in two offsets
        # produces one key rather than two.
        "at": event_time.astimezone(dt.timezone.utc).isoformat(),
        # repr rather than the float itself: json.dumps already round-trips
        # a float exactly, but going through repr makes the encoding
        # explicit and independent of any future dumps() setting.
        "price": repr(float(price)) if price is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def unprotected_reason(event_time: dt.datetime | None) -> str | None:
    """Why this alert has no replay protection, or None if it has some."""
    if event_time is not None:
        return None
    return (
        "the alert carries no `time` field, so two alerts on consecutive bars are "
        "indistinguishable and a duplicate delivery cannot be detected. Add "
        '"time": "{{timenow}}" to the alert message body - the shipped Pine script '
        "already sends it"
    )
