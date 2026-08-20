"""A stable identity for "the same chance to trade".

Two observations should pair only when they are genuinely about the same
opportunity. Pairing on "timestamps are close" would join a champion's
look at token A with a challenger's look at token B thirty seconds later
and call it a controlled comparison.

THREE WAYS TO NAME AN OPPORTUNITY, IN ORDER OF PREFERENCE

1. A canonical id the source already assigns - a scanner run id, a webhook
   event id. When one exists it is the truth: the source has already
   decided what counts as one delivery, and re-deriving that from market
   data can only be a worse guess.

2. Otherwise: the mint, plus the evaluation instant snapped to a fixed
   bucket, plus a coarse market snapshot.

3. Never the database row id. That would make the id depend on insertion
   order, so a replay after a restart would mint a fresh identity for an
   observation already recorded.

WHY THE SNAPSHOT IS COARSE

An earlier version hashed the exact reference price, and that was a bug
waiting to happen: two duplicate scanner calls a second apart normally see
prices that differ in the sixth decimal, so the "duplicate" hashed to a
different id and was recorded as a second, independent sample. Sample
inflation from a retry is exactly what the unique constraint exists to
prevent.

Every numeric component is therefore snapped to a RELATIVE grid
(`SNAPSHOT_TOLERANCE`) before hashing, so quotes within a percent of each
other collapse to one identity while a genuine 30% move in the same minute
correctly stays a separate opportunity - which it is, in this market.

Bucketing has an unavoidable seam: two values a hair apart can still land
on opposite sides of a boundary. When that happens the id splits and the
duplicate is recorded twice, which is the OLD behaviour - the failure mode
degrades to what it was, it does not get worse, and it stays rare instead
of being the normal case.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections.abc import Mapping

# Enough hex to make a collision irrelevant at any volume this bot will
# ever see, short enough to read in a log line.
ID_LENGTH = 24

# Evaluation instants inside the same bucket are the same opportunity.
# Sub-minute jitter between the champion's evaluation and a challenger's
# is an artifact of execution order, not a difference in the opportunity,
# and letting it split the id would mean nothing ever paired.
DEFAULT_BUCKET_SECONDS = 60

# Relative grid for numeric snapshot components. 1% is far wider than the
# noise between two duplicate reads and far narrower than a move that
# makes the setup a different trade.
SNAPSHOT_TOLERANCE = 0.01

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _aware(moment: dt.datetime) -> dt.datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def time_bucket(observed_at: dt.datetime, bucket_seconds: int = DEFAULT_BUCKET_SECONDS) -> int:
    """The start of the bucket containing `observed_at`, as a unix second."""
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")
    seconds = int((_aware(observed_at) - _EPOCH).total_seconds())
    return seconds - (seconds % bucket_seconds)


def quantize(value: float, tolerance: float = SNAPSHOT_TOLERANCE) -> str:
    """Snap a number to a relative grid, as a short stable token.

    Relative rather than absolute because this has to work for a price of
    0.000000031 and a liquidity of 4,100,000 with one rule. Zero and
    negatives are handled explicitly rather than fed to log().
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if value == 0:
        return "0"
    step = math.log1p(tolerance)
    sign = "-" if value < 0 else ""
    return f"{sign}{math.floor(math.log(abs(value)) / step)}"


def _snapshot_token(snapshot: Mapping[str, object] | None) -> str:
    """Flatten a market snapshot into a deterministic string.

    Sorted by key so two callers building the same dict in a different
    order get the same id, which they otherwise would not.
    """
    if not snapshot:
        return ""
    parts = []
    for key in sorted(snapshot):
        value = snapshot[key]
        if value is None:
            parts.append(f"{key}=none")
        elif isinstance(value, bool):
            parts.append(f"{key}={int(value)}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={quantize(float(value))}")
        else:
            parts.append(f"{key}={value}")
    return "|".join(parts)


def opportunity_id(
    token_address: str,
    observed_at: dt.datetime,
    reference_price: float | None = None,
    *,
    event_id: str | None = None,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    snapshot: Mapping[str, object] | None = None,
) -> str:
    """Deterministic id for one evaluation of one token.

    `event_id` wins outright when the caller has one. Otherwise the id
    comes from the mint, the bucketed instant, and the coarse snapshot -
    `reference_price` is folded into that snapshot under the key `price`,
    so passing it positionally still works and still separates a token
    that moved from one that did not.

    Idempotency falls out of using only inputs the opportunity itself
    carries:

        a webhook delivered twice        same payload   -> same id
        a scanner cycle that overlaps    same bucket    -> same id
        a restart replaying a candidate  same inputs    -> same id
    """
    if event_id:
        # Namespaced so a source whose ids happen to look like a mint
        # address can never collide with the derived form.
        return hashlib.sha256(f"event|{event_id}".encode()).hexdigest()[:ID_LENGTH]

    merged: dict[str, object] = dict(snapshot or {})
    if reference_price is not None:
        merged.setdefault("price", reference_price)

    payload = f"{token_address}|{time_bucket(observed_at, bucket_seconds)}|{_snapshot_token(merged)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:ID_LENGTH]
