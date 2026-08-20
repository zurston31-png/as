"""A stable identity for "the same chance to trade".

Two observations should pair only when they are genuinely about the same
opportunity. Pairing on "timestamps are close" would join a champion's
look at token A with a challenger's look at token B thirty seconds later
and call it a controlled comparison.

WHAT MAKES IT STABLE

The id is a hash of the token, the evaluation instant truncated to the
second, and the reference price. All three come from the opportunity
itself, never from a database row id, which gives idempotency for free:

    a webhook delivered twice        same payload, same second -> same id
    a scanner cycle that overlaps    same token, same tick     -> same id
    a restart replaying a candidate  same inputs               -> same id

A genuinely new look at the same token a minute later has a different
second and usually a different price, so it correctly becomes a new
opportunity rather than being swallowed by the old one.

Price is included deliberately. Two evaluations in the same second at
materially different prices are different opportunities - the market
moved between them - and collapsing those would pair a decision made at
one price with an outcome measured from another.
"""
from __future__ import annotations

import datetime as dt
import hashlib

# Enough hex to make a collision irrelevant at any volume this bot will
# ever see, short enough to read in a log line.
ID_LENGTH = 24


def opportunity_id(
    token_address: str,
    observed_at: dt.datetime,
    reference_price: float | None,
) -> str:
    """Deterministic id for one evaluation of one token.

    Truncated to the second: sub-second jitter between the champion's
    evaluation and a challenger's is an artifact of execution order, not a
    difference in the opportunity, and letting it split the id would mean
    nothing ever paired.
    """
    moment = observed_at if observed_at.tzinfo else observed_at.replace(tzinfo=dt.timezone.utc)
    stamp = moment.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    # Price is rounded to 12 significant places rather than used raw, so a
    # float that round-trips through the database with a different last
    # bit does not produce a different id.
    price = f"{reference_price:.12g}" if reference_price is not None else "none"
    payload = f"{token_address}|{stamp}|{price}"
    return hashlib.sha256(payload.encode()).hexdigest()[:ID_LENGTH]
