"""How did a security stop trading? Evidence from independent sources, combined.

Pure functions, no I/O.

Each source contributes evidence with a category:

* ``acquired_cash`` / ``acquired_stock`` / ``acquired_stock_and_cash`` - a merger
  record with its terms (Alpaca corporate actions).
* ``acquired`` - an acquisition without known terms (Wikidata: a new owner or
  parent organisation dated around the last trade).
* ``bankruptcy`` - a bankruptcy event (Wikidata) or a rename to a Nasdaq "Q"
  ticker, the bankruptcy convention (Alpaca).
* ``failure`` - a recorded collapse or failure (Wikidata), for example a bank
  seized by regulators.
* ``redeemed`` - the class was redeemed or retired for cash (SEC Form 25 under
  rule 12d2-2(a)(1)/(a)(2)); for a blank-check company (SPAC), the trust
  liquidation. The amount is not in the notice.
* ``delisted_no_reason`` - only the date the listing ended (supports the date,
  says nothing about the outcome).
* ``resumed`` - the same ticker trades again later, so the "end" may be a long
  trading halt or a reuse of the ticker by another company.
* ``fragment`` - a very short run of trades of a thinly traded instrument.

Rules (written down before looking at the results):

1. Evidence of two different kinds of exit (acquisition, redemption, failure)
   together: ``conflict``, no value.
2. Acquisition only: outcome acquired. Value = cash per share when a merger
   record gives it and no record shows a stock part; otherwise unknown (None)
   and flagged. A cash record next to a stock or stock-and-cash record of the
   same deal (Alpaca records Diamond Offshore's 2024 merger both ways) means
   cash plus shares: the cash alone would understate the value.
   Exception: a redemption together with a cash acquisition and nothing else.
   Both say holders were paid cash; Alpaca records a SPAC's trust liquidation
   as a "cash merger" at the trust value, which SEC files as a redemption.
   Outcome redeemed, value = that cash per share.
3. Redemption only: outcome redeemed, value unknown (None) and flagged.
4. Failure or bankruptcy only: value 0.0, the agreed default for a confirmed
   bankruptcy or failure. Equity holders sometimes recover (old Hertz did); the
   note says so.
5. Only ``resumed``/``fragment`` or no evidence at all: ``unresolved``, value
   None. An unknown exit is never treated as a total loss.

Status: ``confirmed`` when two independent sources support the outcome,
``single_source`` with one.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

ACQUISITION = {"acquired_cash", "acquired_stock", "acquired_stock_and_cash", "acquired"}
FAILURE = {"bankruptcy", "failure"}
REDEMPTION = {"redeemed"}
AMBIGUOUS = {"resumed", "fragment"}

#: Evidence dated on or after this day is out of scope and ignored.
SCOPE_END = date(2025, 1, 1)


@dataclass(frozen=True)
class Evidence:
    source: str
    category: str
    event_date: date | None
    detail: str
    reference: str | None = None
    cash_per_share: float | None = None


@dataclass(frozen=True)
class Outcome:
    outcome: str
    status: str
    assumed_value: float | None
    note: str


def resolve_outcome(evidence: Sequence[Evidence]) -> Outcome:
    acquisitions = [e for e in evidence if e.category in ACQUISITION]
    failures = [e for e in evidence if e.category in FAILURE]
    redemptions = [e for e in evidence if e.category in REDEMPTION]
    cash_paid = [e for e in acquisitions if e.category == "acquired_cash"]
    if redemptions and acquisitions and not failures and len(cash_paid) == len(acquisitions):
        cash = next((e for e in cash_paid if e.cash_per_share is not None), None)
        if cash is not None:
            return Outcome(
                "redeemed",
                "confirmed"
                if len({e.source for e in (*redemptions, *cash_paid)}) >= 2
                else "single_source",
                cash.cash_per_share,
                f"cash redemption, {cash.cash_per_share} per share ({cash.source} records it as a "
                "cash merger; SEC as a redemption)",
            )
    exits = [group for group in (acquisitions, redemptions, failures) if group]
    if len(exits) > 1:
        return Outcome(
            "conflict",
            "conflict",
            None,
            "sources disagree: "
            + "; ".join(f"{e.source}: {e.category}" for group in exits for e in group),
        )
    if acquisitions:
        sources = {e.source for e in acquisitions}
        status = "confirmed" if len(sources) >= 2 else "single_source"
        cash = next(
            (
                e
                for e in acquisitions
                if e.category == "acquired_cash" and e.cash_per_share is not None
            ),
            None,
        )
        kinds = {e.category for e in acquisitions}
        if "acquired_cash" in kinds and kinds & {"acquired_stock", "acquired_stock_and_cash"}:
            kinds = {"acquired_stock_and_cash"}
            cash = None
        outcome = next(
            (
                k
                for k in ("acquired_cash", "acquired_stock_and_cash", "acquired_stock")
                if k in kinds
            ),
            "acquired",
        )
        if cash is not None:
            return Outcome(
                outcome,
                status,
                cash.cash_per_share,
                f"cash per share {cash.cash_per_share} ({cash.source})",
            )
        note = (
            "paid in acquirer shares; value needs the acquirer's price"
            if outcome != "acquired"
            else "acquired, terms unknown: value not assumed"
        )
        return Outcome(outcome, status, None, note)
    if redemptions:
        sources = {e.source for e in redemptions}
        return Outcome(
            "redeemed",
            "confirmed" if len(sources) >= 2 else "single_source",
            None,
            "class redeemed or retired for cash (a SPAC returns its trust value); "
            "amount not in the evidence: value not assumed",
        )
    if failures:
        sources = {e.source for e in failures}
        status = "confirmed" if len(sources) >= 2 else "single_source"
        outcome = "failure" if any(e.category == "failure" for e in failures) else "bankruptcy"
        return Outcome(
            outcome,
            status,
            0.0,
            "total loss assumed (agreed default for a confirmed bankruptcy or failure; "
            "equity holders occasionally recover)",
        )
    if any(e.category in AMBIGUOUS for e in evidence):
        return Outcome(
            "resumed_or_fragment",
            "unresolved",
            None,
            "the ticker trades again later or the run is very short: not treated as an exit loss",
        )
    dated = [e for e in evidence if e.category == "delisted_no_reason"]
    note = "no source explains the end" + (
        f"; listing end dated {dated[0].event_date} ({dated[0].source})" if dated else ""
    )
    return Outcome("unknown", "unresolved", None, note + ": value not assumed")


# =============================================================================
# Wikidata
# =============================================================================

US_EXCHANGE_WORDS = ("new york stock exchange", "nasdaq", "nyse", "american stock exchange", "cboe")


def _day(text: str) -> date | None:
    try:
        return date.fromisoformat(text[:10]) if text and text[0].isdigit() else None
    except ValueError:
        return None


def _pairs(text: str) -> list[tuple[str, date | None]]:
    out = []
    for part in text.split("|"):
        if not part.strip("@ "):
            continue
        label, _, when = part.rpartition("@")
        out.append((label or part, _day(when)))
    return out


def _value(row: dict, key: str) -> str:
    return row.get(key, {}).get("value", "")


def _between(day: date | None, low: date, high: date) -> bool:
    return day is not None and low <= day <= high and day < SCOPE_END


def wikidata_matches(
    bindings: Iterable[dict], first: date, last: date, *, ticker_shared: bool = False
) -> list[dict]:
    """Rows describing a US listing under the ticker that overlaps the period ``first``..``last``.

    Pass the period during which the security used this ticker. When the ticker
    belonged to more than one security (``ticker_shared``), a row without any
    listing date cannot be placed in time and is left out.
    """
    rows = []
    for b in bindings:
        if not any(word in _value(b, "exchLabel").lower() for word in US_EXCHANGE_WORDS):
            continue
        start, end = _day(_value(b, "tStart")), _day(_value(b, "tEnd"))
        if ticker_shared and start is None and end is None:
            continue
        if start is not None and (start >= SCOPE_END or start > last + timedelta(days=30)):
            continue
        if end is not None and end < first - timedelta(days=30):
            continue
        rows.append(b)
    return rows


def wikidata_evidence(rows: Iterable[dict], last: date) -> list[Evidence]:
    """Exit evidence from matching Wikidata rows, for a security whose last trade is ``last``."""
    evidence: list[Evidence] = []
    for b in rows:
        item, label = _value(b, "item").rsplit("/", 1)[-1], _value(b, "itemLabel")
        for kind in ("owners", "parents"):
            for owner, when in _pairs(_value(b, kind)):
                if _between(when, last - timedelta(days=250), last + timedelta(days=30)):
                    evidence.append(
                        Evidence(
                            "wikidata",
                            "acquired",
                            when,
                            f"{label}: {'owned by' if kind == 'owners' else 'parent'} {owner} "
                            f"from {when}",
                            item,
                        )
                    )
        for event, when in _pairs(_value(b, "events")):
            low = event.lower()
            if "bankrupt" in low and _between(
                when, last - timedelta(days=400), last + timedelta(days=60)
            ):
                evidence.append(
                    Evidence("wikidata", "bankruptcy", when, f"{label}: {event} on {when}", item)
                )
            elif ("collapse" in low or "failure" in low) and _between(
                when, last - timedelta(days=30), last + timedelta(days=60)
            ):
                evidence.append(
                    Evidence("wikidata", "failure", when, f"{label}: {event} on {when}", item)
                )
        for key, what in (("tEnd", "listing ended"), ("dissolvedAt", "dissolved")):
            when = _day(_value(b, key))
            if _between(when, last - timedelta(days=10), last + timedelta(days=30)):
                evidence.append(
                    Evidence(
                        "wikidata", "delisted_no_reason", when, f"{label}: {what} {when}", item
                    )
                )
    return evidence
