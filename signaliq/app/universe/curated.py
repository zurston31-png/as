"""Corrections entered by hand, each with its source and a check against prices.

Used only where no automated source has the fact. Every entry is reported, and
each is checked against market prices when the batch runs; an entry that the
prices contradict is flagged, not silently applied.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CuratedSpinOff:
    parent_symbol: str
    ex_date: date
    #: (child ticker, child shares received per parent share)
    children: tuple[tuple[str, float], ...]
    reference: str


CURATED_SPIN_OFFS: tuple[CuratedSpinOff, ...] = (
    CuratedSpinOff(
        parent_symbol="UTX",
        ex_date=date(2020, 4, 3),
        children=(("CARR", 1.0), ("OTIS", 0.5)),
        reference=(
            "United Technologies' separation of Carrier and Otis, completed 2020-04-03 with "
            "the Raytheon merger: one Carrier share for each UTC share and one Otis share for "
            "every two UTC shares, as announced by the companies at completion. Entered by hand "
            "because Alpaca has no spin-off record for it and records the day as a 1-for-2 "
            "reverse split instead. Not yet checked against the SEC filing (SEC access pending "
            "a contact address)."
        ),
    ),
)
