"""Security type classification from independent evidence. Pure functions, no I/O.

Sources, each recorded as evidence with its own detail:

* ``openfigi`` - Bloomberg's OpenFIGI mapping of the ticker (securityType and
  securityType2). OpenFIGI describes the most recent holder of a ticker, so its
  answer is only attached to the security that used the ticker last.
* ``wikidata`` - "instance of" classes of the company item that listed under the
  ticker during the security's trading period (public company, bank,
  exchange-traded fund, ...).
* ``symbol_suffix`` - Nasdaq's fifth-letter convention for five-letter tickers:
  W warrant, R right, U unit, X mutual fund.

A type is ``confirmed`` when two independent sources agree, ``single_source``
with one, ``conflict`` when sources disagree, and ``unknown`` with none. Only
confirmed or single-source ``common`` securities may join the universe.
"""

from collections.abc import Iterable
from dataclasses import dataclass

TYPES = (
    "common",
    "etf",
    "warrant",
    "right",
    "unit",
    "preferred",
    "fund",
    "adr",
    "partnership",
    "unknown",
)
#: Types the common-stock universe accepts.
UNIVERSE_TYPES = frozenset({"common"})

_FIGI_TYPE2 = {
    "common stock": "common",
    "warrant": "warrant",
    "right": "right",
    "unit": "unit",
    "preference": "preferred",
    "preferred": "preferred",
    "depositary receipt": "adr",
    "mutual fund": "fund",
    "closed-end fund": "fund",
    "open-end fund": "fund",
    "ltd part": "partnership",
    "partnership shs": "partnership",
}
_FIGI_TYPE = {
    "etp": "etf",
    "equity wrt": "warrant",
    "right": "right",
    "unit": "unit",
    "adr": "adr",
    "gdr": "adr",
    "preference": "preferred",
    "pfd": "preferred",
    "closed-end fund": "fund",
    "open-end fund": "fund",
    "mlp": "partnership",
    "ltd part": "partnership",
    "reit": "common",
    "common stock": "common",
    "tracking stk": "common",
    "dutch cert": "adr",
    "ny reg shrs": "adr",
}


def figi_type(security_type: str | None, security_type2: str | None) -> str:
    """Our type for an OpenFIGI record. securityType wins for ETPs, warrants, ADRs and REITs."""
    first = (security_type or "").strip().lower()
    second = (security_type2 or "").strip().lower()
    if first in ("etp", "equity wrt", "adr", "gdr", "reit", "mlp", "pfd", "preference"):
        return _FIGI_TYPE[first]
    if second in _FIGI_TYPE2:
        return _FIGI_TYPE2[second]
    return _FIGI_TYPE.get(first, "unknown")


def suffix_type(symbol: str) -> str | None:
    """Type implied by Nasdaq's fifth-letter convention (five-letter tickers only)."""
    if len(symbol) != 5 or not symbol.isalpha():
        return None
    return {"W": "warrant", "R": "right", "U": "unit", "X": "fund"}.get(symbol[-1])


_WIKIDATA_COMMON = (
    "public company",
    "business",
    "enterprise",
    "bank",
    "bank holding company",
    "holding company",
    "financial group",
    "company",
    "corporation",
    "conglomerate",
)
_WIKIDATA_OTHER = {
    "exchange-traded fund": "etf",
    "closed-end fund": "fund",
    "mutual fund": "fund",
    "investment trust": "fund",
}


def wikidata_type(instance_labels: Iterable[str]) -> str | None:
    labels = [label.strip().lower() for label in instance_labels if label.strip()]
    for label in labels:
        for key, value in _WIKIDATA_OTHER.items():
            if key in label:
                return value
    if any(label in _WIKIDATA_COMMON or label.endswith(" company") for label in labels):
        return "common"
    return None


@dataclass(frozen=True)
class TypeVote:
    source: str
    security_type: str


def resolve_type(votes: Iterable[TypeVote]) -> tuple[str, str]:
    """(type, status) from independent votes. ``declared`` votes do not count."""
    counted = [v for v in votes if v.source != "declared" and v.security_type != "unknown"]
    kinds = {v.security_type for v in counted}
    if not kinds:
        return "unknown", "unknown"
    if len(kinds) > 1:
        return "unknown", "conflict"
    kind = kinds.pop()
    sources = {v.source for v in counted}
    return kind, "confirmed" if len(sources) >= 2 else "single_source"


_NAME_NOISE = frozenset(
    {
        "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC",
        "LLC", "LP", "THE", "HOLDINGS", "HOLDING", "GROUP", "CLASS", "CL", "SHS", "SHARES",
        "COM", "COMMON", "NEW", "DEL", "NV", "SA", "AG", "TRUST", "ETF", "FUND", "OF", "AND",
        "US", "USA",
    }
)  # fmt: skip


def name_tokens(name: str) -> set[str]:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in name.upper())
    return {word for word in cleaned.split() if word not in _NAME_NOISE and len(word) > 1}


def names_agree(first: str, second: str) -> bool:
    """Whether two company names share a meaningful word.

    ``META PLATFORMS INC`` and ``Meta Platforms`` agree.

    Abbreviations defeat it (``INTL BUSINESS MACHINES`` / ``IBM``); a disagreement
    is therefore reported, never used to pick a type.
    """
    return bool(name_tokens(first) & name_tokens(second))
