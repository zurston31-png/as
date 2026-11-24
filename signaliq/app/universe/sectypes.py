"""Security type classification from independent evidence. Pure functions, no I/O.

Sources, each recorded as evidence with its own detail:

* ``sec_cover`` - the class title the issuer itself gives the ticker on the cover page of
  its 10-K, 10-Q, 8-K, 20-F or 40-F ("Class A common stock", "American depositary shares
  ...", "Common units representing limited partner interests"), from filings made while
  the security traded under that ticker (2019 on). The issuer's own, dated description.
* ``sec_text`` - a hand-read SEC passage (``app.universe.curated``), re-verified each run.
* ``openfigi`` - Bloomberg's OpenFIGI mapping of the ticker (securityType and
  securityType2). OpenFIGI describes the ticker's holder *today*, so its answer is used
  only for the security that used the ticker last, and only when its company name agrees
  with a name known for the security's own period (Wikidata, SEC cover pages).
* ``sec_form25`` - the class of security named in the SEC delisting notice
  (Form 25 / 25-NSE) filed when the security stopped trading (``app.universe.sec``).
* ``symbol_suffix`` - Nasdaq's fifth-letter convention for five-letter tickers:
  W warrant, R right, U unit, X mutual fund.
* ``wikidata`` - "instance of" classes of the company item listed under the ticker during
  the security's period. Wikidata describes the *company*, not the listed security: a
  public company, a bank, a real estate investment trust are all ``company``, which is
  consistent with common stock, ADRs, partnership units, preferred shares, units,
  warrants and notes alike; an exchange-traded fund, an exchange-traded note or a
  closed-end fund is a security type of its own.

Explicit rules (``CLASSIFICATION_RULES``, printed in every report):

1. **Common stock**: common or ordinary shares, including REITs (a REIT is a tax status;
   its listed shares are common shares or shares of beneficial interest), tracking stock,
   and blank-check companies (SPACs). Only common stock can join the universe.
2. **ADR**: American depositary shares or receipts, New York registry shares. Excluded by
   design (the universe is US common stock), however the company is described.
3. **ETF / fund / ETN**: exchange-traded funds, closed-end and mutual funds, exchange-traded
   notes (debt of the issuing bank). Never common, whatever the issuer's name.
4. **Warrant, right, unit, preferred, partnership, debt**: from the class title, OpenFIGI
   or the fifth letter. A SPAC's units and warrants are not its common shares.
5. **SPACs** are common stock by type; they are kept out of the universe by a separate,
   point-in-time rule (SIC 6770 on file by the decision date, ``evidence.blank_check_on``),
   because the same shares become an operating company's at the merger.
6. **Disagreement**: when sources name different types, only the issuer's own SEC
   description settles it: a hand-read SEC passage (``sec_text``) first, then the
   cover-page class title (``sec_cover``). The type is then ``resolved`` and the outvoted
   evidence is kept. Without either, or when the issuer's own descriptions disagree (an
   ADR that became an ordinary listing), the type stays in ``conflict``: never a guess.

Status: ``confirmed`` when two independent sources agree (the SEC sources count once; a
Wikidata ``company`` confirms only common stock), ``single_source`` with one,
``resolved`` as above, ``conflict``, or ``unknown`` with none. Confirmed, single-source and
resolved common stocks may join the universe.
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
    "debt",
    "etn",
    "unknown",
)
#: Wikidata's vote for an operating company (any kind of listed security but a fund).
COMPANY = "company"
#: Types a company (``COMPANY``) can have issued; funds and notes it cannot.
COMPANY_TYPES = frozenset(
    {"common", "adr", "partnership", "preferred", "unit", "warrant", "right", "debt"}
)
#: Statuses whose type is backed by evidence and may be used for membership.
USABLE_STATUSES = frozenset({"confirmed", "single_source", "resolved"})
#: The issuer's own SEC descriptions, which alone may settle a disagreement (lower first).
DECIDING_SOURCES = {"sec_text": 0, "sec_cover": 1}
CLASSIFICATION_RULES = (
    "common stock (incl. REITs, tracking stock, SPAC shares): the only type the universe takes",
    "ADR (depositary shares/receipts, NY registry shares): excluded by design",
    "ETF, closed-end/mutual fund, ETN: never common, whatever the issuer's name",
    "warrant, right, unit, preferred, partnership, debt: class title, OpenFIGI or fifth letter",
    "SPACs: common by type, excluded by the point-in-time SIC 6770 rule, not by type",
    "Wikidata describes the company: 'company' (incl. REIT, bank) fits any non-fund type",
    "OpenFIGI describes today's holder: used only if its name matches the security's period",
    "disagreement: settled only by the issuer's SEC text (sec_text, then sec_cover); else conflict",
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


_WIKIDATA_COMPANY = (
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
    "exchange-traded note": "etn",
    "exchange traded note": "etn",
    "closed-end fund": "fund",
    "mutual fund": "fund",
    "investment trust": "fund",
}


def _is_reit(label: str) -> bool:
    return "real estate investment trust" in label or label == "reit"


def wikidata_type(instance_labels: Iterable[str]) -> str | None:
    """``company`` for an operating company (a REIT included), a fund type for a fund."""
    labels = [label.strip().lower() for label in instance_labels if label.strip()]
    for label in labels:
        if _is_reit(label):
            continue  # "real estate investment trust" is not an "investment trust" fund
        for key, value in _WIKIDATA_OTHER.items():
            if key in label:
                return value
    if any(
        _is_reit(label) or label in _WIKIDATA_COMPANY or label.endswith(" company")
        for label in labels
    ):
        return COMPANY
    return None


def cover_class_type(title: str | None) -> str | None:
    """Our type for the class title an issuer gives on its cover page; None if unclear."""
    text = " ".join((title or "").lower().split())
    if not text:
        return None
    if "exchange traded note" in text or "exchange-traded note" in text or " etn" in f" {text}":
        return "etn"
    if "partner" in text:
        return "partnership"
    if text.startswith("unit") or (" units" in f" {text}" and "warrant" in text):
        return "unit"
    if "warrant" in text:
        return "warrant"
    if text.startswith("right"):
        return "right"
    if "preferred" in text or "preference" in text:
        return "preferred"
    if "depositary" in text or "depository" in text or "registry share" in text:
        return "adr"
    if any(w in text for w in ("% notes", "notes due", "debentures", "% senior")):
        return "debt"
    if any(
        w in text
        for w in (
            "common stock",
            "common share",
            "ordinary share",
            "beneficial interest",
            "capital stock",
        )
    ):
        return "common"
    return None


@dataclass(frozen=True)
class TypeVote:
    source: str
    security_type: str


def source_family(source: str) -> str:
    """Sources from the same origin count once: every ``sec_*`` source is SEC."""
    return "sec" if source.startswith("sec") else source


def resolve_type(votes: Iterable[TypeVote]) -> tuple[str, str]:
    """(type, status) from independent votes under the rules above. ``declared`` votes
    do not count."""
    counted = [v for v in votes if v.source != "declared" and v.security_type != "unknown"]
    if not counted:
        return "unknown", "unknown"
    company = [v for v in counted if v.security_type == COMPANY]
    specific = [v for v in counted if v.security_type != COMPANY]
    if not specific:
        return "common", "single_source"  # an operating company, nothing else known
    kinds = {v.security_type for v in specific}
    if company and not kinds <= COMPANY_TYPES:
        # A company is not a fund: its vote disagrees, as a vote for common stock.
        specific += [TypeVote(v.source, "common") for v in company]
        company = []
        kinds = {v.security_type for v in specific}
    if len(kinds) == 1:
        kind = kinds.pop()
        families = {source_family(v.source) for v in specific}
        if company and kind == "common":
            families.add("wikidata")
        return kind, "confirmed" if len(families) >= 2 else "single_source"
    deciding = [v for v in specific if v.source in DECIDING_SOURCES]
    if deciding:
        best = min(DECIDING_SOURCES[v.source] for v in deciding)
        top = {v.security_type for v in deciding if DECIDING_SOURCES[v.source] == best}
        if len(top) == 1:
            return top.pop(), "resolved"
    return "unknown", "conflict"


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
