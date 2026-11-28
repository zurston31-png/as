"""Two proposed rules, off by default, switched on only to measure their impact.

Neither rule is adopted. Each is enabled by an environment variable for a measurement run
into its own database; with both off (the default), every result is exactly Stage 1.1's
(proved by the neutral run of Phase 1).

* ``SIGNALIQ_RULE_DATED_TYPES=1`` - **dated security types.** A security whose own cover
  pages, read at every dated filing in its issuer evidence, give more than one class type
  (CRH: American depositary shares on its 20-F of 2020, ordinary shares later) takes, at
  each decision, the type of the latest such cover page filed on or before the decision.
  If cover pages filed after that one (by the decision) were not read, the type is unknown:
  never guessed. Before its first such cover page, or for a security whose cover pages
  agree, the type is the undated one, as now. Pages are only read from the cache.
* ``SIGNALIQ_RULE_COREGISTRANT=1`` - **co-registrants.** When two companies' cover pages
  claim one ticker over the same period and they share filings (a parent and a subsidiary
  filing one document), the company is chosen by the shared cover page itself: first the
  one whose own row title lists the ticker ("PG&E Corporation common stock ... PCG"), then
  the one named first among the registrants. If the cover page decides neither way, the
  current rule stands (the company with more cover pages).
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from app.universe import sec_history as sh
from app.universe import sectypes

DATED_TYPES_VARIABLE = "SIGNALIQ_RULE_DATED_TYPES"
COREGISTRANT_VARIABLE = "SIGNALIQ_RULE_COREGISTRANT"
NOT_READ = "not in the cache: not read"


def dated_types_enabled() -> bool:
    return os.environ.get(DATED_TYPES_VARIABLE, "").strip() == "1"


def coregistrant_enabled() -> bool:
    return os.environ.get(COREGISTRANT_VARIABLE, "").strip() == "1"


def enabled_rules() -> list[str]:
    return [
        name
        for name, on in (
            ("dated security types", dated_types_enabled()),
            ("co-registrant choice", coregistrant_enabled()),
        )
        if on
    ]


# =============================================================================
# Dated security types
# =============================================================================


@dataclass(frozen=True)
class DatedClass:
    """The class title a company's cover page gives a ticker, on its filing date."""

    filed: date
    symbol: str
    cik: str
    form: str
    title: str | None
    kind: str | None  # None: not one recognised class, or not conclusive (see why)
    why: str
    url: str


def cover_classes(client, points_by_sid) -> dict[int, list[DatedClass]]:
    """Every cover page in each security's issuer evidence, with the class it gives the ticker.

    The same reading as the undated rule (``evidence.resolve_types``): the row of the ticker
    under "Trading Symbol(s)", one class only, and on a 20-F only an ADS title is conclusive.
    """
    forms: dict[str, dict[str, str]] = {}
    out: dict[int, list[DatedClass]] = {}
    for sid, points in points_by_sid.items():
        dated = []
        for p in sorted((p for p in points if p.how == "cover_page"), key=lambda p: p.filed):
            if p.cik not in forms:
                company = sh.submissions(client, p.cik)
                forms[p.cik] = {
                    acc.replace("-", ""): form
                    for form, _, acc in (company.filings if company else ())
                }
            form = forms[p.cik].get(p.url.split("/")[-2], "")
            text = client.cover_text(p.url)
            if text is None:
                # Never requested here: a page not in the cache is recorded as not read.
                dated.append(
                    DatedClass(p.filed, p.symbol, p.cik, form, None, None, NOT_READ, p.url)
                )
                continue
            if not sh.cover_page_symbol(text, p.symbol):
                dated.append(
                    DatedClass(
                        p.filed,
                        p.symbol,
                        p.cik,
                        form,
                        None,
                        None,
                        "ticker not listed on this page",
                        p.url,
                    )
                )
                continue
            title = sh.cover_page_class(text, p.symbol)
            kind = sectypes.cover_class_type(title)
            why = "" if kind else "not one recognised class"
            if title is None:
                why = "the ticker is listed on more than one row"
            elif kind and kind != "adr" and form.startswith("20-F"):
                why, kind = (
                    f"a {form} names the registered shares, not whether they trade as ADSs",
                    None,
                )
            dated.append(DatedClass(p.filed, p.symbol, p.cik, form, title, kind, why, p.url))
        out[sid] = dated
    return out


def type_changes(classes: dict[int, list[DatedClass]]) -> dict[int, list[DatedClass]]:
    """Securities whose conclusive cover pages give more than one type, with all their pages
    (conclusive, inconclusive and not read), in filing order."""
    return {
        sid: dated for sid, dated in classes.items() if len({c.kind for c in dated if c.kind}) > 1
    }


def dated_type(pages: list[DatedClass], day: date) -> tuple[str | None, str]:
    """The type on ``day`` from cover pages filed on or before it, and why.

    The latest conclusive page decides. If cover pages filed after it (and by ``day``) were
    not read, the class may have changed unseen, so the type is ``unknown`` (never guessed).
    None before the first conclusive page: the undated type applies.
    """
    latest, unread = None, 0
    for c in pages:
        if c.filed > day:
            break
        if c.kind:
            latest, unread = c, 0
        elif c.why == NOT_READ:
            unread += 1
    if latest is None:
        return None, "no conclusive cover page yet"
    if unread:
        return "unknown", f"{unread} cover pages after {latest.filed} not read"
    return latest.kind, f"{latest.form} cover page of {latest.filed}"


def type_on(pages: list[DatedClass], day: date) -> str | None:
    return dated_type(pages, day)[0]


def type_runs(pages: list[DatedClass]) -> list[tuple[str, DatedClass, DatedClass, int]]:
    """Consecutive conclusive pages of one type: (type, first page, last page, number)."""
    runs: list[list[DatedClass]] = []
    for c in (c for c in pages if c.kind):
        if runs and runs[-1][0].kind == c.kind:
            runs[-1].append(c)
        else:
            runs.append([c])
    return [(r[0].kind, r[0], r[-1], len(r)) for r in runs]


# =============================================================================
# Co-registrants
# =============================================================================

_DROP = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC",
    "LLC", "LP", "L", "P", "NV", "N", "V", "SA", "AG", "THE", "DE", "MD", "NY",
}  # fmt: skip


def name_tokens(name: str) -> list[str]:
    """A company name as comparable words: '&' read as AND, legal forms and states dropped."""
    text = re.sub(r"/[A-Z]{2,3}/?", " ", name.upper()).replace("&", " AND ")
    return [t for t in re.split(r"[^A-Z0-9]+", text) if t and t not in _DROP]


def _text_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", text.upper().replace("&", " AND ")) if t]


def _positions(tokens: list[str], name: list[str], longer: list[str] | None) -> list[int]:
    """Where ``name`` occurs in ``tokens``, leaving out occurrences that are the start of
    ``longer`` (the other company's name, when this one is a prefix of it)."""
    found = []
    for i in range(len(tokens) - len(name) + 1):
        if tokens[i : i + len(name)] != name:
            continue
        if longer and len(longer) > len(name) and tokens[i : i + len(longer)] == longer:
            continue
        found.append(i)
    return found


REGISTRANT_BLOCK = 600
#: The cover's own heading lines that come just before the registrants' names.
_COVER_HEADINGS = re.compile(
    r"earliest\s+event\s+reported\)?:?\s*[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}"
    r"|fiscal\s+year\s+ended:?\s*[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}"
    r"|(?:quarterly\s+)?period\s+ended:?\s*[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}"
    r"|commission\s+file\s+numbers?:?\s*[\d-]+",
    re.I,
)


def registrant_block(text: str) -> str:
    """The registrants' names as printed on the cover: the text before the first "Exact name
    of registrant" heading and after the cover's last heading line before it (the date of
    report, the fiscal period or the commission file number). Hidden inline-XBRL values
    that precede the cover (entity names in any order) are left out. Empty if not found."""
    found = re.search(r"exact\s+name\s+of\s+(the\s+)?registrants?", text, re.I)
    if not found:
        return ""
    block = text[max(0, found.start() - REGISTRANT_BLOCK) : found.start()]
    headings = list(_COVER_HEADINGS.finditer(block))
    return block[headings[-1].end() :] if headings else ""


@dataclass(frozen=True)
class CoregistrantChoice:
    symbol: str
    kept_cik: str
    other_cik: str
    changed: bool
    reason: str
    url: str


def _cached_cover(session, url: str) -> str | None:
    """A cover page from the cache only, never requested and never recorded as missing, so
    the rule cannot stop or alter the issuer step's own completeness check."""
    from app.universe import sec, sec_cache

    text = sec_cache.lookup(session, sec.COVER_PREFIX + sec.cache_key(url))
    if text is not None:
        return text
    whole = sec_cache.lookup(session, sec.cache_key(url))
    return None if whole is None else sec.cover_snippet(whole)


def _vote(text: str, symbol: str, a: list[str], b: list[str]) -> tuple[str | None, str]:
    """One shared cover page's answer: "a", "b" or None, and how it was read."""
    title = sh.cover_page_class(text, symbol) if sh.cover_page_symbol(text, symbol) else None
    row = _text_tokens(title or "")
    own_a = bool(a) and bool(_positions(row, a, b))
    own_b = bool(b) and bool(_positions(row, b, a))
    if own_a != own_b:
        return ("b" if own_b else "a"), f"row '{title}' names it"
    tokens = _text_tokens(registrant_block(text))
    pa, pb = _positions(tokens, a, b), _positions(tokens, b, a)
    if pa and pb and pa[0] != pb[0]:
        return ("b" if pb[0] < pa[0] else "a"), "named first among the registrants"
    return None, f"row '{title}' names neither; registrant names found: {bool(pa)}, {bool(pb)}"


def choose(client, symbol: str, kept: list, other: list) -> tuple[bool, CoregistrantChoice]:
    """Whether ``other`` should replace ``kept`` (both lists of issuer points), and why.

    Every shared filing whose cover page is in the cache votes; the answer must be
    unanimous, otherwise the current rule stands.
    """

    def accession(p):
        return p.url.split("/")[-2]

    mine = {accession(p): p for p in kept}
    theirs = {accession(p): p for p in other}
    shared = sorted(set(mine) & set(theirs), key=lambda a: mine[a].filed)

    def keep(reason, url=""):
        return False, CoregistrantChoice(symbol, kept[0].cik, other[0].cik, False, reason, url)

    if not shared:
        return keep("no shared filing (not co-registrants of one document): current rule")
    votes = []
    unread = 0
    for acc in shared:
        mp, tp = mine[acc], theirs[acc]
        text = _cached_cover(client.session, mp.url) or _cached_cover(client.session, tp.url)
        if text is None:
            unread += 1
            continue
        # Each company by the name it filed under then (United Continental Holdings became
        # United Airlines Holdings during the overlap).
        votes.append((_vote(text, symbol, name_tokens(mp.name), name_tokens(tp.name)), mp.url))
    decided = [(v, how, url) for (v, how), url in votes if v]
    sides = {v for v, _, _ in decided}
    summary = (
        f"{len(shared)} shared filings, {len(votes)} read ({unread} not in the cache), "
        f"{len(decided)} decided"
    )
    if not decided:
        detail = votes[-1][0][1] if votes else "none read"
        return keep(
            f"undecided: {summary}; last read: {detail}: current rule",
            votes[-1][1] if votes else "",
        )
    if len(sides) > 1:
        counts = {s: sum(1 for v, _, _ in decided if v == s) for s in sides}
        return keep(
            f"undecided: shared cover pages disagree ({summary}; kept {counts.get('a', 0)}, "
            f"other {counts.get('b', 0)}): current rule",
            decided[-1][2],
        )
    side, how, url = decided[-1]
    replace = side == "b"
    winner, loser = (other, kept) if replace else (kept, other)
    reason = f"{'the other' if replace else 'the kept'} company: {how} ({summary}, unanimous)"
    return replace, CoregistrantChoice(symbol, winner[0].cik, loser[0].cik, replace, reason, url)


#: Every choice made by the co-registrant rule in this process, for the report.
CHOICES: dict[int, list[CoregistrantChoice]] = defaultdict(list)
