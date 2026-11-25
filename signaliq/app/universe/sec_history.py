"""SEC history: full-text search, company filing history and point-in-time industry codes.

Read-only, through the same cache as ``app.universe.sec`` (source ``sec``).

* **Full-text search** (efts.sec.gov, filings from 2001 on): which companies wrote a
  ticker in their filings while the security traded, for example "(NASDAQ: MDGS)" in a
  press release. A hit proposes a company; it is not proof on its own.
* **Submissions** (data.sec.gov): a company's name, former names with dates, and its
  filings (form, date, accession number).
* **Filing headers**: the SIC code assigned to the company *when that filing was made*.
  SEC's company record only shows today's SIC, which is wrong for a blank-check company
  (SIC 6770) that later merged with an operating business. Headers give the SIC as of
  each date, so a membership decision only uses what was on file by then.
"""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from app.universe import sec as _sec
from app.universe.sec import ARCHIVES, SCOPE_END, SUBMISSIONS_PAGE_URL, CachedSec, SecUnavailable
from app.universe.sectypes import _NAME_NOISE

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
BLANK_CHECK_SIC = "6770"


# =============================================================================
# Parsing (pure)
# =============================================================================


@dataclass(frozen=True)
class TextHit:
    cik: str
    name: str
    form: str
    filed: date
    adsh: str
    filename: str
    sic: str | None
    file_type: str = ""

    @property
    def url(self) -> str:
        folder = self.adsh.replace("-", "")
        return f"{ARCHIVES}/Archives/edgar/data/{int(self.cik)}/{folder}/{self.filename}"


def parse_full_text(text: str) -> list[TextHit]:
    """Hits of a full-text search, one per (company, document); filings from 2025 on dropped."""
    try:
        hits = json.loads(text)["hits"]["hits"]
    except (ValueError, KeyError, TypeError):
        return []
    out = []
    for hit in hits:
        source = hit.get("_source", {})
        adsh, _, filename = hit.get("_id", "").partition(":")
        try:
            filed = date.fromisoformat(source.get("file_date", ""))
        except ValueError:
            continue
        if filed >= SCOPE_END:
            continue
        names = source.get("display_names") or []
        sics = source.get("sics") or []
        for i, cik in enumerate(source.get("ciks") or []):
            name = re.sub(r"\s+\(CIK \d+\)$", "", names[i]).strip() if i < len(names) else ""
            out.append(
                TextHit(
                    cik.zfill(10),
                    re.sub(r"\s+\([A-Z0-9.\-, ]+\)$", "", name).strip(),
                    source.get("form", ""),
                    filed,
                    adsh,
                    filename,
                    sics[i] if i < len(sics) else None,
                    source.get("file_type") or "",
                )
            )
    return out


@dataclass(frozen=True)
class FormerName:
    name: str
    start: date | None
    end: date | None


@dataclass(frozen=True)
class Submissions:
    cik: str
    name: str
    sic: str | None
    tickers: tuple[str, ...]
    former_names: tuple[FormerName, ...]
    filings: tuple[tuple[str, date, str], ...]  # (form, filed, accession), newest first


def _day(text: str | None) -> date | None:
    try:
        return date.fromisoformat((text or "")[:10])
    except ValueError:
        return None


#: Older submission pages that end before this date are not read: nothing in scope
#: (2016-2024, with a year of SIC history before a security's first trade) needs them.
OLDEST_NEEDED = date(2014, 1, 1)


def _block_filings(block: dict) -> list[tuple[str, date, str]]:
    return [
        (form, filed, accession)
        for form, day, accession in zip(
            block.get("form", []),
            block.get("filingDate", []),
            block.get("accessionNumber", []),
            strict=False,
        )
        if (filed := _day(day)) is not None and filed < SCOPE_END
    ]


def older_pages(text: str) -> list[str]:
    """Names of the company's older submission pages that can hold filings in scope."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    files = (data.get("filings") or {}).get("files") or [] if isinstance(data, dict) else []
    return [
        f["name"]
        for f in files
        if f.get("name") and not ((to := _day(f.get("filingTo"))) and to < OLDEST_NEEDED)
    ]


def parse_submissions(text: str, pages: Sequence[str] = ()) -> Submissions | None:
    """A company's record from its submissions file and the texts of its older pages.

    SEC's file lists only the most recent filings (about a thousand); a heavy filer's
    2016-2024 filings are on the older pages, so without them the company looks as if it
    had filed nothing then (Toronto-Dominion Bank).
    """
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or "cik" not in data:
        return None
    rows = _block_filings(data.get("filings", {}).get("recent", {}))
    for page in pages:
        try:
            rows += _block_filings(json.loads(page))
        except (ValueError, AttributeError):
            return None
    seen: set[str] = set()
    filings = tuple(
        row
        for row in sorted(rows, key=lambda r: (r[1], r[2]), reverse=True)
        if not (row[2] in seen or seen.add(row[2]))
    )
    return Submissions(
        str(data["cik"]).zfill(10),
        data.get("name", ""),
        data.get("sic") or None,
        tuple(data.get("tickers") or ()),
        tuple(
            FormerName(f.get("name", ""), _day(f.get("from")), _day(f.get("to")))
            for f in data.get("formerNames") or ()
        ),
        filings,
    )


def header_sic(text: str, cik: str | None = None) -> str | None:
    """The SIC code of company ``cik`` in a filing's SEC header (as assigned at filing).

    A header can list several companies (a merger filing names the filer and the subject
    company); only the block that holds ``cik`` is read. Without ``cik``, the first.
    """
    if cik is not None:
        blocks = re.split(r"<(?:FILER|SUBJECT-COMPANY|FILED-BY|REPORTING-OWNER|ISSUER)>", text)
        own = [b for b in blocks[1:] if re.search(rf"<CIK>\s*0*{int(cik)}\b", b)]
        if not own:
            return None
        text = own[0]
    found = re.search(r"<ASSIGNED-SIC>\s*(\d{4})", text) or re.search(
        r"STANDARD INDUSTRIAL CLASSIFICATION:.*?\[(\d{4})\]", text
    )
    return found.group(1) if found else None


def sic_on(changes: Sequence[tuple[date, str | None]], day: date) -> str | None:
    """SIC in force on ``day`` from (filing date, SIC) points: the last filing on or before it."""
    known = [sic for filed, sic in sorted(changes, key=lambda c: c[0]) if filed <= day and sic]
    return known[-1] if known else None


# =============================================================================
# Cached access
# =============================================================================


def full_text(
    sec: CachedSec, phrase: str, start: date, end: date, forms: str | None = None
) -> list[TextHit]:
    """Full-text search for an exact phrase in filings dated ``start``..``end`` (before 2025)."""
    end = min(end, SCOPE_END - timedelta(days=1))
    params = {
        "q": f'"{phrase}"',
        "dateRange": "custom",
        "startdt": start.isoformat(),
        "enddt": end.isoformat(),
    }
    if forms:
        params["forms"] = forms
    text = sec.get(FTS_URL, params)
    return parse_full_text(text) if text else []


#: SEC's cover-page table ("Title of each class / Trading Symbol(s) / Name of each
#: exchange") is on 10-K, 10-Q, 8-K, 20-F and 40-F filings made since mid-2019.
COVER_FORMS = "10-K,10-Q,8-K,20-F,40-F"
COVER_FIRST = date(2019, 1, 1)
#: How far after "Trading Symbol" a ticker may stand to be read as listed in that table.
COVER_WINDOW = 600


#: An exchange named right after a ticker in the cover table.
_EXCHANGE_AFTER = re.compile(
    r"^[\s\u200b\u00a0,;:|()\-]{0,12}(?:the\s+)?(?:nasdaq|new\s+york\s+stock\s+exchange|nyse|cboe|bats"
    r"|chicago\s+stock\s+exchange|nyse\s+american|american\s+stock\s+exchange)",
    re.I,
)


def cover_page_symbol(text: str, ticker: str, window: int = COVER_WINDOW) -> str | None:
    """The cover-page fragment where a filing lists ``ticker`` under "Trading Symbol(s)".

    Within ``window`` characters after the column heading, the ticker must stand alone, in
    capitals, and be followed by the exchange it is registered on, as in the table's rows
    ("Common Stock XOM New York Stock Exchange"). A ticker merely written near the table
    ("AT&T", "NYSE", a parent company named in the next paragraph) does not count.
    """
    pattern = re.compile(rf"(?<![A-Za-z0-9.&'\-]){re.escape(ticker)}(?![A-Za-z0-9&'\-])")
    for heading in re.finditer(r"Trading\s+Symbol", text, re.I):
        after = text[heading.end() : heading.end() + window]
        for found in pattern.finditer(after):
            if _EXCHANGE_AFTER.match(after[found.end() : found.end() + 60]):
                return text[heading.start() : heading.end() + found.end() + 30].strip()
    return None


_HEADINGS = re.compile(
    r"(?:Title\s+of\s+each\s+class|Trading\s+Symbol\s*\(?s?\)?|Name\s+of\s+each\s+exchange"
    r"\s+on\s+which\s+registered|Name\s+of\s+exchange\s+on\s+which\s+registered)",
    re.I,
)
_EXCHANGE_NAME = re.compile(
    r"(?:the\s+)?(?:nasdaq(?:[\w\s]{0,40}?(?:market|llc))?|new\s+york\s+stock\s+exchange(?:\s+llc)?"
    r"|nyse(?:\s+(?:american|arca|mkt))?(?:\s+llc)?|cboe[\w\s]{0,20}?(?:exchange|llc))",
    re.I,
)


def cover_page_class(text: str, ticker: str, window: int = COVER_WINDOW) -> str | None:
    """The title of the class a filing's cover page lists under ``ticker`` ("Class A
    common stock, par value $0.0001 per share", "American depositary shares, each
    representing 20 Class A common shares"): the row's text before the ticker. None when
    the ticker is listed on more than one row (it does not name one class)."""
    pattern = re.compile(rf"(?<![A-Za-z0-9.&'\-]){re.escape(ticker)}(?![A-Za-z0-9&'\-])")
    for heading in re.finditer(r"Trading\s+Symbol", text, re.I):
        start = max(0, heading.start() - 200)
        after = text[heading.end() : heading.end() + window]
        rows = [
            m
            for m in pattern.finditer(after)
            if _EXCHANGE_AFTER.match(after[m.end() : m.end() + 60])
        ]
        if len(rows) > 1:
            return None  # the ticker on several rows (common and preferred, ADS and shares)
        for found in rows:
            row = text[start : heading.end() + found.start()]
            # The row starts after the headings and after the previous row's exchange.
            cut = 0
            for m in (*_HEADINGS.finditer(row), *_EXCHANGE_NAME.finditer(row)):
                cut = max(cut, m.end())
            title = " ".join(row[cut:].replace("\u200b", " ").split()).strip(" :;,()*")
            return title[-200:] or None
    return None


def cover_page_hits(
    sec: CachedSec,
    ticker: str,
    first: date,
    last: date,
    cik: str | None = None,
    *,
    phrase: bool = True,
) -> list[TextHit]:
    """Filings (cover-page forms) dated ``first``..``last`` holding the ticker and the
    "Trading Symbol" heading, optionally only those of company ``cik``. From 2019 on.

    A hit only proposes a company: ``cover_page_symbol`` on the document confirms it.
    """
    first = max(first, COVER_FIRST)
    last = min(last, SCOPE_END - timedelta(days=1))
    if first > last:
        return []
    params = {
        # Without ``phrase``, the ticker alone: SEC's index misses the heading in some
        # filers' cover pages (ViacomCBS), whose documents still show the table.
        "q": f'"{ticker}" "Trading Symbol"' if phrase else f'"{ticker}"',
        "dateRange": "custom",
        "startdt": first.isoformat(),
        "enddt": last.isoformat(),
        "forms": COVER_FORMS,
    }
    if cik:
        params["ciks"] = str(int(cik)).zfill(10)
    text = sec.get(FTS_URL, params)
    if cik and text and (last - first).days > 31 and _truncated(text):
        # One company with more filings than a page of results: split the period, so none
        # is missed. (Without a company the answer only proposes candidates; one page will do.)
        middle = first + (last - first) / 2
        return cover_page_hits(sec, ticker, first, middle, cik, phrase=phrase) + cover_page_hits(
            sec, ticker, middle + timedelta(days=1), last, cik, phrase=phrase
        )
    hits = parse_full_text(text) if text else []
    return [h for h in hits if cik is None or h.cik == str(int(cik)).zfill(10)]


def _truncated(text: str) -> bool:
    """Whether a full-text answer holds fewer documents than it matched."""
    try:
        body = json.loads(text)["hits"]
        return int(body["total"]["value"]) > len(body["hits"])
    except (ValueError, KeyError, TypeError):
        return False


def submissions(sec: CachedSec, cik: str) -> Submissions | None:
    """The company's record with every filing in scope, older pages included.

    None when the record (or, cache only, one of its pages) is not available: a partial
    filing list is never returned. A listed page SEC does not serve stops the run
    (``SecUnavailable``) rather than being read as "no filings".
    """
    text = sec.get(SUBMISSIONS_URL.format(cik=str(int(cik)).zfill(10)))
    if not text:
        return None
    pages = []
    for name in older_pages(text):
        page = sec.get(SUBMISSIONS_PAGE_URL.format(name=name))
        if page is None:
            return None
        if not page:
            raise SecUnavailable(f"SEC listed but did not serve {name}")
        pages.append(page)
    return parse_submissions(text, pages)


def filing_sic(sec: CachedSec, cik: str, accession: str) -> str | None:
    folder = accession.replace("-", "")
    text = sec.get(f"{ARCHIVES}/Archives/edgar/data/{int(cik)}/{folder}/{accession}.hdr.sgml")
    return header_sic(text, cik) if text else None


def sic_history(sec: CachedSec, cik: str, first: date, last: date) -> list[tuple[date, str | None]]:
    """(filing date, SIC) points that pin down every SIC change between ``first`` and ``last``.

    Reads the header of the last filing before ``first`` (the SIC in force at the start)
    and of the last filing up to ``last``; when they differ, halves the interval until
    each change is located between two consecutive filings. A handful of requests per
    company. Only filings made by then are ever used for a date, so nothing leaks from
    the future.
    """
    company = submissions(sec, cik)
    if company is None:
        return []
    dated = sorted({(filed, accession) for _, filed, accession in company.filings})
    before = [f for f in dated if f[0] < first]
    inside = [f for f in dated if first <= f[0] <= last]
    points: list[tuple[date, str]] = []
    if before:
        inside = [before[-1], *inside]
    if not inside:
        return []

    def read(i: int) -> str | None:
        return filing_sic(sec, cik, inside[i][1])

    def search(lo: int, hi: int, lo_sic: str | None, hi_sic: str | None) -> None:
        if lo_sic == hi_sic or hi - lo <= 1:
            points.append((inside[hi][0], hi_sic))
            return
        mid = (lo + hi) // 2
        mid_sic = read(mid)
        search(lo, mid, lo_sic, mid_sic)
        search(mid, hi, mid_sic, hi_sic)

    first_sic = read(0)
    points.append((inside[0][0], first_sic))
    if len(inside) > 1:
        search(0, len(inside) - 1, first_sic, read(len(inside) - 1))
    # One point per filing date (a header without a SIC gives None; the first wins).
    by_day: dict[date, str | None] = {}
    for day, sic in sorted(points, key=lambda p: (p[0], p[1] or "")):
        by_day.setdefault(day, sic)
    return sorted(by_day.items())


def ticker_candidates(
    sec: CachedSec, ticker: str, first: date, last: date, phrases: Iterable[str] | None = None
) -> dict[str, list[TextHit]]:
    """Companies whose filings wrote the ticker in an exchange context while it traded.

    Phrases like "NASDAQ: XYZ" or "NYSE: XYZ" avoid matching an ordinary word ("RATE").
    """
    phrases = phrases or (
        f"NASDAQ: {ticker}",
        f"NYSE: {ticker}",
        f"NYSE American: {ticker}",
        f"NYSE Arca: {ticker}",
    )
    found: dict[str, list[TextHit]] = {}
    for phrase in phrases:
        for hit in full_text(sec, phrase, first - timedelta(days=30), last + timedelta(days=30)):
            found.setdefault(hit.cik, []).append(hit)
    return found


def _lead_word(name: str) -> str | None:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in name.upper())
    words = [w for w in cleaned.split() if w not in _NAME_NOISE and len(w) > 1]
    return words[0] if words else None


plain_text = _sec.plain_text  # one definition: cover snippets must read exactly the same


def own_ticker_mention(text: str, ticker: str, company: str, window: int = 200) -> str | None:
    """The sentence fragment where ``company`` names ``ticker`` as its own, or None.

    "FSB Bancorp, Inc. (NASDAQ: FSBC)" counts for FSB Bancorp. "Sears Holdings Corp.
    (formerly NASDAQ: SHLD)" in another company's director biography does not count for
    that company: the name right before the ticker must be the filer's own, and
    "formerly" disqualifies it.
    """
    lead = _lead_word(company)
    if lead is None:
        return None
    pattern = rf"\b(?:NASDAQ|NYSE(?:\s+(?:American|Arca|MKT))?)\s*:\s*{re.escape(ticker)}\b"
    for match in re.finditer(pattern, text, re.I):
        before = text[max(0, match.start() - window) : match.start()]
        if "formerly" in before[-30:].lower():
            continue
        words = "".join(ch if ch.isalnum() else " " for ch in before.upper()).split()
        if lead in words:
            return text[max(0, match.start() - window) : match.end() + 1].strip()
    return None


def _preference(hit: TextHit) -> tuple[int, date]:
    if hit.file_type.upper().startswith("EX-99"):
        return (0, hit.filed)
    if hit.form in ("8-K", "6-K", "425", "497", "497K"):
        return (1, hit.filed)
    return (2, hit.filed)


#: Name-to-ticker distance for proposing a delisting candidate (a Form 25 must still
#: match) and the stricter one for naming a security's issuer, which has no second gate.
CANDIDATE_WINDOW = 200
ISSUER_WINDOW = 60


def confirm_writer(
    sec: CachedSec,
    hits: Sequence[TextHit],
    ticker: str,
    docs: int = 2,
    window: int = CANDIDATE_WINDOW,
):
    """(hit, fragment) where the candidate names the ticker as its own, from its own filing.

    Reads at most ``docs`` documents, press releases first.
    """
    for hit in sorted(hits, key=_preference)[:docs]:
        fragment = own_ticker_mention(plain_text(sec.get(hit.url)), ticker, hit.name, window)
        if fragment:
            return hit, fragment
    return None


#: At most this many full-text candidates per ticker, the most frequent writers first.
MAX_TEXT_CANDIDATES = 8


def ranked_candidates(found: dict[str, list[TextHit]]) -> list[str]:
    """CIKs ordered by number of hits (ties by CIK), capped at ``MAX_TEXT_CANDIDATES``."""
    order = sorted(found, key=lambda cik: (-len(found[cik]), cik))
    return order[:MAX_TEXT_CANDIDATES]
