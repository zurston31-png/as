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

import html
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from app.universe.sec import ARCHIVES, SCOPE_END, CachedSec
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


def parse_submissions(text: str) -> Submissions | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or "cik" not in data:
        return None
    recent = data.get("filings", {}).get("recent", {})
    filings = tuple(
        (form, filed, accession)
        for form, day, accession in zip(
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("accessionNumber", []),
            strict=False,
        )
        if (filed := _day(day)) is not None and filed < SCOPE_END
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


def submissions(sec: CachedSec, cik: str) -> Submissions | None:
    text = sec.get(SUBMISSIONS_URL.format(cik=str(int(cik)).zfill(10)))
    return parse_submissions(text) if text else None


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


def plain_text(raw: str | None) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(raw or "")).split())


def own_ticker_mention(text: str, ticker: str, company: str) -> str | None:
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
        before = text[max(0, match.start() - 200) : match.start()]
        if "formerly" in before[-30:].lower():
            continue
        words = "".join(ch if ch.isalnum() else " " for ch in before.upper()).split()
        if lead in words:
            return text[max(0, match.start() - 200) : match.end() + 1].strip()
    return None


def _preference(hit: TextHit) -> tuple[int, date]:
    if hit.file_type.upper().startswith("EX-99"):
        return (0, hit.filed)
    if hit.form in ("8-K", "6-K", "425", "497", "497K"):
        return (1, hit.filed)
    return (2, hit.filed)


def confirm_writer(sec: CachedSec, hits: Sequence[TextHit], ticker: str, docs: int = 2):
    """(hit, fragment) where the candidate names the ticker as its own, from its own filing.

    Reads at most ``docs`` documents, press releases first.
    """
    for hit in sorted(hits, key=_preference)[:docs]:
        fragment = own_ticker_mention(plain_text(sec.get(hit.url)), ticker, hit.name)
        if fragment:
            return hit, fragment
    return None


#: At most this many full-text candidates per ticker, the most frequent writers first.
MAX_TEXT_CANDIDATES = 8


def ranked_candidates(found: dict[str, list[TextHit]]) -> list[str]:
    """CIKs ordered by number of hits (ties by CIK), capped at ``MAX_TEXT_CANDIDATES``."""
    order = sorted(found, key=lambda cik: (-len(found[cik]), cik))
    return order[:MAX_TEXT_CANDIDATES]
