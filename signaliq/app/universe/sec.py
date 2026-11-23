"""SEC EDGAR cross-check: delisting notices, filings and company records. Read-only.

What SEC adds, and how it is used:

* **Form 25 / 25-NSE** (notice of removal from listing). Filed by the issuer (25)
  or the exchange (25-NSE) when a class of securities is delisted. It names the
  issuer (CIK), the **class of security** ("Common Stock", "Warrants", "Units",
  "6.25% Notes due 2027", ...) and the **rule** used:

  - 17 CFR 240.12d2-2(a)(1)/(a)(2): the class was redeemed or retired (cash).
  - 12d2-2(a)(3): the class was replaced by other securities or cash (a merger).
  - 12d2-2(a)(4): all rights in the class were extinguished.
  - 12d2-2(b): the exchange removed it for not meeting its rules.
  - 12d2-2(c): the issuer withdrew it voluntarily.

* **8-K items** around the last trade: 1.03 bankruptcy or receivership; 2.01
  completion of an acquisition together with 3.03 or 5.01 (rights changed,
  change in control) means the company was acquired; 3.01 is a delisting notice.

* **Company record**: SIC industry code and name (detail only; SIC describes
  the issuer, not each security, so it is never used as the security's type).

How a security is matched to an SEC company, conservatively:

1. Search keys come from anywhere (the security's tickers, company names known
   from other sources, names in the batch definition). A key only proposes
   candidate companies; it is not evidence.
2. A candidate is accepted only if it filed a Form 25 or 25-NSE close to the
   security's last trade (from ``WINDOW_BEFORE`` days before to ``WINDOW_AFTER``
   days after). A company that holds the ticker today but delisted nothing then
   is rejected, which handles reused tickers.
3. Filings dated 2025 or later are ignored.

Every response is cached in ``external_responses`` (source ``sec``) so a report
can be reproduced without asking SEC again.
"""

import hashlib
import html
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
from sqlalchemy.orm import Session

from app.universe.models import ExternalResponse

USER_AGENT = "SignalIQ-Research universe-validation/0.2 signaliq.research@gmail.com"
BROWSE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
ARCHIVES = "https://www.sec.gov"
PAUSE_SECONDS = 0.15  # about 6 requests per second; SEC allows at most 10
WINDOW_BEFORE = 45
WINDOW_AFTER = 90
SCOPE_END = date(2025, 1, 1)
MAX_CANDIDATES_PER_NAME = 6
BANKRUPTCY_LOOKBACK = 365

#: Company names for known cases whose ticker SEC no longer resolves. Search keys only:
#: a name proposes candidates, and only a delisting notice at the right time confirms one.
SEARCH_NAMES: dict[str, tuple[str, ...]] = {
    "SHLD": ("sears holdings corp",),
    "FTR": ("frontier communications",),
    "MON": ("monsanto co /new/",),  # "monsanto co" jumps to the pre-2000 company
    "TWX": ("warner media",),  # Time Warner Inc. was renamed Warner Media, LLC in 2018
    "CELG": ("celgene corp",),
    "BBBY": ("bed bath",),
    "REV": ("revlon inc",),
    "PRTY": ("party city holdco",),
    "EXPR": ("express, inc",),
    "ENDP": ("endo international",),
    "DISCA": ("discovery, inc",),
    "FRC": ("first republic bank",),
    "CRC": ("california resources corp",),
    "DO": ("diamond offshore drilling",),
}


# =============================================================================
# Parsing (pure)
# =============================================================================


@dataclass(frozen=True)
class Filing:
    form: str
    filed: date
    items: str
    href: str


@dataclass(frozen=True)
class Company:
    cik: str
    name: str
    sic: str | None
    sic_description: str | None
    filings: tuple[Filing, ...]


def _date(text: str) -> date | None:
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError:
        return None


def parse_company_feed(text: str) -> Company | None:
    """A company's filing feed (browse-edgar output=atom for one CIK)."""
    cik = re.search(r"<cik>(\d+)</cik>", text)
    if cik is None:
        return None

    def pick(tag: str) -> str | None:
        found = re.search(rf"<{tag}>([^<]*)</{tag}>", text)
        return found.group(1).strip() if found else None

    filings = []
    for entry in re.findall(r"<entry>(.*?)</entry>", text, re.S):
        form = re.search(r"<filing-type>([^<]*)<", entry)
        filed = re.search(r"<filing-date>([^<]*)<", entry)
        href = re.search(r"<filing-href>([^<]*)<", entry)
        items = re.search(r"<items-desc>([^<]*)<", entry)
        if form and filed and href and _date(filed.group(1)):
            filings.append(
                Filing(
                    form.group(1).strip(),
                    _date(filed.group(1)),
                    items.group(1) if items else "",
                    href.group(1).strip(),
                )
            )
    return Company(
        cik.group(1).zfill(10),
        html.unescape(pick("conformed-name") or ""),
        pick("assigned-sic"),
        html.unescape(pick("assigned-sic-desc") or "") or None,
        tuple(filings),
    )


def parse_company_search(text: str) -> list[tuple[str, str]]:
    """(name, CIK) pairs from a company-name search (one pair if SEC went straight to a company)."""
    single = parse_company_feed(text)
    if single is not None:
        return [(single.name, single.cik)]
    pairs = []
    for entry in re.findall(r"<entry>(.*?)</entry>", text, re.S):
        name = re.search(r"<title>([^<]*)</title>", entry)
        cik = re.search(r"CIK=(\d+)", entry)
        if name and cik:
            pairs.append((html.unescape(name.group(1)).strip(), cik.group(1).zfill(10)))
    return pairs


@dataclass(frozen=True)
class RemovalNotice:
    issuer_cik: str
    issuer_name: str
    exchange: str
    class_description: str
    rule: str
    signed: date | None
    reason_text: str = ""


def parse_form25(xml: str, reason_text: str = "") -> RemovalNotice | None:
    """Fields of a Form 25 / 25-NSE primary document."""
    issuer = re.search(r"<issuer>(.*?)</issuer>", xml, re.S)
    exchange = re.search(r"<exchange>.*?<entityName>([^<]*)</entityName>", xml, re.S)
    if issuer is None:
        return None
    cik = re.search(r"<cik>(\d+)</cik>", issuer.group(1))
    name = re.search(r"<entityName>([^<]*)</entityName>", issuer.group(1))
    description = re.search(r"<descriptionClassSecurity>([^<]*)<", xml)
    rule = re.search(r"<ruleProvision>([^<]*)<", xml)
    signed = re.search(r"<signatureDate>([^<]*)<", xml)
    return RemovalNotice(
        issuer_cik=(cik.group(1) if cik else "").zfill(10),
        issuer_name=html.unescape(name.group(1)).strip() if name else "",
        exchange=html.unescape(exchange.group(1)).strip() if exchange else "",
        class_description=html.unescape(description.group(1)).strip() if description else "",
        rule=rule.group(1).strip() if rule else "",
        signed=_date(signed.group(1)) if signed else None,
        reason_text=" ".join(reason_text.split())[:600],
    )


def class_types(description: str) -> set[str]:
    """Every security type a Form 25 class description mentions (a notice can cover several)."""
    text = description.lower()
    found = set()
    if re.search(r"\bwarrants?\b", text):
        found.add("warrant")
    if re.search(r"\brights?\b", text):
        found.add("right")
    if re.search(r"\bunits?\b", text):
        found.add("unit")
    if re.search(r"\b(notes?|debentures?|bonds?)\b", text):
        found.add("debt")
    if re.search(r"exchange[- ]traded|\betf\b|\betn\b|\bfund\b|beneficial interest", text):
        found.add("etf")
    if re.search(r"american depositary|\bads\b|\badrs?\b", text):
        found.add("adr")
    elif "common" in text or "ordinary share" in text:
        found.add("common")
    if re.search(r"preferred|depositary shares", text) and "american depositary" not in text:
        found.add("preferred")
    return found


def class_type(description: str, symbol: str) -> str | None:
    """The type a notice gives the security traded as ``symbol``, or None when unclear.

    A notice covering "Class A common stock and warrants" describes both the common
    shares and the warrants. Nasdaq's fifth letter of the security's own ticker (W,
    R, U) picks the warrant, right or unit; otherwise the common class is meant.
    A description naming several types and no common class is left unclear.
    """
    found = class_types(description)
    suffix = (
        {"W": "warrant", "R": "right", "U": "unit"}.get(symbol[-1:]) if len(symbol) == 5 else None
    )
    if suffix and suffix in found:
        return suffix
    if "common" in found:
        return "common"
    return next(iter(found)) if len(found) == 1 else None


def notice_type(descriptions: Iterable[str], symbol: str) -> str | None:
    """The type the delisting notices of one security give it, or None when unclear.

    An issuer often delists several classes at once (DISCA: Series A, B and C common;
    Time Warner: common stock and notes). The security's own ticker suffix decides
    for a warrant, right or unit; otherwise the common class is the one that traded
    under a plain ticker. Several other types and no common class: unclear.
    """
    kinds = {class_type(d, symbol) for d in descriptions} - {None}
    suffix = (
        {"W": "warrant", "R": "right", "U": "unit"}.get(symbol[-1:]) if len(symbol) == 5 else None
    )
    if suffix in kinds:
        return suffix
    if "common" in kinds:
        return "common"
    return kinds.pop() if len(kinds) == 1 else None


def looks_like_fund(company: str) -> bool:
    """The issuer's name says it is a fund (closed-end funds call their shares "common shares")."""
    return bool(re.search(r"\b(fund|etf|etn)\b", company, re.I))


_SUFFIXES = (
    (r"\bcorporation\b", "corp"),
    (r"\bincorporated\b", "inc"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
)


def search_variants(name: str) -> list[str]:
    """The name, plus the form EDGAR's prefix search is likelier to match.

    EDGAR matches the start of its conformed name ("LINKEDIN CORP"), so a share-class
    tail ("LINKEDIN CORP - A") or a spelled-out suffix ("LinkedIn Corporation") finds
    nothing. Candidates found this way still need a timely delisting notice.
    """
    text = name.lower()
    text = re.sub(r"\s+-\s+[a-z]$|\s+(class|cl|series)\s+[a-z]$", "", text)
    for pattern, short in _SUFFIXES:
        text = re.sub(pattern, short, text)
    text = " ".join(re.sub(r"[.,]", " ", text).split())
    return [name] if text == name.lower() else [name, text]


def rule_meaning(rule: str) -> tuple[str, str]:
    """(category for exit evidence, plain-language meaning) of a 12d2-2 provision."""
    compact = rule.replace(" ", "").lower()
    if "12d2-2(a)(3)" in compact:
        return "acquired", "class replaced by other securities or cash (merger) [12d2-2(a)(3)]"
    if "12d2-2(a)(1)" in compact or "12d2-2(a)(2)" in compact:
        return "redeemed", "class redeemed or retired [12d2-2(a)(1)/(2)]"
    if "12d2-2(a)(4)" in compact:
        return "delisted_no_reason", "all rights in the class extinguished [12d2-2(a)(4)]"
    if "12d2-2(b)" in compact:
        return "delisted_no_reason", "removed by the exchange for not meeting its rules [12d2-2(b)]"
    if "12d2-2(c)" in compact:
        return "delisted_no_reason", "withdrawn voluntarily by the issuer [12d2-2(c)]"
    return "delisted_no_reason", f"rule not recognised: {rule or 'none given'}"


def eight_k_category(items: str) -> str | None:
    """Exit category implied by an 8-K's items, or None."""
    numbers = set(re.findall(r"\d\.\d\d", items))
    if "1.03" in numbers:
        return "bankruptcy"
    if "2.01" in numbers and ({"3.03", "5.01"} & numbers):
        return "acquired"
    return None


_CASH_TERM = re.compile(
    r"right to receive[^.]{0,80}?\$\s?([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?)\s*"
    r"(?:in cash|per share in cash|in cash per share|, in cash|, without interest)",
    re.I,
)


def cash_terms(text: str) -> tuple[float | None, str]:
    """Cash per share stated in a merger 8-K, only when no stock component appears near it.

    Returns (amount or None, the sentence it came from). Fragile by nature, so the
    sentence is kept as evidence for review.
    """
    plain = " ".join(re.sub(r"<[^>]+>", " ", html.unescape(text)).split())
    for match in _CASH_TERM.finditer(plain):
        # Sentences end at a period followed by a space (not the one in "$41.25" or "0.5").
        start = plain.rfind(". ", 0, match.start()) + 1
        end = plain.find(". ", match.end())
        sentence = plain[start : end if end != -1 else match.end() + 200].strip()
        received = sentence[sentence.lower().find("right to receive") :]
        if re.search(
            r"\bshares? of\b|(?:\bone|(?<![$.,\d])\d+(?:\.\d+)?)\s+"
            r"(?:of a\s+)?(?:[a-z]+\s+){0,3}shares?\b"
            r"|fraction of a share",
            received,
            re.I,
        ):
            return None, sentence[:400]  # part of the payment is in stock
        return float(match.group(1).replace(",", "")), sentence[:400]
    return None, ""


# =============================================================================
# Cached access
# =============================================================================


class CachedSec:
    """SEC requests through the research database cache. ``network=False`` reads the cache only."""

    def __init__(
        self, session: Session, *, network: bool, refresh: bool = False, sleep=time.sleep
    ) -> None:
        self.session, self.network, self.refresh, self.sleep = session, network, refresh, sleep
        self.requests = 0
        self.failures: list[str] = []
        self._http = (
            httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) if network else None
        )

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def get(self, url: str, params: dict | None = None) -> str | None:
        key = url + (
            "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items())) if params else ""
        )
        if len(key) > 128:  # a stable digest: the cache must answer the same key in every run
            key = key[:96] + "#" + hashlib.sha256(key.encode()).hexdigest()[:24]
        row = self.session.get(ExternalResponse, ("sec", key))
        if row is not None and not self.refresh:
            return row.response
        if not self.network:
            return None
        for _attempt in range(4):
            self.sleep(PAUSE_SECONDS)
            response = self._http.get(url, params=params)
            self.requests += 1
            if response.status_code in (200, 404):
                break
            self.sleep(10)
        text = response.text if response.status_code == 200 else ""
        if response.status_code not in (200, 404):
            self.failures.append(key)
            return None
        self.session.merge(ExternalResponse(source="sec", request_key=key, response=text))
        return text

    def company(self, cik_or_ticker: str, before: date, form: str = "") -> Company | None:
        """A company's newest 100 filings before ``before``, optionally of one form type.

        EDGAR matches ``form`` as a prefix ("25" also returns 25-NSE and 25/A). Asking
        per form type matters for busy filers, whose newest 100 filings of all types
        may not reach back to the dates that matter (Monsanto, Time Warner).
        """
        text = self.get(
            BROWSE_URL,
            {
                "action": "getcompany",
                "CIK": cik_or_ticker,
                "type": form,
                "owner": "include",
                "dateb": before.strftime("%Y%m%d"),
                "count": "100",
                "output": "atom",
            },
        )
        return parse_company_feed(text) if text else None

    def search(self, name: str) -> list[tuple[str, str]]:
        text = self.get(
            BROWSE_URL,
            {
                "action": "getcompany",
                "company": name,
                "type": "",
                "owner": "include",
                "count": "40",
                "output": "atom",
            },
        )
        return parse_company_search(text) if text else []

    def removal_notice(self, filing: Filing) -> RemovalNotice | None:
        index = self.get(filing.href)
        if not index:
            return None
        docs = re.findall(r'href="(/Archives/edgar/data/[^"]+)"', index)
        primary = next((d for d in docs if d.endswith("primary_doc.xml") and "/xsl" not in d), None)
        reason = next(
            (
                d
                for d in docs
                if d.lower().endswith(".txt") and not re.search(r"\d{10}-\d{2}-\d{6}\.txt$", d)
            ),
            None,
        )
        xml = self.get(ARCHIVES + primary) if primary else None
        reason_text = self.get(ARCHIVES + reason) if reason else ""
        if not xml:
            return None
        cleaned = re.sub(r"<[^>]+>", " ", reason_text or "")
        return parse_form25(xml, cleaned)

    def document_text(self, filing: Filing) -> str:
        index = self.get(filing.href)
        if not index:
            return ""
        docs = [
            d
            for d in re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', index)
            if "-index" not in d
        ]
        text = self.get(ARCHIVES + docs[0]) if docs else ""
        return text or ""


# =============================================================================
# Matching a security to SEC filings
# =============================================================================


@dataclass
class SecFinding:
    cik: str | None = None
    company: str | None = None
    sic: str | None = None
    notices: list[tuple[Filing, RemovalNotice]] = field(default_factory=list)
    eight_ks: list[Filing] = field(default_factory=list)
    other: list[Filing] = field(default_factory=list)
    cash: tuple[float | None, str] = (None, "")
    candidates_checked: list[str] = field(default_factory=list)
    note: str = ""


_REMOVAL_FORMS = ("25", "25-NSE", "25/A", "25-NSE/A")


def find(sec: CachedSec, tickers: Sequence[str], names: Iterable[str], last: date) -> SecFinding:
    """Match a security that last traded on ``last`` to the SEC company that delisted it then."""
    before = min(last + timedelta(days=WINDOW_AFTER + 30), SCOPE_END - timedelta(days=1))
    low, high = (
        last - timedelta(days=WINDOW_BEFORE),
        min(last + timedelta(days=WINDOW_AFTER), SCOPE_END - timedelta(days=1)),
    )
    finding = SecFinding()
    seen: set[str] = set()
    candidates: list[str] = []
    for ticker in tickers:
        company = sec.company(ticker, before)
        if company is not None and company.cik not in seen:
            seen.add(company.cik)
            candidates.append(company.cik)
    keys = list(dict.fromkeys(v for name in names for v in search_variants(name)))
    for name in keys:
        for _, cik in sec.search(name)[:MAX_CANDIDATES_PER_NAME]:
            if cik not in seen:
                seen.add(cik)
                candidates.append(cik)
    matches = []
    for cik in candidates:
        removals_feed = sec.company(cik, before, "25")
        finding.candidates_checked.append(cik)
        if removals_feed is None:
            continue
        removals = [
            f for f in removals_feed.filings if f.form in _REMOVAL_FORMS and low <= f.filed <= high
        ]
        if removals:
            matches.append((removals_feed, removals))
    if not matches:
        finding.note = (
            f"no Form 25 between {low} and {high} from {len(candidates)} candidate companies"
        )
        return finding
    if len(matches) > 1:
        finding.note = "several companies filed a Form 25 in the window: " + ", ".join(
            f"{c.name} ({c.cik})" for c, _ in matches
        )
    company, removals = matches[0]
    finding.cik, finding.company, finding.sic = company.cik, company.name, company.sic
    for filing in removals:
        notice = sec.removal_notice(filing)
        if notice is not None:
            finding.notices.append((filing, notice))
    # A bankruptcy can come many months before the delisting (Revlon filed in June 2022
    # and was delisted in October); an acquisition closes around the last trade.
    eight_k = sec.company(company.cik, before, "8-K")
    for f in eight_k.filings if eight_k else ():
        kind = eight_k_category(f.items)
        start = last - timedelta(days=BANKRUPTCY_LOOKBACK if kind == "bankruptcy" else 60)
        if kind and start <= f.filed <= high:
            finding.eight_ks.append(f)
    terminations = sec.company(company.cik, before, "15")
    finding.other = [
        f
        for f in (terminations.filings if terminations else ())
        if last - timedelta(days=60) <= f.filed <= high
    ]
    acquisition = next(
        (f for f in finding.eight_ks if eight_k_category(f.items) == "acquired"), None
    )
    if acquisition is not None:
        finding.cash = cash_terms(sec.document_text(acquisition))
    return finding
