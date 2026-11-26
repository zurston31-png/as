"""Cross-check of the candidate pool against every delisting notice filed with the SEC.

The completeness proof covers the tickers in the candidate pool. A company that none of the
pool's sources names would be invisible to it. This check reads EDGAR's quarterly form
indexes (one small file per quarter, 2019-Q4 to 2024-Q4), lists every Form 25 and 25-NSE
filed in that period, and classifies each filer:

* ``in_batch``  - the company is an issuer in the stage's issuer evidence (its security
  was given daily data);
* ``in_pool``   - one of its tickers is in the candidate pool (so the screen and the
  completeness proof judged it);
* ``gap``       - its tickers are known and none is in the pool: the pool missed it;
* ``unresolved`` - no ticker is known for it from the cache (SEC's record lists none, as
  is usual after a delisting); listed for review, never counted as covered.

Network: the indexes and filing records are read through the SEC cache. Without
``--fetch`` nothing is requested; if an index is not cached the check reports NOT RUN.
Fetching them is new data and needs approval (Phase 2).

    python scripts/form25_pool_check.py --db STAGE.db [--fetch] [--report F]
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
REMOVAL_FORMS = {"25", "25-NSE"}
FIRST, LAST = (2019, 4), (2024, 4)
PERIOD = (date(2019, 10, 1), date(2024, 12, 31))


@dataclass(frozen=True)
class Removal:
    form: str
    company: str
    cik: str
    filed: date
    path: str


def quarters(first=FIRST, last=LAST) -> list[tuple[int, int]]:
    out, (y, q) = [], first
    while (y, q) <= last:
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


_TAIL = re.compile(
    r"^(?P<rest>.*?)\s+(?P<cik>\d+)\s+(?P<filed>\d{4}-\d{2}-\d{2})\s+(?P<path>\S+)\s*$"
)


def parse_form_index(text: str) -> list[Removal]:
    """Form 25 and 25-NSE lines of an EDGAR form.idx, filed in the period.

    Each line ends with the CIK, the filing date and the file name, read from the right;
    the form type is the first column (it can contain spaces: "SC 13G"), whose width comes
    from the header line ("Form Type   Company Name ...").
    """
    lines = text.splitlines()
    try:
        header = next(i for i, line in enumerate(lines) if line.startswith("Form Type"))
    except StopIteration:
        raise ValueError("not an EDGAR form index: no 'Form Type' header") from None
    company_at = lines[header].index("Company Name")
    out = []
    for line in lines[header + 2 :]:
        found = _TAIL.match(line)
        if not found:
            continue
        form = line[:company_at].strip()
        if form not in REMOVAL_FORMS:
            continue
        filed = date.fromisoformat(found["filed"])
        if PERIOD[0] <= filed <= PERIOD[1]:
            company = found["rest"][company_at:].strip()
            out.append(
                Removal(form, company, str(int(found["cik"])).zfill(10), filed, found["path"])
            )
    return out


def classify(
    removals: list[Removal],
    issuer_tickers: dict[str, set[str]],
    pool: set[str],
    tickers_of,
) -> dict[str, list[tuple[Removal, set[str]]]]:
    """Each delisting filer as in_batch, in_pool, gap or unresolved (see the module doc).

    ``tickers_of(cik)`` returns the tickers known for a company from the cache, or None
    when its record is not cached (then the filer is unresolved)."""
    out: dict[str, list[tuple[Removal, set[str]]]] = defaultdict(list)
    for r in removals:
        if r.cik in issuer_tickers:
            out["in_batch"].append((r, issuer_tickers[r.cik]))
            continue
        tickers = tickers_of(r.cik) or set()
        if not tickers:
            out["unresolved"].append((r, set()))
        elif tickers & pool:
            out["in_pool"].append((r, tickers & pool))
        else:
            out["gap"].append((r, tickers))
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--fetch", action="store_true", help="request indexes not in the cache")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.database import create_session_factory, session_scope
    from app.universe import sec
    from app.universe.database import open_research_database

    lines: list[str] = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    issuer_tickers: dict[str, set[str]] = defaultdict(set)
    for cik, symbol in db.execute("select distinct cik, symbol from issuer_evidence"):
        issuer_tickers[cik].add(symbol)
    pool = {r[0] for r in db.execute("select symbol from candidates")}
    engine = open_research_database(
        f"sqlite:///{args.db.resolve().as_posix()}", read_only=not args.fetch
    )
    with session_scope(create_session_factory(engine)) as s:
        client = sec.CachedSec(s, network=args.fetch)
        removals, missing = [], []
        for year, quarter in quarters():
            text = client.get(INDEX_URL.format(year=year, quarter=quarter))
            if text is None:
                missing.append(f"{year}-Q{quarter}")
                continue
            removals += parse_form_index(text)
        if missing:
            say(
                f"NOT RUN: {len(missing)} of {len(quarters())} quarterly form indexes are not in "
                f"the cache ({', '.join(missing)}). Fetching them (--fetch) is new data and "
                "needs approval."
            )
            if args.report:
                args.report.write_text("\n".join(lines) + "\n")
            return 2

        def tickers_of(cik):
            text = client.get(SUBMISSIONS_URL.format(cik=cik))
            if not text:
                return None
            return {t.upper() for t in json.loads(text).get("tickers", []) if t}

        groups = classify(removals, issuer_tickers, pool, tickers_of)
    say(f"Form 25 / 25-NSE filed {PERIOD[0]} to {PERIOD[1]}: {len(removals)}")
    for name in ("in_batch", "in_pool", "gap", "unresolved"):
        say(f"  {name}: {len(groups.get(name, []))}")
    for name in ("gap", "unresolved"):
        say(f"\n{name.upper()}")
        for r, tickers in groups.get(name, []):
            say(
                f"  {r.filed} {r.form} CIK {r.cik} {r.company}: "
                f"tickers {sorted(tickers) or 'none known'}"
            )
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 1 if groups.get("gap") else 0


if __name__ == "__main__":
    sys.exit(main())
