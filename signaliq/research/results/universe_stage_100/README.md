# Universe Stage 1 (100 securities)

Stage 1 of the staged universe expansion (100, then 250, then 500 securities). The full
validation report is added here when the stage run completes.

## SEC filing lists: switch to SEC's JSON service

During the Stage 1 run, SEC's older company-filings page
(`www.sec.gov/cgi-bin/browse-edgar`) failed many requests with a passing 503 ("File
Unavailable"), at times three requests in four. Filing lists asked **by CIK** (the Form 25,
8-K and Form 15 lists that `sec.find` reads) now come from SEC's JSON service
(`data.sec.gov/submissions`, `CachedSec.company_json`). Lists asked **by ticker**, and
company-name searches, stay on browse-edgar.

No universe selection rule, research assumption, membership logic or exit-value treatment
changed. Only the source of one kind of list changed.

### Equivalence test before switching

`filing_source_check.txt` comes from `scripts/compare_filing_sources.py --live`. It compares
every browse-edgar company filing list in the cache with the same list built from the JSON
service. The comparison covers:

- the CIK;
- each filing's accession number, filing date and form type;
- the 8-K item numbers;
- the order of the list and its length (at most 100);
- the company name and SIC code.

| Lists asked by CIK (switched) | Count |
| --- | --- |
| Identical (filings, order, name, SIC) | 920 |
| Different, explained (see below) | 5 |
| Unexplained | **0** |

**The 5 explained differences.** browse-edgar was asked again live, retried until it
answered. Today it answers **exactly** the JSON list in all 5 cases, so the cached
browse-edgar answers were wrong when fetched:

- Three were missing a real delisting notice:
  - Ernexa Therapeutics' Form 25 of 2021-10-22;
  - CIK 1280784's Form 25-NSE of 2021-07-02, missing from two lists.
- Two contained filings made after the list's cut-off date:
  - Seagen's 2023 Form 25-NSE in a list cut off on 2021-02-19;
  - 2021–2022 8-Ks in CIK 1486159's list cut off on 2021-02-06.

An unanswered request is never counted as a match.

**Rules the JSON list follows, each confirmed by the comparison:**

- the form type is a prefix match ("25" also finds 25-NSE);
- filings dated on the cut-off day are included;
- the newest 100 are kept;
- filings of one day are listed in the order SEC accepted them;
- SEC's SIC "0000" (none assigned) is shown as no SIC, as browse-edgar shows it.

**Lists asked by ticker (211) are not switched, and why.** For 84 retired tickers, such as
ATVI, TWTR, SIVB and FRC, browse-edgar resolves the ticker to its company. SEC's JSON ticker
list knows only today's tickers, so it cannot. `sec.find` uses only the CIK of a ticker
lookup.

**68 cached lists use a retired cache-key format** (Python's per-process `hash()`, from
before the sha256 key). No run can look them up, and each of their 29 companies also has
lists under the current key. They were not compared.

**Difficult cases.** These come from the v5 batch, using lists asked by CIK:

| Case | Identical lists |
| --- | --- |
| Bankruptcies and failures | 65 of 65 |
| Redemptions and other delistings | 22 of 22 |
| Reused tickers (kept-separate securities) | 56 of 56 |
| Acquisitions | 123 of 124 |

The one acquisition difference is the Seagen list above.

First Republic and Signature Bank file with the FDIC, not the SEC, so they have no SEC
issuer and no CIK lists. Their ticker lookups stay on browse-edgar and are unchanged.

### Impact on findings

`filing_source_impact.txt` comes from `scripts/filing_source_impact.py`, which is
cache-only. It recomputes every ended security's SEC findings once from the cached
browse-edgar lists and once with the lists asked by CIK built from JSON. The findings
compared are the matched company, its delisting notices, the exit 8-Ks, the Form 15s and
the cash terms.

| | Ended securities | Comparable | Findings changed |
| --- | --- | --- | --- |
| v5 validation batch | 66 | 66 | **0** |
| Stage 1, when switched | 163 | 126 | **0** |

The other 37 Stage 1 securities did not yet have all their answers cached. They are
reported as not comparable, not as unchanged.

**What is kept.**

- Every cached browse-edgar answer stays in the cache. For lists asked by CIK, those answers
  are no longer read.
- The v5 archive stays reproducible at its own commit (`f26e0b0`).
- Rerunning today's code on the v5 database would need the JSON answers. Those are held in
  the stage database, and with them the v5 findings are identical.

### SEC request handling (same session)

- Every request is retried, up to 30 times, through passing 503s and dropped connections.
- A request SEC never answers stops the run (`SecUnavailable`). It is never read as "no
  filings".
- Answers are committed every 25 requests, so a stopped run resumes from what it has.
