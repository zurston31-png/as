# Universe Stage 1 (100 securities)

Stage 1 of the staged universe expansion (100, then 250, then 500 securities). Research
data only: no strategy, no backtest, no optimization, no parameter freezing, and no evidence
dated 2025 or later. The IEX database and the Strategy A archive are untouched.

**Status: complete; awaiting review. Stage 2 has not started.** The full report is
`validation_report.txt`. Open findings that need a decision are listed below. They are
not resolved here.

## Files

| File | What it is |
| --- | --- |
| `validation_report.txt` | The full Stage 1 validation report (`scripts/universe_validation_batch.py --stage 100`) |
| `reproducibility.txt` | Two cache-only runs: SHA-256 of each report, byte comparison, requests made |
| `merge_audit.txt` | `scripts/audit_merges.py`: every trading-gap merge and rejection rechecked |
| `outcome_audit.txt` | `scripts/audit_outcomes.py`: every exit value traced to its evidence |
| `curated_quotes_live_check.txt` | Every hand-read SEC quote re-found in the live SEC document |
| `excluded_type_impact.txt` | `scripts/excluded_type_impact.py`: members if conflicting or unknown types counted as common |
| `issuer_gap_check.txt` | `scripts/issuer_gap_check.py`: the 9 common stocks with no SEC issuer |
| `filing_source_check.txt` | Equivalence test of SEC's two filing-list sources (below) |
| `filing_source_impact.txt` | SEC findings recomputed with each filing-list source (below) |
| `protected_data_check.txt` | IEX database fingerprint and Strategy A archive checksums |

## Stage 1 results

### Pool, screen and completeness

- Candidate pool: 18,490 tickers from four sources:
  - SEC's ticker list;
  - Alpaca corporate actions;
  - Wikidata listings;
  - the named validation cases.
- Today's ticker list is one source among four, not the universe.
- 11,063 tickers traded in at least one month of 2016–2024.
- Daily data was fetched for 998 tickers:
  - 840 that ranked in the top 300 in some month;
  - 158 added by the completeness proof, which adds any ticker whose dollar-volume bound could reach the 100th member.
- Completeness rounds: [158, 0]. **None left at the end.**
- Every one of the 54 named validation cases was found by at least one automated source.

### Securities

- 1,041 securities: 883 trading, 158 ended before 2025. 98 have more than one ticker.
- 1,073 permanent ids are registered: 1,041 active and 32 merged into another id, each recorded.

### Membership (point in time)

- Rule: top 100 eligible common stocks by 63-session median dollar volume, $5 minimum raw price, 126 sessions of history, 50 active sessions. Rule fingerprint `57433ff596a5b47f`.
- 59 monthly decisions, 2020-01-31 to 2024-11-29, each with exactly 100 members.
- 207 distinct members, all common stock: 199 with the type confirmed by two sources, 8 by one source.
- Look-ahead tests:
  - recompute with data cut off at each decision: **59 of 59 identical**;
  - every price and volume after 2022-06-30 multiplied by 10: **earlier decisions unchanged**;
  - members chosen after their last trade: **0**;
  - latest evidence date of every kind is before 2025.
- 339 decisions excluded a blank-check company (SIC 6770). Each exclusion uses only a SIC filed by the decision date (0 unsupported).
- 7 members later ended:

  | Member | Outcome | Confidence | Exit value |
  | --- | --- | --- | --- |
  | ATVI | acquired for cash | confirmed | 95.00 |
  | TWTR | acquired for cash | confirmed | 54.20 |
  | PXD | acquired for stock | confirmed | 269.51 |
  | DISCA | acquired for stock | one source | 24.78 |
  | AGN | acquired for stock and cash | confirmed | **none** |
  | WORK | acquired for stock and cash | confirmed | **none** |
  | RTN | acquired | one source | **none** |

### Exit outcomes (158 ended securities)

| Confidence | Count |
| --- | --- |
| Two independent sources | 84 |
| **One source (lower confidence)** | **39** |
| **Unresolved** | **34** (11 resumed or fragment, 23 unknown) |
| Not an exit (trading gap) | 1 |
| Conflict | 0 |

| Value basis | Count |
| --- | --- |
| Stated cash (SEC 8-K or Alpaca, quoted) | 52 |
| Verified deal terms (cash + shares × acquirer's raw close on the effective day, within 10% of the target's last close) | 7 |
| **Assumed zero** (bankruptcy or failure confirmed; shareholder recovery **not** confirmed) | **19** |
| No value | 80 |

- The 16 cash deals stated by both Alpaca and the SEC 8-K agree on the amount, 16 of 16.
- Every value traces to its evidence: 78 of 78.
- FRC and SBNY are failures with an assumed zero, each from one source. They are labelled lower confidence.

### Audits

All of these pass, except the duplicate audit (1 case, finding 1 below):

| Audit | Result |
| --- | --- |
| Duplicate-security audit (report section 9) | **FAILS: 1 possible duplicate without a recorded reason** (YNDX / NBIS) |
| Ticker-days held by two securities | 0 |
| Same-ticker successions with a recorded decision | 34 of 34 |
| Trading-gap merges, independently rechecked (`merge_audit.txt`) | 32 of 32 merges pass; 34 of 34 rejections supported |
| Exit values (`outcome_audit.txt`) | 0 problems; 59 payouts and 19 assumed zeros kept apart |
| Hand-read SEC quotes, re-found live (`curated_quotes_live_check.txt`) | 42 of 42 in 28 SEC documents |
| SEC findings, browse-edgar vs JSON lists (`filing_source_impact.txt`) | 0 changed |
| IEX database (`protected_data_check.txt`) | 15,861 bars, fingerprint `e66890e79788fc78`, unchanged |
| Strategy A archive | 13 of 13 files match `SHA256SUMS` |
| Reproducibility (`reproducibility.txt`) | Two cache-only runs **byte-identical**, SHA-256 `f8aa470e…de4a10`; 0 requests to SEC, OpenFIGI and Wikidata |

## Open findings for review

None of these is resolved here, and nothing was changed to make the report pass.

1. **Duplicate audit failure: YNDX (id 616) and NBIS (id 898).**
   - The two share one issuer (CIK 1513845). Yandex N.V. was renamed Nebius Group N.V. on 2024-08-16.
   - YNDX last traded on 2022-02-25. Its outcome is unresolved, with no value.
   - NBIS traded from 2024-10-21. There was no Form 25 or Form 15. Alpaca records a name change on 2024-08-21. The price runs 18.94 → 20.00.
   - The gap is 665 sessions, beyond the 250-session merge rule, so the rule keeps them apart, but no reason is recorded.
   - Neither was ever a member, so Stage 1 membership is unaffected.
   - **Decision needed:** a documented exception merging them, or a documented keep-separate.
2. **Members that ended with no exit value: AGN, WORK (stock and cash) and RTN (one source).**
   - Verified deal terms would be needed. None is assumed.
3. **Nine common stocks have no identified SEC issuer** (`issuer_gap_check.txt`). With no issuer there is no SIC, so the blank-check exclusion cannot run for them. The v5 batch had none. Causes:
   - The issuer step for still-trading stocks uses **SEC's ticker list as of today (2026)**, which has moved since 2024:
     - SQ (now XYZ), IAC, YY, FEAC→SKLZ: the ticker is no longer listed;
     - XOM: now held by a new ExxonMobil Holdings Corp with no filings in 2016–2024;
     - TD: the cached submissions list only 2025+ filings, because only SEC's most recent block of filings is read.
   - Ended stocks skip that step and have no Form 25 match: CBS, SMCI and COHR. Today's COHR is II-VI's renamed company, correctly not used.
   - Impact:
     - SQ (52 decisions) and XOM (59) are members. The cache holds their issuers' current records: Block, Inc. (formerly Square, Inc.), SIC 7372, and Exxon Mobil Corp, SIC 2911. Neither is a blank-check SIC, but these are current codes, not point-in-time.
     - **FEAC was a blank-check company** (Flying Eagle Acquisition, before its merger with Skillz on 2020-12-17). It was eligible at the 2020-10-30 and 2020-11-30 decisions because its SIC was unknown. It ranked 645 and 643, so it was not a member, and it would not be one at 250 or 500 either. A blank-check company ranked higher would not be caught.
     - The other six were never members. Several were eligible and could matter at a larger size: IAC (best rank 192) and SKLZ (208) at 250; YY (463), TD (330) and COHR (549) at 500. They are operating companies. The risk is only that a blank-check company among them would go unnoticed.
   - **Decision needed** before Stage 2: whether to resolve issuers by the CIK from the ticker's own period (browse-edgar ticker lookup or Wikidata) and read SEC's older submission pages. That would change issuer identification, so it is not done here.
4. **Types in conflict (73) or unknown (30)** (`excluded_type_impact.txt`). Treated as common, 17 would have been members:
   - 15 ADRs (BABA, TSM, PDD, ASML, JD, SE, BIDU, ARM, BNTX, VALE, FUTU, BILI, LI, XPEV, LK). ADRs are outside the universe by design.
   - NVDL (an ETF) and TVIX (an ETN).
   - No REIT and no unknown-type operating company reaches the top 100.
   - Wikidata types several REITs as "fund" (CCI, DLR, SPG, AVB, INVH, CONE, AAIC). They are excluded as conflicts and may matter at 250 or 500.
   - 33 ended securities are kept out only because of type. In Stage 1 only TVIX (an ETN) would have been a member.
5. **Validation case META resolved to METV.** The validation ticker matched the Roundhill Metaverse ETF, which held the META ticker before 2022. The type is ETF, so it is excluded. Explained, not a failure.
6. **Carried forward, still open:**
   - 39 one-source outcomes remain lower confidence.
   - 19 assumed-zero recoveries need sensitivity testing.
   - 34 unresolved outcomes stay flagged, with no value.
   - From v5: YGF and WDRW are unresolved, and FSB stays excluded.
   - A clean 100-security stage is not proof that 250 or 500 will be clean.
7. **Data checks reported, not failures:**
   - 9 Alpaca splits contradicted by prices (not applied);
   - 3 magnitude mismatches (held out of membership as "price adjustment under review", 118 decisions);
   - 11 large raw moves with no recorded split.
   - Each is listed in report section 5.

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
| Stage 1, when switched (mid-run) | 163 | 126 | **0** |
| Stage 1, final | 158 | 110 | **0** |

At the switch, the other 37 Stage 1 securities did not yet have all their answers cached.
In the final state, 48 are not comparable. For all 48, every JSON answer is cached; what is
missing is a browse-edgar list that was never requested, because those lists were asked after
the switch. They are reported as not comparable, not as unchanged.

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
