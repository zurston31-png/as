# Universe Stage 1.1 (100 securities): cleanup

Stage 1.1 is a cleanup of Stage 1 before any expansion. Stage 1 (v1.0) is preserved
unchanged in `../universe_stage_100/`. Research data only: no strategy, no backtest, no
optimization, no parameter freezing, and no evidence dated 2025 or later. The IEX database
and the Strategy A archive are untouched.

**Status: complete; awaiting review. Stage 2 has not started.**

The selection rule, the membership logic and the exit-value treatment are unchanged:

- top 100 common stocks by 63-session median dollar volume;
- point-in-time decisions;
- one-source outcomes stay lower confidence;
- assumed zeros stay apart from payouts;
- unresolved outcomes stay without a value.

What changed is the evidence behind identity, issuer and type, as listed below.

## Results (final run A8, commit c280338; reproduced byte for byte by run B)

Files in this folder:

| File | What it is |
| --- | --- |
| `validation_report.txt` | the full Stage 1.1 report (run A8) |
| `reproducibility.txt` | run A8 against run B: reports and every table compared |
| `comparison_v1_0_to_v1_1.txt` | every difference from Stage 1.0, security by security |
| `merge_audit.txt` | trading-gap merges, YNDX/NBIS and the five handovers, re-derived from raw data |
| `outcome_audit.txt` | every exit outcome and value against its evidence |
| `deal_audit.txt` | the 11 deal-term values recomputed from the quotes and the acquirer's close |
| `issuer_audit.txt` | the issuer evidence, independently re-read (`scripts/audit_issuers.py`) |
| `issuer_gap_check.txt` | common stocks without an issuer, and the issuer behind every member decision |
| `excluded_type_impact.txt` | members if conflicting or unknown types were counted as common |
| `protected_data_check.txt` | the IEX database and the Strategy A archive, verified unchanged |

**Every audit passes:**

- **Merges:**
  - 33 of 33 merges pass every check.
  - 35 of 35 rejections are independently supported.
  - 6 of 6 rename decisions pass.
  - YNDX→NBIS (1 of 1 identity exception) is one security: one registrant (CIK 1513845), no Form 25 or Form 15 in between, neither ticker
    trading during the suspension, close 18.94 → 20.00.
  - All 5 handovers pass every check.
- **Outcomes:** 0 problems.
- **Deals:** 11 of 11 values recompute from the quotes.
- **Issuers:** PASS.
  - Replayed from the cache with 0 requests to SEC: identical to what was stored.
  - 2,023 cover-page run ends re-read, each listing the ticker.
  - 38 of 38 own-filing documents and 97 of 97 delisting notices re-read and confirmed.
  - Evidence fingerprint `7a037ec50c449f21`.
- **Issuer gaps:** 0 common stocks without an issuer and without a documented reason.

**Membership:**

- 59 decisions (2020-01-31 to 2024-11-29), 100 members each, 208 distinct members, all
  common stock (confirmed).
- Recomputed from data cut off at each decision: 59 of 59 identical.
- Prices after 2022-06-30 multiplied by 10: earlier decisions unchanged.
- Members chosen after their last trade: 0.
- Latest date of every input: before 2025.

**Against Stage 1.0** (`comparison_v1_0_to_v1_1.txt`): 9 of 59 decisions differ, and
distinct members go from 207 to 208.

| Decision | In | Out | Why |
| --- | --- | --- | --- |
| 2020-06-30 | SPG | TMUS (rank 101) | SPG is a REIT: no longer read as a fund |
| 2020-09-30 | GOLD (Barrick, id 1088) | GOLD (joined Randgold/Barrick series) | handover cut: Barrick is its own security |
| 2021-01-29 | CVS | AZN | AZN is an ADR on its own 20-F cover page |
| 2021-02-26 to 2021-07-30 (6) | PARA | AZN, DHR, TDOC, AMC, NVAX, SPGI (rank 101) | v1.0 took PARA's issuer from today's ticker list: Parabolic Technologies, a SPAC (SIC 6770). ViacomCBS's own cover pages give Paramount Global (CIK 813828, SIC 4833) |

Other changes:

- Types: conflicts fall from 73 to 4 and unknowns from 30 to 17.
- Exit values: deal terms from 7 to 11 (AGN, WORK, RTN, Randgold).
- Issuer at the last trade:
  - 9 found only now (CBS, SQ, COHR, TD, Barrick and others);
  - 13 lost, mostly ETF trusts whose issuer came from today's ticker list, plus DD;
  - 16 with a different CIK (AXON, BLK, both CZRs and others).

**Exit values of the members that later ended** (all 7 have a value, and none is assumed):

| Member | Last trade | Terms | Acquirer close on the effective date | Value | Target's last close |
| --- | --- | --- | --- | --- | --- |
| AGN | 2020-05-08 | $120.30 + 0.8660 ABBV | ABBV 83.96 (2020-05-08) | 193.0094 | 193.02 |
| WORK | 2021-07-20 | $26.79 + 0.0776 CRM | CRM 242.11 (2021-07-21) | 45.5777 | 45.20 |
| RTN | 2020-04-02 | 2.3348 RTX | RTX 49.93 (2020-04-03) | 116.5766 | 116.96 |
| PXD | 2024-05-02 | 2.3234 XOM | XOM 116.00 (2024-05-03) | 269.5144 | 269.62 |
| DISCA | 2022-04-08 | 1 WBD | WBD 24.78 (2022-04-11) | 24.78 | 24.43 |
| ATVI | 2023-10-12 | $95.00 cash | — | 95.00 | — |
| TWTR | 2022-10-27 | $54.20 cash | — | 54.20 | — |

- Each effective date is quoted from the target's own 8-K on completion. The deal audit
  re-reads the quote and the acquirer's raw SIP close on that day.
- RTN and DISCA are single-source outcomes (lower confidence). The other five are
  confirmed by two sources.

**Outcomes of all 153 ended securities:**

| Confidence | Count |
| --- | --- |
| two sources | 87 |
| one source (lower confidence) | 38 |
| unresolved (no value) | 28 |

| Value basis | Count |
| --- | --- |
| stated cash | 52 |
| deal terms | 11 |
| assumed zero (event confirmed, recovery not) | 19 |
| none | 71 |

Every one of the 82 values is traced to its evidence.

**Duplicates:**

- 0 ticker-days held by two securities.
- 35 same-ticker successions, each with a recorded decision.
- 7 same-issuer pairs kept separate, each with its reason.
- 1 documented identity exception (YNDX/NBIS).
- 5 handovers.

**SEC requests:** run A8 made 0 requests; the cache stood at 38,120 answers at the start
and at the end. Run A5 made the last new requests: 6 cover pages, all stored and counted
(see `reproducibility.txt`).

**Reproducibility:**

- Two full runs, A8 and B, produced byte-identical reports (sha256 `24741a6c…`) and
  identical fingerprints for all 21 tables.
- The issuer step replayed from the cache alone reproduces the stored evidence exactly.

### Open items (documented, not hidden)

- **Form 25 conservatism.** CBS→VIAC, PSAC→FFIE and SGAM→RDBX stay two securities each.
  Their registrant filed a Form 25 at the switch, for an exchange transfer or for a SPAC's
  units. None of the three earlier securities was ever a member.
- **Co-registrant choice.** Where a parent and a subsidiary file the same documents, the
  issuer can be the subsidiary (PCG, URI, QSR, HTZ, CHTR/CCO Holdings). Same filings and
  same industry, so no decision changes.
- **CRH** stays in conflict: ADR on its 20-F until 2023, ordinary shares on its 8-K of
  December 2024. AABA, SOF→SOFR and INFO are the other three conflicts. 17 types are
  unknown.
  - Counted as common, only TVIX (an ETN) would have been a member, for 4 decisions; it
    would have displaced FI, ACN and EBAY (`excluded_type_impact.txt`).
- **Handover earlier holders without a value.** Caesars Entertainment Corp (CZR), Healthcare
  Realty (HR) and Physicians Realty (DOC) have confirmed acquisitions but no exit value.
  Their terms were not hand-read, and none was a member.
- **GOLD (Barrick).** Its issuer rests on a full-text filing dated after the cut ("later
  document"). Before a security's first evidence the earliest evidence is used, and the
  report counts every decision where that happens.
- **DD** (E. I. du Pont, CIK 30554, last trade 2017-08-31) has no issuer now. Its only
  evidence was a Form 25 match whose notice was not read. Its single-source "acquired"
  outcome rests on that company's 8-K of 2017-09-01, and it traded before the first
  decision.
- **One-source and unresolved outcomes:**
  - 38 one-source outcomes stay lower confidence;
  - 19 assumed zeros stay apart from payouts;
  - 28 unresolved outcomes have no value.
- **The security type applies to the whole history.** It comes from sources of the period
  where available (cover pages from 2020), but a type is not dated, so a later
  reclassification would apply retroactively. This is unchanged from Stage 1.0.

## 1. YNDX / NBIS: one security, by a documented exception

Stage 1's duplicate audit failed on Yandex N.V. (YNDX) and Nebius Group (NBIS). They share
one SEC registrant (CIK 1513845), but the 665 sessions without a trade exceed the 250-session
trading-gap rule.

SEC filings by the registrant itself, quoted word for word and re-verified on every run,
establish one continuous security:

- Nasdaq suspended trading in Yandex N.V.'s Class A shares on 2022-02-28 (6-K, March 2022).
- The company renamed itself Nebius Group N.V., "(NASDAQ: YNDX, to be changed to NBIS)"
  (6-K, 2024-08-16).
- Nasdaq determined that trading "will resume" after the divestment of the Russian assets
  (6-K, 2024-10-17).
- Trading in "the Company's Class A ordinary shares" resumed on 2024-10-21 (6-K, 2024-10-21).

The 250-session rule itself is not changed. The link is a named exception
(`CURATED_IDENTITY_LINKS` in `app/universe/curated.py`):

- It joins exactly two segments: YNDX's last trade on 2022-02-25 and NBIS's first trade
  on 2024-10-21.
- It applies only while every quote is found.
- `scripts/audit_merges.py` checks it independently, from raw data:
  - both ends are real trades;
  - neither ticker traded in between;
  - both segments belong to one security;
  - that security's issuer is CIK 1513845;
  - the registrant's own filing list shows no Form 25 or Form 15 in between.

The original records are kept. NBIS's permanent id (898) is registered as "merged into"
YNDX's (616), as for every trading-gap merge. The suspension stays visible: the security has
no bars for 665 sessions, and any later use must treat it as untradeable then.

Tests in `tests/test_universe_merges.py`:

- the exception joins only the named segments;
- a rename alone does not bridge the gap;
- a missing quote leaves the two apart.

## 2. Point-in-time issuer identification

In Stage 1, a still-trading security took as its issuer the company that holds its ticker
in SEC's ticker list *today* (2026), if that company had filed during the period. Tickers
move:

- SQ is now XYZ;
- XOM now points to a new ExxonMobil holding company;
- IAC passed from IAC/InterActiveCorp to a new IAC in 2020.

Nine common stocks were left without an issuer, and a SPAC (FEAC) went undetected.

Now an issuer is dated evidence (`issuer_evidence` table, `evidence.issuer_points`):

- **cover_page**: the company's own 10-K, 10-Q, 8-K, 20-F or 40-F lists the ticker under
  "Trading Symbol(s)", followed by its exchange. SEC has required that cover table since
  2019. The document is read to confirm each company's first and last filing with the
  ticker. A company holds a ticker for one stretch of time, so the end is found by halving.
- **form25**: the delisting notice at the last trade. It counts only when the notice itself
  was read and names the company as the issuer (see section 7).
- **full_text**: before 2019, or for a filer without the table, a filing in which the
  company names the ticker as its own ("(NYSE: XOM)" right after its name). Every
  significant word of the company's name must stand right before the ticker. U.S. Gold
  Corp's proxy statement names a director's former employer, "Barrick Gold Corporation
  (NYSE: GOLD)", and that does not make U.S. Gold the holder of GOLD.

Candidate companies come from:

- the delisting notice;
- SEC's ticker list as of today;
- a cover-page search not limited to one company;
- companies that wrote the ticker in an exchange context;
- the companies found for the security's other tickers (a SPAC and the business it became
  share a CIK).

**A candidate is accepted only on its own cover pages. Today's ticker list proposes
companies but never decides.** When two companies' cover pages overlap in time (a REIT and
its operating partnership filing together), the one with more filings is kept, and the
overlap is reported. The exception is a company whose own cover pages go on for more than
120 days beyond the other's. That is a new holder whose filings only touch the old one's
around the closing (HTA taking HR in July 2022), and its own stretch is kept.

Known limitation: for co-registrants that file the same documents, "more filings" can pick
the subsidiary over the parent. PG&E's utility over PG&E Corp, United Rentals North
America, Restaurant Brands' partnership and Hertz Corp are examples. Both are in the same
filings and the same industry, so no blank-check or membership decision changes. All 78
overlaps are listed in section 4e.

**The issuer on a day** is the company of the latest evidence on or before that day
(`evidence.issuer_on`). The blank-check test uses that company's SIC as of that day. The SIC
is read only from filing headers dated on or before the day.

- Before a security's first evidence, the earliest evidence is used, and the report counts
  every decision where that happens.
- Trading-gap decisions compare the issuers at the gap: the earlier security's at its last
  trade, the later security's at its first.
- SEC's older submission pages are now read, so a heavy filer (Toronto-Dominion Bank) no
  longer looks as if it filed nothing in 2016–2024.

Look-ahead tests (`tests/test_universe_issuers.py`):

- dropping every SIC filing and every issuer document dated after a day never changes that
  day's blank-check answer;
- the issuer follows a ticker that passes between companies;
- today's ticker list cannot decide;
- a search hit whose cover page does not list the ticker is not evidence;
- a SPAC and its successor share one company across the rename.

## 3. Deal terms for AGN, WORK and RTN

Terms are quoted from each target's own 8-K on completion. Each value is cash plus shares
times the acquirer's raw SIP close on the effective date, and is used only if it is within
10% of the target's last close (the existing rule):

| Target | Terms per share | Effective | Acquirer price |
| --- | --- | --- | --- |
| AGN (Allergan) | $120.30 cash + 0.8660 AbbVie | 2020-05-08 | ABBV close |
| WORK (Slack) | $26.79 cash + 0.0776 Salesforce | 2021-07-21 | CRM close |
| RTN (Raytheon) | 2.3348 UTC shares (UTC renamed Raytheon Technologies, RTX) | 2020-04-03 | RTX close |

For RTN, UTC spun off Carrier and Otis to its own holders of record. Raytheon holders did
not receive them, so RTX's close on the merger day is the value.

## 4. Security classification rules

The rules are printed at the top of section 3 of the report (`app/universe/sectypes.py`):

1. **Common stock** includes REITs, tracking stock and SPAC shares. It is the only type the
   universe takes.
2. **ADRs** (depositary shares or receipts, New York registry shares) are excluded by design.
3. **ETFs, closed-end and mutual funds, and ETNs** are never common, whatever the issuer is
   called.
4. **Warrants, rights, units, preferred, partnership units and debt** come from the class
   title, OpenFIGI or Nasdaq's fifth letter.
5. **SPACs** are common by type. They are excluded by the point-in-time SIC 6770 rule, not by
   type.
6. **Wikidata describes the company, not the listed security.** "Public company", "bank" and
   "real estate investment trust" all mean *company*, which fits common stock, ADRs,
   partnership units and anything else a company issues; only a fund or a note
   contradicts it.
   - In Stage 1, "real estate investment trust" matched "investment trust" and was read as
     a fund. That is how CCI, DLR, SPG, AVB, INVH, CONE and AAIC fell into conflict; all are
     now confirmed common stock.
7. **New source, `sec_cover`:** the class title the issuer gives the ticker on its own cover
   page. It votes only when the title names exactly one class:
   - not when the ticker is on two rows (KSU's preferred and common; TIGR's shares and
     ADSs);
   - not when a column layout runs several titles together ("Common Stock ... 1.900% Notes
     Due 2022", Walmart and Costco 2019);
   - only from cover pages filed in the membership window (2020 on): a type applies to
     every decision, and Blackstone's partnership units became common stock in July 2019;
   - on a 20-F only an explicit ADS title counts, because a 20-F registers the underlying
     shares with a footnote that they trade as ADSs (TSM).
8. **OpenFIGI** describes today's holder of a ticker.
   - It is set aside when its name disagrees with the issuer's names in SEC filings of the
     period, and an ETF answer agrees with an issuer that is a fund trust.
   - For a ticker the security no longer used at the end of 2024, it is also set aside when
     its name disagrees with Wikidata.
   - For a ticker still in use, a disagreement with Wikidata alone is a conflict, not a
     choice.
9. **Disagreements are settled only by the issuer's own SEC description:** a hand-read SEC
   passage first, then the cover-page class title. Such a type is `resolved`, and the
   outvoted evidence is kept and listed in the report. Otherwise it stays in `conflict`
   (for example, CRH: ADR until 2023, ordinary shares after). Nothing is guessed.

`confirmed`, `single_source` and `resolved` common stocks may be members. Each type keeps
its sources (`type_evidence`), including the votes not counted, with the reason.

## 5. Renames confirmed by the issuer

The first full Stage 1.1 run found five securities split in two at a rename. Alpaca's rename
records were dated 20–48 sessions after the real switch (CBS→VIAC, PSAC→FFIE, SGAM→RDBX,
BTX→ERNA), or missing (LB→BBWI). Stage 1.0 could not see this without issuers.

`evidence.decide_issuer_renames` treats two consecutive securities under different tickers
(at most 10 sessions apart) as one security only if all of these hold:

- one registrant's cover pages list the old ticker within 120 days before the switch and the
  new one within 120 days after;
- no delisting notice was filed, neither one matched to the security nor one in the
  registrant's own filing list from 45 days before the switch;
- no Alpaca merger acquired the old ticker;
- the price continues within 0.5–2×.

Every decision, including each rejection, is recorded with its reasons. `audit_merges.py`
rechecks it from raw data.

The rule is conservative: a Form 25 at the switch keeps two securities apart even when it
marks only an exchange transfer (CBS left the NYSE as ViacomCBS listed on Nasdaq) or a SPAC's
units (PSAC, SGAM). Those three are listed as open items below.

## 6. Tickers that passed to another company without a pause

The comparison against Stage 1.0 found five securities that were really two companies.
In each case a ticker passed to a new holder from one session to the next. SIP keeps
returning one uninterrupted series for the ticker, so Stage 1.0 joined the two companies:

| Ticker | Earlier holder, until | New holder, from | How the earlier holder left |
| --- | --- | --- | --- |
| CZR | Caesars Entertainment Corp, 2020-07-20 | Eldorado Resorts (ERI), 2020-07-21 | acquired by Eldorado |
| IR | Ingersoll-Rand plc, 2020-02-28 | Gardner Denver (GDI), 2020-03-02 | renamed itself Trane (TT) |
| HR | Healthcare Realty Trust, 2022-07-20 | Healthcare Trust of America (HTA), 2022-07-21 | acquired by HTA |
| DOC | Physicians Realty Trust, 2024-02-29 | Healthpeak (PEAK), 2024-03-04 | acquired by Healthpeak |
| GOLD | Randgold Resources ADSs, 2018-12-28 | Barrick Gold (NYSE, formerly ABX), 2019-01-02 | acquired by Barrick |

The joined series were wrong in three ways:

- the acquirer looked as if it had ended ("ERI acquired", "HTA acquired");
- the target's real exit was missing;
- GOLD showed a false crash from 82.89 to 13.10, which was in fact the 6.1280 exchange
  ratio.

**The rule** (`segments.handover_cuts`) needs two Alpaca records at the same time:

- a rename X → T onto a ticker T that is still trading;
- T's earlier holder's own exit: a merger naming T as the acquired company, or a rename
  away from T.

The run of T is then cut at the new holder's first trade, and the rename links that part
to the new holder's own history. A late rename date alone never cuts. GOLD has no Alpaca
rename, so it is a documented exception (`CURATED_HANDOVERS`). It is applied only while its
four passages from Randgold's own 6-Ks are found, including "Randgold's NASDAQ trading
ticker "GOLD" will be transferred to Barrick on the NYSE". Randgold's exit value comes from
quoted terms: 6.1280 Barrick shares per share, and one ADS per share.

`scripts/audit_merges.py` re-derives every handover from the raw bars and raw events with
separate code. For each one it checks that:

- the stored securities split the run exactly at the cut;
- the new holder's security came from the renamed ticker;
- the issuer evidence differs across the cut and continues the new holder's own company.

Every place where the securities split one uninterrupted run must be one of these
handovers.

## 7. How the issuer evidence is fetched, cached and checked

- **Prefetch:** `scripts/prefetch_issuer_evidence.py` runs the pipeline's issuer step with six
  securities in flight. It stays under SEC's limit with one request every 0.15 s across all
  workers, and retries as the pipeline does. It stores every answer in the same cache. The
  stage run then reads them from the cache.
- **Every request is accounted for:** the report lists, for each SEC step (curated
  documents, evidence, issuers) and added up over every pass, the HTTP responses (retries
  included), the answers stored and the failures. It then checks that the cache grew by
  exactly the answers stored. Before this, only the evidence step was shown, and only its
  last pass: run A5 reported 0 requests while the issuer step had fetched 6 cover pages
  (the CCO Holdings and Biohaven Ltd documents the end-trim below needed).
- **Cover pages are cached as snippets:** only the plain text up to just past the "Trading
  Symbol" table is stored (a few KB, `CachedSec.cover_text`), never the whole 10-K. A
  filing already cached whole gives its snippet without a request.
- **A missing answer stops the run:** the issuer step reads the cache through `StrictCache`,
  which stops the run on any answer it needs but cannot find, rather than reading it as
  "no issuer". A request SEC never answers stops the run too.
- **Resuming is safe:** answers are keyed by request and replaced, never appended, so a rerun
  cannot duplicate them. The issuer tables are rebuilt from the cache on every run.
- **Independent audit:** `scripts/audit_issuers.py` uses separate code and checks that:
  - the issuer step replayed from the cache alone matches what was stored;
  - every evidence row is in the company's own EDGAR folder and filing list, and inside the
    ticker's period;
  - the first and last cover page of each run, re-read, list the ticker;
  - every own-filing (full_text) document, re-read, shows the company naming the ticker
    as its own;
  - every delisting notice (form25), re-read, is a Form 25 in the company's own filing
    list, filed near the last trade, and names that company as the issuer;
  - every member decision has an issuer;
  - every blank-check exclusion, and every member, re-derives from SIC filed by the
    decision date.

  It also prints a fingerprint of the evidence, so runs can be compared.
- **Search refinements found during the runs:**
  - spread probes plus halving find a company whose first filings predate SEC's cover table
    (McDonald's);
  - a stretch of a ticker period with no evidence is searched for other holders (Apache
    Corp before APA Corp);
  - a candidate whose cover heading SEC's index misses is read through its filings naming
    the ticker (ViacomCBS);
  - both ends of a company's stretch must be documents that list the ticker themselves.
    The halving lets one document without the table pass on the strength of its neighbour
    (an exhibit filed as the main document). At the start of a stretch, that had let IAC's
    10-Q of May 2020 open IAC's MTCH stretch, though its cover lists IAC. IAC's first cover
    page listing MTCH came after the separation in July 2020. This gave IAC the MTCH ticker
    in May–June 2020, and moved 102 other stretches one filing early. The same holds for a
    new holder's stretch cut out of its run by the overlap rule (CCO Holdings for CHTR,
    Biohaven Ltd for BHVN). `audit_issuers.py` found both, and tests reproduce both.
  - a Form 25 is evidence only when its notice was read. SEC's match accepts a company
    that filed any Form 25 around the last trade. For 35 securities the notice could not
    be read (no machine-readable primary document), yet the match had been turned into
    evidence. For 33 of them the security's own cover pages or filings independently name
    the same company. AXON did not: Hercules Capital, a lender whose filings mention
    Axovant's ticker, filed a Form 25 for its own notes the day AXON last traded, and
    became AXON's issuer. AXON's issuer is now Axovant Sciences (its own 2016 filing). The
    35th, DD (E. I. du Pont de Nemours, CIK 30554, last trade 2017-08-31), has no other
    issuer evidence and now has no issuer; it traded only before the first membership
    decision, so no decision reads it. Such a company is now only a candidate, accepted on
    its own filings like any other. Report section 4b marks each such match "candidate
    only", with the issuer from dated evidence. Its 8-Ks are still read as exit evidence:
    for the 33 they are the issuer's own; AXON's candidate has none; DD's single-source
    "acquired" outcome rests on CIK 30554's 8-K of 2017-09-01 (the DowDuPont merger). The audit had
    skipped Form 25 rows; it now re-reads every notice: in the company's own folder and
    filing list as a Form 25, filed near the last trade, naming the company as issuer.
