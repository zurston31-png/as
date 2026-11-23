# SIP universe validation batch (research data only)

The same 136-ticker batch, re-run with SEC EDGAR as an additional independent source.
**No strategy was run, nothing was backtested, no parameter was frozen, and no bars
or corporate actions from 2025 or later were read.** Evidence dated 2025 or later is
ignored, and the tests check that.

The IEX market data and the Strategy A archive are untouched:

- IEX bars: identical hash (15,861 bars).
- Strategy A: all archive checksums OK.

The batch lives in a separate database (`data/sip_research.db`, git-ignored).

| File | What it is |
| --- | --- |
| [`validation_report.txt`](validation_report.txt) | v3, the current report (adds section 4b, the SEC cross-check) |
| [`validation_report_v2.txt`](validation_report_v2.txt) | v2: OpenFIGI and Wikidata, before SEC |
| [`validation_report_v1.txt`](validation_report_v1.txt) | The first batch |

Two consecutive runs from the cache give byte-identical reports.

## Sources

| Source | Used for | Coverage in this batch | Access |
| --- | --- | --- | --- |
| Alpaca corporate actions | Mergers with terms, renames (incl. Q = bankruptcy), splits, spin-offs | Mergers and renames dense from 2020; spin-offs dense from 2023 (120 records 2016-2024) | Existing |
| OpenFIGI (Bloomberg) | Security type of a ticker's current holder | 75 of 113 tickers | Read-only, no key |
| Wikidata | Company records: owners and parents with dates, bankruptcy and collapse events, listing end dates, company class | 60 of 113 tickers | Read-only |
| **SEC EDGAR (new)** | Delisting notices (Form 25 / 25-NSE): class of security and delisting rule; 8-K items on bankruptcy and completed mergers; cash terms from the merger 8-K | 54 of 90 ended securities matched | Read-only; 433 requests in the batch runs plus about 270 in exploratory probes; 0 failures |
| Nasdaq ticker convention | 5th letter W/R/U/X = warrant/right/unit/mutual fund | Five-letter tickers | Rule |
| Hand-entered | UTX spin-off terms, with source text and a price check | 1 entry | `app/universe/curated.py` |

All answers are stored in the research database and reused, so the report can be
reproduced; `--refresh` asks again.

SEC requests identify SignalIQ in the User-Agent with the dedicated research contact
address, as SEC requires. They only read public EDGAR pages: company filing lists,
company-name searches, Form 25 documents and merger 8-Ks. Nothing was submitted, and
no account was created.

## 0. SEC cross-check

Code: [`app/universe/sec.py`](../../../app/universe/sec.py). Tests:
[`tests/test_universe_sec.py`](../../../tests/test_universe_sec.py) (42, fixtures only).

**How a security is matched, conservatively:**

1. **Search keys only propose candidate companies.** They come from the tickers, the
   company names from Wikidata and OpenFIGI, and 14 names in the batch definition.
2. **A candidate is accepted only if it filed a Form 25 or 25-NSE close to the last
   trade** (45 days before to 90 days after) and before 2025. This rejects today's
   holder of a reused ticker: MON is now Monument Circle, which delisted nothing in
   2018.
3. **Type:** the notice's class of security ("Common Stock", "Ordinary Shares",
   "Warrants", "6.25% Notes"...). A notice covering several classes ("Class A common
   stock and warrants") gives common to a plain ticker and warrant to a ...W ticker.
   Notices for the issuer's other classes (Time Warner's debentures) are not counted.
   A fund's "Common Shares" (GMZ) do not count as common stock. SIC is recorded as
   detail only.
4. **Outcome**, from the delisting rule and the 8-Ks:
   - 12d2-2(a)(3) (merger) or an 8-K with items 2.01 + 3.03/5.01: acquired.
   - (a)(1)/(a)(2): redeemed, a new outcome with no assumed value.
   - (b), the exchange removed it: says nothing about why.
   - An 8-K with item 1.03 within a year before the delisting: bankruptcy.
   - Cash per share only when the merger 8-K states cash with no stock component.
   All SEC filings count as one source.

**Rules added after reading the results**, each with a test:

- **SPAC liquidation:** Alpaca records a SPAC's trust liquidation as a "cash merger" at
  the trust value, and SEC as a redemption. That is one cash exit, not a conflict
  (UTAA $10.68, BACA $10.68).
- **A cash record beside a stock record of the same deal means cash plus shares.**
  This fixed an error carried over from v2. VMW had been valued at $142.50 and Diamond
  Offshore (2024) at $5.65, the cash part alone. Both are now cash-and-stock
  acquisitions with no value assumed.
- **Search-name variants:** EDGAR matches the start of its official name, so
  "LinkedIn Corporation" is also searched as "linkedin corp". This added LinkedIn
  ($196 cash) and J.C. Penney (bankruptcy) as SEC confirmations.

### The 30 cases you asked about

**19 securities of unknown type in v2: 8 resolved, 11 remain unknown.**

| id | Ticker | Last trade | SEC company (CIK, SIC) | Notice | Type now | Outcome now |
| --- | --- | --- | --- | --- | --- | --- |
| 14 | CRC | 2020-07-15 | California Resources (1609253, 1311) | 25-NSE 2020-07-31, Common Stock, (b); 8-K 1.03 | common | bankruptcy, 0.0 |
| 18 | DO | 2020-04-24 | Diamond Offshore (949039, 1381) | 25-NSE 2020-05-12, Common Stock, (b); 8-K 1.03 | common | bankruptcy, 0.0 |
| 27 | FTR | 2020-04-23 | Frontier Communications (20520, 4813) | 25-NSE 2020-04-29, Common stock, (b); 8-K 1.03 | common | bankruptcy, 0.0 |
| 38 | MON | 2018-06-06 | Monsanto Co /NEW/ (1110783, 2870) | 25-NSE 2018-06-07, Common Stock, (a)(3); merger 8-K | common | acquired for cash, $128.00 |
| 52 | SHLD | 2018-10-23 | Sears Holdings (1310067, 5311) | 25-NSE 2018-11-15, Common Stock, (b); 8-K 1.03 | common | bankruptcy, 0.0 |
| 80 | CFII→VIEW | 2024-04-04 | View, Inc. (1811856, 3211) | 25-NSE 2024-04-25, Class A common stock and warrants, (b); 8-K 1.03 | common | bankruptcy, 0.0 (confirmed with Wikidata) |
| 98 | BACA | 2024-08-14 | Berenson Acquisition I (1869673, 6770 blank check) | 25-NSE 2024-10-16, Class A Common Stock, (a)(1) | common | redeemed, no value |
| 109 | YGF | 2024-02-08 | YanGuFang International (1875496, 2000) | 25-NSE 2024-04-09, Ordinary Shares, (b) | common | **unresolved**: Nasdaq delisting, no reason |

All 8 are single-source common (SEC only), so they become eligible for membership.

| id | Ticker | Last trade | Why SEC did not resolve it |
| --- | --- | --- | --- |
| 21 | ENRJ | 2018-03-26 | No candidate company: the ticker is unknown to SEC and no source names the company |
| 46 | RATE | 2017-11-08 | Same; the ticker trades again later |
| 65 | WDRW | 2016-03-11 | Same; a 30-session fragment, ticker reused later |
| 66 | MDGS | 2016-01-26 | One trading session; the ticker resolves to a different company |
| 70, 72 | OILD | 2016, 2020 | The ticker resolves to the ETN issuer (BMO). ETNs are notes and file no Form 25 under the ticker |
| 71 | FSBC | 2020-04-30 | Today's FSBC is Five Star Bancorp (a reuse). Alpaca shows the 2020 holder was acquired by Evans Bancorp (EVBN) for cash and stock, but no source names that company |
| 74 | JMPD→JMPNL | 2021-09-10 | Baby bonds (exchange-listed notes) of JMP Group; no company candidate |
| 76 | SMMC | 2019-10-29 | No candidate company |
| 83, 92 | SHLD | 2021 | Post-bankruptcy fragments after Sears was delisted in 2018 (16 to ~60 sessions); no Form 25 then |

They stay excluded from the common-stock universe as unknown type. At least FSBC is
an acquired common stock, so excluding it is a small survivorship bias (see the
limitations below).

**11 common stocks with an unresolved outcome in v2: 9 resolved, 2 remain unresolved.**

| id | Ticker | Last trade | SEC company | Evidence | Outcome now |
| --- | --- | --- | --- | --- | --- |
| 7 | BBBY | 2023-05-02 | 20230930-DK-Butterfly-1 (886158), formerly Bed Bath & Beyond | 25-NSE 2023-07-10, (b); 8-K 1.03 on 2023-04-24 | bankruptcy, 0.0 |
| 10 | CELG | 2019-11-20 | Celgene (816284) | 25-NSE (a)(3); merger 8-K | acquired, terms not extracted (cash plus BMS shares plus a CVR), no value |
| 17 | DISCA | 2022-04-08 | Warner Bros. Discovery (1437107) | 25-NSE for Series A, B, C common, (a)(3); merger 8-K | acquired, no value |
| 20 | ENDP | 2022-08-25 | Endo International (1593034) | 25-NSE 2022-09-14, Ordinary Shares, (b); 8-K 1.03 | bankruptcy, 0.0 |
| 22 | EXPR | 2024-03-06 | EXP OldCo Winddown (1483510), formerly Express | 25-NSE 2024-03-14, (b); 8-K 1.03 | bankruptcy, 0.0 |
| 42 | PRTY | 2023-01-17 | Party City Holdco (1592058) | 25-NSE 2023-02-03, (b); 8-K 1.03 | bankruptcy, 0.0 |
| 47 | REV | 2022-10-20 | Revlon (887921) | 25-NSE 2022-10-21, (b); 8-K 1.03 June 2022 | bankruptcy, 0.0 |
| 59 | TWX | 2018-06-14 | Warner Media LLC, formerly Time Warner Inc (1105705) | 25-NSE 2018-06-15, (a)(3); merger 8-K | acquired, terms not extracted (cash plus AT&T shares), no value |
| 87 | MON | 2022-12-23 | Monument Circle Acquisition (1828325, blank check) | 25-NSE, (a)(1) | redeemed (SPAC), no value |
| 26 | FRC | 2023-04-28 | none | First Republic Bank filed with the FDIC, not the SEC. Wikidata only dates its dissolution | **unresolved** |
| 30 | HTBX→NHWK→SCPX | 2024-06-14 | none | SCPX trades again later (a halt or a reuse); no Form 25 in the window | **unresolved** |

All 9 resolved outcomes are single-source (SEC). The 6 bankruptcies take the agreed
total-loss default. The acquisitions and redemptions carry no value.

**A note on Time Warner and Monsanto:** the first SEC probe missed both. SEC's name
search jumps to older companies of the same name: "monsanto co" goes to the
pre-2000 company and "time warner inc" to the pre-2001 one. Their search names in
the batch definition are now "monsanto co /new/" and "warner media". These are
search keys only; the delisting notice dated the day after the last trade is what
confirms the match.

## 1. Exit outcomes: all ended securities

The rules are written in [`app/universe/outcomes.py`](../../../app/universe/outcomes.py):

- **Evidence of two different exits** (acquisition, redemption, failure): conflict,
  no value. The exception is a SPAC cash redemption (see section 0).
- **Acquisition:** cash value only when a merger record states it and no record shows
  a stock part; otherwise no value.
- **Redemption:** no value unless a merger record states the cash.
- **Confirmed bankruptcy or failure:** the agreed total loss.
- **Anything else:** unresolved, with no value. An unknown exit is never a total loss.

| Outcome | Confirmed (2 sources) | Single source | Unresolved |
| --- | --- | --- | --- |
| Acquired for cash | 13 | 4 | |
| Acquired for stock / stock and cash | 6 / 4 | 0 / 1 | |
| Acquired, terms unknown | | 6 | |
| Redeemed (SPAC trust or class retired) | 2 (UTAA, BACA $10.68) | 3 | |
| Bankruptcy | 9 | 9 | |
| Failure | 1 (SVB) | 1 (Signature Bank) | |
| Ticker resumed later or short fragment | | | 24 |
| No explanation | | | 7 |

- **Overall:** 90 ended securities. 35 are confirmed (v2: 5), 24 are single-source
  (v2: 37), and 31 are unresolved (v2: 49). There are no conflicts.
- **Values:** 39 have a value (19 cash amounts, including 2 SPAC redemptions, and 20
  total losses) and 51 have none.

**The 31 unresolved:**

- **Common stocks: FRC, SCPX, YGF.** These are the ones that matter for membership.
  FRC was a member until 2023-03-31.
- **Unknown type, not eligible (10):** ENRJ, RATE, WDRW (2016), MDGS, OILD ×2,
  JMPD→JMPNL, SMMC, SHLD ×2, from the table above.
- **SPAC warrants and rights, not eligible:** NOVVW ×10, CETUR ×3 and HSACW ×2 are
  fragments whose ticker trades on and off; IMVTW and ITHXW have no explanation.
- **WDRW (2020), an ETF: no explanation.**

None of them is assigned a value.

## 2. Spin-offs: UTX and the Carrier/Otis separation

- **Alpaca's error:** Alpaca records 2020-04-03 as a **1-for-2 reverse split** of UTX.
  The prices contradict it: a reverse split should double the price, but it fell from
  $86.01 to $49.93. Every recorded split is now checked against the prices on either
  side. A contradicted record is not applied; this one is rejected.
- **The correction:** the separation is entered by hand as 1 Carrier and ½ Otis per UTX
  share, with its source text. Using real closes on the ex-date (CARR $16.92, OTIS
  $47.32), a holder's position went from $86.01 to $49.93 + $40.58 = **$90.51, +5.2 %**.
  An ordinary day; the record passes.
- **The factor:** UTX history before the ex-date is divided by (49.93 + 40.58) / 49.93 =
  1.8127. The adjusted series now shows exactly the holder's +5.23 % on that day.
  - A simple dividend-style factor would have shown +9.9 %, because the distribution
    was almost half the share price. That error was found and fixed during this work.
  - The price check rules out gross errors but cannot prove the exact ratio (½ Carrier +
    ½ Otis would also look plausible). The entry stays flagged until it is checked
    against the filing.
- **Other spin-offs:** Alpaca's 120 spin-off records are applied the same way, with the
  same price check. None belonged to a batch security.

**Split checks:** 47 splits of batch securities were checked against prices.

| Verdict | Count | Splits | Treatment |
| --- | --- | --- | --- |
| Consistent | 42 | | Applied |
| Contradicted | 2 | UTX 2020, LILAK 2016 (a "1-for-8" the price ignored) | Not applied |
| Size mismatch | 3 | Hertz 2016 (spin-off of Herc the same day), Top Ships 2017 ×2 (reverse splits during a collapse) | Applied, flagged; security held out of membership until reviewed |

## 3. Security types

Rules: [`app/universe/sectypes.py`](../../../app/universe/sectypes.py). SEC's class of
security is a new, independent vote (`sec_form25`).

| Type | Confirmed | Single source |
| --- | --- | --- |
| Common | 58 (v2: 31) | 19 (v2: 38) |
| ETF | 1 | 10 |
| Warrant | 5 | 13 |
| Right | 2 | 3 |
| Fund / ADR | | 1 / 1 |
| Unknown | | 11 (v2: 19), all ended |

**Conflicts:** none in the final run.

- SEC first produced one conflict, GMZ: OpenFIGI said closed-end fund and the notice
  said "Common Shares". Fixed by the fund-issuer rule; GMZ is now fund
  (single-source).
- **Rules carried over from v2:** OpenFIGI's answer only goes to the last user of a
  ticker; it is name-checked against Wikidata for tickers no longer used; Wikidata
  rows are matched to the dates the ticker was used.

**Membership:** the 43 distinct members are all common stocks: 36 confirmed, 7
single-source. The no-look-ahead proofs pass:

- 59 of 59 decisions are identical when recomputed with data cut off at each decision.
- Changing every price after mid-2022 changes no earlier decision.
- 0 securities were chosen after their last trade.

## Unresolved problems and limitations (before the full universe)

1. **11 securities still have no type.** They are all ended, mostly ETNs, notes and
   tiny or reused tickers. At least one, FSBC, is an acquired common stock, and its
   exclusion is a small survivorship bias. SEC cannot find a company when no source
   names it and the ticker now belongs to someone else.
   - **Next step:** EDGAR full-text search for the ticker in filings of the time, or
     SEC's historical company-ticker data, if available. Both would need your
     approval.
2. **3 ended common stocks have no outcome:**
   - **FRC** filed with the FDIC, not the SEC. Its fate (FDIC seizure, sale to
     JPMorgan, shareholders wiped out) is public knowledge, but none of the
     automated sources state it.
   - **SCPX** resumed trading under the same ticker.
   - **YGF** was removed by Nasdaq. The notice text gives the determination but not a
     usable reason.

   **Your decision:** whether to enter FRC as a curated failure (with a source text,
   like UTX), or keep it unresolved.
3. **Many outcomes rest on SEC alone:** 24 single-source overall, 18 of them from SEC.
   SEC is a primary source, but its classifications come from parsing: the 8-K item
   numbers and the 12d2-2 rule cited.
4. **Values for stock mergers and cash-plus-stock deals are not computed** (11
   securities, including CELG, TWX, VMW and DO 2024). They need the acquirer's price
   and the exchange ratio. Cash terms are extracted from an 8-K only when cash alone
   is paid. Old Hertz shows that the total-loss default can be wrong.
5. **SPACs:** 5 redeemed securities (SPAC shares and one warrant), and blank-check
   companies (SIC 6770) pass the common-stock filter as common.
   - **Your decision:** whether SPAC shares belong in the universe at all. The SIC
     code makes them easy to exclude.
6. **SEC matching has gaps** (36 of 90 ended securities unmatched). Most are SPAC
   instrument fragments, ETNs, bank filers (FDIC: FRC, Signature Bank) and companies
   no source names. Red Hat and Mellanox were not matched either: no candidate
   company filed a Form 25 in the window under the names tried. Their outcomes come
   from Alpaca and Wikidata.
7. **Unchanged from v2:**
   - The UTX ratios are hand-entered and only price-checked.
   - Alpaca's split and spin-off records are incomplete before 2023.
   - Wikidata coverage is uneven.
   - No SIP history before 2016, and no OTC prices after a delisting.
8. **Disclosure (unchanged):** one exploratory Alpaca news request without dates
   returned three 2026 headlines. They were not stored or used.

## Reproduce

```bash
python scripts/universe_validation_batch.py --report validation_report.txt
```

The first run asks SEC (about 430 requests at about 6 per second); later runs read the
cache.
