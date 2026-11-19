# SIP universe validation batch (research data only)

The same 136-ticker batch as before, re-run with independent evidence for exit
outcomes, spin-offs and security types. **No strategy was run, nothing was
backtested, no parameter was frozen, and no bars or corporate actions from 2025 or
later were read.** Evidence dated 2025 or later is ignored, and the tests check
that.

The IEX market data and the Strategy A archive are untouched:

- IEX bars: identical hash (15,861 bars).
- Strategy A: 3 runs, 186 trades, no freeze, all archive checksums OK.

The batch lives in a separate database (`data/sip_research.db`, git-ignored).

| File | What it is |
| --- | --- |
| [`validation_report.txt`](validation_report.txt) | v2, the current report |
| [`validation_report_v1.txt`](validation_report_v1.txt) | The first batch, kept for comparison |

Two consecutive runs give byte-identical reports.

## Sources

| Source | Used for | Coverage in this batch | Access |
| --- | --- | --- | --- |
| Alpaca corporate actions | Mergers with terms, renames (incl. Q = bankruptcy), splits, spin-offs | Mergers and renames dense from 2020; spin-offs dense from 2023 (120 records 2016-2024) | Existing |
| OpenFIGI (Bloomberg) | Security type of a ticker's current holder | 75 of 113 tickers | New, read-only, no key |
| Wikidata | Company records: owners and parents with dates, bankruptcy and collapse events, listing end dates, company class | 60 of 113 tickers | New, read-only |
| Nasdaq ticker convention | 5th letter W/R/U/X = warrant/right/unit/mutual fund | Five-letter tickers | Rule |
| Hand-entered | UTX spin-off terms, with source text and a price check | 1 entry | `app/universe/curated.py` |

All requests sent only tickers and company identifiers, under a generic User-Agent.

OpenFIGI and Wikidata answers are stored in the research database and reused, so
the report can be reproduced; `--refresh` asks again.

Wikidata was in an outage during this work and allowed one query per minute; it
also returned one 502 error. The fetcher retries, never guesses, and reports any
ticker it could not look up (none in the final run).

**SEC is paused** until you provide a dedicated contact address. Alpaca's news API was
also tried: it returns nothing for 2016-2024 on this account.

## 1. Exit outcomes: independent evidence, unknowns kept unresolved

The rules were written before the results were seen
([`app/universe/outcomes.py`](../../../app/universe/outcomes.py)):

- **Acquisition evidence and failure evidence together:** conflict, no value.
- **Acquisition:** cash value only when a merger record states it; otherwise no value.
- **Confirmed bankruptcy or failure:** the agreed total loss.
- **Anything else:** unresolved, with no value. An unknown exit is never a total loss.

| Outcome | Confirmed (2 sources) | Single source | Unresolved |
| --- | --- | --- | --- |
| Acquired for cash | 4 (ATVI $95, HZNP $116.50, MLNX $125, VMW $142.50) | 16 | |
| Acquired for stock / stock and cash | 1 (ALXN) | 7 | |
| Acquired, terms unknown | | 2 (LinkedIn by Microsoft, Red Hat by IBM, from Wikidata) | |
| Bankruptcy | | 9 | |
| Failure | | 2 (SVB, Signature Bank, from Wikidata) | |
| Ticker resumed later or short fragment | | | 31 |
| No explanation | | | 18 |

- **Values:** of the 90 ended securities, 31 have a value (20 cash payouts, 11 total
  losses) and 59 have none.
- **No conflicts:** no source ever contradicts another.
- **The first batch was wrong on 56 securities:** it had called them "unexplained" and
  assumed a total loss. Now known acquisitions (LinkedIn, Red Hat) are acquisitions,
  and the rest stay unresolved instead of being counted as losses.

**Unresolved common stocks (11):** BBBY, FRC, Party City, Revlon, Endo, Express, Celgene,
Time Warner, Discovery, Monsanto, SCPX. Several are well-known failures (BBBY, FRC,
Party City, Revlon) or acquisitions (Celgene, Time Warner, Monsanto). Wikidata only
dates their listing end, without a reason, so they are flagged rather than guessed.
SEC Form 25 and merger filings are the expected way to resolve them.

The other 38 unresolved securities are warrants, rights, an ETF, securities of unknown
type, or fragments of thinly traded instruments. None of them can enter the
common-stock universe.

**Known risk in "confirmed" bankruptcies:** old Hertz shareholders were in fact paid out.
The total-loss default is wrong for them, which the planned sensitivity test must cover.

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

Rules: [`app/universe/sectypes.py`](../../../app/universe/sectypes.py).

- Only common stock enters the universe. ETFs, warrants, rights, units, preferred
  shares, funds, ADRs, partnerships and unknown types are excluded.
- A type is **confirmed** when two independent sources agree, and **single-source** with
  one. A declaration in the batch file is not counted as evidence.
- Sources that disagree give a **conflict**, never a guess.

| Type | Confirmed | Single source |
| --- | --- | --- |
| Common | 31 | 38 |
| ETF | 1 | 10 |
| Warrant | 5 | 13 |
| Right | 1 | 4 |
| Fund / ADR | | 1 / 1 |
| Unknown | | 19 (all ended) |

No conflicts remain. Getting there needed three safeguards against reused tickers,
each with a test:

- **OpenFIGI's answer goes only to the security that used the ticker last.**
- **For a ticker the security no longer used at the end of 2024, OpenFIGI's company
  name must agree with Wikidata's company for that period.** FB is now a ProShares
  ETF; it is rejected for Facebook.
- **Wikidata records are matched to the dates the ticker was used, and undated records
  are ignored for tickers several securities shared.** Meta Platforms is no longer
  matched to the 2021 META ETF, and Sears no longer to the 2023 SHLD ETF.

## Unresolved problems and limitations (before the full universe)

1. **Types missing for old holders of reused tickers.** All 19 unknown-type
   securities are ended companies. They include old Chesapeake (CHK), old California
   Resources (CRC), old Diamond Offshore (DO), Frontier, Monsanto and Sears. The
   current sources describe only today's holder of a ticker, and undated records
   cannot be placed in time. Excluding them removes failed and acquired companies
   specifically: **survivorship bias through the back door.** It must be fixed before
   the full universe. The best fix is SEC: a company ID and industry code, with
   dates. OpenFIGI's text search was tried and does not find old holders by ticker.
2. **11 ended common stocks have no outcome**, including BBBY and FRC (see 1. above).
   They would enter a backtest with no exit value. The expected fix is SEC Form 25
   plus merger filings.
3. **Most resolved outcomes rest on one source** (37 single-source, 5 confirmed).
4. **Stock-merger values are not computed** (they need the acquirer's price). Total
   loss on bankruptcy is wrong in known cases (old Hertz).
5. **The UTX ratios are hand-entered and only price-checked.** Spin-offs before 2023
   are sparse in Alpaca, so others may be missing. The price-jump check found no
   unexplained drop in this batch apart from the cases listed.
6. **Alpaca's split events are incomplete** (ENRJ 2016), and Alpaca's adjusted prices
   include later splits (TOPS, NUWE, OILD) and the false UTX split. SignalIQ never
   filters on Alpaca's adjusted prices.
7. **Wikidata is a present-day snapshot with uneven coverage** (60 of 113 tickers;
   small companies are thin). Its dates are sometimes announcement dates, not
   completion dates (LinkedIn: owner from June 2016, deal closed December 2016).
8. **Not fixable with this data:** no SIP history before 2016 and no OTC prices after a
   delisting.
9. **Disclosure:** one exploratory request to Alpaca's news API, sent without a date
   range, returned three Apple headlines from 2026. They were not stored or used, and
   every later request had an explicit date range ending 2024-12-31.

## Reproduce

```bash
python scripts/universe_validation_batch.py --report validation_report.txt
```
