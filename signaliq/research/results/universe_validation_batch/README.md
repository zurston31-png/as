# SIP universe validation batch (research data only)

A test of the identity, ticker-history, corporate-action, price-adjustment and
point-in-time membership logic for a broader US stock universe. It used 136
tickers before any full download. **No strategy was run, nothing was backtested,
no parameter was frozen, and no data from 2025 or later was read.**

The IEX market data and the Strategy A results were not touched: the batch lives
in a separate database (`data/sip_research.db`, git-ignored, with its own
migrations and version table), and the market data database is byte-identical to
before.

[`validation_report.txt`](validation_report.txt) is the full output of
`python scripts/universe_validation_batch.py`. Two consecutive runs produce
byte-identical reports and identical database contents.

## Batch

| Selection | Tickers | How chosen |
| --- | --- | --- |
| Known cases | 54 | By hand, each with the real-world outcome written down beforehand: 9 active (overlap with the IEX data), 3 bank failures, 18 bankruptcies, 18 acquisitions, 5 renames, 1 reused ticker |
| Random | 46 | Seeded draw (seed 20261119) from Alpaca's historical corporate-action events: 12 cash-merger targets, 12 stock-merger targets and 12 renamed tickers from 2020-2024, and 10 tickers that split in 2016-2018 |
| Discovered | 36 | Every ticker a rename connects to a batch ticker, followed until no new ticker appears |

Today's ticker list was not used anywhere.

## What was proven

- **Security ids.** The 136 tickers produced 124 securities. A security is a chain of
  ticker listings joined by rename events. All 54 known cases came out right: renames
  joined (FB→META, UTX→RTX, FISV→FI, ANTM→ELV, CHK→EXE, BOWX→WE, YRCW→YELL, FI→XPRO),
  and reused or re-listed tickers split into separate securities:
  - CHK: old Chesapeake to 2020-06-26; new company from 2021-02-10.
  - HTZ: old Hertz to 2020-10-29; new company from 2021-11-09.
  - META: an ETF in 2021-06..2022-01, later METV; Facebook's history sits under FB.
  - Also CRC, DO, MON and FI.
- **Filler bars.** When a ticker stops trading, Alpaca keeps returning one bar per session
  with zero volume, zero trades and the last price repeated, until the ticker trades
  again, possibly for another company. 8,171 such bars were found in 48 tickers. They
  are never used as prices, and a silence of more than 10 sessions ends a listing.
- **Ticker history against Alpaca's own company mapping.** For 96 of 124 securities,
  every real trade equals Alpaca's company-mapped series. Every difference is a gap or
  an error in Alpaca's mapping, not in ours. Alpaca maps FI before 2023 to Frank's
  International (verified: $1.68 on 2020-09-30, while Fiserv traded as FISV at $103).
  Alpaca's mapping also does not reach the earlier history of many renamed or re-listed
  securities (old Hertz: 0 of 1,216 sessions).
- **Raw against adjusted prices.** Our own split adjustment, built from Alpaca's split
  events, equals Alpaca's split-adjusted close within 0.1 % on 116,715 of 124,185
  sessions. Each of the 8 securities that differ is explained under problems 4 (splits) and 7
  (FI and RATE mapping). UTX matches only because both sides apply the same mis-recorded split
  (problem 3).
  Of 156 raw day-to-day moves beyond +80 % / −45 %, 45 are recorded splits, 79 are real
  moves (Alpaca's adjusted series moves too), 1 is a missing split, and 31 have no
  adjusted price to compare.
- **Raw prices for filters.** In 285 membership decisions the $5 price filter answers
  differently on split-adjusted prices. Each one would let a later split decide an
  earlier membership.
- **Point-in-time membership** (validation rule: monthly, the 25 most-traded eligible
  batch securities by 63-session median raw dollar volume; raw price ≥ $5; ≥ 126
  sessions of history; ETFs declared as such excluded):
  - 59 decisions, 2020-01-31 to 2024-11-29.
  - Recomputing every decision from data cut off at that decision gives identical
    results (59 of 59).
  - Multiplying every later price and volume by 10 changes no earlier decision.
  - No security was chosen after its last trade.
  - 24 of 50 distinct members later ended: SIVB, SBNY, FRC, BBBY, SAVE, BIG, TWTR,
    ATVI and others were members until their last month. Survivorship is visible, not
    removed.

## Unresolved data problems (must be settled before the full universe)

1. **SEC Form 25 cross-check is blocked.** www.sec.gov is reachable, but SEC refuses
   requests whose User-Agent does not declare a contact email ("Undeclared Automated
   Tool"). A real contact address is needed, which needs the owner's permission. None
   was invented.
2. **Exit reasons are missing for most delistings: 56 of 90 ended securities are
   "unexplained".** Under the agreed default, all 56 are treated as a total loss.
   - **Real acquisitions treated as failures:** LinkedIn (2016, $196 cash), Monsanto
     (2018, $128), Time Warner (2018), Red Hat (2019, $190), Celgene (2019). Alpaca has
     no merger events before about 2020.
   - **Discovery→WBD (2022):** no event at all. Discovery was a membership candidate
     until 2022-03.
   - **Old Hertz:** shareholders were paid out, a known exception to total loss.
   - **SPACs:** some cash events are dated weeks after the last trade, for example BACA,
     39 sessions later.
   - **Price can't decide it.** SIVB last traded at $106 and SBNY at $70, because both
     collapsed while halted.

   A second, independent source of exit reasons is needed. Candidates are SEC Form 25
   plus merger filings, which run into problem 1.
3. **Spin-offs are absent, and one is mis-recorded.** Alpaca returned no spin-off events
   for the batch. It records United Technologies' spin-off of Carrier and Otis
   (2020-04-03) as a 1-for-2 "reverse split". The value of the spun-off shares accounts
   for the −42 % raw drop that day; a real 1-for-2 reverse split would instead make the
   move about +16 %. Both Alpaca's and our adjusted UTX history before that date are
   wrong. Spin-offs need their own source and handling.
4. **Split events are incomplete, and Alpaca's adjusted prices include later splits.**
   - **Missing split events:** two splits Alpaca applies are not in its event list
     (ENRJ 2016, 1:25), or were recorded so that a first rule missed them (fixed for
     Frank's→Expro).
   - **Later splits:** TOPS, OILD and NUWE differ from Alpaca's adjusted series by one
     constant factor on every session through 2024, which fits a split after 2024 that
     Alpaca applies to all history.

   SignalIQ must compute its own adjustment from dated events and must never filter on
   Alpaca's adjusted prices.
5. **Thinly traded instruments break into many ids.** One warrant ticker (NOVVW) became
   11 securities and CETUR became 5, because the 10-session silence rule splits them.
   Each fragment end currently looks like an exit with a total loss. A rule is needed for
   a ticker that resumes with no sign of a different company.
6. **Security type is unknown.** Alpaca gives no reliable security type. 14 of 50 members
   have no known type, and several are in fact ETFs (AIVL, BSJN, DES, OILD) or other
   non-common securities. The random draw also picked 10 mutual-fund tickers (5 letters
   ending in X) from merger events; they have no SIP bars. Common shares must be
   identified before the universe is built.
7. **Alpaca's own company mapping is unreliable for history**: problem 2 above and the
   FI and RATE cases. Securities must keep being built from literal tickers plus rename
   events, never from Alpaca's default mapping.
8. **Not fixable with this data.** Two limits remain: no SIP history before 2016, and no
   prices after a delisting to the over-the-counter market (all Q tickers have no SIP
   bars).

## Reproduce

```bash
python scripts/universe_validation_batch.py --report validation_report.txt
```
