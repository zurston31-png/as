# The 2021-10-25 anomaly (read-only investigation, stored data only)

Nothing was fetched. Sources: the Stage 1.1 database (version 1, opened `immutable=1`), the raw
Phase 1b responses (version 2), cached SEC submissions JSON. Evidence in `oct25/`.

## Answer

**It is a gap in Alpaca's data, not a market event and not our ingestion.** On 2021-10-25,
Alpaca's daily SIP bars for 218 liquid tickers, nearly all NYSE-listed, hold only the opening and
closing auction prints. The same defect affects 2021-04-19 (499 tickers; Alpaca repaired it
between 2026-11-26 and 2026-11-27) and a **third day nobody had flagged, 2022-03-08** (47 tickers).
2021-10-25 and 2022-03-08 are still broken in Alpaca's data today.

## Evidence

1. **Scan of every session, 2016-2024** (`oct25_scan.txt`). A bar is suspect when the ticker's
   median trade count over its previous 20 sessions is at least 1,000 and the bar has at most 5
   trades. Three sessions stand out; the other 13 sessions with any suspect bar have 1 or 2 each.

   | Session | Suspect | Liquid tickers | Share |
   | --- | --- | --- | --- |
   | 2021-04-19 | 499 | 892 | 55.9% |
   | 2021-10-25 | 218 | 894 | 24.4% |
   | 2022-03-08 | 47 | 902 | 5.2% |

2. **Shape: auction prints only** (`oct25_shape.txt`). 2021-10-25: trade counts 2-5 (104 bars
   with exactly 3); in 194 of 218 bars the high and low are just the open and the close; volume is
   7% / 10% / 15% (quartiles) of the ticker's 20-session median, about what the opening and closing
   auctions alone trade. Example: AFL 2021-10-22 18,305 trades, **2021-10-25 3 trades**, 2021-10-26
   28,035 trades.

3. **Exchange: NYSE-listed names** (`oct25_exchanges.txt`; listing from the cached SEC
   submissions JSON, which is the *current* listing, not point-in-time). 2021-10-25, where the
   exchange is known: 189 suspect NYSE against 3 Nasdaq; Nasdaq-listed tickers are almost
   untouched (3 of 269). Only part of NYSE is affected (189 of 449 NYSE names), with no
   alphabetical clustering. The 3 "Nasdaq" exceptions (DPZ, EVRG, ROP) may have been NYSE-listed
   in 2021 and transferred later; the stored data cannot confirm this. The 26 "unknown" are mostly
   companies no longer listed.

4. **Not our ingestion** (`oct25_versions.txt`). The raw Alpaca responses of 2026-11-27 contain
   the same auction-only bars: 215 of the 218 suspect bars on 2021-10-25 are identical to Stage
   1.1's stored rows (OHLC, volume, trades), and 217 still have at most 5 trades. 46 of 47 on
   2022-03-08 are identical. Our stored rows equal the provider's answer.

5. **Not a market-wide event.** Consolidated (SIP) data includes every venue's trades, so even
   with NYSE itself down, NYSE-listed stocks trade elsewhere. A consolidated day with 3 trades for
   AFL, CL or INFY is not plausible, and Alpaca has already repaired the identical defect on
   2021-04-19. A direct confirmation needs new data (below).

## Effect on the universe (Stage 1.1, version 1)

Nine decisions have one of the three days in their 63-session window. My recomputation
reproduces every stored median exactly (e.g. 741 of 741 eligible on 2021-10-29). With Alpaca's
real repaired bars for 2021-04-19 it reproduces the actual result: no member changes
(`oct25_membership.txt`).

| Decision | Bad day in window | Members changed: bars left out | Members changed: neighbour average |
| --- | --- | --- | --- |
| 2021-04-30 | 04-19 | +TDOC -HON | none |
| 2021-05-28 | 04-19 | +AMC -NCLH | none |
| 2021-06-30 | 04-19 | +DAL -CHTR | +ACN +DAL -CHTR -NCLH |
| 2021-10-29 | 10-25 | none | none |
| 2021-11-30 | 10-25 | none | none |
| **2021-12-31** | 10-25 | **+SPGI -TMUS** | **+SPGI -TMUS** |
| 2022-03-31, 04-29, 05-31 | 03-08 | none | none |

Both stand-in methods are imperfect: for 2021-04-19 Alpaca's real repair changed nothing, while
the two methods swapped 1 or 2 members at the 100th place. So:

- 2021-10-29, 2021-11-30 and the three 2022 decisions are robust to the defect;
- **2021-12-31 is uncertain**: SPGI (NYSE, defective bar) and TMUS (Nasdaq) sit at the 100th
  place, and both methods put SPGI in. Only real repaired data can settle it;
- the medians and ranks of every NYSE-listed security in those windows are slightly low.

## What would confirm it (NOT fetched; needs your approval)

Four read-only Alpaca market-data requests, all for 2021-2022 data, recorded into a new response
store (never into a protected one):

| # | Request | Count | What it shows |
| --- | --- | --- | --- |
| 1 | `GET https://data.alpaca.markets/v2/stocks/bars?symbols=AFL&timeframe=1Min&start=2021-10-25T13:30:00Z&end=2021-10-25T20:00:00Z&feed=sip&adjustment=raw&limit=10000` | 1 | whether Alpaca holds intraday minutes for an affected ticker (390 expected) or only the two auctions |
| 2 | `GET https://data.alpaca.markets/v2/stocks/bars?symbols=AFL,CL,INFY&timeframe=1Day&start=2021-10-25&end=2021-10-26&feed=iex&adjustment=raw&limit=10000` | 1 | the same day from Alpaca's separate IEX feed: thousands of IEX trades would prove the stocks traded normally |
| 3 | as 1, with `symbols=AJG` and 2022-03-08 (`start=2022-03-08T14:30:00Z&end=2022-03-08T21:00:00Z`) | 1 | the same for the third day |
| 4 | as 2, with `symbols=AJG,AME,AVB&start=2022-03-08&end=2022-03-09` | 1 | the same for the third day |

Expected total: **4 requests** (each fits on one page). An independent non-Alpaca source would be
stronger; none is connected.

## Not done

- No new data fetched; no repair or substitution of any bar. Stage 1.1 unchanged.
- The 13 sessions with 1-2 suspect bars were not examined (likely single-stock halts or thin
  days).
- The listing exchange is today's, not 2021's.
- Whether to exclude, repair or flag these days in future versions is your decision. A
  data-quality check for "liquid ticker with auction-only bar" would catch this automatically.
