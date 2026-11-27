# Alpaca requests awaiting your approval (NOT executed)

Purpose: confirm the 2021-10-25 and 2022-03-08 defects from a second view of the same days, and
measure SPGI's real 2021-10-25 volume, which decides whether the 2021-12-31 membership (TMUS at
the 100th place, SPGI at the 101st, 0.49% apart) is right.

All four are read-only market-data requests for 2021-2022 (no 2025+ data, no trading endpoint),
written exactly as the project's client sends them. Each fits on one page (limit 10,000).
**Expected total: 4 requests**, plus retries if Alpaca returns 429 or 5xx (every attempt would be
logged).

| # | Exact request | Expected size | What it decides |
| --- | --- | --- | --- |
| 1 | `GET https://data.alpaca.markets/v2/stocks/bars?symbols=SPGI,AFL&timeframe=1Min&start=2021-10-25T13:30:00Z&end=2021-10-25T20:00:00Z&limit=10000&adjustment=raw&feed=sip&currency=USD&sort=asc` | at most 2 × 391 bars | whether Alpaca's SIP minute bars hold the whole session for two defective tickers (09:30-16:00 EDT). If they do, their summed volume is the real day's volume: SPGI's settles the 2021-12-31 question, and the defect is in Alpaca's daily aggregation only |
| 2 | `GET https://data.alpaca.markets/v2/stocks/bars?symbols=AFL,CL,INFY,SPGI&timeframe=1Day&start=2021-10-25T00:00:00Z&end=2021-10-25T23:59:59Z&limit=10000&adjustment=raw&feed=iex&currency=USD&sort=asc` | 4 bars | the same day from Alpaca's separate IEX feed: thousands of IEX trades would show the stocks traded normally (IEX volume is only IEX's share, so it cannot repair the bar) |
| 3 | `GET https://data.alpaca.markets/v2/stocks/bars?symbols=AJG,AME&timeframe=1Min&start=2022-03-08T14:30:00Z&end=2022-03-08T21:00:00Z&limit=10000&adjustment=raw&feed=sip&currency=USD&sort=asc` | at most 2 × 391 bars | as 1, for the third day (09:30-16:00 EST) |
| 4 | `GET https://data.alpaca.markets/v2/stocks/bars?symbols=AJG,AME,AVB&timeframe=1Day&start=2022-03-08T00:00:00Z&end=2022-03-08T23:59:59Z&limit=10000&adjustment=raw&feed=iex&currency=USD&sort=asc` | 3 bars | as 2, for the third day |

How they would run, once approved:

- through the project's client in record mode, into a **new, empty** store
  `data/phase1d_anomaly/alpaca_responses.db`, with every HTTP attempt logged (the first live use
  of G2);
- nothing is written to any research database; no bar is repaired or replaced. The answers are
  reported, and any treatment of the defective bars is a separate decision for you.

An independent non-Alpaca source (a second vendor) would be stronger evidence. None is
connected.
