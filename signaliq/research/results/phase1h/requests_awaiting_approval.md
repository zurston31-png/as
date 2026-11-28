# Request awaiting your explicit approval (not sent)

Purpose: settle whether K (Kellogg) on 2018-03-26 is a provider data defect
(`research/results/phase1g/k_2018-03-26.txt`). Nothing has been fetched.

## The one request

```
https://data.alpaca.markets/v2/stocks/bars?symbols=K,GIS&timeframe=1Min&start=2018-03-26T13:30:00Z&end=2018-03-26T20:00:00Z&limit=10000&adjustment=raw&feed=sip&currency=USD&sort=asc
```

- **Data:** SIP 1-minute bars, regular session only (09:30-16:00 New York; daylight saving time
  had started on 2018-03-11, so 13:30-20:00 UTC). Read-only market data from 2018, nothing after
  2024.
- **Size:** about 780 bars (up to 390 per symbol); one page, maybe two.
- **Control:** GIS (General Mills, same sector) traded normally that day (46,103 trades in its
  daily bar). If GIS comes back with about 390 minute bars and K does not, the feed works for
  that day and only K's data is missing.

## How it would run (same safeguards as Phase 1d)

- **Store:** a new, empty response store, `data/phase1h_k/alpaca_responses.db`. The run
  refuses to start if the file exists, and every HTTP attempt is logged.
- **Writes:** none to any research database. Nothing is repaired or replaced.
- **Approved list:** the run checks afterwards that every request sent equals this URL. Then
  the store is added to the protected databases.
- **Script:** `research/results/phase1e/scripts/run_anomaly_requests.py`, adapted to this URL
  and store.

## How the answer would be read

| Result for K | Meaning | Status |
| --- | --- | --- |
| Minute bars hold only the opening and closing prints (about 6 trades) while GIS has a full day | trades are missing for K only, as for AFL, SPGI, AJG and AME | **confirmed**, by the same standard used for those four in Phase 1f. The missing trades are still not recoverable from Alpaca |
| Minute bars hold about 20,000 trades | the daily bar lost trades that the minute data has | **confirmed**, with the stored minute data as the evidence |
| Request fails | reported as it failed | unchanged |

No membership decision depends on this bar.
