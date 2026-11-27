# Phase 1e results: the four approved Alpaca requests

Exactly the four approved read-only requests were sent, nothing else. They went into a new,
empty store, `data/phase1d_anomaly/alpaca_responses.db` (now protected), with every HTTP
attempt logged. No research database was written; no bar was repaired, replaced or
overwritten. Protected data unchanged (`protected_data_check.txt`).

## Request log and raw-data checks

| # | Request | Attempts | Result |
| --- | --- | --- | --- |
| 1 | SIP 1-minute bars, SPGI and AFL, 2021-10-25 09:30-16:00 New York | 1 (200) | 4 bars |
| 2 | IEX daily bars, AFL, CL, INFY, SPGI, 2021-10-25 | 1 (200) | **0 bars** (`{"bars": {}}`) |
| 3 | SIP 1-minute bars, AJG and AME, 2022-03-08 09:30-16:00 New York | 1 (200) | 3 bars |
| 4 | IEX daily bars, AJG, AME, AVB, 2022-03-08 | 1 (200) | **0 bars** |

- 4 requests, 4 attempts, all status 200, 0 retries, 0 failures, one page each.
- Each request key equals the approved URL exactly (`run_output.txt`); the client log is
  `request_log.txt`.
- Store verify PASS: 4 bodies intact, every response has its logged 200 attempt, attempts (4)
  equal the log's request lines (4). G2 holds for this run.
- Store sha256 `4f157cc873d81628c40ab7be5786d3ebd360fe82b93b0e2ae7e3bb8aea76ffda`; body
  checksums in `raw_data_checks.txt`. The raw bodies stay in the store, not in git (Alpaca
  licence).
- The two empty IEX answers are valid responses, reported as returned. Nothing was substituted.

## Minute bars against the daily bars (`minute_vs_daily.txt`)

| Ticker | Day | Minute bars (of 390) | Minute trades / volume | Daily bar trades / volume |
| --- | --- | --- | --- | --- |
| AFL | 2021-10-25 | 2 (09:30, 16:00) | 3 / 332,794 | 3 / 332,794 (equal) |
| SPGI | 2021-10-25 | 2 (09:30, 16:00) | 3 / 134,677 | 3 / 134,677 (equal) |
| AJG | 2022-03-08 | 1 (09:30 only) | 1 / 7,950 | 5 / 161,099 (closing print missing from the minutes) |
| AME | 2022-03-08 | 2 (09:30, 16:00) | 2 / 125,600 | 3 / 125,700 (one 100-share trade missing from the minutes) |

## What the new evidence establishes, date by date

**2021-04-19.** No request was made for it (none approved). Alpaca's own repair is in dataset
version 2. Stage 1.1 (version 1) keeps the defective bars. With the repaired bars, no member
changes.

**2021-10-25.**
- Alpaca's SIP minute data holds only the opening and closing auction prints for SPGI and AFL,
  and they add up exactly to the defective daily bars. The missing trades are absent from
  Alpaca's underlying data, not only from its daily totals.
- Alpaca's IEX feed has no bar at all for AFL, CL, INFY or SPGI that day. Our IEX database
  (fetched 2026-11-17) has normal IEX bars that day for 10 other, non-NYSE tickers, so Alpaca's
  IEX feed lacks these names too.
- This is a separate-feed comparison only. IEX is one exchange, and an empty answer says
  nothing about the consolidated volume.
- **Established:** Alpaca cannot supply the missing trades in any feed we tried. A repair would
  need a non-Alpaca source.

**2022-03-08.**
- Same pattern: AME has only its two auction prints; AJG has only its opening print. AJG's
  closing print (153,149 shares) is in the daily bar but not in the minute bars, so Alpaca's own
  minute and daily data disagree.
- IEX has no bar for AJG, AME or AVB.
- Under both stand-in methods, the three decisions whose windows hold this day keep the same
  members.

## 2021-12-31: SPGI against TMUS (`spgi_tmus_2021-12-31.txt`)

| | Rank | Member | 63-session median of close × volume |
| --- | --- | --- | --- |
| TMUS | 100 | yes | 628,197,191 |
| SPGI | 101 | no | 625,129,380 |

- SPGI's 2021-10-25 bar (3 trades, $60.3M) is the lowest of its 63 window values. Its neighbouring
  sessions were $670.2M and $1,155.7M.
- Whatever the real value, it can raise SPGI's median by one step at most, to $637.8M.
- **SPGI would rank above TMUS if its real 2021-10-25 dollar volume was above $628.2M**
  (about 1.40 million shares at the $447.37 close), roughly its own median day.
- **A corrected ranking is not supported:** no source we have holds the real volume. The
  original ranking stands in Stage 1.1, and 2021-12-31 stays uncertain.
- A full correction would also need the real bars of the other affected eligible securities
  (211 on that day), since others near the 100th place could move as well.

## Limits that stay visible

- The detector (proposed G1) finds near-empty bars only. Partial loss on other days would not
  be caught, so the historical data is not shown to be clean.
- G1 still fails for Stage 1.1: 767 flagged bars without an approved treatment.
- No independent (non-Alpaca) source was available; none is connected.
