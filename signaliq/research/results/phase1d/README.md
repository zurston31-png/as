# Phase 1d results: historical data defects, audit requirements, attempt-logging test

Approved scope only: stored data, a local fake provider, documents. **No request was sent to
Alpaca or SEC.** No bar was repaired or replaced. No Stage 2, no expansion, no Strategy A rerun,
no 2025+ data. Protected data unchanged (`protected_data_check.txt`).

## 1. Historical data defects

**The 13 days with 1-2 suspect bars are false alarms of my first scan rule, not data defects:**

| Days | Ticker | What it is | Eligible at any decision |
| --- | --- | --- | --- |
| 2022-12-13 (1) | RSX | VanEck Russia ETF: one trade after 9 months of suspension (last normal bars Feb-Mar 2022). Real | no (ETF) |
| 2023-01-30 to 02-10 (6) | SPAQ | ticker reused by a new thin ETF (security 878, from 2023-01-30); the "normal" trades were the old SPAC's (security 848) | no |
| 2024-10-14 to 10-24 (6) | INFO | ticker reused by a new thin fund (security 897, from 2024-10-10); the "normal" trades were IHS Markit's (security 637) | no |

The rule now uses the **security**, not the ticker, and counts missing sessions as 0. On Stage
1.1 it flags **767 bars on exactly three days and nowhere else** (`audit_auction_bars_stage_1_1.txt`).
Every bar is listed in `suspect_bars.tsv`: date, ticker, normal trade count (20-session median),
flagged trade count, volume share, version-2 status, exchange, and decisions affected.

| Day | Bars | Normal trades (median) | Flagged trades | Volume vs normal | Version 2 (Alpaca 2026-11-27) | Exchange now | Eligible / member at a covering decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-04-19 | 499 | 23,169 | 1-3 | 1% | **repaired** (all 499) | NYSE 400, Nasdaq 18, unknown 80 | 451 / 67 |
| 2021-10-25 | 221 | 15,654 | 2-5 | 10% | **still broken** (220 of 221) | NYSE 182, Nasdaq 3, unknown 36 | 211 / 1 (MDT, member 2021-12-31) |
| 2022-03-08 | 47 | 12,413 | 2-5 | 10% | **still broken** (46 of 47) | NYSE 39, Nasdaq 1, unknown 7 | 47 / 0 |

**Membership impact** (`membership_impact.txt`; the recomputation reproduces every stored
median):

| Decision | With Alpaca's repair | Bars left out | Neighbour average | 100th/101st gap |
| --- | --- | --- | --- | --- |
| 2021-04-30 | none | +TDOC -HON | none | 0.02% |
| 2021-05-28 | none | +AMC -NCLH | none | 0.43% |
| 2021-06-30 | none | +DAL -CHTR | +ACN +DAL -CHTR -NCLH | 0.12% |
| 2021-10-29, 2021-11-30 | (no repair exists) | none | none | 1.63%, 0.28% |
| **2021-12-31** | (no repair exists) | **+SPGI -TMUS** | **+SPGI -TMUS** | **0.49%** |
| 2022-03-31, 04-29, 05-31 | (no repair exists) | none | none | 0.67-1.14% |

For 2021-04-19, Alpaca's real repair changed no member, while both stand-in methods did. So
the stand-ins overstate the impact, and only real data can settle 2021-12-31. The exact
requests are in `requests_awaiting_approval.md` (4 requests, not executed).

## 2. Audit requirements G1 and G2

The final wording and exact measurement are in `research/plans/stage2_thresholds.md`, section G.
**Not adopted.** G1 is measured by `scripts/audit_auction_bars.py` (3 tests); Stage 1.1 today
**fails G1** (767 unexplained bars). G2 is measured by attempt counts against the client log, by
`response_store_tool.py verify` (two new consistency checks, 1 new test), and by replay.

## 3. Controlled record/replay test of attempt logging

`scripts/attempt_log_controlled_test.py`: the real client over real HTTPS against a local fake
provider (fake credentials), then replay with the server stopped and the network blocked.
**18 of 18 checks PASS** (`attempt_log_controlled_test.txt`):

- 15 attempts logged = 15 requests the server received = 15 client log lines;
- order and status of each attempt: 500, 429, 200; dropped connection, read timeout, 200; a 422;
  4 × 503; 4 dropped connections;
- 7 failed bodies stored intact;
- replay: the same bars, and the same AlpacaRequestError 422, AlpacaServerError 503 and
  AlpacaNetworkError, with 0 network requests; store unchanged.

**Negative control:** with the logger deliberately losing 500 responses, the test **FAILS**
(`attempt_log_negative_control.txt`). No full or live recording was run.

## 4. External storage

`research/plans/durable_archive_plan.md`, "Exact requirements": four variable names, the exact key
permissions (B2 capabilities; AWS IAM policy), and the verification procedure (safety probes:
the bucket is not public, a delete is refused, retention is set; then upload, fresh download,
F4). **Nothing is stored off this container.**

Tests: 990 passed.
