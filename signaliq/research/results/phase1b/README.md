# Phase 1b results: raw Alpaca response storage, record/replay proof, archive recovery

Scope: Phase 1 follow-up only. No Phase 2, no 250-stock run, no Strategy A change, no 2025+ data,
no SEC requests. Every file here holds reports, counts and hashes only; no raw market data.

## 1. Record/replay proof (recording #3)

Driver: `scripts/phase1b_genuine.sh` (with `scripts/run_counted.py` and
`scripts/compare_membership.py`); full output in `record_replay/driver_output.txt`.

| Check | Result |
| --- | --- |
| Response store before the recording | **absent** (the driver refuses to start if it exists) |
| Fresh databases | record and replay databases each newly seeded from Stage 1.1's parent (every table equal to the parent), separate files (different inodes) |
| Recording | 2026-11-27 08:37:40-10:27:47 UTC, exit 0 |
| Live requests to Alpaca | **2,095** (`fetched_live=2095`; 2,095 "Alpaca request" log lines; 2,095 distinct requests) |
| Live responses | **2,095**, all status 200; **0 non-200, 0 retries** (no attempt above 1) |
| Pagination | every page is a separate request and a separate stored response (106 corporate-actions pages, 1,989 bars pages) |
| Answered from the store during the recording | 74, every one a repeat of a request already fetched live earlier in the same run (`record_replay/record_from_store_check.txt`) |
| Store verify | 2,095 responses, 0 damaged bodies, **PASS** |
| Store sha256 | `108d2a177926916376e9199fc86e5477401f66d806f552bc0d4a7123bc4cb6dc` (497 MB; bodies 1,900,477,670 bytes; fingerprint 61544d16fc7e61e4) |
| Offline replay | 10:28:03-12:08:10 UTC, exit 0. Proxy pointed at 127.0.0.1:9, no credentials, `NoNetworkTransport` |
| Live requests during replay | **0** (`fetched_live=0`; 0 "Alpaca request" log lines); 2,169 = 2,095 + 74 answered from the store |
| Store after replay | sha256 unchanged |
| Stage report | record and replay **byte-identical** (sha256 d30c69b9a6e9865f…) |
| Table fingerprints | **21 of 21 identical**, standard and logical |
| Membership | **59 of 59 decisions identical**, every column of every row (61,124 rows; 100 members each) |
| Completeness audit | **PASS** on both, output identical (and identical to the Stage 1.1 audit) |

**Failure in the driver, fixed by a rerun:** the driver called `audit_completeness.py --db` with a
`sqlite:///` URL instead of a file path, so both audits failed to open the database (exit 1) and the
driver's "COMPLETENESS AUDITS IDENTICAL" line only compared two identical error messages. That
line is not evidence. Both audits were rerun with file paths (`record_replay/completeness_*.txt`):
both PASS, with identical output.

**Retry gap (open before commercial use):** only status-200 responses are stored. This recording
had 0 failed attempts, so nothing was lost, but every HTTP attempt, including failed ones, should
be stored before commercial use.

## 2. Earlier recordings (interrupted, not used)

| Recording | Fetched | Requests | Why incomplete |
| --- | --- | --- | --- |
| #1 | 2026-11-27 00:10-01:30 | 2,095 live, all 200, 0 retries | the background task was killed at its 2-hour limit before the run finished |
| #2 | 2026-11-27 02:20-03:39 | 2,095 live, all 200, 0 retries | the container rebooted (idle reclaim, back at 08:15) before the run finished |

Neither store was reused; recording #3 started from no store. Their sha256 are in
`record_replay/set_aside_files_sha256.txt`; the files were deleted afterwards to free disk.

Before that, a restarted record run was serving responses from #1's store; it was not a genuine
recording, was stopped and discarded.

#1 and #2 made the same 2,095 requests. 1,421 bodies differ byte for byte, all only in the order of
symbols in the JSON; parsed, 0 content differences. Asking Alpaca again does not give the same bytes.

## 3. Alpaca revision since Stage 1.1

The recording does not reproduce Stage 1.1 exactly: Alpaca repaired session 2021-04-19 (504
tickers whose bars held 1-3 trades), changed SPY 2019-08-12, and restored NEE/FTV's adjusted close,
between 2026-11-26 08:45 and 2026-11-27 00:10 UTC. Members are identical in all 59 decisions;
ranks and medians changed in 3 decisions. Details: `alpaca_revision_2026-11-27.txt`. Stage 1.1 is
unchanged.

## 4. Archive recovery (local test only)

Details: `archive_roundtrip.txt`; evidence in `archive/`. **Local test only: nothing is stored off
this container.**

- Stage 1.1 through a local bare git repository and a fresh clone: 100 of 100 sha256, 21 of 21
  table fingerprints, PASS. Restored from the clone: logical fingerprints 21 of 21 identical,
  completeness audit PASS with output identical to Stage 1.1's. The standard fingerprint of
  `external_responses` differs by design (the restore compresses SEC answers).
- Response store: export (8 files, largest 79.6 MB), verify PASS, restore, store verify PASS with
  the same fingerprint 61544d16fc7e61e4 and 0 damaged bodies.
- Still required: the same checks on a copy downloaded from real off-machine storage (F4).

## 5. Protected data

`protected_data_check.txt`: **PASS** (2026-11-27 12:19 UTC)

- `data/universe_stage.db` sha256 aca7478991c5ebe7…, `data/universe_stage_v1_0_tables.db`
  55bdc101068cbad7…, `data/signaliq.db` 3bba6d15f0fbd11f…, all equal to the hashes taken
  before Phase 1;
- IEX: 15,861 bars, fingerprint e66890e79788fc78;
- Strategy A archive: 13 of 13 sha256 OK;
- Stage 1.1: 21 of 21 table fingerprints identical;
- 0 commits and 0 uncommitted changes in the Stage 1.0, Stage 1.1 and Strategy A archives since
  Phase 1 began (b4190ef).

## 6. Other files

- `coregistrant_sec_requirements.txt`: the exact SEC requests the co-registrant measurement
  would need (2 expected, 12 at most). None fetched.
- Tests: 977 passed at HEAD 2efe7b4.
