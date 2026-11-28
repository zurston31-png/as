# Phase 1c results: protection, the 2021-10-25 anomaly, attempt logging, storage plan

Approved scope only. No Stage 2, no 250-stock run, no Strategy A rerun, no parameter freezing,
no 2025+ data, no new requests to Alpaca or SEC.

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Protect `sip_research.db`; classify every database; unsafe defaults fail closed | done; live check PASS | `protection_live_check.txt`, `../../plans/database_inventory.md`, commit 5aff84a |
| 2 | Investigate 2021-10-25 (read-only, stored data) | Alpaca data gap: auction-only bars for 218 liquid, mostly NYSE-listed tickers; same defect on 2021-04-19 (repaired by Alpaca) and on **2022-03-08** (new, still broken); not our ingestion; membership robust except 2021-12-31 (SPGI/TMUS at the 100th place) | `oct25_investigation.md`, `oct25/`, commit 201cc20 |
| 3 | Preserve failed responses | every HTTP attempt logged (status or network error, failed bodies); recorded failures replay as the same error, offline; prepared only, no recording run | `tests/test_response_store.py`, commit 3060dfa |
| 4 | Finish the external storage plan | complete; waiting for your bucket; nothing stored off the container | `../../plans/durable_archive_plan.md`, commit 7f63fe2 |
| - | Two dataset versions kept separately, nothing overwritten | version 1 = Stage 1.1 database; version 2 = Phase 1b raw responses (now protected) | `../../plans/database_inventory.md` |
| - | Protected data | PASS; `sip_research.db` and the Phase 1b store hashed as new baselines | `protected_data_check.txt` |

Tests: 986 passed (980 before item 3).
