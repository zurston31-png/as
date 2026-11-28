# Research databases: inventory and classification

Every database the research workflow uses, and what may happen to it. Enforced in code by
`PROTECTED_DATABASES` in `app/database.py` (checked by `tests/test_universe_protection.py` and
`tests/test_response_store.py`; live check in `research/results/phase1c/protection_live_check.txt`).

## Protected: never written by any workflow

| File | What it is | Size | How it is enforced |
| --- | --- | --- | --- |
| `data/universe_stage.db` | Stage 1.1 research database (version 1, below) | 20 GB | research workflow opens it read-only only (`mode=ro`); writers refuse it |
| `data/universe_stage_v1_0_tables.db` | Stage 1.0 tables | 11 MB | same |
| `data/signaliq.db` | IEX market data, Strategy A's input | 4.4 MB | same; ingestion, `run_strategy.py` and `validate.py` refuse it, and `init_db` refuses to migrate it |
| `data/sip_research.db` | validation batch v2 database | 1.4 GB | same (**added in Phase 1c**; it was the default research database before) |
| `data/phase1b_record/alpaca_responses.db` | raw Alpaca responses of the Phase 1b recording (version 2, below) | 497 MB | record mode refuses it; replay mode (read-only) only (**added in Phase 1c**) |
| `data/phase1d_anomaly/alpaca_responses.db` | raw responses of the four approved anomaly requests (4 requests, 4 attempts; sha256 4f157cc8…) | 24 KB | same (**added in Phase 1e**) |
| `data/stage_1_1_hybrid/universe_stage.db` | hybrid dataset (version 3, below); sha256 7357aa5f… (d08e272e… before the Phase 1g note fix); replacement log `replacements.json` beside it (sha256 040de2ee…) | 508 MB | research writers refuse it (**added in Phase 1f**); `build_hybrid.py` refuses to run if it exists. Not self-contained: its SEC answers stay in Stage 1.1, its read-only parent |

The Strategy A archive (`research/results/strategy_a/`, 13 files under `SHA256SUMS`) and the
Stage 1.0/1.1 result folders are protected through git and checked by the protected-data check.

## Writable: only when named explicitly, and only new files

| Variable | Used by | Default |
| --- | --- | --- |
| `SIGNALIQ_SIP_RESEARCH_DATABASE_URL` | every research script (`open_research_database`) | **none: unset stops the run** (`NoResearchDatabase`). Before Phase 1c it defaulted to `data/sip_research.db`, and `--stage` runs defaulted to `data/universe_stage.db` |
| `SIGNALIQ_DATABASE_URL` | ingestion, `run_strategy.py`, `validate.py`, the API | `data/signaliq.db`, which is protected, so the writers stop unless a new file is named. The API only reads (its start-up migration step refuses a protected file that needs migrating) |
| `SIGNALIQ_ALPACA_RESPONSES` (+ `_MODE=record`) | Alpaca client | off; a protected store is refused in record mode |
| `SIGNALIQ_SIP_RESEARCH_PARENT_DB` | layered read-only parent | none; opened read-only at the SQLite level |

## Temporary: may be deleted; nothing depends on them

| File | What it is |
| --- | --- |
| `data/phase1_neutral/universe_stage.db` (500 MB) | Phase 1 neutral reproduction (2026-11-26). Holds Alpaca data as Alpaca served it on 2026-11-26 (between the two versions below), with no raw responses. Its fingerprints are in `research/results/phase1/`. Kept until you decide; not protected |
| `data/phase1f_v2_rebuild/universe_stage.db` (523 MB, sha256 ca1cfa35…) | version 2 rebuilt by offline replay on 2026-11-28 (21 of 21 tables and the report identical to the Phase 1b recording); input to the hybrid build. Can be rebuilt at any time |
| `data/phase1b_replay/` (empty) | the replay database was deleted after the comparisons (sha256 in `research/results/phase1b/record_replay/set_aside_files_sha256.txt`) |
| `data/universe_stage.db-shm`, `-wal` | SQLite's side files for the Stage 1.1 database; the `-shm` timestamp moves when it is read |
| scratchpad databases | test copies; outside `data/` |

## Dataset versions (never overwritten, never substituted)

| Version | Source | Kept as | Differs from the other |
| --- | --- | --- | --- |
| **1: Stage 1.1** (built 2026-11-25) | Alpaca as served before 2026-11-26; raw responses were never stored | `data/universe_stage.db` + archive manifest `research/results/phase1/archive_manifest_stage_1_1.json` | broken 2021-04-19 bars (1-3 trades for 504 tickers), old SPY 2019-08-12 |
| **2: Phase 1b recording** (2026-11-27 08:38-10:01 UTC) | raw Alpaca responses, every body with its sha256 | `data/phase1b_record/alpaca_responses.db` (sha256 108d2a17…, fingerprint 61544d16fc7e61e4) + export manifest `research/results/phase1b/archive/response_store_manifest.json` | repaired 2021-04-19, revised SPY 2019-08-12; same 100 members in all 59 decisions |

| **3: hybrid** (built 2026-11-28 03:02-03:10 UTC) | Stage 1.1, with its 505 security / 504 ticker bars of 2021-04-19 replaced from version 2 (every replacement logged with old and new values, response key, body sha256 and fetch time); 2021-10-25 and 2022-03-08 kept as stored, flagged, and bounded | `data/stage_1_1_hybrid/universe_stage.db` | from version 1: 2021-04-19 only (bars) and the 8 notes fixed in Phase 1g; from version 2: SPY 2019-08-12 only (not in the approved scope); membership identical to version 2. Three decisions labelled uncertain: `research/results/phase1g/uncertain_decisions.tsv` |

Version 2's universe database is not kept as a file: it is rebuilt exactly from the stored
responses by an offline replay (proved byte-identical in Phase 1b; its table fingerprints are in
`research/results/phase1b/record_replay/tables_record.txt`). Any analysis must say which version
it used. Stage 1.1 stays the reference until you decide otherwise.

Neither version is stored off this container yet (see `durable_archive_plan.md`).
