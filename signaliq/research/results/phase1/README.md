# Phase 1: preparation code and tests

Approved scope: preparation code and tests only. Not started: Stage 2, any new data, the
250-stock universe, Strategy A, parameter freezing, and any 2025+ data. Neither proposed rule
has been adopted. Both are off by default and were only switched on to measure them.

## Summary

| Requirement | Result |
| --- | --- |
| Stage 1.1 data cannot be modified by the new workflow | **PASS** |
| The unchanged 100-stock setup reproduces Stage 1.1 byte for byte | **FAIL**: 3 diagnostic report lines and 1 diagnostic column differ, because Alpaca revised its split-adjusted prices for NEE and FTV. All 59 decisions, the membership table and 20 of 21 tables are identical. See section 2. |
| The completeness audit independently reproduces its result | **PASS** |
| All protected-data fingerprints unchanged | **PASS** (section 8) |
| Dated security types: measured at 100, every changed classification shown | done, not adopted (section 4) |
| Co-registrant issuer rule: measured at 100, every affected security shown | cache-only listing done: 2 of 78 change. **Full run NOT COMPLETED**: it needs 1 or more SEC pages never fetched. Not adopted (section 5). |
| Durable storage: recovered and verified | **local demonstration PASS; off-machine storage NOT done**, because it needs your action (section 6) |
| Every exact threshold, what it measures, what happens on failure | `research/plans/stage2_thresholds.md`; nothing in it is approved |
| Tests | 966 passed; ruff: 12 findings, all in `tests/test_validation.py`, which no commit since Stage 1.0 has touched (section 9) |

## 1. Protection of Stage 1.1 (`protection_live_check.txt`)

- `open_research_database` refuses to open the Stage 1.1 database, the Stage 1.0 tables, or the
  IEX database unless it is in read-only mode. It raises `ProtectedDatabase` before any
  connection is made.
- In read-only mode the engine uses SQLite `mode=ro`, so SQLite itself refuses writes.
- New runs write only to their own database. They read Stage 1.1's SEC answers through a
  separate read-only connection with `PRAGMA query_only=ON`.
- Live checks:
  - pointing the stage run at any of the three protected databases exits with code 1 and
    writes no report;
  - a write attempt in read-only mode is refused with "attempt to write a readonly database".
- Unit tests: `tests/test_universe_protection.py` (12 tests).
- The sha256 of the Stage 1.1 database file was taken before Phase 1 and compared again
  after all runs (section 8).

## 2. Neutral reproduction (`neutral_reproduction.txt`, `neutral_validation_report.txt`)

**Method.** The unchanged N=100 setup (no rule switched on) was run into a fresh database,
seeded from Stage 1.1 and verified at 21 of 21 tables. SEC was offline: answers came from the
Stage 1.1 cache, and the run made 0 SEC requests. Alpaca was queried live for the same
symbols and period as Stage 1.1.

**Result: not byte for byte.**

| What | Result |
| --- | --- |
| Report | 3 of 2,395 lines differ, all in the diagnostic section "our split adjustment against Alpaca's split-adjusted close" |
| Tables | 20 of 21 identical, including `membership` (61,124 rows), `split_checks`, `securities`, `candidates` and the bars |
| `security_bars` | Only the `alpaca_adjusted_close` column differs, in 2,288 rows |

The two securities involved:

- **NEE (id 374)**, 1,213 rows, 2016-01-04 to 2020-10-26. Stage 1.1 stored close ÷ 4, which is
  correct for the 4:1 split of 2020-10-27. Alpaca now returns the unadjusted close, so its
  "split-adjusted" series falls 75% overnight on 2020-10-27.
- **FTV (id 636)**, 1,075 rows, 2016-07-05 to 2020-10-08. Stage 1.1 stored close × 0.8368, a
  factor for the Vontier spin-off of 2020-10-09. Alpaca no longer applies it.

**How the cause was confirmed.** The same historical bars were requested directly through the
project's Alpaca client (2026-11-26, `adjustment=split`, SIP feed). The result matches the new
values. The arithmetic also accounts for the whole report difference: 1,213 NEE sessions are
now off and 1,075 FTV sessions are now within tolerance, a net change of −138, which equals
1,758,774 − 1,758,912.

**Why decisions are unaffected.** This column is Alpaca's own adjusted price. We store it only
to cross-check our split adjustment (`checks.adjustment_differences` and the jump listing). No
eligibility, ranking, split check or membership decision reads it. Our own split factors come
from our split records and are unchanged.

**What this means.** A rerun that re-requests live provider data cannot be guaranteed
byte-identical, because the provider can change history. Stage 1.1's own reproducibility test
(two runs, A8 and B) passed because Alpaca did not change in between.

Options, none adopted:

1. Store the raw Alpaca responses in the database and archive, and have reruns read them back.
2. Keep live re-requests, and report provider-side changes as a separate, explained section.

The Alpaca revision itself is a data-quality finding: Alpaca's split-adjusted NEE history is
now wrong.

## 3. Completeness audit (`completeness_audit_*.txt`)

`scripts/audit_completeness.py` is separate code. It reads the database through sqlite3
read-only and recomputes everything from the stored monthly and daily bars. On Stage 1.1 it
finds:

- every one of the 18,490 tickers in the pool has its monthly fetch recorded;
- 18,490 of 18,490 screen ranks match;
- selection agrees with rank;
- 5,900 of 5,900 member medians match;
- the premises of the bound hold;
- 0 of 17,492 left-out tickers could have been members (the closest was at 1.001× the
  threshold);
- 158 of 158 added tickers recompute.

**RESULT: PASS.** The output on the neutral database is byte-identical to the output on Stage
1.1. On the restored archive it also passes (section 6). Tests show that it fails on
incomplete data (`tests/test_audit_completeness.py`).

## 4. Dated security types, measured, NOT adopted (`dated_types_measurement.txt`)

**The rule.** At each decision, use the latest conclusive cover page filed on or before it. If
cover pages filed after that one were not read, the type is "unknown".

**Limitation.** Of 71,585 cover pages in the issuer evidence, 4,074 are in the cache and were
read. The other 67,511 were never fetched, so their class titles are not known. Reading them
would be new data, which Phase 1 does not fetch.

Securities whose read pages show more than one type: 3.

- **BX**: a partnership on 2019-05-09 (10-Q, "Common units representing limited partner
  interests"); common from 2019-07-05 (8-K, "Class A Common Stock"). The change is before the
  first decision (2020-01-31).
- **CRH**: an ADR from 2020-03 to 2021-03; common on 2024-12-30. The switch cannot be dated:
  too many pages in between were not read.
- **PRU**: Hawaiian Electric cover pages appear in its issuer evidence. That is an
  issuer-evidence error, not a type change. It is listed so that it is not missed.

**Effect.**

- Membership with the rule on: **0 of 59 decisions differ**.
- With the rule off, the recompute is identical to Stage 1.1 (61,124 security-decisions).
- Because so many pages are unread, the rule would mark BX and PRU "unknown" at most
  decisions. Every decision is listed with the page that dates it.

Recommendation: do not adopt it until the cover pages it needs are read.

## 5. Co-registrant issuer rule, measured, NOT adopted (`coregistrant_choices.txt`)

**The rule.** When two companies file a shared cover page, a vote decides which one is the
ticker's issuer:

1. The ticker's row on the cover page names one company: that company wins.
2. Otherwise, the company named first in the registrant block wins.

The vote must be unanimous across all cached shared filings. Otherwise the current rule
stands. Pages are read only from the cache.

**Offline listing.** 78 overlapping ticker periods:

| Outcome | Count | Details |
| --- | --- | --- |
| Changed | 2 | **URI**: United Rentals North America becomes United Rentals, Inc. (2 pages decided, unanimous). **HTZ**: Hertz Corp becomes Hertz Global Holdings (9 pages decided, unanimous). Both are the listed parent and look correct. |
| Same as the current rule | 41 | the cover page decides the same company |
| Undecided | 35 | the current rule stands. Examples: **UAL** (4:1, not unanimous) and **PCG** (stays the subsidiary). |
| Answers missing | 0 | |

The first version of the rule was wrong on CHTR, CMCSA and UAL. It used the earliest company
name and read the hidden XBRL header. Both were fixed before this listing.

**Full run with the rule on: NOT COMPLETED** (`coregistrant_full_run.txt`).

The run stopped at the issuer step with `SecUnavailable`. That is the designed behaviour: a
missing SEC answer stops the run instead of becoming "no issuer".

- **Cause:** under the rule, URI's issuer becomes United Rentals, Inc. (CIK 0001067701). Its SIC
  history needs the overflow page `CIK0001067701-submissions-001.json`. Stage 1.1 never
  fetched that page, because Stage 1.1 used the subsidiary.
- **Not fetched:** fetching it is new data, which Phase 1 does not do.
- **Unknown:** whether later steps need more uncached answers. That can only be learned by
  fetching.
- **Downstream effects** on membership and outcomes are therefore not measured.

To finish the measurement, one approved SEC request (the page above) is needed, plus any
further pages the run then reports. Each would be fetched read-only and counted.

## 6. Durable storage (`archive_recovery_demo.txt`, `archive_manifest_stage_1_1.json`)

**Tooling.** `scripts/archive_database.py` has three commands:

- **export**: writes gzip JSONL chunks of at most 256 MB uncompressed. Every file is under
  95 MB.
- **verify**: checks the sha256 of every file and the per-table fingerprints, including an
  all-columns fingerprint.
- **restore**: rebuilds a working database from the files.

**Local demonstration on Stage 1.1: PASS.**

1. Export: 100 files, 1,642 MB.
2. Verify: 0 problems, and all 21 tables match the live database.
3. Restore: 2.8 GB, with SEC answers compressed. The logical fingerprints of all 21 tables
   match Stage 1.1.
4. The completeness audit passes on the restored database.
5. 25 of 25 large SEC documents read back identically through `CachedSec`.

The restored database was then deleted to free disk. The completeness and document checks on
it were recorded in the session output, not saved to a file.

**Not done: off-machine storage.** This container is temporary, so a copy here is not durable.

- `zurston31-png/as` is public, so the archive must not go there.
- There are no cloud credentials in this environment.

Recommendation, which needs your action:

1. Create a new **private** GitHub repository just for the archive. Every file is under
   GitHub's 100 MB limit, and the total is about 1.6 GB.
2. Grant this session push access.
3. Then I push the archive, make a fresh clone, verify, restore, and compare fingerprints and
   the completeness audit.

That is the demonstration required before Phase 2.

- **Alternatives:** a private Hugging Face dataset, or a cloud bucket with a token added as an
  environment secret.
- **Licensing:** the archive contains Alpaca market data, so Alpaca's data licence applies to
  where it is stored. A private repository is the conservative choice.

## 7. Other code (tests only, nothing fetched)

- **Nesting test.** `compare_stage_versions.py --nesting K` checks that a larger run's top K
  equal the smaller run's members. Output without the flag is unchanged.
- **EDGAR Form 25 pool cross-check.** `scripts/form25_pool_check.py`. **NOT RUN**: it needs 21
  quarterly EDGAR indexes, which are new data. Without `--fetch` it reports "NOT RUN".

## 8. Protected data (`protected_data_check.txt`)

**RESULT: PASS** (2026-11-26, after all Phase 1 runs).

- **Protected database files:** the sha256 of the Stage 1.1 database, the Stage 1.0 tables and
  the IEX database are identical to the hashes taken before Phase 1.
- **IEX:** 15,861 bars, fingerprint `e66890e79788fc78`.
- **Strategy A:** 13 of 13 files OK.
- **Stage 1.1 tables:** 21 of 21 fingerprints identical.
- **Git:** 0 commits and 0 uncommitted changes touching the Stage 1.0, Stage 1.1 or Strategy A
  archives since Phase 1 began (base b4190ef).

## 9. Tests and lint (`test_results.txt`)

- `pytest`: **966 passed**.
- `ruff format --check`: all files formatted.
- `ruff check`: 12 findings, all SIM905 in `tests/test_validation.py`. That file was last
  changed in commit 1be4c56, before Stage 1.0, and Phase 1 did not touch it. Every file changed
  since Stage 1.0 passes. Not fixed, because it is outside the approved scope.

## 10. Data requests made in Phase 1

- **SEC:** 0 requests in the runs, which were offline.
- **Alpaca:**
  - the neutral run and the co-registrant run (until it stopped) re-requested the same
    historical bars Stage 1.1 used;
  - 8 small direct requests confirmed the NEE and FTV change.

  No new symbols, no new period, and nothing dated 2025 or later. No trading or paper-API
  access.

## Failures and open items

1. **Byte-for-byte reproduction: FAIL.** The cause is an Alpaca revision of its split-adjusted
   close for NEE and FTV (section 2). Decisions are unaffected. A decision on storing raw
   provider responses is needed.
2. **Co-registrant full run: NOT COMPLETED.** It stopped by design on an uncached SEC page
   (URI's new issuer). Finishing it needs your approval to fetch it (section 5).
3. **Off-machine durable storage: not done.** It needs a private repository or bucket from you
   (section 6).
4. **Form 25 pool cross-check: NOT RUN**, because it needs new data.
5. **Dated types rule:** limited by 67,511 unread cover pages; not adopted.
6. **Co-registrant rule:** not adopted; 35 cases undecided.
7. **PRU issuer evidence** contains Hawaiian Electric cover pages: an evidence error to
   investigate.
8. **Thresholds:** none approved. The Form 25 cross-check has no threshold yet.
9. **Lint:** 12 findings in an untouched test file.
