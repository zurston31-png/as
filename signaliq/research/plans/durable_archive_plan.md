# Durable research archive: plan

Status: **plan only. Nothing has been pushed anywhere.** No private repository is assumed to
exist or to be reachable from this session. The research archive must never go to
`zurston31-png/as`, which is public.

## Why

The research data exists only on this container's disk. The container is temporary, so the
data can be lost at any time. The NEE/FTV revision shows that lost provider data cannot be
fetched again as it was: Alpaca now serves different history. The data itself has to be
preserved, not only the code and the reports.

## What must be preserved

| Item | Where now | Size | Notes |
| --- | --- | --- | --- |
| Stage 1.1 research database | `data/universe_stage.db` | 21.0 GB on disk; 1.6 GB as an archive (100 files) | the reference for every later version; includes the SEC cache |
| Stage 1.0 tables | `data/universe_stage_v1_0_tables.db` | 11 MB | archived Stage 1.0 |
| IEX market data | `data/signaliq.db` | 4.5 MB | Strategy A's input |
| Validation batch v2 database | `data/sip_research.db` | 1.4 GB | earlier research; **not on the protected list**, and it is the default database when no URL is set (see Risks) |
| Raw Alpaca responses of the Phase 1b record run | `data/phase1b_record/alpaca_responses.db` | measured at export | needed for byte-for-byte replay |
| Reports, audits, manifests, Strategy A results | in git (`research/`) | small | already in the public repository; no raw data |

**Stage 1.1 has no raw Alpaca responses.** They were never stored, and Alpaca has since
changed some of them. The values Stage 1.1 received, including the original NEE and FTV
adjusted closes, survive only as the parsed rows in the Stage 1.1 database. That is why the
Stage 1.1 database is the item that matters most.

## Requirements

1. **Private.** Visibility is checked by the tool before every push; a push to a public
   destination is refused.
2. **Off this machine**, in at least one copy. Two independent copies are recommended: for
   example, a private repository plus a copy on your own computer or drive.
3. **Verified after storage, from a fresh copy, never from the local files:**
   - every file's sha256 matches the manifest;
   - the restored databases match every table fingerprint;
   - the completeness audit passes;
   - the response store verifies, and a replay reproduces the record run.
4. **Never altered.** A new version is a new export with its own manifest. Nothing is
   overwritten. Git history keeps every version.
5. **Licence.** The archive holds Alpaca market data, and Alpaca's market data agreement
   restricts redistribution. A private store that only you can read is the conservative
   choice. Please check the terms for whichever destination you pick.

## Options

| Option | Fits the files? | Integrity | What you would set up |
| --- | --- | --- | --- |
| **A. New private GitHub repository** (recommended) | yes: every archive file is under 95 MB (GitHub's limit is 100 MB per file); about 1.6 GB plus the response store | git hashes every object; our manifest adds sha256 and table fingerprints; full history | create the repository as Private; install the Claude GitHub App on it; tell me its name |
| B. Private Hugging Face dataset | yes | per-file hashes; history | a private dataset and an access token added as an environment secret for its host |
| C. Cloud bucket (GCS or S3) with object versioning | yes | provider checksums plus our manifest; versioning and retention lock available | a bucket and a credential scoped to it, added as an environment secret |
| D. Your own computer or drive | yes | our manifest | once A exists, `git clone` the private repository onto your computer or drive and run `archive_database.py verify` there; best used as the second copy |

Recommended: **A as the primary copy, D as the second.**

## Procedure (once you have set up a destination)

1. **Export:**
   - `scripts/archive_database.py export` for each database;
   - copy the raw response store as is, plus its `response_store_tool.py summary`.
2. **Check visibility.** I confirm through the GitHub tools that the repository is private.
   If it is public or unknown, nothing is pushed.
3. **Push** with git. Every file is under 95 MB, so no Git LFS is needed. GitHub refuses a
   single push larger than 2 GB, and recommends keeping repositories small (ideally under
   1 GB; under 5 GB strongly recommended). Please check these figures in GitHub's current
   documentation. The full archive is about 2.1 GB, so it goes up in at least two pushes:
   Stage 1.1 in one, and the response store and small databases in another. Each new
   version adds roughly its own compressed size, so the repository will pass 5 GB after
   about two more full versions. At that point a new repository per version, or option C,
   is the better home.
4. **Fresh clone** into an empty directory.
5. **Verify the clone:**
   - `archive_database.py verify` (sha256 of every file, row counts, fingerprints);
   - `archive_database.py restore`;
   - `table_fingerprints.py --logical` compared with the reference;
   - `audit_completeness.py` on the restored Stage 1.1;
   - `response_store_tool.py verify`;
   - a replay run compared byte for byte with the record run.
6. **Record the result** in `research/results/` (hashes only, no data) and commit it to this
   repository.

Steps 3 to 5 were rehearsed in this session against a local bare git repository standing in
for the private remote (`research/results/phase1b/archive_roundtrip.txt`).

## What I need from you

- Create the destination: a private repository under your account or organization, for
  example `signaliq-research-data`. Make sure "Private" is selected.
- Install the Claude GitHub App on that repository, then tell me its name. I will attach it to
  the session with push access and check its visibility before pushing anything.
- Decide whether you also want a second copy (option D).

## Risks found while planning

1. **`data/sip_research.db` is unprotected** and is the default research database when
   `SIGNALIQ_SIP_RESEARCH_DATABASE_URL` is not set. A script started without that variable
   would write into the validation batch v2 data. Recommended: add it to the protected list.
   Not done, because it is outside the approved scope.
2. **Read-only readers touch `data/universe_stage.db-shm`.** That is SQLite's shared-memory
   index for WAL mode. The database file itself is unchanged (sha256 checked), but the `-shm`
   file's timestamp moves. Opening the parent with `immutable=1` would avoid even that. Not
   done.
