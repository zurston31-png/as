# Durable research archive: plan

Status: **plan only. No research data is stored off this container.** No bucket or private
repository is assumed to exist or to be reachable from this session. The research archive must never go to
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

## Recommendation: object storage, plus an independent second copy

GitHub's own documentation says Git is not designed to be a backup tool, limits a single push
to 2 GB and each file to 100 MB, and recommends repositories under 1 GB (under 5 GB strongly
recommended). At about 2.1 GB, and growing with every version, the archive belongs in object
storage, not in Git.

**Primary copy: a private object storage bucket**, with:

- **object versioning on**, so an overwrite or deletion keeps the earlier version;
- **an object lock or retention policy** (for example, keep every object for at least a year), so
  that even a mistaken or malicious delete cannot remove an archived version during that time;
- **a credential scoped to that one bucket, without delete permission**: it can list, read and
  add files only. A session holding it cannot destroy the archive;
- public access blocked at the bucket or account level.

Providers that offer all of this include Backblaze B2, AWS S3, Google Cloud Storage and
Cloudflare R2. At 2 to 3 GB the storage cost is small at any of them; please check current
pricing. How this session would authenticate differs by provider, and must be confirmed when
you choose one:

- S3, R2 and GCS's S3-compatible access sign each request with a secret key. That key would have
  to be an environment variable that commands in sessions can read, which is another reason to
  give it no delete permission.
- Backblaze B2's own API starts with a username/password-style key exchange, which may be
  possible to store as a network secret that this session never sees. I have not tested this.

**Second copy: independent of the first**, so that no single account, provider or mistake can
lose both. In order of preference:

1. a bucket at a **different provider, under a different account**, which you copy into yourself
   from the first bucket (for example with `rclone`). No session credential ever touches it;
2. a copy on **your own computer or an external drive**, downloaded from the first bucket and
   checked there with `scripts/archive_database.py verify`.

Not recommended as the primary copy: a private GitHub repository (Git is not a backup tool, and
the size limits above apply). It remains acceptable as a small extra copy of the manifests and
hashes, which are already in git.

## Procedure (once a bucket exists)

1. **Export:**
   - `scripts/archive_database.py export` for each database;
   - export the raw response store the same way (it is a SQLite file, so it is split into files
     under 95 MB like the others), plus its `response_store_tool.py summary`.
2. **Check that the bucket is private** before the first upload; nothing is uploaded otherwise.
3. **Upload** every file and the manifests under a new version prefix (for example
   `stage_1_1/2026-11-25/`). Nothing is ever overwritten.
4. **Download into an empty directory** (or verify each object's provider checksum against the
   manifest's sha256), never trusting the local files.
5. **Verify the downloaded copy:**
   - `archive_database.py verify` (sha256 of every file, row counts, fingerprints);
   - `archive_database.py restore`;
   - `table_fingerprints.py --logical` compared with the reference;
   - `audit_completeness.py` on the restored Stage 1.1;
   - `response_store_tool.py verify`;
   - a replay run compared byte for byte with the record run.
6. **Record the result** in `research/results/` (hashes only, no data) and commit it to this
   repository.
7. **Second copy:** you copy the bucket to the second destination and verify it the same way.

## What has and has not been proved

A rehearsal in this session (`research/results/phase1b/archive_roundtrip.txt`) pushed the Stage
1.1 archive into a local bare git repository, cloned it fresh over the `file://` transport, and
verified the clone against the live database. **That proves the archive format and the
verification workflow. It does not prove durability:** the bare repository and the clone were
on this same container's disk, and both disappear with it. **No research data is stored off this
container yet.** It is not safe until step 5 passes on a copy downloaded from a real off-machine
destination (threshold F4).

## What I need from you

1. Choose a provider and create a **private bucket** with versioning and an object lock or
   retention policy, and public access blocked.
2. Create a credential limited to **that bucket**, with list, read and write permissions and **no
   delete**. Add it to this environment's settings yourself; never paste it into the chat. Tell
   me the provider and the bucket name, and I will tell you exactly which setting it goes in.
3. Choose the second copy: a second provider and account, or your own drive.

## Risks found while planning

1. **`data/sip_research.db` is unprotected** and is the default research database when
   `SIGNALIQ_SIP_RESEARCH_DATABASE_URL` is not set. A script started without that variable
   would write into the validation batch v2 data. Recommended: add it to the protected list.
   Not done, because it is outside the approved scope.
2. **Read-only readers touch `data/universe_stage.db-shm`.** That is SQLite's shared-memory
   index for WAL mode. The database file itself is unchanged (sha256 checked), but the `-shm`
   file's timestamp moves. Opening the parent with `immutable=1` would avoid even that. Not
   done.
