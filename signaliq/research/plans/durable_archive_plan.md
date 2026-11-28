# Durable research archive: plan

Status: **plan complete, waiting for your bucket. No research data is stored off this
container.** No bucket or private repository is assumed to exist or to be reachable from this
session. The research archive must never go to `zurston31-png/as`, which is public.

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
| Validation batch v2 database | `data/sip_research.db` | 1.4 GB | earlier research; protected since Phase 1c |
| Raw Alpaca responses of the Phase 1b recording (dataset version 2) | `data/phase1b_record/alpaca_responses.db` | 497 MB; 498 MB as an archive (8 files) | rebuilds version 2 exactly by replay |
| Reports, audits, manifests, Strategy A results | in git (`research/`) | small | already in the public repository; no raw data |

**Stage 1.1 has no raw Alpaca responses.** They were never stored, and Alpaca has since
changed some of them. The values Stage 1.1 received, including the original NEE and FTV
adjusted closes, survive only as the parsed rows in the Stage 1.1 database. That is why the
Stage 1.1 database is the item that matters most.

## Requirements

1. **Private.** Visibility is checked by the tool before every push; a push to a public
   destination is refused.
2. **Off this machine**, in two independent copies (below).
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
- Backblaze B2's own API starts with a username/password-style key exchange that a network
  secret could hold, but it uploads to hosts that change per upload, which an exact-host
  allowlist cannot cover. So it is not usable here (see "Concrete setup").

**Second copy: independent of the first**, so that no single account, provider or mistake can
lose both. In order of preference:

1. a bucket at a **different provider, under a different account**, which you copy into yourself
   from the first bucket (for example with `rclone`). No session credential ever touches it;
2. a copy on **your own computer or an external drive**, downloaded from the first bucket and
   checked there with `scripts/archive_database.py verify`.

Not recommended as the primary copy: a private GitHub repository (Git is not a backup tool, and
the size limits above apply). It remains acceptable as a small extra copy of the manifests and
hashes, which are already in git.

## Concrete setup (recommended)

How this session can authenticate decides the design. Environment network secrets add a fixed
header to requests for one exact host; they cannot sign S3 requests (every S3 request carries
its own signature), and Backblaze's native API uploads to hosts that change per upload, which an
exact-host allowlist cannot cover. So the session needs an **S3-compatible access key in two
environment variables**, readable by commands in sessions of that environment. That is
acceptable only because the key cannot delete and the bucket's object lock protects every
version.

**Primary: Backblaze B2 (or AWS S3 if you already use AWS).**

1. Create a private bucket, e.g. `signaliq-research-archive`, with **Object Lock enabled at
   creation** (it cannot be added later on B2) and a default retention, e.g. 1 year, governance
   or compliance mode. Keep all versions (no lifecycle rule that deletes).
2. Create an application key **restricted to that bucket** with listFiles, readFiles and
   writeFiles, **without deleteFiles** (on AWS: an IAM user whose policy allows only
   `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` on that bucket, with Block Public Access on).
3. Make a **personal** environment for this repository (not the Default, not an organization
   environment) and add the four variables listed under "Exact requirements" yourself. Never
   paste them into the chat.
4. Tell me the bucket name and the endpoint (e.g. `s3.us-west-004.backblazeb2.com`). I will ask
   for that one host to be allowed in the environment's network settings (you approve it).

**Second copy: independent of the first, done by you, not by a session.**

- Preferred: a bucket at another provider under another account (e.g. AWS S3 if the primary is
  B2), filled from your own computer with `rclone copy` from the first bucket, then checked with
  `scripts/archive_database.py verify`; or
- an external drive: download the version prefix with `rclone copy` and verify it there.

No session credential ever reaches the second copy.

**Done means (threshold F4):** a fresh download from the bucket into an empty directory passes
step 5 below, and the result is committed. Until then nothing is called durable.

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

## Exact requirements (before any credential is added)

Nothing below has been run. The tool is `rclone` (installed with `apt-get install -y rclone` in
the session; not yet tested here), reading its settings from environment variables so that no
configuration file holds the key.

**Environment variables** (a personal environment for this repository; you add them, never in
the chat):

| Variable | Secret | Value |
| --- | --- | --- |
| `SIGNALIQ_ARCHIVE_KEY_ID` | yes | the key's ID (B2 "keyID", AWS "access key ID") |
| `SIGNALIQ_ARCHIVE_KEY_SECRET` | yes | the key's secret (B2 "applicationKey", AWS "secret access key") |
| `SIGNALIQ_ARCHIVE_BUCKET` | no | e.g. `signaliq-research-archive` |
| `SIGNALIQ_ARCHIVE_ENDPOINT` | no | e.g. `s3.us-west-004.backblazeb2.com` (B2) or `s3.us-east-1.amazonaws.com` (AWS) |

The endpoint host also has to be allowed in the environment's network settings (I propose
that change; you approve it).

**Permissions of the key, exactly:**

- Backblaze B2 application key: bucket = only the archive bucket; capabilities `listBuckets`,
  `listFiles`, `readFiles`, `writeFiles`. **Not** `deleteFiles`, `writeBucketRetentions`,
  `bypassGovernance`, `writeFileRetentions`, `writeFileLegalHolds` or any key-management
  capability.
- AWS: an IAM user with only this policy (no console access):
  `{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::BUCKET"},{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::BUCKET/*"}]}`
- Bucket: private, Object Lock on with a default retention (e.g. 1 year), versioning on (implied
  by Object Lock), public access blocked.

**Verification procedure** (each step must pass before the next; the results are committed as
hashes and counts only):

1. *Safety checks, before any research data is sent:*
   - an unauthenticated `curl -sI https://$SIGNALIQ_ARCHIVE_ENDPOINT/$SIGNALIQ_ARCHIVE_BUCKET/`
     returns 401 or 403 (the bucket is not public);
   - upload a 1-line test object `_probe/probe.txt`, then try to delete it with the key: the
     delete **must be refused**. If it succeeds, stop: the key can delete;
   - read the test object's retention: a retain-until date must be set (object lock active).
2. *Upload:* `rclone copy` each export directory (Stage 1.1, Stage 1.0, IEX, `sip_research.db`,
   the Phase 1b response store) to `<bucket>/<dataset>/<export date>/`, with `--immutable` (never
   overwrite) and `--checksum`.
3. *Fresh download:* `rclone copy` the same prefixes into a new, empty directory (never reusing
   the local export).
4. *F4 checks on the downloaded copy:* `scripts/archive_database.py verify` (sha256 of every
   file against its manifest, row counts, fingerprints); `restore`;
   `scripts/table_fingerprints.py --logical` equal to the reference for all 21 Stage 1.1 tables;
   `scripts/audit_completeness.py` PASS; `scripts/response_store_tool.py verify` PASS with
   fingerprint 61544d16fc7e61e4; the IEX restore gives 15,861 bars and fingerprint
   e66890e79788fc78.
5. *Record:* commit the counts, hashes and results under `research/results/`. Only then is the
   archive called durable.

Disk: the Stage 1.1 export (1.6 GB) plus its restore (2.8 GB) needs about 4.5 GB free at once;
the container has 5.0 GB, so the datasets are verified one at a time.

## What has and has not been proved

A rehearsal in this session (`research/results/phase1b/archive_roundtrip.txt`) pushed the Stage
1.1 archive into a local bare git repository, cloned it fresh over the `file://` transport,
verified the clone (100 sha256, 21 table fingerprints), restored it (21 of 21 logical
fingerprints, completeness audit PASS), and exported, verified and restored the response store
(same fingerprint 61544d16fc7e61e4). **That proves the archive format and the
verification workflow. It does not prove durability:** the bare repository and the clone were
on this same container's disk, and both disappear with it. **No research data is stored off this
container yet.** It is not safe until step 5 passes on a copy downloaded from a real off-machine
destination (threshold F4).

## What I need from you

Steps 1-4 of "Concrete setup": the bucket with Object Lock, the bucket-scoped key without
delete, the two environment variables in a personal environment, and the bucket name and
endpoint. Then choose the second copy (another provider and account, or your own drive).

## Risks found while planning

1. ~~`data/sip_research.db` is unprotected~~ **Fixed in Phase 1c**: it is protected, and the
   research workflow has no default database (see `database_inventory.md`).
2. **Read-only readers touch `data/universe_stage.db-shm`.** That is SQLite's shared-memory
   index for WAL mode. The database file itself is unchanged (sha256 checked), but the `-shm`
   file's timestamp moves. Opening the parent with `immutable=1` would avoid even that. Not
   done.
