# Off-container backup: exact configuration, for your review

Status: **for review only.** Nothing has been created, no credential requested, nothing
uploaded. Pick one primary (A: Backblaze B2, or B: AWS S3). The second copy (C) is made by you,
never by a session. Do not put any credential in git or in the chat.

## What is backed up

One version prefix per dataset, never overwritten:

| Dataset | Archive (from `scripts/archive_database.py export`) | Size |
| --- | --- | --- |
| Stage 1.1 (dataset version 1) | 100 files + manifest | 1.6 GB |
| Stage 1.0 tables | export | ~5 MB |
| IEX market data (`signaliq.db`) | export | ~2 MB |
| `sip_research.db` (validation batch v2) | export | ~0.4 GB (estimate) |
| Phase 1b raw responses (dataset version 2) | 8 files + manifest | 0.5 GB |
| Phase 1d anomaly responses | export | <1 MB |
| Hybrid dataset (version 3, when approved) | export | ~0.2 GB (estimate) |

Layout: `<bucket>/<dataset>/<export date>/...`, for example `stage_1_1/2026-11-28/0001-batch_symbols.jsonl.gz`.

## The four environment variables (same for A and B)

Added by you in a **personal** environment for this repository (not the Default, not an
organization environment):

| Variable | Secret | Example |
| --- | --- | --- |
| `SIGNALIQ_ARCHIVE_KEY_ID` | yes | B2 keyID / AWS access key ID |
| `SIGNALIQ_ARCHIVE_KEY_SECRET` | yes | B2 applicationKey / AWS secret access key |
| `SIGNALIQ_ARCHIVE_BUCKET` | no | `signaliq-research-archive-<something unique>` |
| `SIGNALIQ_ARCHIVE_ENDPOINT` | no | `s3.us-west-004.backblazeb2.com` / `s3.us-east-1.amazonaws.com` |

The endpoint host must also be allowed in the environment's network settings; I will propose
exactly that one host for you to approve. Environment variables are readable by commands in
that environment's sessions. That is why the key below can never delete, and the bucket locks
every version.

## A. Backblaze B2

**Bucket** (web console, Buckets, Create a Bucket):

| Setting | Value |
| --- | --- |
| Files in bucket | **Private** |
| Default encryption | **Enable** (SSE-B2) |
| Object Lock | **Enable** (choose it when creating the bucket) |
| Default retention | mode **Governance**, period **365 days** (see "Retention" below) |
| Lifecycle settings | **Keep all versions of the file** (the default); no rule that deletes or hides |

**Application key.** The console's "Read and Write" key type includes deleting, so create the
key with the B2 command-line tool on your own computer, listing the capabilities explicitly:

```
b2 key create --bucket <bucket> signaliq-archive listBuckets,listFiles,readFiles,writeFiles
```

- Restricted to that one bucket.
- Capabilities exactly: `listBuckets`, `listFiles`, `readFiles`, `writeFiles`.
- **Not:** `deleteFiles`, `bypassGovernance`, `writeFileRetentions`, `writeFileLegalHolds`,
  `writeBucketRetentions`, `writeBuckets`, `deleteBuckets`, any key capability (`writeKeys`,
  `deleteKeys`, `listKeys`), `shareFiles`.

Not verified here: whether this exact set is enough for every S3-API call below, and whether a
delete without a version ID is refused or only hides the file. The probe (step V1) checks both;
any surprise stops the procedure.

## B. AWS S3

**Bucket** (S3 console, Create bucket):

| Setting | Value |
| --- | --- |
| Region | your choice, e.g. `us-east-1` |
| Object Ownership | **ACLs disabled (Bucket owner enforced)** |
| Block Public Access | **all four settings on** |
| Bucket Versioning | **Enable** |
| Default encryption | **SSE-S3** |
| Object Lock (Advanced settings) | **Enable** |
| Default retention | mode **Governance**, **365 days** |
| Lifecycle rules | none |

**IAM user** `signaliq-archive`: no console access, one access key, and only this policy
(replace `BUCKET`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:ListBucketVersions",
       "s3:GetBucketVersioning", "s3:GetBucketObjectLockConfiguration"],
     "Resource": "arn:aws:s3:::BUCKET"},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion",
       "s3:PutObject", "s3:GetObjectRetention"],
     "Resource": "arn:aws:s3:::BUCKET/*"},
    {"Effect": "Deny", "Action": ["s3:DeleteObject", "s3:DeleteObjectVersion",
       "s3:BypassGovernanceRetention", "s3:PutObjectRetention", "s3:PutObjectLegalHold",
       "s3:PutBucket*", "s3:DeleteBucket*", "s3:PutLifecycleConfiguration",
       "s3:PutBucketPolicy", "s3:PutObjectAcl"],
     "Resource": ["arn:aws:s3:::BUCKET", "arn:aws:s3:::BUCKET/*"]},
    {"Effect": "Deny", "Action": "s3:*",
     "Resource": ["arn:aws:s3:::BUCKET", "arn:aws:s3:::BUCKET/*"],
     "Condition": {"Bool": {"aws:SecureTransport": "false"}}}
  ]
}
```

With versioning on, `PutObject` on an existing name adds a new version; the earlier one stays,
locked.

## Retention: Governance or Compliance

- **Governance (recommended):** the session's key cannot delete or shorten retention. You, with
  your own administrator account and the bypass permission, can still remove a mistaken upload.
- **Compliance:** nobody, including you and the provider's account owner, can delete any version
  before the period ends, and you pay for it until then. It is the strongest protection, but a
  mistake cannot be undone.
- 365 days is a starting point. Longer is safer, at a small storage cost.

## C. Second copy (yours, independent)

Either a bucket at the **other** provider under a **different** account, or an external drive.
You copy into it from your own computer, using credentials no session ever sees, then run
`scripts/archive_database.py verify` on that copy.

## Verification commands (run by me after you set up; nothing has been run)

Tool: the AWS command-line interface, which speaks to both B2 and S3. It is installed in a
throwaway virtual environment (`python3 -m venv /tmp/awscli && /tmp/awscli/bin/pip install
awscli`; not yet tested here).

```bash
export AWS_ACCESS_KEY_ID="$SIGNALIQ_ARCHIVE_KEY_ID" AWS_SECRET_ACCESS_KEY="$SIGNALIQ_ARCHIVE_KEY_SECRET"
export AWS_DEFAULT_REGION="$(echo "$SIGNALIQ_ARCHIVE_ENDPOINT" | cut -d. -f2)"
E="https://$SIGNALIQ_ARCHIVE_ENDPOINT"; B="$SIGNALIQ_ARCHIVE_BUCKET"; A=/tmp/awscli/bin/aws
```

**V1. Safety probe (no research data yet). Every line must give the stated result, or I stop.**

1. `curl -s -o /dev/null -w '%{http_code}' "$E/$B/"` gives 401 or 403: the bucket is not public.
2. `$A s3api get-bucket-versioning --endpoint-url "$E" --bucket "$B"` gives `Enabled`, and
   `$A s3api get-object-lock-configuration ...` shows the default retention you chose.
3. `$A s3api put-object --endpoint-url "$E" --bucket "$B" --key _probe/probe.txt --body probe.txt`
   returns a `VersionId` (V).
4. `$A s3api get-object-retention ... --key _probe/probe.txt --version-id V` shows a mode and a
   `RetainUntilDate`.
5. `$A s3api delete-object ... --key _probe/probe.txt --version-id V` **fails** (AccessDenied).
6. `$A s3api delete-object ... --key _probe/probe.txt` (no version) fails on S3. On B2 it may
   only hide the file; either is acceptable if step 7 passes.
7. `$A s3api get-object ... --key _probe/probe.txt --version-id V back.txt` and `cmp probe.txt
   back.txt`: version V is still there, unchanged.

**V2. Upload.** For each dataset export directory, first check that the prefix is empty
(`$A s3 ls "s3://$B/<dataset>/<date>/" --endpoint-url "$E"` lists nothing), then run
`$A s3 cp --recursive <export dir> "s3://$B/<dataset>/<date>/" --endpoint-url "$E"`.

**V3. Fresh download** into a new, empty directory:
`$A s3 cp --recursive "s3://$B/<dataset>/<date>/" <empty dir> --endpoint-url "$E"`.

**V4. F4 on the downloaded copy only:**
- `scripts/archive_database.py verify <dir>`: sha256 of every file against the manifest;
- `restore`, then `scripts/table_fingerprints.py --logical`: 21 of 21 Stage 1.1 tables equal to
  the reference;
- `scripts/audit_completeness.py` PASS;
- `scripts/response_store_tool.py verify` PASS, with fingerprint 61544d16fc7e61e4 (Phase 1b)
  and 18b8b83bebcc77a5 (Phase 1d);
- the IEX restore gives 15,861 bars, fingerprint e66890e79788fc78.

Disk allows one dataset at a time (the Stage 1.1 download plus its restore need about 4.5 GB).

**V5. Record** the counts, hashes and results in `research/results/` and commit them. Only then
is the archive called durable.
