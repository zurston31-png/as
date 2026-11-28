# Stage 2 plan: universe of 250 securities (preparation only)

**Status: plan for review. Nothing in this document has been run.** The approval received
covers writing this plan only. Stage 2 work starts only after the plan is approved, and each
phase below ends at a gate that needs its own approval.

Unchanged constraints:

- No Strategy A, no backtests, no optimization, no parameter freezing.
- No market data or evidence dated 2025 or later.
- No trading or paper-API access.
- The IEX database, the Strategy A archive, Stage 1.0 and Stage 1.1 are not modified.
- Today's ticker list never decides the historical universe; IEX and SIP stay separate.
- One-source outcomes stay lower confidence; assumed zeros stay apart from payouts;
  unresolved outcomes keep no value.

Figures marked *measured* come from the Stage 1.1 database, read-only. Figures marked
*estimate* are projections, to be replaced by measurements at the gates.

## 0. Starting point and the constraints that shape the plan

| Item | Stage 1.1 | Source |
| --- | --- | --- |
| Securities in the batch | 1,036 | measured |
| Membership decisions | 59 monthly, 2020-01 to 2024-11, 100 members each | measured |
| Distinct members | 208 | measured |
| Candidate pool | 18,490 tickers (10,370 with a monthly rank) | measured |
| Tickers ranked within 3 × N in some month: N = 100 / N = 250 | 840 / 1,960 | measured (`candidates.best_rank`) |
| Tickers added by the completeness proof at N = 100 | 158 | measured |
| SEC cache | 38,120 answers, 20.4 GB | measured |
| of which whole documents | 10,591 answers, 18.9 GB (1.8 MB each on average) | measured |
| of which cover-page snippets | 7,441 answers, 0.16 GB | measured |
| Free disk in this session | 8.1 GB | measured (`df`) |

Three constraints follow:

1. **Disk.** The Stage 1.1 database cannot be copied, because a copy failed once by filling
   the disk. Stage 2 can't add whole documents at Stage 1.1's rate either: at 1.8 MB each,
   about 4,500 more would use all the free space.
2. **Durability.** The Stage 1.1 database and its SEC cache live only in this cloud
   container, which is temporary. Git holds the reports, the audits and the table
   fingerprints, not the data. Section 5 covers this.
3. **Breadth.** "Top 250 by 63-session median dollar volume" is a liquid, large-company
   universe. Stage 2 makes it larger. It will not make it a broad-market sample, and the
   report must not imply that it does (section 6).

## 1. How the universe expands from 100 to 250

The selection rule is unchanged except for its size. It keeps the top N eligible common
stocks by 63-session median dollar volume, a $5 minimum raw price, 126 sessions of history
and 50 active sessions in the window. N goes from 100 to 250, so the rule fingerprint
changes and Stage 2 has a version of its own.

The expansion reuses the mechanism that proved Stage 1 complete (`app/universe/stage.py`):

1. **Candidate pool.** Same sources as Stage 1:
   - SEC's exchange-listed ticker list;
   - Alpaca corporate actions 2016–2024;
   - Wikidata US listings with dates;
   - the named validation cases.

   No source is dropped and today's list alone never decides. The pool of 18,490 tickers
   does not depend on N.
2. **Monthly screen.** A ticker gets daily bars if it ranks within 3 × N = 750 by monthly
   traded value in any month that feeds a decision. That is 1,960 tickers (measured),
   against 840 at N = 100.
3. **Completeness proof.** For every ticker left out, an upper bound on its 63-session
   median dollar volume is computed from monthly bars at each decision. Any ticker whose
   bound reaches the 250th member gets daily data, and the stage repeats until none is
   left. The 250th member's dollar volume is lower than the 100th's, so more tickers will
   be added: *estimate* 300–600, against 158 at N = 100.
4. **Evidence per security, unchanged:** types with every source kept, point-in-time
   issuers with dated evidence, exit outcomes, trading-gap and rename decisions, handovers,
   and deal terms.

*Estimates:*

- about 2,300–2,600 tickers with daily data and 2,500–2,900 securities, against 1,036;
- about 1,400–1,800 new securities needing SEC evidence;
- about 450–550 distinct members (Stage 1.1 had 2.1 distinct members per seat; that ratio
  may fall as N grows).

## 2. How point-in-time membership is preserved

All Stage 1 protections carry over unchanged and must pass again:

- Membership reads only raw daily bars up to the decision, the security type, held-out
  split checks, and SIC and issuer evidence filed by then. It never reads exit outcomes,
  exit values or any later price.
- Recompute with data cut off at each decision: 59 of 59 identical.
- Every price after 2022-06-30 multiplied by 10: earlier decisions unchanged.
- Members chosen after their last trade: 0.
- Blank-check exclusions re-derived from SIC on file by the decision date
  (`audit_issuers.py`).
- Latest date of every input: before 2025.

New check, the **nesting test**: the top 100 of each Stage 2 decision must equal the
Stage 1.1 membership of that decision.

- The ranking rule is the same, and the completeness proof at N = 100 already showed that
  no ticker outside the Stage 1.1 batch could have been a top-100 member.
- A difference can therefore come only from evidence that changed for a security in both
  batches: a new issuer candidate, a new type source, a new identity decision.
- Every difference is listed with its cause and resolved before the gate. A difference is
  never absorbed silently.
- `compare_stage_versions.py` gains this check.

## 3. How unresolved types and outcomes are handled

**Types.** Stage 1.1 has 4 conflicts and 17 unknown types. Stage 2 will add more, because
lower ranks hold more leveraged products, foreign listings and short-lived companies.

- Rules unchanged: only `confirmed`, `resolved` and `single_source` common stock can be
  a member, and an unknown or conflicting type is never guessed.
- `excluded_type_impact.py` runs on every Stage 2 run and lists each unknown or
  conflicting security that would have been a member if counted as common.
- **Gate rule:** every such would-be member gets a documented decision before the gate:
  - resolved from the issuer's own SEC text (a quoted passage, re-verified on every run);
  - or confirmed as not common stock (TVIX at N = 100 is an ETN);
  - or left unresolved and listed with its decisions and the members it would displace.

  Securities that are never near membership may stay unknown, counted and listed.

**Dated types (limitation 3).** Today a type applies to the whole history. Proposed, as a
separately approvable item, in preparation phase P1:

- A type becomes dated only where the issuer's own dated cover pages show a change. For
  example, CRH's 20-F shows ADSs until 2023 and its 8-K of December 2024 shows ordinary
  shares.
- Each decision then reads the type in force on its date, using only cover pages filed by
  then.
- Securities without such evidence keep their undated type, flagged as undated in the
  report.
- The change is first measured at N = 100 (see P1) and its impact reported before it is
  adopted.

**Co-registrants (limitation 2).** A parent and a subsidiary can file the same documents.
Proposed rule, separately approvable:

- When co-registrants overlap, prefer the registrant whose own cover-page row lists the
  ticker. If both rows do, prefer the one named first on the combined cover page (usually
  the parent; the five known cases are checked by hand before the rule is relied on).
- Known cases: PCG, URI, QSR, HTZ and CHTR/CCO Holdings.
- The rule is tested on those five, measured at N = 100, and rechecked for every overlap by
  `audit_issuers.py`.

**Outcomes.** Unchanged:

- One-source outcomes stay lower confidence.
- Assumed zeros stay apart from payouts.
- Unresolved outcomes keep no value.
- Each value is traced to its evidence.

At N = 100, 7 members later ended, and all have a traced value. At N = 250 the
*estimate* is 20–40.

**Gate rule:** every member that later ended has either a traced value (stated cash or
deal terms re-verified against the acquirer's raw close on the effective date) or is listed
as unresolved with its reason. Where an acquisition's terms are missing, they are read by
hand from the target's own 8-K, as for AGN, WORK and RTN. Non-members may remain
unresolved and are counted.

## 4. How completeness is proved

Within the pool: the completeness proof of section 1, rounds until none is left, as now.

New in Stage 2:

1. **An independent audit of the proof.** A new `scripts/audit_completeness.py`, written as
   separate code, re-derives each excluded ticker's upper bound from the stored monthly
   bars. It checks that no excluded ticker's bound reaches the 250th member at any
   decision, and that every added ticker appears with the decision that added it.
2. **A check of the pool itself.** The proof covers only tickers in the pool. A company
   that no source names would be invisible. The cross-check:
   - read EDGAR's quarterly form indexes from 2019-Q4 to 2024-Q4 (public, read-only, about
     21 small files);
   - list every Form 25 and 25-NSE filer;
   - map each filer to its tickers through the issuer evidence;
   - report every delisted company with a ticker that is not in the pool, and any of them
     that might have been near membership.

   This catches companies that left exchanges in the membership period without an Alpaca
   corporate action. Its result is reported whatever it shows.
3. **Unchanged from Stage 1:**
   - every SEC request is accounted for, and the cache grows by exactly the answers stored;
   - a missing answer stops the run (`StrictCache`) instead of becoming "no issuer";
   - issuers are re-read independently (`audit_issuers.py`);
   - 0 common stocks without an issuer and without a documented reason;
   - duplicate identities: 0 ticker-days held twice, and every same-ticker succession
     recorded.
4. **Reproducibility:** two full runs with byte-identical reports and identical
   fingerprints for every table.

## 5. How the Stage 1.1 archive is protected

- **A separate database.** Stage 2 writes only to a new `data/universe_stage_250.db`.
- **A read-only layered cache.** Stage 2 reads SEC and other cached answers first from its
  own cache, then from the Stage 1.1 database opened read-only (SQLite `mode=ro`). New
  answers go only to the Stage 2 cache. Nothing is copied, so the disk limit holds, and the
  Stage 1.1 database is never written.
- **A guard in code.** Opening a protected database (Stage 1.1, Stage 1.0 tables, IEX) for
  writing raises an error. A test covers it.
- **Proof that the new path is neutral.** In phase P1, the unchanged N = 100 rule is run
  into a fresh database through the layered cache. Its report and table fingerprints must
  equal Stage 1.1's (`reproducibility.txt`) byte for byte, before any rule change is
  measured.
- **Checks before and after every Stage 2 run:**
  - the 21 Stage 1.1 table fingerprints;
  - IEX at 15,861 bars and fingerprint `e66890e79788fc78`;
  - Strategy A `SHA256SUMS` 13 of 13;
  - git shows no commit touching `research/results/universe_stage_100/`,
    `universe_stage_100_v1_1/` or `strategy_a/`.
- **Compressed storage in the Stage 2 cache.** New whole documents are stored compressed,
  and only in the new database. Cover pages stay snippets. Stage 1.1 rows keep their
  format.
- **A disk guard.** A run stops cleanly, with its accounting, if free disk falls below
  2 GB. It resumes from the cache without duplicating anything.
- **Durability, which needs your decision.** If this container is reclaimed, the Stage 1.1
  data cache (20 GB) is lost. Git keeps the reports, audits and fingerprints, so a rebuild
  could be verified against them, but every answer would have to be fetched again.
  Options:
  - (a) accept the risk, since SEC filings do not change and a rebuild is checkable;
  - (b) provide durable storage (a bucket or drive) for a compressed export of the cache;
  - (c) work in a session with more disk.

  Recommendation: (b) if available, otherwise (a), documented.

## 6. What Stage 2 does not claim

- **It is not broad-market coverage.** The report will show, per decision:
  - the SIC division mix;
  - the dollar-volume range from the 1st to the 250th member;
  - turnover between decisions;
  - members that later ended.

  It will be compared with Stage 1.1, which makes concentration and skew visible.
- **No strategy result** of any kind.

## 7. Phases, gates and estimates

| Phase | Work | Network | Ends with |
| --- | --- | --- | --- |
| P1. Preparation (code and tests) | separate database, layered read-only cache, write guard, compressed storage, disk guard, request accounting carried over; `audit_completeness.py`; EDGAR Form 25 pool cross-check; nesting test; optional dated types and co-registrant rule | none, except EDGAR quarterly indexes (about 21 files) | the N = 100 neutral-path run equal to Stage 1.1 byte for byte; then, separately, the measured impact at N = 100 of each optional rule; full tests pass; **gate 1** |
| P2. Fetch | Alpaca monthly and daily SIP bars for the screen and the completeness rounds; SEC evidence, read-only (contact `signaliq.research@gmail.com`) | Alpaca market data, SEC | counts of requests, answers stored and failures per step; disk used; nothing ranked or reported yet; **gate 2** |
| P3. Stage 2 run and audits | two full runs at N = 250; all audits (merges, outcomes, deals, issuers, issuer gaps, excluded types, completeness, pool cross-check); comparison with Stage 1.1, including the nesting test; protected-data checks | none (cache only) | Stage 2 report, archived in `research/results/universe_stage_250/`; **gate 3: stop for review** |

*Estimates*, to be replaced by measurements at each gate:

- **SEC requests:** Stage 1.1 needed about 37 cached answers per security. For 1,400–1,800
  new securities that is about 50,000–65,000 requests. SEC's limit allows at most 10 per
  second and the prefetch runs at about 6.7 per second, so that is at least 2–3 hours, and
  more likely 4–8 hours with retries and the cover-page searches.
- **Disk:** under 6 GB, with snippets and compression. The P2 gate reports the real figure.
- **One full run at N = 250:** about 3 hours, against 70 minutes at N = 100. Audits about
  30 minutes.

## 8. Decisions requested

1. Approve phase P1 (preparation code and tests, no Stage 2 run).
2. Dated types (section 3): adopt if the N = 100 measurement is acceptable, or keep as a
   documented limitation.
3. Co-registrant rule (section 3): adopt if the N = 100 measurement is acceptable, or keep
   as a documented limitation.
4. Durability of the cache (section 5): option (a), (b) or (c).
5. Gate thresholds as written:
   - completeness: 0 tickers left that could have been members;
   - nesting: every difference explained;
   - would-be members with an unknown type: each with a documented decision;
   - members that later ended: each valued or listed as unresolved.
