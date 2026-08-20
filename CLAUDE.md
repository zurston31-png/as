# Working on this repo

## Always hand over the build — do not wait to be asked

After **every** push to this repo, without being prompted:

1. Run `./scripts/package.sh` — it builds `dist/memecoin-bot-<stamp>-<sha>.zip`
   from `git archive HEAD`, so the bundle is exactly what is committed.
2. Send that zip to the user with `SendUserFile`.

The user has asked not to have to request the new files each time. Treat a
push without a delivered bundle as an unfinished handover.

`git archive` is used deliberately rather than zipping the working
directory: the working tree carries `.env`, the runtime database, backups
and `__pycache__`, and a bundle built from it would ship live secrets.

## The project is frozen for a paper-collection run

Do not change any of these without the user explicitly asking:

scoring formulas · thresholds · scoring weights · exit policy · fees ·
slippage model · regime classifier · liquidity classifier · horizon
definitions · risk assumptions

All of them are covered by the strategy version hash (`app/strategy/version.py`),
including the ones that live in code. Editing one mints a new version and
splits the dataset, which is exactly what the freeze exists to prevent. A
change worth making becomes a **new challenger**, not an edit to the champion.

Fixing a genuine bug or red CI is always in scope.

## Non-negotiables

- **Paper only.** `LIVE_TRADING` and `LIVE_EXECUTION_ACKNOWLEDGED` stay
  `false`. No wallet keys, no real funds, no live-order execution.
- **Never fabricate data.** No invented API responses, historical prices,
  test results, or sample observations. A measurement that cannot be taken
  is recorded as unmeasurable, never as zero.
- **Never weaken a safety filter to improve a number.** The rug, security,
  data-quality and risk gates are not tuning candidates.
- **Do not claim an edge unless recorded evidence supports it.** The bot
  currently has zero trading observations; say so plainly rather than
  reporting numbers from an empty dataset.

## Conventions

- Pure Python for indicators — no numpy or pandas (compiled-wheel failures).
- `SessionLocal` is `autoflush=False`. A query issued before a `flush()`
  will not see pending rows; this has caused real bugs more than once.
- Migrations are additive only (`ALTER TABLE ADD COLUMN`, then indexes).
- Tests: when one fails, work out whether the test or the code is wrong and
  record *why* in the docstring. Do not loosen an assertion to get green.
- Run the full suite before pushing. Report the real number.
