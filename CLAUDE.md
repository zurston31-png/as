# Working on this repo

## Handing over the build

The VPS deploys from a **git clone** at `/root/memecoin-bot-live` and
updates itself: a systemd timer runs `deploy/auto_update.sh` every 15
minutes, which pulls, rebuilds, and rolls back on its own if the new
container is unhealthy. Pushing to `claude/memecoin-trading-bot-im07pf` is
therefore the whole handover — **do not send a zip after every push.**

Build one with `./scripts/package.sh` only when asked, or when the git
path is unavailable. It uses `git archive HEAD` rather than zipping the
working directory, deliberately: the working tree carries `.env`, the
runtime database, backups and `__pycache__`, and a bundle built from it
would ship live secrets.

What the auto-updater will NOT deploy, and why it matters when writing a
commit that is about to go live unattended:

- anything that changes the **strategy version hash** — that splits the
  collection dataset, so it aborts and resets rather than deploying
- anything where `LIVE_TRADING` or `LIVE_EXECUTION_ACKNOWLEDGED` is not
  false in the built image
- a non-fast-forward, a failed build, or a container that fails its health
  check within 90s (that last one restores the previous commit)

So a strategy change is not merely discouraged by this file — it will be
refused by the machine, and the bot will stay on the old commit until
someone deploys it deliberately.

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
