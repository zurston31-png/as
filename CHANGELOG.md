# Autopilot session — 21 August 2026

Nine commits on `claude/memecoin-trading-bot-im07pf`, from `b21a6ab` to
`2d71d3b`. 47 files changed, ~6,600 lines added.

**Strategy version is unchanged at `v-83c77cda` throughout.** Nothing in
this session altered a scoring formula, threshold, weight, exit policy,
fee, the slippage model, or either classifier. The paper-collection
dataset is not split.

**Test suite: 1,591 passing**, of which **184 are new this session**
across 11 new test files. Every commit was pushed and CI was green on
every one checked.

---

## What this session found

Seven genuine defects. The suite that existed before this session passed
against every one of them, so each got a test reproducing the exact
condition rather than a nearby one. They are listed before the features
because they matter more.

### 1. Two exits racing could sell one position twice — `c642a20`

`close_position` read `position.qty`, awaited the sell, and only then
marked the row closed. The position monitor's stop-loss and a TradingView
sell alert are two callers of that path on one event loop, so a stop
firing while a sell alert was in flight had **both** of them sell the same
position and **both** credit the proceeds. The paper account was paid
twice for a position it held once, with two sell legs against one entry.

Unlike the entry race there was no downstream check to catch it, because
"is there an open position?" was true for both. `partial_close_position`
had the identical race, and a partial against a full exit was the same
collision.

Fixed with `reserve_exit`, keyed on the **position id** rather than the
mint — one reservation per position is what makes a partial and a full
exit mutually exclusive rather than merely each unique. Verified: 4 of the
11 new race tests fail without the guard.

### 2. A background worker could die silently — `ab36e0e`

Every worker opened its session with `db = SessionLocal()` on the line
*above* its try block. A full disk, a vanished database file or an
exhausted pool went straight past the handler, out of the loop, and killed
the task. asyncio does not report a task that dies unawaited, and
`main.py` holds a reference to every task until shutdown — so nothing was
printed at all.

For the position monitor, the only thing in the bot that fires a
stop-loss, that meant **exits could stop being checked in complete
silence** while the dashboard showed a green health panel.

All six loops now run through `app/monitor/supervisor.py`.

### 3. Three crashes and a fail-open — `2d71d3b`

- **`price_acceleration` read one bar further back than it checked for.**
  Guarded on `len(closes) < 60`, then read `closes[-61]`. 60 is the
  boundary the guard was written around, and `extract` calls this outside
  any handler — so a token sitting on it lost its entire feature set every
  pass. Reproduced at 59/60/61 bars: only 60 raised.

- **`scored_event` was unbound when live scoring was off.** Assigned only
  inside `if LIVE_SIGNAL_SCORE_ENABLED:` but read by the early engine,
  whose guard is the independent `EARLY_SIGNAL_ENABLED`. The read raised
  `UnboundLocalError` inside a `try` that logs and continues, so the early
  engine stopped working for every candidate and nothing said why.

- **The Pine payload emitted invalid JSON for sub-$1 prices.** Pine's `#`
  is an *optional* digit, so `0.00042` formatted as `.00042` — and a bare
  leading dot is not a valid JSON number, so the webhook rejected the
  whole alert. Memecoin prices are nearly always below 1, making this the
  normal case rather than an edge case.

- **The Jupiter live path treated "confirmed" as "filled."** A slippage
  revert confirms at the requested commitment and carries a non-`None`
  `err`; the old code ignored it and booked a position the wallet never
  held. It also reported a sell's **USDC leg** as `filled_qty` with
  tokens-per-USDC as the price — the inverse of the contract the paper
  engine sets — so a $100 sale would have booked $2,000 of proceeds and a
  partial exit would have subtracted 100 from a position holding 2,000
  tokens.

  This path is unreachable while the live flags are off, and it cannot be
  tested against a real chain from here. The 17 new tests drive the
  decision logic with stubs and prove exactly that: which failures are
  caught, what a reverted transaction is treated as, and how the legs map
  onto `SwapResult`. They prove nothing about behaviour against a real
  Solana RPC.

### 4. Paper-only was advised, not enforced — `2d71d3b`

`CLAUDE.md`'s first non-negotiable was held only by the two execution
backends and a preflight report. The operator-facing entry points — the
launcher, the scanner one-shot, the test-signal sender — would run against
a live-configured `.env` and print "PAPER mode", because they read
`LIVE_TRADING` and never `LIVE_EXECUTION_ACKNOWLEDGED`. A deployment with
`LIVE_TRADING=false` and the acknowledgement left on is one restart away
from real orders.

`app/safety/paper_only.py` is now the single answer, reading both flags.
The launcher's check matters most on the kept-`.env` path, where an
existing file survives untouched.

README's "Going-live checklist" and the VPS guide's "Going live" section —
which told the operator to fund a wallet and set `LIVE_TRADING=true` —
are replaced with a paper-run checklist. A test greps the docs so they
cannot come back.

### 5. A documented safety mechanism did not exist — `c642a20`

`app/concurrency.py`'s docstring claimed `reserved_elsewhere` made the
two-process case fail loudly. No such function existed, and never had. The
scope note now says plainly that the guard covers one process and cannot
detect a second.

---

## Risk work, all behind one disabled flag

`RISK_EQUITY_AWARE_DAILY_LOSS` defaults to **false**. Production risk
behaviour is unchanged by this session.

### Daily loss, measured as an equity drawdown — `5545d68`

An audit of `evaluate_daily_loss` found three ways the limit was weaker
than its name:

1. **Open positions were invisible.** Only `pnl_usd` on trades closed
   today was summed, so a book down $400 unrealized reported a daily loss
   of $0 and the bot kept buying. The limit bit only after the damage was
   crystallised — the moment it can no longer prevent anything.
2. **The reference was a constant.** `PORTFOLIO_STARTING_BALANCE_USD`
   never re-based, so on an account fallen to $600 the "5% limit" was
   still $50 — 8.3% of what was left. It grew *more* permissive exactly as
   the account shrank.
3. **It was never consulted on the buy path.** The only caller ran after a
   close.

What the audit did **not** find matters as much: **fees are not missing.**
`pnl_usd` is computed from fill prices and the fill model bakes the fee
into the fill price, so subtracting `Trade.fee_usd` would charge every fee
twice. A test pins that so nobody "fixes" it later.

`app/risk/daily_loss.py` measures `day_start_equity − current_equity`
instead. That single subtraction captures realized and unrealized at once
and **cannot double count, because there are no two terms to add.**

### Worst-case open risk — `6a81d43`

`app/risk/open_risk.py` asks: if every stop in the book were hit right
now, plus the trade being considered, would the day's limit be breached?

The trap is double counting. A position bought at $1.00, now at $0.90,
with a stop at $0.85 has already lost $0.10 — and that dime is already
inside the day's drawdown. Only the $0.05 from here to the stop is still
ahead. So stop risk is measured **from the current mark, never from
entry**, and the two figures are disjoint by construction.

Base assumes every stop fills exactly at its stop price — a floor, not a
forecast. Stress adds the deterministic part of the same fill model that
would price the real exit, against the **lowest pool depth actually
recorded** for that position. Nothing invented.

Three risks are named rather than modelled as zero, because nothing
recorded can quantify them: a gap through the stop, a liquidity pull, and
correlated stop-outs into one falling market. **The stress figure is a
floor on the bad case, not a ceiling.**

### How the flag interacts with the freeze

Enabling it mints a new strategy version. A new opt-in digest list means a
flag that decides nothing while off is free to add, but splits history the
moment it decides something — so the frozen dataset is preserved now and
the split happens automatically if the rule is ever switched on.

---

## Replay protection — `058e911`

The webhook had no memory. Post the same alert twice — a retry after a
timeout the bot actually handled, a proxy replaying a buffered request —
and it was processed twice.

Signals now carry a deterministic idempotency key over the canonical mint,
the direction, the alert's own bar timestamp and the price — never the
arrival time, which is the one thing a replay changes. **The uniqueness is
a UNIQUE index, not a SELECT-then-INSERT**: those are two statements, and
the gap between them is exactly the window a duplicate lands in. A test
inserts a duplicate behind the application's back to prove it.

An alert with no `time` gets a NULL key and no protection, deliberately:
without it, two genuine alerts on consecutive bars are byte-identical, and
**suppressing a real signal is a worse failure than processing a
duplicate.** NULL means "cannot be checked", not "no duplicate found".

The index is declared as an `Index` rather than `unique=True` so
`app/migrations.py` creates it on existing databases — a constraint that
only appeared on a fresh database would leave every real deployment
unprotected.

---

## Evidence and analysis

Nothing here tunes a filter or moves a threshold. All of it surfaces
evidence only.

### Rejection analytics — `8fd0a88`, `18274dd`

A new `/rejections` page, with three views ordered by increasing
authority:

- **Where candidates stop.** One row per **distinct mint**, at the deepest
  stage it reached. A token re-evaluated forty times counts once. A mint
  that failed a cooldown at 09:00 and bought at 11:00 was not stopped by
  the risk gate.
- **What each check uniquely contributes.** Every pre-screen check runs on
  every candidate and all five verdicts are recorded, so ablation is a set
  operation over rows on disk — no simulation. If liquidity rejects 480
  mints and 478 also fail three other checks, removing it changes the
  funnel by two.
- **Whether rejecting them was right.** The only question that can justify
  moving a threshold, and the only one needing outcomes. On this dataset
  it correctly reports that no check has 30 measured outcomes on both
  sides yet.

The page states the tempting misreading twice: the stage that stops the
most candidates is nearly always the earliest one, because it is the only
one that runs on everything. That is arithmetic, not evidence.

### Evidence grades — `20d2c1b`

Every statistic now carries **INSUFFICIENT / EARLY / USABLE / STRONG**.

The boundaries are derived, not chosen. For a proportion the 95%
confidence half-width is `1.96·√(0.25/n)`, so 30, 100 and 385 are where it
crosses 18, 10 and 5 percentage points. The width is reported alongside
the label, and a test asserts each boundary against the formula.

Two things the grade is **not**, both tested:

- **It is not evidence of an edge.** A precisely measured loss grades
  STRONG, and STRONG then means the bot is confidently losing.
- **It is not a promise for returns.** The ladder is derived for a
  proportion; memecoin returns are heavy-tailed and converge far more
  slowly, so a mean's caveat says its grade is an optimistic ceiling.

A report is graded by its **weakest** performance measure, never an
average, so three well-sampled numbers cannot launder one resting on four
observations.

### Resolver health — `20d2c1b`

`coverage()` says how much of the dataset is filled in; it cannot say
whether the thing filling it is alive. A worker that died an hour ago and
a young dataset both show coverage below 100%.

`app/analysis/resolver_health.py` reports the backlog — with the **oldest**
overdue row, since a hundred rows one minute late is the batch about to
run and one row six hours late is a fault — the lateness resolutions are
landing at, and rows sealed without a measurement **split by reason**: a
dead price feed is the market, lateness is this worker, and only one is
fixable.

"Late" comes from the same function the resolver uses, so the health check
and the resolver cannot disagree about which observations were lost. An
empty dataset reports IDLE, never HEALTHY.

---

## What was audited and found sound

Reported as findings-of-no-finding rather than skipped:

- **Canonical token identity.** Every dedup, exposure and cooldown lookup
  already goes through `instrument_key`. Added structural guards that read
  the source, so a future `filter_by(symbol=...)` in the risk path fails a
  test rather than silently pooling two unrelated assets under one
  exposure cap.
- **Rug/security screening.** Every no-data path already fails closed —
  no address, no scanner data, token not found. `RUGCHECK_ENABLED=false`
  is the one legitimate pass-open path; it warns and stamps the report so
  a trade taken without screening is identifiable afterwards.
- **Hard risk ceilings.** No combination of `.env` values produces a
  manager past its ceilings. Added one test for the collective property,
  alongside the existing per-clamp tests.

---

## Still open

- **The Jupiter fill is estimated from the quote, not read from the
  chain.** `SwapResult.fill_estimated_from_quote` labels it rather than
  passing it off as a measurement. Reading the executed amounts needs the
  transaction's token balance deltas, which this backend does not parse.
  Live path only — unreachable while the flags are off.
- **The VPS is still running bundle `b21a6ab`.** Everything in this
  changelog is on the branch and in the delivered zips, but nothing has
  been deployed — that needs your terminal.
- **`paper.py` feeds `price_change_1h_pct` into a parameter named
  `volatility_1h_pct`.** Flagged in an earlier session and deliberately
  not changed: the slippage model is frozen.
- **Zero trading observations.** Nothing in this session produced
  evidence of an edge, and the bot still has none. Every analysis page
  added here currently reports INSUFFICIENT, which is the correct answer.
