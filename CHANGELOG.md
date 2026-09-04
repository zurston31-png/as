# Auto-update covers both deployment paths — 4 September 2026

The updater shipped earlier today only handled the Docker path, while
`deploy/vps_setup.md` documents two: Docker (section 1) and systemd + venv
(section 2). Half the documented deployments would have self-updated and
half would not, with nothing saying which.

`deploy/auto_update.sh` now detects the mode and handles both — build and
container swap on Docker, `pip install` into the venv and `systemctl
restart` on systemd — behind the same refusal set. `MODE=docker|systemd`
overrides detection. The freeze and paper-only checks still run against
the NEW code before it replaces the running service: in a throwaway
container on the Docker path, from the updated checkout on the systemd
path.

Section 4 of the guide now opens by pointing at the timer instead of
leading with the manual procedure. The manual steps stay, because they are
still needed for the first deploy, for the case where the automatic update
deliberately refuses, and for when something has gone wrong enough to want
to drive it by hand.

Verified: the no-op path was run end to end on this checkout and exited
cleanly at `up to date at d4b0c98, nothing to do`. The deploy paths
themselves cannot be exercised from here — there is no access to the VPS —
which is why the install instructions lead with a dry run.

Deploy tooling and documentation only. **Strategy version unchanged at
`v-83c77cda`. Suite: 1,730 passing.**

---

# The VPS updates itself, and refuses to — 4 September 2026

`deploy/auto_update.sh` plus a systemd timer poll the branch every 15
minutes and redeploy when it moves. The owner asked for this after
noticing that pushing to GitHub did not change what the bot was running.

Most of the file is the part that does NOT deploy. Automatic deployment
puts code on a running trading bot with nobody in between, and the value
of this project is one clean dataset under ONE frozen configuration, so
the script aborts rather than proceeding when:

- the **strategy version hash would change**. The freeze, enforced
  mechanically rather than by convention: a new hash splits the dataset,
  and that is not a decision a timer gets to make at 4am.
- `LIVE_TRADING` or `LIVE_EXECUTION_ACKNOWLEDGED` is not false in the
  built image.
- the pull is not a fast-forward — divergence means someone worked by
  hand, and an automatic merge would bury it.
- the build fails (old container never stopped), or the new container is
  unhealthy within 90s (previous commit restored, rebuilt, brought back).

The version and paper-only checks run against the **new image, in a
throwaway container, before it replaces the running one**, so a bad deploy
is caught before it serves anything. The database is copied to
`/data/deploy-backups/` before any restart and lives outside the clone.

Also corrected: the "send a zip after every push" rule in `CLAUDE.md`. It
was written when a zip was the only way onto the box, and survived the
switch to a git clone two days ago — so every push since produced a file
card the owner had no use for. Replaced with the git-pull workflow and a
note that the auto-updater will refuse a frozen-config change outright.

Deploy tooling and documentation only. No application code touched.
**Strategy version unchanged at `v-83c77cda`. Suite: 1,730 passing.**

---

# Two unreachable analyses, made reachable — 4 September 2026

## Monte Carlo: the second mode was implemented but unusable

`app/analysis/monte_carlo.py` has always supported two resampling modes.
BOOTSTRAP (with replacement) answers *what range of OUTCOMES is consistent
with this edge*. SHUFFLE (order only) reorders the exact trades, so every
path ends at the same total by construction, and answers *given this edge,
how bad could the RIDE have been* — the survivability question, and the
one that says whether a different ordering of the same trades would have
breached the daily loss limit.

Only bootstrap was reachable. `scripts/performance_report.py --mc-mode
{bootstrap,shuffle}` now selects; bootstrap remains the default so no
historical report changes meaning.

## Out-of-sample and walk-forward could never be answered at all

Both criteria read *"no analysis run yet"* permanently, however many times
`scripts/run_backtest.py --walk-forward` was run — nothing carried a
backtest result into the gate. `--evidence-out` now writes one and
`--backtest-evidence` reads it.

**Connecting them creates a hazard bigger than the feature, and most of
this work is the guard.** The backtester DEFAULTS to synthetic candles.
Run on this machine, a default `--walk-forward` produced 3 of 3 profitable
windows and a profitable 12-trade out-of-sample window — enough to flip
both criteria green on a market that never existed. So a run records where
its candles came from, and `app/analysis/backtest_evidence.py` refuses
anything that is not real market history: synthetic outright, unrecognised
sources fail-closed, stale schema, missing fields, and arithmetic that
cannot be true (more profitable windows than windows). A refusal returns
None with a printed reason rather than raising, because the correct
response is to keep reporting the criteria as unmeasured — not to crash
and not to substitute a pass.

Read-side only: `app/analysis/`, `scripts/`. No scoring, threshold,
weight, exit policy, fee, slippage, classifier or horizon touched.
**Strategy version unchanged at `v-83c77cda`.**

14 new tests across `tests/test_backtest_evidence.py` and
`tests/test_report_mc_mode_and_evidence.py`, including the synthetic
figures above pinned end to end. **Suite: 1,730 passing.**

---

# Holding time reported at the scale the strategy trades at — 1 September 2026

The repaired holding-time panel immediately showed why it needed to be
repaired, and then showed a second problem:

    average / median     0.1h / 0.1h
    shortest / longest   0.0h / 0.4h
    by holding time
      <1h    24 trades    62%    -1.38

The champion closes positions in MINUTES. At one decimal in hours every
trade rounds to two or three indistinguishable values, "0.0h" spans ten
seconds to three minutes, and a breakdown whose first edge is 1h cannot
separate a book whose LONGEST hold is 24 minutes. A breakdown that puts
every trade in one bucket is not a breakdown.

- `format_duration_hours()` renders a duration in its natural unit
  (7s / 6m / 3.7h / 1.1d), used by the CLI report and the dashboard.
  Returns None for None so callers keep their own "n/a" and no
  unmeasured span becomes "0s".
- Holding-time buckets are now `[5m, 15m, 30m, 1h, 4h, 12h, 1d, 3d]`,
  with each edge rendered in its own unit. Resolution goes where this
  strategy lives; the long end is kept so a slower challenger still fits.
- `_build_breakdown` groups and sorts by interval INDEX rather than by
  parsing digits back out of rendered labels. The old string-parsing sort
  needed a tiebreak to keep the open-ended bucket first and would have
  broken outright on a mixed-unit label like "30m-1h".

Read-side only. `app/analysis/` is not covered by the strategy version
hash (which spans BEHAVIORAL_SETTINGS, scoring weights, and the regime
and liquidity constants), so no bucket edge here can split the dataset.
**Strategy version unchanged at `v-83c77cda`.**

8 new tests in `tests/test_duration_resolution.py`, one of which pins
that the numeric breakdowns keep their existing labels so the refactor
cannot silently relabel history. Two existing tests in
`test_trade_analytics.py` were UPDATED, not loosened: both asserted the
old edges, both keep an exact full-label assertion, and each records in
its docstring what changed and why. **Suite: 1,716 passing.**

---

# Holding-time summary resolves its entry legs — 1 September 2026

Caught by the upgraded deployment contradicting itself in one report:

    HOLDING TIME
      average / median     n/ah / n/ah        <- summary
    by holding time
      <1h    24 trades                        <- breakdown

Same 24 trades, two answers. `opened_at` is on the buy leg and
`closed_at` on the sell, so measuring the span needs the entry index.
`breakdown_by_holding_time` passed one; `summarize_holding_time` did
not, and its `entries=None` default let that compile silently. This was
a miss in the entry-attribution fix (`ae940c6`) - it threaded the index
through every breakdown and not through the summary beside them.

`summarize_holding_time` now builds the index itself rather than taking
it as an argument, so no caller can forget it again.

Read-side only. **Strategy version unchanged at `v-83c77cda`.**

5 regression tests in `tests/test_holding_time_consistency.py`; 4 of the
5 fail against the previous code. The invariant they hold is that the
summary and the breakdown count the same trades, not any particular
duration. **Suite: 1,708 passing.**

---

# Exit-reason buckets group by rule — 1 September 2026

Found by reading a live deployment's own report rather than by review:
24 closed trades produced 24 exit-reason buckets of one trade each,
every one flagged "too few trades to mean anything", when 21 of them
were the same rule firing 21 times.

`breakdown_by_exit_reason` grouped on the raw `close_reason` string, and
every exit reason renders the numbers that fired it —
`trend reversal: lower highs after peak $0.00024290`. No two exits fire
at the same price, so the grouping never grouped anything, and the most
useful row the report could print was the one row it could not.

Buckets now key on the exit RULE, with prices and percentages replaced
by a placeholder (`app/analysis/trade_analytics.py::exit_reason_rule`).
The raw reason on the trade row is untouched; this changes only how the
breakdown aggregates. On the deployment that surfaced it, 24 buckets
become 4: 21 trend reversals, 1 momentum loss, 1 partial profit-take,
1 dev-wallet exit.

Read-side only — no scoring, threshold, weight, exit policy, fee,
slippage, or classifier change. **Strategy version unchanged at
`v-83c77cda`.**

8 regression tests in `tests/test_exit_reason_grouping.py`, written
against the verbatim strings that deployment emitted. 7 of the 8 fail
against the previous code. **Suite: 1,703 passing.**

---

# Autopilot session — 21 August 2026

Fourteen commits on `claude/memecoin-trading-bot-im07pf`, from `b21a6ab`
to the cost-aggregation fixes. 55 files changed, ~7,500 lines added.

**Strategy version is unchanged at `v-83c77cda` throughout.** Nothing in
this session altered a scoring formula, threshold, weight, exit policy,
fee, the slippage model, or either classifier. The paper-collection
dataset is not split.

**Test suite: 1,681 passing**, of which **273 are new this session**
across 17 new test files. Every commit was pushed and CI was green on
every one checked.

---

## What this session found

Twenty genuine defects. Two came from a second review round, two from a
third and one from a fourth; **four of those five were fail-opens, a
late-firing safety check, or a wrong figure in code written earlier in
this same session**. Three more came from auditing the post-mortem
record, and two more from asking whether that record's weighting bug
existed anywhere else - it existed in both siblings. One reported
finding was checked and rejected; two more were declined with reasons.
The suite that existed before this session passed
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

### 5. A drained pool looked perfectly stable — round 2

`liquidity_stability` collected depth readings with `if o.liquidity_usd`,
which is falsy for `0.0`. Every zero reading was dropped, so the swing was
computed over only the surviving non-zero depths — and a pool that went
from $50,000 to zero came out with a stability of **1.0, perfectly
stable.** Verified against the old code: it really did report 1.0.

Two siblings in the same file, same root cause — a measurement that could
not be taken, or a genuine zero, treated as interchangeable:

- `txn_rate_change` guarded on `buys_1h` alone and then folded a missing
  `sells_1h` to zero with `or 0`, publishing a half-reported observation
  as a complete one. `pressure()` two functions below already required
  both counts.
- Four detail strings used `if value` rather than `if value is not None`,
  so a genuine 0.0 ratio was described as "volume or liquidity missing"
  when both had been measured.

Fixing the liquidity filter also exposed a latent `ZeroDivisionError` when
every reading is zero — now reported as unmeasurable rather than as the
1.0 that would have claimed perfect stability for a pool that no longer
exists.

### 6. The paper-only guard did not recognise `LIVE_TRADING=1` — round 2

A fail-open in the guard added earlier in this same session. The launcher
and test-signal sender are stdlib-only, so they parse `.env` by hand — and
they matched only the literal string `true`. pydantic-settings accepts
`1`, `yes`, `on`, `y`, `t` and `True` as well, so a bot configured with
any of those was live while the guard waved it through, reporting a safety
it was not providing.

Both now share a parser checked against pydantic's actual behaviour, skip
commented lines, handle `export FOO=bar`, and treat an unparseable value
as ENABLED — a refusal path should not resolve "I cannot tell" as "safe".

### 7. A documented safety mechanism did not exist — `c642a20`

`app/concurrency.py`'s docstring claimed `reserved_elsewhere` made the
two-process case fail loudly. No such function existed, and never had. The
scope note now says plainly that the guard covers one process and cannot
detect a second.

---

### 8. Recorded slippage was a function of the trade's return — post-mortem audit

The worst of the three, because it does not crash, does not look wrong,
and produces a confident false finding.

`build_postmortem` took the **mean** execution cost across a position's
legs and subtracted **every leg's fees** divided by the **entry**
notional. Those two terms are not commensurable: the fee term grows with
the exit notional, the cost term does not. So the column named
`slippage_pct` was partly a measure of how much the trade made.

Measured against the real fill model (0.25% fee + 0.75% impact/spread per
leg, so 0.75% true slippage throughout):

| exit | reported slippage | true |
|---|---|---|
| flat | +0.500% | 0.750% |
| 2x | +0.250% | 0.750% |
| 4x | −0.250% | 0.750% |
| 10x | −1.750% | 0.750% |
| 50x | −11.750% | 0.750% |

A 50x winner recorded execution *paying* 11.75%. Regress execution cost
against outcome on that column and you find "our winners fill better" —
an artefact of arithmetic that would survive review because the column is
named `slippage_pct` and the number is plausible at small returns.

Now measured per leg and then averaged: `execution_cost_pct −
fee_usd/leg_notional`, which is the subtraction `app/shadow/recorder.py`
already did per fill. A leg with no recorded price is **dropped**, not
counted as fee-free — counting it would report its whole cost as slippage
and invent a cost that was never measured.

### 9. Execution cost was emitted as a fraction beside real percents

`Trade.execution_cost_pct` is a fraction on the row — `trade_analytics.py`
says so and `performance.html` multiplies by 100 to display it. The
post-mortem passed it through raw, so `/api/postmortems` returned
`execution_cost_pct: 0.0087` next to `return_pct: 5.2` and
`slippage_pct: 0.74`. Same suffix, one of them 100x off. Now converted at
the boundary, with the unit stated in the module docstring.

### 10. The observation count saturated at 30 and then lied flat

The post-mortem's honesty mechanism: MFE/MAE come from polled prices, so
they are lower bounds, and `samples` was supposed to say how loose those
bounds are. It was `len(position.recent_prices)` — a buffer trimmed to
`MAX_RECENT_PRICE_SAMPLES = 30` on every tick. The water marks it
qualifies are updated on every tick and never trimmed. So a position
priced 30 times and one priced 3,000 times both reported 30, and the
number meant to grade confidence stopped varying with confidence.

Added `Position.price_ticks_observed`, incremented in `record_price_tick`
alongside the water marks it describes. Additive, nullable, **no
default** — existing rows land `NULL`, which reads as "not recorded",
where a `0` would have asserted that the excursion came from no
observations at all. Verified against a database built without the
column: the additive migration adds it and the pre-existing row is
`NULL`.

The post-mortem reports both, because they bound different things:
`samples` is what the momentum exits could see, `price_ticks` is how
tight MFE/MAE are.

Three defects, 17 new tests in `tests/test_postmortem_costs.py`. 13 of
the 17 fail against the previous code; the other 4 cover the new counter.
No existing test asserted either cost figure, which is how both survived
every prior review.

### 11. `LIVE_TRADING = true` walked straight past the guard — round 3

The second fail-open found in the guard I wrote two rounds ago, and the
same mistake in a different place.

python-dotenv — which is what pydantic-settings parses `.env` with —
trims whitespace around the separator, so it reads `LIVE_TRADING = true`
as enabled. Both operator scripts matched the literal prefix
`LIVE_TRADING=`, so they saw nothing, printed their paper-mode banner and
carried on. Verified against the installed parser, not the docs: the
loader returns `{'LIVE_TRADING': 'true'}` and the guard returned `[]`.

A guard that reassures is worse than no guard. Round 2's fail-open was
the same shape — the guard reimplemented value parsing and got a
different answer from pydantic. This one reimplemented key parsing and
got a different answer from dotenv. Now `partition("=")` with the key
trimmed and compared for **equality**, so `LIVE_TRADING_NOTES=true` still
does not match.

### 12. The halt fired one trade late — round 3

`SessionLocal` is `autoflush=False`, and both exit paths reach
`_check_halt_conditions` having only `db.add()`ed the filled sell leg.
`evaluate_consecutive_losses` queries `models.Trade` directly, so the
streak was computed from the previous N trades — without the loss that
should have triggered the halt. The daily-loss check has the same
exposure. A run hitting its loss limit would take one more position
before stopping.

Fixed with `db.flush()` at the top of `_check_halt_conditions` rather
than at the two call sites, so a third exit path cannot reintroduce it by
forgetting. A test asserts it flushes and does **not** commit — the
caller still owns the transaction.

### 13. My own slippage fix still averaged unequal legs — round 4

Review round 4 landed on the commit that fixed defects 8-10, and it was
right.

Removing the return-dependence was necessary but not sufficient: the
replacement took an **unweighted mean over legs**, so a small expensive
leg counted the same as a large cheap one. A $10 scalp-out at 5% and a
$990 exit at 0.5%, alongside a 0% entry leg, average to **1.8333%** over
the three legs (2.75% if the entry is excluded), when the position
actually paid $5.45 on $2,000 traded - **0.2725%**. Off by 6.7x, in the
direction that makes execution look worse the more finely an exit is
scaled out.

(That paragraph first said 2.0% and "more than 7x". Both were wrong -
2.0% is neither the three-leg mean nor the two-leg one. Caught in review
round 5; the test alongside it had the right figure all along.)

Now aggregated in dollars over notional: cost and slippage are summed as
amounts and divided by the notional that produced them.

Worth recording why the round-3 tests missed it: **every leg in them was
the same size and the same cost**, so a weighted and an unweighted mean
agree exactly. The tests confirmed the property they were written for
and were blind to the one next to it. The new test uses legs that differ
in both, and fails against the unweighted version.

### 14. A test I wrote this session was a time bomb

`tests/test_daily_loss.py` pinned `NOW` to a hardcoded `2026-08-21`. The
champion `evaluate_daily_loss` takes no `now` argument — it derives the
day window from the real clock — so once the date rolled over, the
fixture's trade fell outside "today", the day's realized loss summed to
zero, and the boundary tests asserted against an empty window.

It failed the first time the suite ran after midnight UTC, with no code
change involved. The test was wrong, not the code: it pinned a wall-clock
date against a function that reads the wall clock. `NOW` is now anchored
to the current date, and `_closed_trade` stamps the real current instant
by default, which is the only timestamp guaranteed to be inside the
window the champion computes for itself. Reasoning recorded in the
module, per CLAUDE.md.

Checked the rest of the suite for the same shape: other files hardcode
dates but pass them explicitly as `now=`, so they are deterministic. This
was the only one.

### 15. The same weighting defect in two sibling aggregators

Having found it in `build_postmortem`, the obvious question was whether
anything else aggregates the same column. Three modules do, and two of
them had the identical flat-mean bug:

  * `app/analysis/fill_audit.py` — `mean_cost_pct` averaged the rate over
    fills. `FillRecord` already carried `notional_usd`, so nothing needed
    plumbing; it was simply not used.
  * `app/analysis/trade_analytics.py` — `avg_execution_cost_pct` averaged
    the rate over legs while `total_execution_cost_usd` right beside it
    was a correct dollar sum. The two figures could contradict each
    other on the same page.

The second one is the sharper illustration. Its existing test asserted
`avg_execution_cost_pct == 0.007` on the very same two legs it asserted
`total_execution_cost_usd == $2.20` for, over $300 of notional — which is
0.7333%. The test contained the numbers that disproved its own average
and nobody noticed, because 0.7 looks like the mean of 0.6 and 0.8 and
that is the shape the eye checks. The test was wrong, not the code.

Both now weight by notional, and the invariant is pinned directly: the
reported rate times the costed notional must reproduce the reported
dollar cost. A flat mean cannot satisfy that, so the property cannot
silently regress in any of the three.

### 16. Review round 5 — four more, three of them mine from tonight

The first review to complete since `0248f59` (the previous four aborted
because I kept pushing mid-review). It found four real problems.

**A missing fee was read as a zero fee.** The slippage fix from round 4
used `leg.execution_cost_pct * notional - (leg.fee_usd or 0.0)`, so a leg
with no recorded fee had its ENTIRE execution cost booked as slippage -
the largest possible answer, produced from the absence of a measurement.
Straight violation of CLAUDE.md's "a measurement that cannot be taken is
recorded as unmeasurable, never as zero", in code written earlier the
same night. The comment eight lines above it says a leg with no notional
is "dropped rather than counted as fee-free - that would overstate
slippage"; I identified the exact hazard and then walked into it for the
fee. Slippage now carries its own denominator and only legs with a
recorded fee contribute.

**The post-mortem read every trade, not just filled ones.** A failed or
pending sell carrying a `qty` and an `exit_price` moved the size-weighted
exit price, the return, the fees and the cost rates for a fill that never
happened. Now filtered to `TradeStatus.FILLED` at the query.

**A fee-only leg counted as full cost coverage.** In
`trade_analytics.summarize_costs`, a leg with `fee_usd` but no
`execution_cost_pct` incremented `legs_counted`, so `coverage_pct` and
`cost_data_complete` claimed the leg was measured while its spread,
impact and drift were unknown and absent from the dollar total. The fee
is the one component already known from configuration; the parts worth
measuring are exactly the ones missing. Its fee is still summed, but it
now counts as missing cost data.

**The time-bomb fix was still not deterministic.** Defect 14 replaced a
hardcoded date with a date anchored to import time - which is two clock
reads, the import and the champion's own `_day_bounds()` call, and a run
crossing midnight between them still straddles. A narrower window than
the original, and the same bug. An autouse fixture now pins the
champion's window to the module's `NOW`, so one instant governs both.

Also corrected: the round-4 arithmetic in this file (see above).

### 17. Nothing tested that a failed execution gets recorded

The other half of round 5's test request, and a real gap rather than a
documentation one.

Three paths set `TradeStatus.FAILED` and call `notify_error` - the buy,
the full exit and the partial exit. None had a test. The only appearances
of `TradeStatus.FAILED` anywhere in the suite were fixtures setting it up
so something else could be measured; nothing asserted the write path
produces it.

That gap is worse than a missing filled-path test. A fill that goes
unrecorded is caught within minutes because the position is missing from
the dashboard. A FAILURE that goes unrecorded is invisible by
construction: there is no position to be absent, so the row this code
writes is the only evidence the attempt happened, and without it the
funnel's gap between signals and positions becomes unexplainable - the
precise thing `app/pipeline.py` exists to prevent.

On the exit side it is not just a lost record. A sell that fails silently
leaves a position the operator believes is closed: still open, still
carrying risk, absent from no exposure total, and with no alert.

Four tests in `tests/test_execution_failure_recording.py`, all in paper
mode with a stubbed client returning a failed `SwapResult` - no live
flags, no network. They cover the persisted row and its error string, the
alert, the position staying open at full size, and the cash ledger not
moving.

Verified by neutering the branch: with the record and the alert removed,
the recording test fails. The other three still pass, because an early
return also happens to leave state unmutated - they guard a different
invariant, and that is recorded here rather than claimed as four tests
catching one bug.

### Recorded as inspected, not tested

Round 5 asked for a test proving `_execute_swap` maps a `None` swap-build
response onto a failed `SwapResult`. I wrote one, then found it passed
for the wrong reason: under paper-only the function refuses at the
live-execution guard before it ever builds a swap, so the assertion never
reached the branch it named.

Reaching that branch needs `LIVE_TRADING=true`, which this suite will not
set. So the test now asserts what it actually proves - `_execute_swap`
returns a failed `SwapResult` rather than raising - and its docstring
records that the `swap_data is None` branch is verified by reading the
source, not by execution. A vacuous green assertion would have been worse
than an honest gap.

### Not unified, but now locked: the cost columns disagree on units

While fixing the three aggregators it became clear they do not agree on
what `_pct` means:

| field | unit | 1% is |
|---|---|---|
| `Trade.execution_cost_pct` | fraction | `0.01` |
| `trade_analytics.avg_execution_cost_pct` | fraction | `0.01` |
| `fill_audit.mean_cost_pct` | percent | `1.0` |
| `postmortem.execution_cost_pct` | percent | `1.0` |

Two fields with the same suffix, describing the same cost on the same
trade, differing by 100x.

Nothing is displayed wrong today — all five render sites were checked
individually and each matches its own module's convention. So this is a
trap rather than a defect, and it is the trap that already sprang once:
defect 9 above was exactly this, the post-mortem serving a raw fraction
beside real percents.

Deliberately **not** unified. Converting `trade_analytics` to percent
means editing two currently-correct render sites, and rewriting working
display code for naming tidiness is how a correct number becomes a wrong
one. Instead `tests/test_cost_units.py` pins each convention, asserts the
100x gap explicitly, and lists every render site that would have to move
together — so the mismatch cannot drift silently, and whoever unifies it
later starts from a test that names the whole blast radius.

### Not fixed: foreign keys and CHECK constraints on the audit tables

Round 4 also asked for real foreign keys on `Trade.signal_id`,
`Trade.position_id`, `Position.entry_trade_id` and
`RugCheckResult.signal_id`, plus CHECK constraints or enums on
`Trade.side`, `status` and `mode`. Both are reasonable schema advice and
both are declined for the same two verified reasons.

**SQLite has no `ALTER TABLE ADD CONSTRAINT`** — confirmed, it is a
syntax error. Adding either kind of constraint to an existing table
requires the twelve-step rebuild: create a shadow table, copy every row,
drop the original, rename. That is the opposite of the additive-only rule
in CLAUDE.md, and it would run against the database holding the
collection run's audit trail.

**SQLite does not enforce foreign keys by default** — `PRAGMA
foreign_keys` reads `0`, and this app never sets it. So the declarations
alone would be decorative; making them bite means enabling the pragma,
which starts *rejecting writes at runtime*. That is a live behaviour
change to the persistence layer of a frozen collection run, which is
exactly what the freeze is for.

The underlying concern — an orphaned id silently dropping context from a
post-mortem — is real but currently hypothetical: these ids are only ever
written from within the transaction that created the row they point at,
and nothing deletes from these tables. Recorded here rather than fixed,
as a schema change for whenever the collection run ends.

### Not fixed: the Jupiter build-failure claim

Round 3 also reported that a raising `http.post_json` would escape
`_execute_swap` and skip the failed-trade record. It does not.
`request_json` catches `Exception` on every path and returns `None`, and
`_execute_swap` already maps `None` onto a failed `SwapResult`. Left
alone, with a test pinning the contract so the claim does not need
re-litigating.

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
  passing it off as a measurement, and the flag is now persisted on the
  `Trade` row so a quote-derived fill cannot masquerade as a measured one
  in later P&L. Reading the executed amounts needs the transaction's token
  balance deltas, which this backend does not parse. Live path only —
  unreachable while the flags are off.
- **The VPS is still running bundle `b21a6ab`.** Everything in this
  changelog is on the branch and in the delivered zips, but nothing has
  been deployed — that needs your terminal.
- **`paper.py` feeds `price_change_1h_pct` into a parameter named
  `volatility_1h_pct`.** Flagged in an earlier session and deliberately
  not changed: the slippage model is frozen.
- **Zero trading observations.** Nothing in this session produced
  evidence of an edge, and the bot still has none. Every analysis page
  added here currently reports INSUFFICIENT, which is the correct answer.
