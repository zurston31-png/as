# Two definitions the bot has never written down

Status: **specification only. No code in this document has been written.**
It exists so that two changes to the frozen surface can be argued about
before they are made, rather than discovered afterwards in a dataset that
quietly stopped meaning what it used to mean.

Both came out of an external audit of `630e44b`. The four defects that
audit found which were *not* on the frozen surface are already fixed
(`12a5477`), as is the trade-unit counting problem (`5f3a1bd`). What is
left is the two questions that cannot be answered by fixing a bug,
because the bug is that nobody ever decided the answer.

---

## Part 1 — What is `t=0`?

### The three clocks that currently disagree

A single alert produces at least four distinct instants, and the code
presently mixes three of them:

| Instant | What it is | Where it comes from |
|---|---|---|
| `t_bar_open` | The signal bar's opening time | `Signal.tv_timestamp`, from Pine's `time` |
| `t_bar_close` | When the signal bar finished | **not recorded anywhere** |
| `t_score` | When the bot fetched candles and scored | `dt.datetime.now()` in `schedule()` |
| `t_exec` | When the paper fill landed | `Trade.created_at` |

The current forward-return row takes its **baseline price** from the
alert (a `t_bar_close` quantity, see below) and starts its **horizon
clock** at `t_score`. A row labelled "5-minute return" therefore measures
an interval of `5 minutes + (t_score − t_bar_close)`, which is not five
minutes and is not a fixed quantity either — it varies with webhook
latency and provider response time.

### What the Pine script actually sends, and why it matters

This is the part that makes the obvious fix wrong.
`pine/memecoin_signal_strategy.pine`:

```
'","price":' + jsonNum(close, PRICE_FMT) +
 ',"time":"' + str.tostring(time) +
...
alert(buildPayload("buy"), alert.freq_once_per_bar_close)
```

Three facts follow, and they point in different directions:

1. The alert fires **on bar close** (`alert.freq_once_per_bar_close`).
2. `price` is `close` — the closing price of the bar that just ended.
3. `time` is Pine's `time`, which is the bar's **opening** time.

So the price in the payload was realized at `t_bar_close`, but the
timestamp in the payload is `t_bar_open`. They are one full chart
interval apart.

**This is why "just use `tv_timestamp` instead of `now()`" is not the
fix.** It would anchor the clock one entire interval *before* the moment
its own baseline price existed. On a 5-minute chart measuring a 5-minute
horizon, that is a 100% error in the measurement window — worse than the
latency error it was meant to replace, and worse in a way that looks
tidier.

### The missing field

`t_bar_close = t_bar_open + chart_interval`, and **the chart interval is
not in the payload.** `app/schemas.py` has no interval field. It cannot
be inferred:

- `SIGNAL_SCORE_TIMEFRAME` (currently `15m`) is the timeframe the *bot*
  fetches for its own scoring. It has nothing to do with what chart the
  operator attached the alert to, and assuming they match would silently
  fabricate the number in exactly the cases where they don't.

So `t=0` is **not currently computable from recorded data**, for any
signal already collected. That is a finding in its own right, and it
constrains everything below: this cannot be backfilled onto the existing
61-position record. It can only be made true going forward.

### Proposed definition

> **`t=0` is the close of the last bar that had finished when the signal
> was generated: `t0 = tv_timestamp + chart_interval`.**
>
> The information set at `t=0` is every bar that had closed at or before
> `t0`, and nothing else.

Everything else is then forced, which is the point of having a
definition:

- **Baseline price** — `signal.price`, the close of the bar ending at
  `t0`. Already correct; it is the only one of the four quantities that
  is.
- **Predictors** — every candle used for scoring must satisfy
  `candle.timestamp <= t0 − timeframe`, which is exactly what
  `CandleSeries.up_to(t0)` computes. The backtester already does this.
  The live path (`app/signals/live_gate.py`) does not, and is not even
  passed a timestamp to do it with.
- **Outcomes** — a horizon of `h` minutes resolves at `t0 + h`.
- **Latency** — `t_score − t0` and `t_exec − t0` become *recorded
  measurements* rather than invisible contamination. They belong in the
  dataset as columns. Execution happening after `t=0` is not look-ahead;
  it is slippage, and slippage is a thing we already model. Only the
  *information* used to decide must respect `t=0`.

### Why the two halves must be treated differently

This is the part most likely to be got wrong, because both halves look
like "the same fix".

**Half A — the forward-return anchor is a MEASUREMENT change.**
Forward returns are shadow data. No trade is taken or skipped because of
them. Re-anchoring changes what a recorded row *means*, but changes no
behaviour, so it must **not** mint a new strategy version — and it also
must not silently overwrite the meaning of rows already collected.

The additive-migration pattern this repo already uses fits exactly:

- add `ForwardReturn.anchor_policy` (`"scheduled_at"` for every existing
  row, `"bar_close"` for new ones)
- add `ForwardReturn.chart_interval_seconds`, nullable — null for every
  row collected before the Pine script sends it
- analysis filters on `anchor_policy`, and refuses to pool the two

Both eras stay interpretable and neither is fabricated. Rows whose
interval is null are `unmeasurable`, never assumed.

**Half B — the closed-candle cutoff is a BEHAVIOUR change.**
Applying `up_to(t0)` in `live_gate.py` changes which candles the scorer
sees, which changes scores, which changes which tokens are entered. That
is a strategy change in the full sense: it must mint a new version hash,
split the dataset deliberately, and run as a **challenger** against the
champion rather than replacing it. `deploy/auto_update.sh` will refuse to
roll it out unattended, which is the correct behaviour and not an
obstacle to work around.

The honest expectation: the current champion's scores are computed partly
from bars that had not closed. If the challenger scores materially
differently, that is a measure of how much repainting was in the champion
all along — which is worth knowing regardless of which one performs
better.

### Prerequisite work, in order

1. **Pine**: add `"interval":"' + timeframe.period + '"` to the payload.
   Additive, safe, changes no trading logic. Requires the operator to
   delete and recreate every alert — an alert keeps the settings it was
   made with, as the script's own footer already warns.
2. **Schema**: accept and store the interval; nullable, so old alerts
   still validate.
3. **Half A**: the two additive columns and the anchor change, behind no
   flag, since it alters no behaviour.
4. **Half B**: `LIVE_CLOSED_CANDLES_ONLY`, defaulting `false`, added to
   `OPT_IN_BEHAVIORAL_SETTINGS` so it is free while dormant and splits
   history the moment it is switched on.

Steps 1 and 2 are worth doing early regardless of when 3 and 4 happen:
until the interval is recorded, every signal collected is one that can
never have its `t=0` reconstructed.

---

## Part 2 — What is the unit of risk?

### The inconsistency, stated plainly

After `5f3a1bd`:

```
analytics    -> round trips  (one position = one observation)
risk streak  -> exit legs    (one fill     = one observation)
```

`RiskManager.evaluate_consecutive_losses` queries `Trade.pnl_usd` ordered
by `closed_at` and halts on `MAX_CONSECUTIVE_LOSSES` (4) consecutive
negative rows. A partial exit is one of those rows.

### Which direction the error runs

This matters more than the inconsistency itself.

A partial profit-take is, by construction, **a winning row**. It only
fires when the position is up. So a losing position that first banked a
partial contributes `[+small, −large]` to the sequence — and the `+small`
**resets the streak**.

Three consecutive losing positions, the middle one having taken a
partial, read as a streak of **1**, not 3. There is a test pinning this
in `tests/test_round_trips.py`.

So the current behaviour makes the halt **harder** to trigger than
`MAX_CONSECUTIVE_LOSSES = 4` implies. The kill switch is looser than its
own configuration says. That is the unsafe direction for the error to
run, and it is the reason this is worth raising rather than filing as a
tidiness issue.

It is also probably load-bearing right now: the record shows a longest
losing run of 4 at leg level, and the halt did fire. Whether it *should*
have fired earlier depends on how many of those exits were partials —
which is one of the numbers `scripts/performance_report.py` now prints.

### Proposed definition

> **A consecutive-loss streak counts POSITIONS, not fills. A position is
> a loss when the sum of all its exit legs is negative.**

The rule exists to detect "the strategy has stopped working". The
strategy places bets, not fills. A bet that lost money is one loss no
matter how many transactions closed it.

### Why this cannot simply be changed

It changes when the bot halts. That is live risk behaviour, and
`MAX_CONSECUTIVE_LOSSES` is already in `BEHAVIORAL_SETTINGS`. Under the
standing rule — *do not silently activate risk behaviour changes* — the
implementation is:

- `RISK_POSITION_LEVEL_LOSS_STREAK`, default `false`
- added to `OPT_IN_BEHAVIORAL_SETTINGS`, so it hashes to nothing while
  dormant and splits the dataset the moment it is enabled
- the position-level streak computed and **logged** while disabled, so
  the divergence between the two counts can be measured on real data
  before anyone decides to flip it

That last point is the useful part. Rather than arguing about which unit
is right in the abstract, run both for a while and see how often they
disagree. If they never diverge, the question is moot. If they diverge
often, the size of the gap is the argument.

---

## What is NOT proposed here

- Tuning any threshold, weight or scoring formula. Nothing in this
  document is a performance change, and none of it should be evaluated
  on whether it improves a number.
- Backfilling `t=0` onto existing rows. It is not recoverable; the
  interval was never recorded. Pretending otherwise would be fabricating
  data.
- Changing the champion. Half B is a challenger. Part 2 is a dormant
  flag.
- Reading anything as evidence. At 61 exit legs the record establishes
  nothing either way, and none of this changes that. It only makes the
  next 100 observations mean what they claim to mean.
