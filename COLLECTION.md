# Running the collection

The bot's measuring equipment is finished. What it does not have is
observations. This document is how to get them.

Everything below is **paper only**. `LIVE_TRADING` stays `false`, no keys
are involved, and `preflight` refuses to report ready if that ever changes.

---

## Before you start

The bot is built to degrade quietly — a missing price must never take down
the position monitor guarding a position. The cost of that choice is that a
completely non-functional deployment looks exactly like a quiet market: the
app boots, workers start, the scanner logs a cycle every interval, and not
one usable observation is recorded.

So run this **on the machine that will do the collecting**, not on your
laptop:

```bash
python scripts/research.py preflight
```

It exits non-zero if anything would waste the run. It probes each upstream
with a real request rather than reading the recorded health table, which is
empty on a fresh box — precisely when the answer matters most.

Common blockers it catches:

| Result | What it means |
|---|---|
| `price feed` FAIL | API unreachable, blocked, or rate-limited. Not a quiet market — the probe mint certainly has a price. |
| `candles` FAIL | Nothing gets scored, nothing resolves, every candidate is rejected for missing history. |
| `security screening` FAIL | The gate fails **closed**, so this produces *zero* trades, silently, for as long as it lasts. |
| `collection workers` FAIL | A loop is switched off. Silently incomplete dataset rather than a visible failure. |
| `challengers` FAIL | `SHADOW_CHALLENGERS` JSON is malformed. Entries are skipped by design, so the symptom is an arm that never runs. |
| `backups` WARN | Snapshots are on a disk that a redeploy replaces. Every observation is one deploy from gone. |

A hosted sandbox or corporate network will usually fail the three upstream
probes. That is a network policy, not a bug in the bot — collect somewhere
the market-data APIs are reachable.

---

## Starting it

```bash
cp .env.example .env      # then set DASHBOARD_PASSWORD and WEBHOOK_SECRET
python scripts/init_db.py
python scripts/research.py preflight     # must exit 0
docker compose up -d                      # or the systemd unit in deploy/
```

The dashboard refuses to serve on the shipped default password. That is
deliberate: compose publishes a port and the halt/resume controls live
behind that login.

Then leave it alone. The next useful moment is days away.

---

## What is being measured

Three strategies score the **same** opportunities and their decisions are
recorded separately:

| Arm | Entry threshold | Everything else |
|---|---|---|
| `champion` | 65 | — |
| `strict-70` | 70 | identical |
| `loose-60` | 60 | identical |

One variable, so the result is attributable. Same market, same exit rule,
same fill assumptions, different entry threshold. If the answer comes out
in favour of `strict-70`, it is *about the threshold* — not about twenty
parameters that all moved at once.

Two expectancies are reported and they are not interchangeable:

- **Per opportunity** — a decline counts as 0%. Both arms share a
  denominator, so this is the paired contrast, and it is the only one the
  promotion gate reads.
- **Per entered trade** — only the trades that arm actually took.
  Informative, but two self-selected samples, so it never promotes anything.

A selective strategy with a superb per-trade number can still lose on
per-opportunity terms by trading too rarely. Both facts are true at once.

---

## The daily check

```bash
python scripts/research.py collection
```

Exits non-zero on a failure, so it works as a cron. It catches the things
that quietly ruin a run: an arm recording nothing, positions piling up
unresolved, MFE/MAE missing or incoherent, regime data absent, duplicates,
observations spanning two strategy versions.

`/dataset` in the browser is the same picture plus progress toward the
milestone. It deliberately shows **no return figures** — coverage first,
conclusions after.

---

## When to look at results

Not before:

- **500+ paired opportunities per challenger**
- **400+ resolved outcomes**
- **contrast on at least one regime axis** — two different values on the
  *same* axis, not three different full labels

Until then a 70% win rate or a +8% expectancy means nothing, and
`research.py shadow` will say so rather than print a number that invites
belief.

When the sample is there:

```bash
python scripts/research.py shadow          # paired comparison + promotion gate
python scripts/research.py calibration     # does a higher score predict better?
python scripts/research.py counterfactual  # are the filters rejecting the good ones?
python scripts/research.py degradation     # has recent behaviour drifted?
python scripts/research.py integrity       # what must not be counted
```

The promotion gate applies six bars — sample, significance, effect size,
out-of-sample, regime consistency, drawdown — with a multiple-comparison
correction across the challengers. It returns PASS, FAIL, or
INSUFFICIENT_DATA, and the third is not a failure.

---

## The freeze

While collecting, do not change:

scoring formulas · thresholds · weights · exit policy · fees · slippage
model · regime classifier · liquidity classifier · horizon definitions ·
risk assumptions

Any of those changes what an observation *means*. The strategy version hash
covers all of them — including the scoring weights and regime boundaries,
which live in code — so an edit mints a new version and `collection` will
warn that the data now spans two. Old observations stay valid evidence
about the old version; they just must not be averaged with the new ones.

If you do need to change something, treat it as a **new challenger**, not
an edit to the champion.

---

## What never happens automatically

- No live trading. `LIVE_TRADING` and `LIVE_EXECUTION_ACKNOWLEDGED` stay
  false, and there is no wallet or key handling anywhere in the codebase.
- No safety filter is ever weakened by a search. The counterfactual reports
  what the security, data-quality and risk gates cost and is structurally
  incapable of flagging them as tuning candidates. A rug that would have
  paid for eleven minutes is not evidence the rug filter is too strict.
- No threshold is changed by an analysis. Every one of these commands is
  read-only; a finding is a reason for a person to look.
