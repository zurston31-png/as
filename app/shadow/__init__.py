"""Running challengers beside the champion without letting them touch it.

The purpose is PAIRED observations. Comparing two strategies on different
opportunities compares the opportunities as much as the strategies: if the
challenger happened to run through a calmer fortnight, its better numbers
say nothing. Evaluating both on the SAME opportunity removes that, and
what remains is the disagreement - which is the only part worth measuring.

ISOLATION IS STRUCTURAL, NOT POLITE

A challenger writes to `shadow_decisions` and `shadow_positions`. It has no
code path to `positions`, `trades`, the cash ledger, the risk manager's
cooldowns, or any execution client. That is enforced three ways:

  * separate tables, so no existing query has to remember to filter
    hypothetical rows out - and the first one that forgot would silently
    let a challenger move real paper risk
  * the recorder takes no execution client and imports none
  * a test greps this package for the forbidden names

The weaker design - one table with an `is_shadow` flag - fails the moment
anybody writes a query without the flag, and that failure is invisible
until the numbers are already wrong.

WHAT A CHALLENGER IS

A set of parameter overrides, not new code: scoring weights, an entry
threshold, a stop and target. app/signals/scoring.py already takes an
injectable weight map, which is what makes a challenger a configuration
rather than a fork. A challenger that needed new logic would need a new
code path, and a new code path cannot be compared against the champion on
equal terms because it has not been through the same gates.

AN OBSERVATION IS NOT EVIDENCE UNTIL IT IS RESOLVED

Recording what a strategy WOULD have done is only half a measurement. Until
app/shadow/resolver.py runs, every hypothetical entry stays open, every
`return_pct` is NULL, and the paired comparison has two arms of nothing to
compare. The resolver walks post-entry candles through one shared exit rule
(app/shadow/exit_policy.py), so what differs between arms is entry scoring
and nothing else.

It also records fixed horizon returns - 15m, 1h, 4h, 24h - whether or not
the position is still open. Those survive a change to the exit rule, which
is what makes it possible to tell a bad entry apart from a good entry that
a stop cut short.

TWO EXPECTANCIES, KEPT APART

Per OPPORTUNITY (a decline counts as 0%) and per ENTERED TRADE. A selective
strategy with a superb per-trade number can still lose on per-opportunity
terms by trading too rarely, and a busy one can be a worse trader that makes
it up on volume. Only the per-opportunity series is paired - both arms share
the same denominator - so only that one is handed to the promotion gate.

NOTHING HERE PROMOTES ANYTHING

This module produces observations. app/autopilot/promote.py decides. Wiring
promotion into the thing that generates the challengers would let a search
mark its own homework.
"""
