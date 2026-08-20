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

NOTHING HERE PROMOTES ANYTHING

This module produces observations. app/autopilot/promote.py decides. Wiring
promotion into the thing that generates the challengers would let a search
mark its own homework.
"""
