"""The self-improvement loop, and the reason most of it is a brake.

    monitor -> diagnose -> propose -> test -> replay -> compare -> promote
            -> monitor -> roll back if worse

The loop is easy. The hard part, and nearly all of the code here, is
stopping it from confidently making the strategy worse.

WHY THE GATE IS THE PRODUCT

An automated search that keeps generating variations until one beats the
champion is an overfitting machine with a reporting layer. Try two hundred
parameter combinations against a finite history and roughly ten will clear
p < 0.05 on noise alone; the loop will find them, promote them, and attach
a table of statistics that looks exactly like evidence. The dataset does
not have to be small for this to happen - it happens *because* the search
is automated, and it gets worse the harder the loop tries.

So app/autopilot/promote.py corrects for how many challengers were
evaluated, requires the edge to survive out-of-sample, requires it to hold
across market conditions rather than in one lucky regime, and requires the
effect to be large enough to matter after fees. A challenger that fails any
of those is recorded and discarded, not retried with a different seed.

WHAT THIS LOOP MAY AND MAY NOT CHANGE ON ITS OWN

    may     numeric strategy parameters - thresholds, weights, stop and
            target distances, sizing fractions. Bounded, reversible, and
            fully described by a row in the changelog.

    may     bounded operational remedies - back off a rate-limited
            provider, quarantine a data source that keeps failing, halt
            entries when the kill switch trips. All of these only ever
            make the bot MORE conservative.

    may not modify its own source. A loop that edits the code also edits
            the tests that would catch the edit, and a fault in the fixer
            lands in the trading path. Code-level findings are diagnosed,
            explained and logged for a human; app/autopilot/diagnose.py
            produces the report, not a patch.

    may not enable live trading, under any circumstance, by any path.
            Every promotion is to PAPER. The live gates are not reachable
            from this package.

ROLLBACK IS AUTOMATIC, PROMOTION IS EARNED

The asymmetry is deliberate. Promoting requires clearing every bar;
reverting requires only that the live paper result has drifted below what
the challenger promised. Being slow to adopt costs an opportunity. Being
slow to revert compounds.
"""
