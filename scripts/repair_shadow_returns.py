#!/usr/bin/env python3
"""Null the shadow returns that are arithmetic, not outcomes.

WHY THIS EXISTS

`app/shadow/resolver.py` computed `(exit / entry - 1) * 100` with no bound
and no guard on a zero entry price. A mis-scaled price produces a number
that is arithmetically valid and physically impossible, and the comparison
layer had no reason to doubt it: a NULL return is excluded from every
consumer, but a huge one is not. One such row set the champion's mean
per-opportunity return to 10,205,008% and the promotion gate reported
effect sizes and regime breakdowns off it.

Commit ee6d52a stopped new rows being written that way. It could not fix
the rows already on disk, which is what this does.

WHAT IT CHANGES, AND WHAT IT REFUSES TO

For each ShadowPosition whose recorded return is implausible, or whose
entry price could never have produced a return at all:

  - `return_pct` and `gross_return_pct` are set to NULL
  - `exit_reason` is rewritten to say the outcome is unmeasurable, and to
    QUOTE THE ORIGINAL VALUE so nothing is lost by the repair

NULL is the honest value. Every consumer already reads it as "entered,
outcome unknown", which is exactly true of these rows - the position was
real, the number was not. Deleting the rows instead would erase the
evidence that the bot took those entries; overwriting them with 0.0 would
be inventing a flat outcome, which is the failure CLAUDE.md names.

Nothing else is touched. Entry price, exit price, timestamps, the
excursion envelope and the exit policy fingerprint all stay, so a
repaired row can still be inspected and argued with.

THE BOUND IS DELIBERATELY GENEROUS

MAX_PLAUSIBLE_RETURN_PCT is 10,000% - a 100x. A memecoin really can do
that, so the threshold excludes essentially nothing real while catching a
scale fault by orders of magnitude. A row is repaired only when it is
impossible, never when it is merely surprising. Run the dry run and read
the list before applying: if a genuine outcome appears in it, the bound
is wrong and should be argued about rather than worked around.

Usage:
    # see what would change
    python scripts/repair_shadow_returns.py

    # do it
    python scripts/repair_shadow_returns.py --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import models  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.shadow.resolver import MAX_PLAUSIBLE_RETURN_PCT  # noqa: E402


def is_corrupt(row: models.ShadowPosition) -> str | None:
    """Why this row's return cannot be believed, or None if it can.

    Checks the gross figure when there is one: it is the raw
    `(exit/entry - 1) * 100`, so it is the number the fault actually
    lands in. `return_pct` is the same value minus a cost of well under a
    percent, so either would do - but reading the one closest to the
    computation keeps the reason honest about where the fault is.
    """
    if row.return_pct is None and row.gross_return_pct is None:
        return None                       # already unresolved, nothing to do
    if not row.entry_price or row.entry_price <= 0:
        return f"entry price {row.entry_price!r} could never produce a return"

    measured = row.gross_return_pct
    if measured is None:
        measured = row.return_pct
    if measured is not None and abs(measured) > MAX_PLAUSIBLE_RETURN_PCT:
        return (
            f"recorded {measured:+,.0f}% from entry {row.entry_price:.10g} "
            f"to exit {row.exit_price:.10g}"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true",
        help="actually write. Without it this is a dry run.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        rows = db.query(models.ShadowPosition).all()
        corrupt = [(r, why) for r in rows if (why := is_corrupt(r)) is not None]

        resolved = sum(
            1 for r in rows
            if r.return_pct is not None or r.gross_return_pct is not None
        )
        print(f"  shadow positions      {len(rows)}")
        print(f"  with a recorded return{resolved:>6}")
        print(f"  implausible           {len(corrupt)}")

        if not corrupt:
            print()
            print("Nothing to repair. Every recorded return is inside the bound.")
            return 0

        print()
        by_strategy: dict[str, int] = {}
        for row, _ in corrupt:
            by_strategy[row.strategy_id] = by_strategy.get(row.strategy_id, 0) + 1
        for strategy, count in sorted(by_strategy.items()):
            print(f"    {strategy:<24} {count}")

        print()
        print("  rows (up to 20 shown):")
        for row, why in corrupt[:20]:
            print(f"    #{row.id} {row.symbol:<12} {row.strategy_id:<14} {why}")
        if len(corrupt) > 20:
            print(f"    ... and {len(corrupt) - 20} more")

        # What the surviving record looks like, so the point of the repair
        # is visible rather than asserted.
        survivors = [
            r.return_pct for r in rows
            if r.return_pct is not None
            and not any(r is c for c, _ in corrupt)
        ]
        print()
        if survivors:
            mean = sum(survivors) / len(survivors)
            print(f"  after repair: {len(survivors)} usable returns, "
                  f"mean {mean:+.2f}% per resolved position")
        else:
            print("  after repair: NO usable returns remain - the repair would "
                  "empty the dataset rather than clean it, which is a finding "
                  "in itself. Do not apply without understanding why.")

        if not args.yes:
            print()
            print("Dry run. Re-run with --yes to apply.")
            return 0

        for row, why in corrupt:
            row.return_pct = None
            row.gross_return_pct = None
            row.exit_reason = (
                f"unmeasurable - {why}; a price scale fault, not a trade "
                f"(repaired by scripts/repair_shadow_returns.py)"
            )
        db.commit()
        print()
        print(f"Repaired {len(corrupt)} row(s). Their returns are now NULL, which "
              "every consumer reads as 'entered, outcome unknown'.")
        print("Re-run: python -m scripts.research shadow")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
