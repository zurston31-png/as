#!/usr/bin/env python3
"""Reset the paper account to a new starting balance, coherently.

Changing PORTFOLIO_STARTING_BALANCE_USD in `.env` is one line, and on its
own it produces a broken bot rather than a smaller account. Three things
have to move together:

  1. THE CASH LEDGER. It is seeded once, at database creation, and never
     re-read from the setting. Edit the setting alone and the bot keeps
     the old simulated cash while every report claims the new number.

  2. THE RECONCILIATION BASELINE. The accounting check reconstructs the
     balance as `baseline + sells - buys`. It has to be told the ledger
     was reset, and from when, or it reports a discrepancy the size of
     the change - and because the kill switch fails closed on a bad
     ledger, the bot would quietly stop opening positions.

  3. THE DATASET. Position size scales with the balance, so fills and
     P&L are not comparable across the change. The strategy version hash
     now covers the balance once it moves off the collection value, so
     the split is recorded rather than silent - this script prints the
     new label so you can see it happen.

WHY IT REFUSES TO RUN WITH POSITIONS OPEN

A position bought before the reset and sold after it would have its
proceeds counted against a baseline that never funded its purchase. The
books would be permanently off by that trade's cost basis, and the kill
switch would halt the bot for it. Close or wait out the open positions
first; there is no correct way to split a position across the boundary.

Usage:
    # 1. put the new value in .env first, so the process running this
    #    agrees with the process that will trade
    PORTFOLIO_STARTING_BALANCE_USD=250

    # 2. see what would happen
    python scripts/set_starting_balance.py

    # 3. do it
    python scripts/set_starting_balance.py --yes

The old trades are NOT deleted. They stay in the database tagged with the
strategy version that produced them, so `/performance?version=<label>`
still reports the old run in full.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import models  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.services import portfolio  # noqa: E402
from app.strategy.version import current_label  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true",
        help="actually write. Without it this is a dry run.",
    )
    args = parser.parse_args()

    target = settings.PORTFOLIO_STARTING_BALANCE_USD
    db = SessionLocal()
    try:
        open_positions = (
            db.query(models.Position)
            .filter(models.Position.status == models.PositionStatus.OPEN.value)
            .count()
        )
        cash = portfolio.get_cash_balance_usd(db)
        baseline = portfolio.get_ledger_baseline(db)
        closed = (
            db.query(models.Trade)
            .filter(models.Trade.pnl_usd.isnot(None), models.Trade.closed_at.isnot(None))
            .count()
        )

        print(f"configured starting balance : ${target:,.2f}")
        print(f"current ledger baseline     : ${baseline.balance_usd:,.2f}"
              f"{'' if baseline.since is None else f' (since {baseline.since.isoformat()})'}")
        print(f"current cash ledger         : ${cash:,.2f}")
        print(f"closed trades on record     : {closed}")
        print(f"open positions              : {open_positions}")
        print(f"strategy version now        : {current_label()}")
        print()

        if open_positions:
            print(
                f"REFUSING: {open_positions} position(s) are open. A position bought\n"
                f"before the reset and sold after it would leave the books\n"
                f"permanently off by its cost basis, and the kill switch would halt\n"
                f"the bot for it. Wait for them to close, then run this again."
            )
            return 1

        if baseline.balance_usd == target and baseline.since is not None:
            print("Nothing to do: the ledger is already based at this balance.")
            return 0

        if not args.yes:
            print("DRY RUN. This would:")
            print(f"  - set the cash ledger to ${target:,.2f}")
            print(f"  - record a new reconciliation baseline from now")
            print(f"  - leave all {closed} existing closed trades in place, tagged with")
            print(f"    their own strategy version, still readable at /performance")
            print()
            print("Re-run with --yes to apply.")
            return 0

        now = dt.datetime.now(dt.timezone.utc)
        portfolio.set_ledger_baseline(db, target, since=now)
        from app.state import set_state

        set_state(db, portfolio.CASH_KEY, float(target))
        db.commit()

        print(f"Ledger reset to ${target:,.2f}, baseline recorded from {now.isoformat()}.")
        print(f"Collection restarts under strategy version {current_label()}.")
        print()
        print("Restart the bot so nothing is holding the old balance in memory:")
        print("  docker compose restart bot")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
