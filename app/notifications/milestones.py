"""Announce collection milestones as the closed-trade count crosses them.

The validation gate refuses to judge the strategy below 100 closed trades
(`MIN_CLOSED_TRADES` in app/analysis/validation.py), which makes 100 the
number the whole paper run is waiting on. Until now nothing said when it
arrived: the count lives in a database on the VPS behind an authenticated
dashboard, so noticing meant remembering to go and look.

That is a poor way to learn something you have been waiting two weeks for,
and it is worse than it sounds - the interesting moment is not "100 trades
exist" but "the gate will now answer", and someone who checks a day late
has a day of unexamined trades sitting on top of the sample they meant to
read.

WHY IT ANNOUNCES SEVERAL NUMBERS, NOT JUST 100

25 and 50 are not thresholds for anything; they are progress. A run that
reports nothing for a fortnight is indistinguishable from a run that
stopped collecting, and the operator's own confidence in the machine is
part of what this is for. The larger ones exist because a sample keeps
getting more informative after the gate first speaks.

WHAT IT DOES NOT DO

It reports the size of the record and nothing about its quality. No P&L,
no win rate, no verdict - a milestone message is not the place to make a
claim about the strategy, and a number arriving on someone's phone reads
as a result whether or not it is one. The gate on the performance page is
where the record gets judged.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app import models
from app.state import get_state, set_state

logger = logging.getLogger(__name__)

# Ascending. 100 is the validation gate's minimum; the rest are progress
# markers on the way to it and beyond.
MILESTONES: tuple[int, ...] = (25, 50, 100, 200, 500, 1000)

# The highest milestone already announced. Persisted rather than held in
# memory so a restart - or the 15-minute auto-updater swapping the
# container - does not re-announce a number the operator already saw.
STATE_KEY = "collection_milestone_announced"

# The gate's minimum, restated here so the message can explain what the
# number means. Imported lazily in `_message` to keep this module free of
# an app.analysis dependency at import time.
_GATE_MINIMUM = 100


def count_closed_trades(db: Session) -> int:
    """Closed legs, counted the way the dashboard and the default
    performance report count them: a realized P&L and a close time.

    Pooled across strategy versions on purpose - that is what the report
    does without a `?version=` filter, and a milestone that counted a
    different denominator from the page it points at would be a bug
    disguised as a notification.
    """
    return (
        db.query(models.Trade)
        .filter(models.Trade.pnl_usd.isnot(None), models.Trade.closed_at.isnot(None))
        .count()
    )


def _message(milestone: int, closed: int) -> str:
    if milestone >= _GATE_MINIMUM:
        meaning = (
            f"The validation gate needs {_GATE_MINIMUM} closed trades before it "
            f"will judge the record, so it can now answer. Open /performance "
            f"for the verdict - it may well still be EXPERIMENTAL or FAILING, "
            f"and that is a real answer too."
        )
    else:
        remaining = _GATE_MINIMUM - closed
        meaning = (
            f"Still collecting - {remaining} more before the validation gate "
            f"will judge the record."
        )
    return f"\U0001f4cf {closed} closed trades.\n{meaning}"


async def announce_if_crossed(db: Session) -> int | None:
    """Announce the highest milestone newly reached. Returns it, or None.

    Flushes first. `SessionLocal` is `autoflush=False`, so a sell leg the
    caller has only `db.add()`ed is invisible to the count below and the
    hundredth trade would announce on the hundred-and-first. The callers
    reach here via `_check_halt_conditions`, which flushes too; this does
    not rely on that, because a third exit path could be written that does
    not.

    Never raises. A notification is not worth failing a trade over, and
    the exit path this sits in has already moved real (paper) money by the
    time it is called.
    """
    try:
        db.flush()
        closed = count_closed_trades(db)

        announced = get_state(db, STATE_KEY, None)

        if not isinstance(announced, int):
            # First run against this database. Record what has already
            # been passed WITHOUT announcing it: the record was 33 trades
            # deep when this was written, and a bot that greets its
            # operator with "25 closed trades" about a fortnight of
            # history is announcing the past. Silence here means the next
            # message is a real crossing.
            passed = [m for m in MILESTONES if m <= closed]
            set_state(db, STATE_KEY, max(passed) if passed else 0)
            return None

        # The highest milestone at or below the current count. Comparing
        # against the whole list rather than the previous count means a
        # jump - a batch of partial exits, or a database restored mid-run -
        # announces the milestone it landed past instead of skipping it.
        reached = [m for m in MILESTONES if announced < m <= closed]
        if not reached:
            return None
        milestone = max(reached)

        set_state(db, STATE_KEY, milestone)

        from app.notifications.notifier import notifier

        await notifier.notify_milestone(_message(milestone, closed))
        logger.info("collection milestone announced: %d closed trades", milestone)
        return milestone
    except Exception:
        logger.exception("milestone check failed; continuing")
        return None
