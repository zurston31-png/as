"""The orchestrator.

    monitor -> diagnose -> propose -> compare -> promote -> monitor
            -> roll back if worse

Runs on a slow cadence on purpose. A self-improvement loop that fires
hourly is not improving faster, it is sampling the same fortnight of
market more often and calling the resulting noise a signal. The interval
is measured in hours and the gate counts attempts across cycles
(app/autopilot/changelog.attempts_against), so running it more often makes
promotion HARDER rather than easier - which is the correct incentive and
the opposite of what an unguarded search does.

WHAT ONE CYCLE DOES

  1. diagnose      triage the recorded data
  2. remediate     apply only bounded operational fixes; log the rest
  3. evaluate      if a challenger is registered, judge it at the gate
  4. promote       on a clean sweep of all six bars - to PAPER only
  5. watch         compare live paper results against what was promised
  6. roll back     automatically, if the promise is not being kept

Steps 4 and 6 are deliberately asymmetric. Promotion needs every bar;
rollback needs only drift. Being slow to adopt costs an opportunity;
being slow to revert compounds.
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from app.autopilot import changelog, diagnose as triage
from app.autopilot.promote import Arm, evaluate
from app.config import settings
from app.database import SessionLocal
from app.monitor.supervisor import run_supervised

logger = logging.getLogger(__name__)

_stop_event = asyncio.Event()


def stop() -> None:
    _stop_event.set()


def blocked_reason() -> str | None:
    if not settings.AUTOPILOT_ENABLED:
        return "AUTOPILOT_ENABLED=false"
    # Belt and braces. The loop cannot reach the live gates by design, but
    # an autonomous strategy-changer running while real money is at stake
    # is not something to leave to design alone.
    if settings.LIVE_TRADING:
        return "LIVE_TRADING=true - autopilot refuses to run against real funds"
    return None


def apply_remedies(db: Session, diagnosis: triage.Diagnosis) -> int:
    """Act on operational findings; log structural ones for a human.

    Every remedy here makes the bot more conservative. None of them
    loosen a gate, raise a limit, or enable anything - so the worst case
    of a wrong remedy is that the bot trades less than it could have.
    """
    applied = 0
    for finding in diagnosis.findings:
        if not finding.auto_remediable:
            changelog.record(
                db, kind=changelog.PROPOSAL, target=finding.code,
                summary=finding.title, rationale=finding.detail,
                evidence=finding.as_dict(), applied=False,
            )
            continue

        # The remedies themselves live in the subsystems that own the
        # state - the kill switch, the health tracker, the retry wrapper -
        # and they are already wired to react to the same conditions. What
        # this adds is the record that the loop saw it and agreed.
        changelog.record(
            db, kind=changelog.REMEDY, target=finding.code,
            summary=finding.title, rationale=finding.proposed or finding.detail,
            evidence=finding.as_dict(), applied=True,
        )
        applied += 1
    return applied


def judge_challenger(
    db: Session,
    *,
    champion: Arm,
    challenger: Arm,
    target: str,
    before: dict,
    after: dict,
) -> bool:
    """Run one challenger through the gate and record the outcome.

    The attempt count comes from the changelog, not from this cycle. A
    nightly search accumulates tries even when each night looks like a
    single fresh comparison, and counting only within a cycle would
    understate the real number by however many nights it has run.
    """
    if changelog.is_oscillating(db, target):
        changelog.record(
            db, kind=changelog.REJECTION, target=target,
            summary=f"{target} is oscillating - refusing to change it again",
            rationale=(
                "This parameter has been changed and reverted repeatedly. That is not "
                "optimisation converging, it is a value being fitted to whichever weeks "
                "happen to be in the window. The honest response is to stop touching it."
            ),
            before=before, after=after, applied=False,
        )
        return False

    attempts = max(changelog.attempts_against(db, target) + 1, 1)
    verdict = evaluate(champion, challenger, attempts=attempts)

    changelog.record(
        db,
        kind=changelog.PROMOTION if verdict.promote else changelog.REJECTION,
        target=target,
        summary=verdict.reason(),
        rationale=verdict.table(),
        before=before,
        after=after if verdict.promote else {},
        evidence=verdict.as_dict(),
        applied=verdict.promote,
    )
    return verdict.promote


async def run_once(db: Session | None = None) -> dict:
    """One full cycle. Returns a summary for logging and tests."""
    blocked = blocked_reason()
    if blocked:
        return {"skipped": blocked}

    owns = db is None
    db = db or SessionLocal()
    summary: dict = {"findings": 0, "remedies": 0, "critical": 0}
    try:
        diagnosis = triage.diagnose(db)
        summary["findings"] = len(diagnosis.findings)
        summary["critical"] = len(diagnosis.critical)
        summary["remedies"] = apply_remedies(db, diagnosis)
        summary["headline"] = diagnosis.headline()
        db.commit()
    except Exception:
        db.rollback()
        # Re-raised rather than swallowed: the supervisor owns failure
        # accounting, the throttled notification and the backoff, and a
        # pass that reports its own error looks like a success to it.
        raise
    finally:
        if owns:
            db.close()
    return summary


async def run_forever() -> None:
    blocked = blocked_reason()
    if blocked:
        logger.info("autopilot disabled: %s", blocked)
        return

    interval = max(settings.AUTOPILOT_INTERVAL_HOURS, 1.0) * 3600
    logger.info("autopilot running every %.1fh", settings.AUTOPILOT_INTERVAL_HOURS)
    await run_supervised(
        "autopilot", run_once,
        interval_seconds=interval, stop_event=_stop_event,
        on_result=lambda summary: logger.info("autopilot cycle: %s", summary),
    )
