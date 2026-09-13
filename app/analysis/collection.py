"""Is the collection run actually producing usable observations?

A paper run that looks healthy from the logs can still be recording
garbage: a challenger that silently failed to load, positions that open
and never resolve, a regime column that is NULL on every row. None of
those raise. They surface weeks later as a comparison that cannot be run,
by which point the time is spent.

So this answers the boring operational questions ON THE RECORDED DATA,
mechanically, one command at a time:

    are decisions being written for every configured strategy?
    do the champion and each challenger actually pair?
    do hypothetical positions resolve, or do they pile up open?
    do MFE and MAE populate, and are they internally coherent?
    are regime and liquidity present, and do they VARY?
    are horizon returns landing - measured or explicitly unmeasurable?
    are there duplicates, or observations from mixed strategy versions?

THREE STATES, NEVER TWO

PASS, FAIL, and INSUFFICIENT_DATA. An empty dataset is not a passing
dataset, and a check that has nothing to look at says so rather than
returning green - a green board on zero rows is the single most expensive
thing this file could do.

IT NEVER FIXES ANYTHING

Read-only, by construction. A collection run that quietly repaired its own
data would be laundering the defect it was built to expose.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.shadow.challengers import CHAMPION_ID, enabled

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
INSUFFICIENT = "INSUFFICIENT_DATA"

# The milestone this run is aiming at: paired opportunities per challenger.
TARGET_PAIRS = 500

# Below this many resolved positions, per-arm expectancy is anecdote and
# the checks that read outcomes say so instead of grading them.
MIN_RESOLVED_TO_JUDGE = 30

# A resolved position's net return must sit inside the path the candles
# recorded. The tolerance absorbs the round-trip cost, which is charged on
# top of the gross move and so can push the net figure just below MAE.
PATH_TOLERANCE_PCT = 5.0


@dataclass
class Check:
    name: str
    status: str
    detail: str
    counts: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "counts": dict(self.counts)}


@dataclass
class CollectionReport:
    checks: list[Check] = field(default_factory=list)
    paired: dict[str, int] = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def blocked(self) -> list[Check]:
        return [c for c in self.checks if c.status == INSUFFICIENT]

    @property
    def progress_pct(self) -> float:
        """How far the slowest arm is toward the paired-sample milestone.

        The SLOWEST, deliberately. A comparison is limited by whichever
        challenger has the least data, so averaging the arms would report
        progress the experiment does not have.
        """
        if not self.paired:
            return 0.0
        return min(min(self.paired.values()) / TARGET_PAIRS * 100, 100.0)

    def verdict(self) -> str:
        if self.failures:
            return (
                f"{len(self.failures)} check(s) FAILED. The data being collected right now is "
                "not fully usable - fix the pipeline before spending more days on it."
            )
        if self.blocked:
            return (
                f"Nothing has failed, but {len(self.blocked)} check(s) have too little data to "
                "grade. That is the expected state early in a run."
            )
        return (
            f"All checks pass. Collection is healthy; the remaining requirement is time "
            f"({self.progress_pct:.0f}% of the way to {TARGET_PAIRS} paired opportunities on "
            "the slowest arm)."
        )

    def as_dict(self) -> dict:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "paired": dict(self.paired),
            "target_pairs": TARGET_PAIRS,
            "progress_pct": round(self.progress_pct, 1),
            "failures": len(self.failures),
            "verdict": self.verdict(),
        }


def _expected_strategies() -> list[str]:
    return [CHAMPION_ID] + [c.strategy_id for c in enabled()]


def _check_decisions(db: Session) -> Check:
    """Every configured strategy is writing rows.

    A challenger whose JSON failed to parse is logged and skipped, which is
    the right behaviour - but weeks later the only visible symptom is an
    arm with no data, and by then the run is wasted.
    """
    counts = dict(
        db.query(models.ShadowDecision.strategy_id, func.count())
        .group_by(models.ShadowDecision.strategy_id)
        .all()
    )
    expected = _expected_strategies()
    if not counts:
        return Check("decisions recorded", INSUFFICIENT,
                     "no shadow decisions yet - the bot has not evaluated an opportunity",
                     {name: 0 for name in expected})

    missing = [name for name in expected if not counts.get(name)]
    if missing:
        return Check("decisions recorded", FAIL,
                     f"configured but recording nothing: {', '.join(missing)}. Check that "
                     "SHADOW_CHALLENGERS parses - a malformed entry is skipped with a log line",
                     counts)
    return Check("decisions recorded", PASS,
                 f"{sum(counts.values())} decisions across {len(counts)} strategies", counts)


def _check_pairing(db: Session, report: CollectionReport) -> Check:
    """The champion and each challenger see the SAME opportunities.

    Pairing is what makes the comparison controlled. If an arm is
    systematically absent from opportunities the champion saw, the two are
    being measured on different flow and no amount of sample size fixes it.
    """
    rows = db.query(
        models.ShadowDecision.opportunity_id, models.ShadowDecision.strategy_id
    ).all()
    if not rows:
        return Check("champion/challenger pairing", INSUFFICIENT,
                     "no opportunities recorded yet", {})

    by_opportunity: dict[str, set[str]] = {}
    for oid, strategy_id in rows:
        by_opportunity.setdefault(oid, set()).add(strategy_id)

    champion_opportunities = sum(
        1 for arms in by_opportunity.values() if CHAMPION_ID in arms
    )
    unpaired = 0
    for name in _expected_strategies():
        if name == CHAMPION_ID:
            continue
        report.paired[name] = sum(
            1 for arms in by_opportunity.values()
            if CHAMPION_ID in arms and name in arms
        )
    for arms in by_opportunity.values():
        if CHAMPION_ID in arms and len(arms) < len(_expected_strategies()):
            unpaired += 1

    counts = {"opportunities": len(by_opportunity),
              "champion": champion_opportunities, "partially_paired": unpaired,
              **report.paired}

    if champion_opportunities and not any(report.paired.values()):
        return Check("champion/challenger pairing", FAIL,
                     "the champion is recording opportunities but no challenger shares any of "
                     "them - the arms are being measured on different flow", counts)
    if unpaired > champion_opportunities * 0.05:
        return Check("champion/challenger pairing", FAIL,
                     f"{unpaired} of {champion_opportunities} champion opportunities are missing "
                     "at least one challenger; a systematically absent arm is not comparable",
                     counts)
    if champion_opportunities < 20:
        return Check("champion/challenger pairing", INSUFFICIENT,
                     f"only {champion_opportunities} opportunities so far - too few to tell a "
                     "pairing fault from a quiet start", counts)
    return Check("champion/challenger pairing", PASS,
                 f"{champion_opportunities} opportunities, every arm present on "
                 f"{champion_opportunities - unpaired}", counts)


def _check_resolution(db: Session) -> Check:
    """Positions close out instead of piling up open forever.

    The failure this catches is the one the resolver was built to fix, so
    it is worth checking that the fix is still working: a position older
    than the maximum hold plus the give-up window has no legitimate reason
    to still be open.
    """
    total = db.query(models.ShadowPosition).count()
    if not total:
        return Check("positions resolve", INSUFFICIENT,
                     "no hypothetical positions opened yet", {"positions": 0})

    resolved = db.query(models.ShadowPosition).filter(
        models.ShadowPosition.return_pct.isnot(None)).count()
    unmeasurable = db.query(models.ShadowPosition).filter(
        models.ShadowPosition.closed_at.isnot(None),
        models.ShadowPosition.return_pct.is_(None)).count()
    still_open = total - resolved - unmeasurable

    deadline = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        hours=settings.MAX_POSITION_AGE_HOURS + settings.SHADOW_UNMEASURABLE_AFTER_HOURS
    )
    stalled = db.query(models.ShadowPosition).filter(
        models.ShadowPosition.closed_at.is_(None),
        models.ShadowPosition.opened_at < deadline,
    ).count()

    counts = {"positions": total, "resolved": resolved, "open": still_open,
              "unmeasurable": unmeasurable, "stalled": stalled}

    if stalled:
        return Check("positions resolve", FAIL,
                     f"{stalled} position(s) are past the maximum hold plus the give-up window "
                     "and still open - the resolver is not running, or is erroring", counts)
    if resolved < MIN_RESOLVED_TO_JUDGE:
        return Check("positions resolve", INSUFFICIENT,
                     f"{resolved} resolved of {total} - below the {MIN_RESOLVED_TO_JUDGE} needed "
                     "before resolution rate says anything", counts)
    return Check("positions resolve", PASS,
                 f"{resolved} resolved, {still_open} open, {unmeasurable} unmeasurable", counts)


def _check_envelope(db: Session) -> Check:
    """MFE and MAE are present on resolved positions, and are coherent.

    Coherence matters as much as presence. MFE below MAE is impossible, and
    a net return outside the recorded path means the exit price came from
    somewhere the candles never went - both are corruption rather than a
    surprising market.
    """
    resolved = db.query(models.ShadowPosition).filter(
        models.ShadowPosition.return_pct.isnot(None)).all()
    if not resolved:
        return Check("MFE / MAE populate", INSUFFICIENT,
                     "no resolved positions to inspect", {"resolved": 0})

    missing = [p.id for p in resolved
               if p.max_favorable_pct is None or p.max_adverse_pct is None]
    inverted = [p.id for p in resolved
                if p.max_favorable_pct is not None and p.max_adverse_pct is not None
                and p.max_favorable_pct < p.max_adverse_pct]
    off_path = [
        p.id for p in resolved
        if p.max_favorable_pct is not None and p.max_adverse_pct is not None
        and p.gross_return_pct is not None
        and not (p.max_adverse_pct - PATH_TOLERANCE_PCT
                 <= p.gross_return_pct
                 <= p.max_favorable_pct + PATH_TOLERANCE_PCT)
    ]
    counts = {"resolved": len(resolved), "missing": len(missing),
              "inverted": len(inverted), "off_path": len(off_path)}

    if missing:
        return Check("MFE / MAE populate", FAIL,
                     f"{len(missing)} resolved position(s) have no envelope - drawdown analysis "
                     f"would silently skip them (ids {missing[:5]})", counts)
    if inverted or off_path:
        return Check("MFE / MAE populate", FAIL,
                     f"{len(inverted)} inverted and {len(off_path)} with a return outside the "
                     "recorded path - these cannot both be true, so the data is corrupt", counts)
    return Check("MFE / MAE populate", PASS,
                 f"all {len(resolved)} resolved positions carry a coherent envelope", counts)


def _check_context(db: Session) -> Check:
    """Regime and liquidity are recorded, and they VARY.

    Presence alone is not enough. The promotion gate's consistency bar
    groups by regime, and a run that only ever saw one condition cannot
    satisfy it however many observations it collects - which is a fact
    worth knowing on day three rather than in week six.
    """
    total = db.query(models.ShadowDecision).count()
    if not total:
        return Check("regime / liquidity context", INSUFFICIENT,
                     "no decisions recorded yet", {})

    with_regime = db.query(models.ShadowDecision).filter(
        models.ShadowDecision.market_regime.isnot(None)).count()
    regimes = {r[0] for r in db.query(models.ShadowDecision.market_regime).distinct()
               if r[0]}
    liquidity = {r[0] for r in db.query(models.ShadowDecision.liquidity_regime).distinct()
                 if r[0]}

    trend_axes, volatility_axes = set(), set()
    for label in regimes:
        parts = label.split("/")
        if parts:
            trend_axes.add(parts[0])
        if len(parts) > 1:
            volatility_axes.add(parts[1])

    counts = {"decisions": total, "with_regime": with_regime,
              "distinct_regimes": len(regimes), "trend_axes": len(trend_axes),
              "volatility_axes": len(volatility_axes),
              "liquidity_bands": len(liquidity)}

    if with_regime < total * 0.9:
        return Check("regime / liquidity context", FAIL,
                     f"only {with_regime} of {total} decisions carry a regime - the consistency "
                     "bar has nothing to group the rest on", counts)
    if total < 50:
        return Check("regime / liquidity context", INSUFFICIENT,
                     f"{total} decisions is too few to judge regime spread", counts)
    thin = [name for name, values in
            (("trend", trend_axes), ("volatility", volatility_axes), ("liquidity", liquidity))
            if len(values) < 2]
    if thin:
        return Check("regime / liquidity context", WARN,
                     f"only one value seen so far on: {', '.join(thin)}. Contrast on at least one "
                     "axis is required before the consistency bar can be satisfied", counts)
    return Check("regime / liquidity context", PASS,
                 f"{len(regimes)} distinct regimes, {len(liquidity)} liquidity bands", counts)


def _check_horizons(db: Session) -> Check:
    """Horizon returns land - as a number, or as an explicit unmeasurable.

    Both outcomes are fine. What is not fine is a missing row, because
    absence cannot be distinguished from "never looked".
    """
    total = db.query(models.ShadowHorizonReturn).count()
    if not total:
        return Check("horizon returns populate", INSUFFICIENT,
                     "no horizon returns recorded yet", {"rows": 0})

    measured = db.query(models.ShadowHorizonReturn).filter(
        models.ShadowHorizonReturn.return_pct.isnot(None)).count()
    unmeasurable = total - measured
    zero_priced = db.query(models.ShadowHorizonReturn).filter(
        models.ShadowHorizonReturn.price_at_horizon == 0).count()

    counts = {"rows": total, "measured": measured,
              "unmeasurable": unmeasurable, "zero_priced": zero_priced}

    if zero_priced:
        return Check("horizon returns populate", FAIL,
                     f"{zero_priced} row(s) recorded a price of zero - a dead feed written as a "
                     "real quote", counts)
    if measured < MIN_RESOLVED_TO_JUDGE:
        return Check("horizon returns populate", INSUFFICIENT,
                     f"{measured} measured of {total} - too few to judge coverage", counts)
    if unmeasurable > measured:
        return Check("horizon returns populate", WARN,
                     f"{unmeasurable} unmeasurable against {measured} measured - the candle feed "
                     "is missing more often than not, which biases the set toward tokens that "
                     "kept trading", counts)
    return Check("horizon returns populate", PASS,
                 f"{measured} measured, {unmeasurable} explicitly unmeasurable", counts)


def _check_duplicates(db: Session) -> Check:
    """No observation is counted twice.

    Unique constraints cover both tables, so this should be structurally
    impossible - which is exactly why it is worth asserting on the real
    data. A database created before a constraint existed would not have it.
    """
    decision_dupes = db.query(
        models.ShadowDecision.opportunity_id, models.ShadowDecision.strategy_id, func.count()
    ).group_by(
        models.ShadowDecision.opportunity_id, models.ShadowDecision.strategy_id
    ).having(func.count() > 1).all()

    horizon_dupes = db.query(
        models.ShadowHorizonReturn.position_id,
        models.ShadowHorizonReturn.horizon_minutes, func.count()
    ).group_by(
        models.ShadowHorizonReturn.position_id, models.ShadowHorizonReturn.horizon_minutes
    ).having(func.count() > 1).all()

    position_dupes = db.query(
        models.ShadowPosition.opportunity_id, models.ShadowPosition.strategy_id, func.count()
    ).group_by(
        models.ShadowPosition.opportunity_id, models.ShadowPosition.strategy_id
    ).having(func.count() > 1).all()

    counts = {"duplicate_decisions": len(decision_dupes),
              "duplicate_horizons": len(horizon_dupes),
              "duplicate_positions": len(position_dupes)}
    if any(counts.values()):
        return Check("no duplicate observations", FAIL,
                     "duplicates found - every average over these rows is weighted wrong and "
                     "nothing downstream will notice", counts)
    return Check("no duplicate observations", PASS, "no duplicates in any shadow table", counts)


def _check_version_mixing(db: Session) -> Check:
    """All observations come from one strategy version.

    Pooling across versions describes a strategy that never existed. The
    check is a WARN rather than a FAIL because the older rows are still
    valid evidence about the older version - they simply must not be
    averaged with the new ones.
    """
    versions = [
        (row[0], row[1]) for row in
        db.query(models.ShadowDecision.strategy_version, func.count())
        .group_by(models.ShadowDecision.strategy_version).all()
    ]
    if not versions:
        return Check("single strategy version", INSUFFICIENT,
                     "no decisions recorded yet", {})
    counts = {label or "unversioned": n for label, n in versions}
    if len(versions) > 1:
        return Check("single strategy version", WARN,
                     f"observations span {len(versions)} strategy versions. Do not pool them - "
                     "filter to one before judging any challenger", counts)
    return Check("single strategy version", PASS,
                 f"all observations from {versions[0][0]}", counts)


def check_collection(db: Session) -> CollectionReport:
    """Run every collection-health check. Read-only."""
    report = CollectionReport()
    report.checks = [
        _check_decisions(db),
        _check_pairing(db, report),
        _check_resolution(db),
        _check_envelope(db),
        _check_context(db),
        _check_horizons(db),
        _check_duplicates(db),
        _check_version_mixing(db),
    ]
    return report
