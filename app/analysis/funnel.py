"""The scanner funnel: how many tokens reached each stage, and where the
rest died.

    FOUND -> PRE-SCREEN -> SECURITY -> SIGNAL -> PAPER BUY -> EXIT

Reading this backwards is the trap it is built to resist. A funnel that
narrows hard is a funnel that is working: the whole design of this bot is
to reject weak setups, and a 99% rejection rate on brand-new memecoin
listings is the expected shape, not a fault. The one legitimate conclusion
to draw from a narrow funnel is about the QUALITY OF THE INPUT - whether
discovery is surfacing anything worth evaluating - and never that the
filters should be loosened to let more through.

The stage counts come from two different places, which is why this module
exists rather than one SQL query:

  ScannedToken.last_stage   what the scanner itself recorded per token,
                            covering discovery and the free pre-screen
  RiskEvent.event_type      what the shared buy path rejected, covering
                            security, signal score, market quality and the
                            risk gates - the same events a TradingView
                            alert generates, because scanner candidates go
                            through the identical code
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models

# Which RiskEvent types belong to which funnel stage. Anything not listed
# is counted under "other" rather than being dropped, so the numbers always
# add up and a newly added rejection reason cannot silently vanish.
STAGE_EVENTS: dict[str, tuple[str, ...]] = {
    "security": ("rug_check_rejected",),
    "market": ("market_quality_rejected", "stale_data_rejected"),
    "signal": ("signal_score_rejected", "signal_score_unavailable"),
    "risk": ("buy_blocked", "exposure_cap_rejected", "price_impact_rejected"),
}


@dataclass
class Stage:
    key: str
    label: str
    reached: int
    rejected_here: int = 0
    note: str = ""

    @property
    def pass_rate(self) -> float | None:
        """Share of what reached this stage that survived it. None when
        nothing reached it - which is not a 0% pass rate."""
        if self.reached <= 0:
            return None
        return (self.reached - self.rejected_here) / self.reached * 100

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "reached": self.reached,
            "rejected_here": self.rejected_here,
            "pass_rate": round(self.pass_rate, 1) if self.pass_rate is not None else None,
            "note": self.note,
        }


@dataclass
class Funnel:
    stages: list[Stage] = field(default_factory=list)
    window_hours: float | None = None
    tokens_seen: int = 0
    positions_open: int = 0
    positions_closed: int = 0
    other_rejections: dict[str, int] = field(default_factory=dict)

    @property
    def widest_drop(self) -> Stage | None:
        """The stage that rejected the most. Where the funnel narrows, not
        where it is broken - the two are easy to confuse and only the
        operator can tell them apart."""
        candidates = [s for s in self.stages if s.rejected_here > 0]
        return max(candidates, key=lambda s: s.rejected_here) if candidates else None

    def as_dict(self) -> dict:
        return {
            "window_hours": self.window_hours,
            "tokens_seen": self.tokens_seen,
            "positions_open": self.positions_open,
            "positions_closed": self.positions_closed,
            "stages": [s.as_dict() for s in self.stages],
            "other_rejections": dict(self.other_rejections),
            "widest_drop": self.widest_drop.key if self.widest_drop else None,
        }


def build_funnel(db: Session, *, window_hours: float | None = 24.0) -> Funnel:
    """Count how far candidates got, over the last `window_hours`.

    `window_hours=None` covers all of history. The default window is short
    on purpose: a lifetime funnel is dominated by whatever the config used
    to be, and the question this answers is almost always "what is
    happening now?".
    """
    since = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
        if window_hours else None
    )

    scanned_q = db.query(models.ScannedToken)
    events_q = db.query(models.RiskEvent)
    signals_q = db.query(models.Signal)
    trades_q = db.query(models.Trade).filter(models.Trade.side == "buy")
    positions_q = db.query(models.Position)

    if since is not None:
        scanned_q = scanned_q.filter(models.ScannedToken.last_evaluated_at >= since)
        events_q = events_q.filter(models.RiskEvent.created_at >= since)
        signals_q = signals_q.filter(models.Signal.received_at >= since)
        trades_q = trades_q.filter(models.Trade.created_at >= since)
        positions_q = positions_q.filter(models.Position.opened_at >= since)

    scanned = scanned_q.all()
    events = events_q.all()
    signals = signals_q.all()
    buys = trades_q.filter(models.Trade.status == models.TradeStatus.FILLED.value).all()
    positions = positions_q.all()

    found = len(scanned)
    prescreen_rejected = sum(1 for t in scanned if t.last_stage == "prescreen")

    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1

    claimed = {name for names in STAGE_EVENTS.values() for name in names}
    stage_totals = {
        stage: sum(counts.get(name, 0) for name in names)
        for stage, names in STAGE_EVENTS.items()
    }
    other = {
        name: n for name, n in counts.items()
        if name not in claimed and name.endswith(("_rejected", "_blocked", "_unavailable"))
    }

    # Candidates that survived the free pre-screen and were handed to the
    # shared buy path. Counted from Signals rather than by subtraction, so
    # a webhook alert (which never touches the scanner) is included and the
    # stage counts describe every entry attempt, not just scanner ones.
    evaluated = len(signals)
    filled_buys = len(buys)

    stages = [
        Stage(
            "found", "FOUND", found, prescreen_rejected,
            note="tokens discovery surfaced and the scanner looked at",
        ),
        Stage(
            "prescreen", "PRE-SCREEN", max(found - prescreen_rejected, 0), 0,
            note="passed the free liquidity/volume/age/txn filters",
        ),
        Stage(
            "evaluated", "EVALUATED", evaluated, stage_totals["security"],
            note="entered the shared buy path (scanner candidates and webhook alerts alike)",
        ),
        Stage(
            "security", "SECURITY", max(evaluated - stage_totals["security"], 0),
            stage_totals["market"],
            note="cleared the rug check and composite security score",
        ),
        Stage(
            "market", "MARKET QUALITY",
            max(evaluated - stage_totals["security"] - stage_totals["market"], 0),
            stage_totals["signal"],
            note="tradeable in practice: real volume, exitable depth, fresh data",
        ),
        Stage(
            "signal", "SIGNAL",
            max(evaluated - sum(stage_totals[k] for k in ("security", "market", "signal")), 0),
            stage_totals["risk"],
            note="scored high enough on the entry setup",
        ),
        Stage(
            "paper_buy", "PAPER BUY", filled_buys, 0,
            note="a simulated position was actually opened",
        ),
    ]

    return Funnel(
        stages=stages,
        window_hours=window_hours,
        tokens_seen=found,
        positions_open=sum(1 for p in positions if p.status == models.PositionStatus.OPEN.value),
        positions_closed=sum(1 for p in positions if p.status != models.PositionStatus.OPEN.value),
        other_rejections=other,
    )
