"""Position state, and the reconciliation that keeps it honest.

The risk engine needs more than a count of open positions. It needs to know
whether you are flat, long, short, or holding a runner that has already taken
a partial and is waiting on its second target - because "one open position" and
"half a position with a breakeven stop" are different risk situations.

Reconciliation matters even in paper mode. The book records what the risk engine
*believes* is open; the broker holds what actually is. If those ever disagree,
something has gone wrong in the pipeline and the right response is to stop
signalling, not to keep sizing new trades against a fiction. The same check is
what you would run against a live broker's own position records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import Position, Side, TradeSignal


@dataclass(slots=True)
class PositionState:
    state: str = "flat"                 # flat | long | short | mixed
    side: Optional[Side] = None
    qty: float = 0.0
    count: int = 0
    awaiting_exit: bool = False         # a runner past TP1 is still exposed
    signal_ids: list[str] = field(default_factory=list)

    @property
    def flat(self) -> bool:
        return self.count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "side": self.side.value if self.side else None,
            "qty": self.qty,
            "count": self.count,
            "awaiting_exit": self.awaiting_exit,
            "signal_ids": list(self.signal_ids),
        }


@dataclass(slots=True)
class Reconciliation:
    ok: bool = True
    checked_at: str = ""
    expected: list[str] = field(default_factory=list)
    actual: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # we think it's open, broker doesn't
    unexpected: list[str] = field(default_factory=list)  # broker has it, we don't
    qty_drift: dict[str, list[float]] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_at": self.checked_at,
            "expected": self.expected,
            "actual": self.actual,
            "missing": self.missing,
            "unexpected": self.unexpected,
            "qty_drift": self.qty_drift,
            "message": self.message,
        }


class PositionBook:
    """What the risk engine believes is open, checked against the broker."""

    def __init__(self) -> None:
        self._expected: dict[str, float] = {}
        self.last_reconciliation = Reconciliation(ok=True, message="not yet checked")

    # ------------------------------------------------------------ bookkeeping

    def record_entry(self, signal: TradeSignal, qty: float) -> None:
        self._expected[signal.id] = qty

    def record_exit(self, signal_id: str) -> None:
        self._expected.pop(signal_id, None)

    def record_partial(self, signal_id: str, remaining: float) -> None:
        if signal_id in self._expected:
            self._expected[signal_id] = remaining

    def clear(self) -> None:
        self._expected.clear()

    @property
    def expected_ids(self) -> list[str]:
        return sorted(self._expected)

    # ---------------------------------------------------------------- state

    @staticmethod
    def state_of(positions: list[Position]) -> PositionState:
        if not positions:
            return PositionState()
        sides = {p.side for p in positions}
        qty = sum(p.remaining for p in positions)
        ids = [p.signal.id for p in positions]
        if len(sides) > 1:
            return PositionState("mixed", None, qty, len(positions),
                                 any(p.tp1_hit for p in positions), ids)
        side = next(iter(sides))
        return PositionState(
            state=side.value.lower(),
            side=side,
            qty=qty,
            count=len(positions),
            awaiting_exit=any(p.tp1_hit for p in positions),
            signal_ids=ids,
        )

    # -------------------------------------------------------- reconciliation

    def reconcile(self, positions: list[Position], now_iso: str) -> Reconciliation:
        actual = {p.signal.id: p.remaining for p in positions}
        expected = dict(self._expected)

        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        drift = {
            sid: [expected[sid], actual[sid]]
            for sid in set(expected) & set(actual)
            if abs(expected[sid] - actual[sid]) > 1e-9
        }

        report = Reconciliation(
            ok=not (missing or unexpected or drift),
            checked_at=now_iso,
            expected=sorted(expected),
            actual=sorted(actual),
            missing=missing,
            unexpected=unexpected,
            qty_drift=drift,
        )
        if report.ok:
            report.message = f"{len(actual)} position(s) match"
        else:
            parts = []
            if missing:
                parts.append(f"{len(missing)} booked but not held")
            if unexpected:
                parts.append(f"{len(unexpected)} held but not booked")
            if drift:
                parts.append(f"{len(drift)} with a size mismatch")
            report.message = "position state disagrees with the broker: " + ", ".join(parts)
        self.last_reconciliation = report
        return report

    def adopt(self, positions: list[Position]) -> None:
        """Take the broker's view as truth. Only ever called deliberately."""
        self._expected = {p.signal.id: p.remaining for p in positions}
