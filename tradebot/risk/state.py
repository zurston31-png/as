"""Persistent risk state.

Kept in a small JSON file so a crash, a restart, or a deliberate kill-switch
trip doesn't hand you a fresh set of daily limits. The day boundary follows the
market session timezone, not UTC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class RiskState:
    trading_day: str = ""
    equity: float = 0.0
    day_start_equity: float = 0.0
    trades_today: int = 0
    realized_today: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: Optional[str] = None      # ISO timestamp
    kill_switch: bool = False
    kill_reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def daily_pnl_pct(self) -> float:
        if not self.day_start_equity:
            return 0.0
        return (self.equity - self.day_start_equity) / self.day_start_equity * 100.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["daily_pnl_pct"] = round(self.daily_pnl_pct, 3)
        return d


class RiskStore:
    def __init__(self, path: str | Path, starting_equity: float) -> None:
        self.path = Path(path)
        self.starting_equity = starting_equity
        self.state = self._load()

    def _load(self) -> RiskState:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                known = {f for f in RiskState.__dataclass_fields__}
                return RiskState(**{k: v for k, v in raw.items() if k in known})
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # corrupt state file - start clean rather than refuse to run
        return RiskState(equity=self.starting_equity, day_start_equity=self.starting_equity)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2))
        tmp.replace(self.path)

    def roll_day(self, today: date) -> bool:
        """Reset the per-day counters when the session date moves forward.

        Only forward: an out-of-order bar or a clock correction must not be
        able to hand you a fresh set of daily limits.
        """
        key = today.isoformat()
        if self.state.trading_day == key or key < self.state.trading_day:
            return False
        if self.state.trading_day:
            self.state.history.append(
                {
                    "day": self.state.trading_day,
                    "trades": self.state.trades_today,
                    "realized": round(self.state.realized_today, 2),
                    "closing_equity": round(self.state.equity, 2),
                }
            )
            self.state.history = self.state.history[-90:]
        self.state.trading_day = key
        self.state.trades_today = 0
        self.state.realized_today = 0.0
        self.state.day_start_equity = self.state.equity or self.starting_equity
        # A kill switch tripped by a daily-loss limit clears with the new day;
        # one a human flipped does not.
        if self.state.kill_switch and self.state.kill_reason.startswith("auto:"):
            self.state.kill_switch = False
            self.state.kill_reason = ""
        self.save()
        return True

    def cooldown_active(self, now: datetime) -> bool:
        if not self.state.cooldown_until:
            return False
        return now < datetime.fromisoformat(self.state.cooldown_until)
