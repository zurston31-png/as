"""Continuous market-data freshness.

Tracking staleness only when the screen-capture backup wakes up is too late -
by then you have already been trading on a frozen feed. This runs on every
bar and every evaluation, and a stale feed blocks signals outright.

Two clocks matter and they fail differently:

* **bar age** - the market timestamp of the newest bar against wall clock.
  Catches a feed that has stopped producing bars.
* **silence** - how long since anything at all arrived, forming bars included.
  Catches a socket that is open but dead.

Outside the trading session both are expected to be large, so staleness is only
enforced while the session is open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .models import utcnow
from .strategy.context import FeedHealth


@dataclass
class FeedMonitor:
    bar_seconds: int
    max_stale_bars: float = 2.5
    max_stale_seconds: float = 0.0     # absolute override; 0 = derive from bar_seconds

    last_bar_ts: Optional[datetime] = None      # market time of the newest bar
    last_message_at: Optional[datetime] = None  # wall clock of the last thing received
    bars_seen: int = 0
    messages_seen: int = 0
    stale_events: int = 0
    _was_stale: bool = False

    @property
    def threshold_seconds(self) -> float:
        if self.max_stale_seconds > 0:
            return self.max_stale_seconds
        return max(self.bar_seconds * self.max_stale_bars, self.bar_seconds + 30)

    def record(self, bar_ts: datetime, is_new_bar: bool, now: Optional[datetime] = None) -> None:
        now = now or utcnow()
        self.messages_seen += 1
        self.last_message_at = now
        if is_new_bar:
            self.bars_seen += 1
        if self.last_bar_ts is None or bar_ts > self.last_bar_ts:
            self.last_bar_ts = bar_ts

    def health(self, in_session: bool = True, now: Optional[datetime] = None) -> FeedHealth:
        now = now or utcnow()
        health = FeedHealth(expected_bar_seconds=self.bar_seconds,
                            received_at=self.last_message_at)

        if self.last_bar_ts is None:
            health.stale = True
            health.reason = "no market data received yet"
            return health

        # The bar's *close* is one interval after its open timestamp.
        health.bar_age_seconds = max(
            0.0, (now - self.last_bar_ts).total_seconds() - self.bar_seconds)
        silence = ((now - self.last_message_at).total_seconds()
                   if self.last_message_at else health.bar_age_seconds)

        if not in_session:
            health.reason = "outside session - freshness not enforced"
            return health

        threshold = self.threshold_seconds
        if health.bar_age_seconds > threshold:
            health.stale = True
            health.reason = (
                f"newest bar closed {health.bar_age_seconds:.0f}s ago, over the "
                f"{threshold:.0f}s limit"
            )
        elif silence > threshold:
            health.stale = True
            health.reason = f"nothing received for {silence:.0f}s"

        if health.stale and not self._was_stale:
            self.stale_events += 1
        self._was_stale = health.stale
        return health

    def to_dict(self, in_session: bool = True) -> dict[str, Any]:
        health = self.health(in_session)
        return {
            **health.to_dict(),
            "last_bar_ts": self.last_bar_ts.isoformat() if self.last_bar_ts else None,
            "bars_seen": self.bars_seen,
            "messages_seen": self.messages_seen,
            "stale_events": self.stale_events,
            "threshold_seconds": round(self.threshold_seconds, 1),
        }
