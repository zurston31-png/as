"""Helpers for the strategy tests.

IMPORTANT - SYNTHETIC TEST FIXTURES ONLY. Every price built here is invented
to exercise a rule. None of it is market data, and no result computed from it
says anything about how a strategy would have performed.
"""

import random
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time

from sqlalchemy.orm import Session, sessionmaker

from app.data.validation import NEW_YORK, trading_days
from app.database import session_scope
from app.models import Bar
from app.strategies.data import DailyBar
from app.strategies.simulator import DayPlan, SizingRules

#: The nine trading sessions from 2 to 12 January 2024.
SESSIONS = [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9, 10, 11, 12)]


def make_bars(prices: Sequence[tuple[float, float, float, float]], sessions=SESSIONS):
    """Daily bars from (open, high, low, close) tuples, one per session."""
    assert len(prices) <= len(sessions)
    return [
        DailyBar(session, open_, high, low, close, 1000)
        for session, (open_, high, low, close) in zip(sessions, prices, strict=False)
    ]


def flat_bars(price: float = 50.0, count: int = len(SESSIONS), sessions=SESSIONS):
    """Bars that never move: open = high = low = close."""
    return make_bars([(price, price, price, price)] * count, sessions)


def random_walk_bars(
    seed: int, sessions: Sequence[date], start: float = 100.0, drift: float = 0.001
):
    """A seeded random walk with consistent highs and lows."""
    generator = random.Random(seed)
    bars, close = [], start
    for session in sessions:
        open_ = close * (1.0 + generator.uniform(-0.01, 0.01))
        close = max(1.0, open_ * (1.0 + drift + generator.uniform(-0.03, 0.03)))
        high = max(open_, close) * (1.0 + generator.uniform(0.0, 0.015))
        low = min(open_, close) * (1.0 - generator.uniform(0.0, 0.015))
        bars.append(
            DailyBar(session, round(open_, 4), round(high, 4), round(low, 4), round(close, 4), 1000)
        )
    return bars


def sessions_between(first: date, last: date) -> list[date]:
    return trading_days(first, last)


class ScriptedStrategy:
    """A strategy whose plans are written out by the test, to exercise the simulator alone.

    ``plans`` maps symbol -> session -> plan; sessions without a plan have no signal.
    """

    name = "scripted_fixture"
    version = "test"

    def __init__(
        self,
        bars: Mapping[str, list[DailyBar]],
        plans: Mapping[str, Mapping[date, DayPlan]],
        *,
        risk_fraction: float = 0.005,
        max_position_fraction: float = 0.20,
        max_positions: int = 5,
    ) -> None:
        self._plans = plans
        self._sizing = SizingRules(risk_fraction, max_position_fraction, max_positions)
        # The simulator hands over one symbol's bars at a time, without the symbol.
        self._symbol_of = {id(items[0]): symbol for symbol, items in bars.items() if items}
        #: Every bar list the simulator handed over, to check what it was allowed to see.
        self.seen: list[list[DailyBar]] = []

    @property
    def sizing(self) -> SizingRules:
        return self._sizing

    def build_plans(self, bars: list[DailyBar]) -> list[DayPlan]:
        self.seen.append(list(bars))
        symbol = self._symbol_of.get(id(bars[0])) if bars else None
        plans = self._plans.get(symbol, {})
        return [plans.get(bar.session, DayPlan(bar.session, False, False)) for bar in bars]


def entry(session: date, *, stop_distance: float, limit: float | None = None, rank: float = 0.0):
    return DayPlan(
        session=session,
        entry_signal=True,
        exit_signal=False,
        stop_distance=stop_distance,
        max_entry_price=limit,
        rank_value=rank,
        indicators={"fixture": True},
    )


def exit_(session: date):
    return DayPlan(session=session, entry_signal=False, exit_signal=True)


def store_daily_bars(
    session_factory: sessionmaker[Session],
    symbol: str,
    bars: Sequence[DailyBar],
    *,
    feed: str = "iex",
    adjustment: str = "split",
) -> None:
    """Insert SYNTHETIC daily bars straight into the test database."""
    with session_scope(session_factory) as session:
        for bar in bars:
            midnight = datetime.combine(bar.session, time(0, 0), tzinfo=NEW_YORK)
            session.add(
                Bar(
                    symbol=symbol,
                    timeframe="1Day",
                    timestamp=midnight.astimezone(UTC),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.close,
                    trade_count=10,
                    source="alpaca",
                    feed=feed,
                    adjustment=adjustment,
                )
            )
