"""Strategy A: momentum breakout on daily bars. Long only.

The rules, exactly as agreed. All of them use completed daily bars only.

Entry signal, judged at the close of session ``t``:
    * close[t] is above the highest high of the 55 sessions before ``t``
      (session ``t`` itself is not part of that high), and
    * close[t] is above the 200-session average close (which includes ``t``).
    Both comparisons are strict: equal is not above.

Entry:
    Buy at the open of the next session. The order is not filled if that open
    is more than 1 ATR above close[t] (the price ran away overnight).

Stop loss:
    Entry fill minus 2 ATR. The ATR is the 20-session average true range as it
    stood at the close of session ``t`` and is fixed for the life of the trade.

Target:
    None. There is no profit target: a position is held until the exit rule or
    the stop ends it.

Exit signal, judged at the close of a session ``u`` while a position is open:
    close[u] is below the lowest low of the 20 sessions before ``u``.
    Sell at the open of the next session.

Position size, decided at the close of session ``t`` (before the entry price is known):
    shares = floor(0.5 % of the strategy's equity / (2 ATR)), but never more
    than floor(20 % of equity / close[t]). At most 5 positions are open at
    once. When more signals than free slots occur on one day, the strongest
    55-session price change goes first.

Re-entry:
    No rule of its own: after an exit, the next entry needs a new entry signal.

This module only turns bars into daily plans. Fills, stops, the account and
the bookkeeping are the simulator's job (``simulator.py``).
"""

from dataclasses import asdict, dataclass, fields
from typing import Any

from app.strategies.data import DailyBar
from app.strategies.indicators import atr, prior_highest, prior_lowest, rate_of_change, sma
from app.strategies.simulator import DayPlan, SizingRules

STRATEGY_NAME = "momentum_breakout"

#: Change this whenever the RULES change (not the parameters), so that results
#: produced by different rules can never be confused.
STRATEGY_VERSION = "1.0"


@dataclass(frozen=True)
class MomentumBreakoutParameters:
    """The numbers in the rules above. The defaults are the agreed starting values."""

    breakout_window: int = 55
    exit_window: int = 20
    trend_window: int = 200
    atr_window: int = 20
    stop_atr_multiple: float = 2.0
    max_entry_gap_atr: float = 1.0
    risk_fraction: float = 0.005
    max_position_fraction: float = 0.20
    max_positions: int = 5

    def __post_init__(self) -> None:
        for name in ("breakout_window", "exit_window", "trend_window", "atr_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a whole number of at least 1, got {value!r}")
        if isinstance(self.max_positions, bool) or not isinstance(self.max_positions, int):
            raise ValueError(f"max_positions must be a whole number, got {self.max_positions!r}")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        for name in ("stop_atr_multiple", "max_entry_gap_atr"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be greater than 0, got {getattr(self, name)!r}")
        for name in ("risk_fraction", "max_position_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(
                    f"{name} must be above 0 and at most 1, got {getattr(self, name)!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict[str, str]) -> "MomentumBreakoutParameters":
        """Build parameters from ``name=value`` text, starting from the defaults."""
        types = {field.name: field.type for field in fields(cls)}
        values: dict[str, Any] = {}
        for name, raw in overrides.items():
            if name not in types:
                raise ValueError(
                    f"unknown parameter {name!r}; expected one of: {', '.join(sorted(types))}"
                )
            try:
                values[name] = int(raw) if types[name] is int else float(raw)
            except ValueError:
                raise ValueError(f"parameter {name} needs a number, got {raw!r}") from None
        return cls(**values)


class MomentumBreakout:
    """Strategy A. Holds its parameters and turns one symbol's bars into daily plans."""

    name = STRATEGY_NAME
    version = STRATEGY_VERSION

    def __init__(self, parameters: MomentumBreakoutParameters | None = None) -> None:
        self.parameters = parameters or MomentumBreakoutParameters()

    @property
    def sizing(self) -> SizingRules:
        return SizingRules(
            risk_fraction=self.parameters.risk_fraction,
            max_position_fraction=self.parameters.max_position_fraction,
            max_positions=self.parameters.max_positions,
        )

    def build_plans(self, bars: list[DailyBar]) -> list[DayPlan]:
        """Return one plan per bar, each computed only from that bar and earlier ones."""
        p = self.parameters
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        closes = [bar.close for bar in bars]

        breakout_level = prior_highest(highs, p.breakout_window)
        exit_level = prior_lowest(lows, p.exit_window)
        trend = sma(closes, p.trend_window)
        average_range = atr(highs, lows, closes, p.atr_window)
        strength = rate_of_change(closes, p.breakout_window)

        plans: list[DayPlan] = []
        for i, bar in enumerate(bars):
            level, average, range_, change = (
                breakout_level[i],
                trend[i],
                average_range[i],
                strength[i],
            )
            ready = None not in (level, average, range_, change) and range_ > 0
            entry = bool(ready and bar.close > level and bar.close > average)
            exit_ = exit_level[i] is not None and bar.close < exit_level[i]
            plans.append(
                DayPlan(
                    session=bar.session,
                    entry_signal=entry,
                    exit_signal=exit_,
                    stop_distance=p.stop_atr_multiple * range_ if entry else None,
                    max_entry_price=bar.close + p.max_entry_gap_atr * range_ if entry else None,
                    rank_value=change if entry else None,
                    indicators=(
                        {
                            "close": bar.close,
                            f"prior_{p.breakout_window}_session_high": level,
                            f"sma_{p.trend_window}": average,
                            f"atr_{p.atr_window}": range_,
                            f"change_{p.breakout_window}_sessions": change,
                        }
                        if entry
                        else {}
                    ),
                )
            )
        return plans
