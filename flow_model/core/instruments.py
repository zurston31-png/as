"""Instrument specifications.

An `InstrumentSpec` is the single authority on contract economics: tick
size, tick value, commissions, and what a 1-point move is worth. Position
sizing and the cost model read from here and nowhere else, so there is no
possibility of two modules disagreeing about what NQ is worth.
"""

from __future__ import annotations

from pydantic import Field, computed_field, model_validator

from flow_model.core.model import FrozenModel

from flow_model.core.enums import InstrumentType


class SessionWindow(FrozenModel):
    """A trading window in exchange-local time (HH:MM, 24h)."""


    name: str
    start: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")

    @property
    def start_minutes(self) -> int:
        h, m = self.start.split(":")
        return int(h) * 60 + int(m)

    @property
    def end_minutes(self) -> int:
        h, m = self.end.split(":")
        return int(h) * 60 + int(m)

    @property
    def wraps_midnight(self) -> bool:
        return self.end_minutes <= self.start_minutes

    def contains_minutes(self, minutes_of_day: int) -> bool:
        """Half-open window [start, end)."""
        if self.wraps_midnight:
            return minutes_of_day >= self.start_minutes or minutes_of_day < self.end_minutes
        return self.start_minutes <= minutes_of_day < self.end_minutes


class InstrumentSpec(FrozenModel):
    """Economics of one tradable (or reference) instrument."""


    symbol: str
    instrument_type: InstrumentType
    description: str = ""
    exchange: str = ""
    timezone: str = "America/New_York"
    currency: str = "USD"

    tick_size: float = Field(gt=0, description="Minimum price increment.")
    tick_value: float = Field(
        gt=0, description="Cash value of one tick for one unit (contract/share)."
    )

    commission_per_side: float = Field(ge=0, default=0.0)
    exchange_fee_per_side: float = Field(ge=0, default=0.0)
    # ETFs charge per share rather than per order.
    commission_per_unit: bool = Field(
        default=False,
        description="If true, commission_per_side is multiplied by size.",
    )

    typical_spread_ticks: float = Field(
        gt=0, default=1.0, description="Fallback spread when no quote data exists."
    )
    min_spread_ticks: float = Field(gt=0, default=1.0)

    tradable: bool = Field(
        default=True,
        description=(
            "False for reference-only instruments (e.g. a cash index whose "
            "options flow feeds signals for a futures contract)."
        ),
    )
    execution_proxy: str | None = Field(
        default=None,
        description=(
            "Symbol actually executed when this instrument is signalled. "
            "Required for INDEX instruments marked tradable."
        ),
    )

    allow_fractional_size: bool = Field(default=False)
    min_size: float = Field(gt=0, default=1.0)
    max_size: float = Field(gt=0, default=1_000.0)

    rth: SessionWindow | None = None
    eth: SessionWindow | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def point_value(self) -> float:
        """Cash value of a one-point price move for one unit."""
        return self.tick_value / self.tick_size

    @model_validator(mode="after")
    def _validate(self) -> "InstrumentSpec":
        if self.min_spread_ticks > self.typical_spread_ticks:
            raise ValueError(
                f"{self.symbol}: min_spread_ticks ({self.min_spread_ticks}) exceeds "
                f"typical_spread_ticks ({self.typical_spread_ticks})"
            )
        if self.min_size > self.max_size:
            raise ValueError(f"{self.symbol}: min_size exceeds max_size")
        if (
            self.instrument_type is InstrumentType.INDEX
            and self.tradable
            and not self.execution_proxy
        ):
            raise ValueError(
                f"{self.symbol}: a tradable INDEX must declare an execution_proxy "
                "(an index cannot be traded directly; costs would be fictional)."
            )
        return self

    # --- price helpers -------------------------------------------------

    def round_to_tick(self, price: float) -> float:
        """Round to the nearest valid price increment."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def ticks_to_price(self, ticks: float) -> float:
        return ticks * self.tick_size

    def price_to_ticks(self, price_distance: float) -> float:
        return price_distance / self.tick_size

    def risk_per_unit(self, stop_distance: float) -> float:
        """Cash at risk per unit for a given stop distance in price units."""
        if stop_distance <= 0:
            raise ValueError("stop_distance must be positive")
        return stop_distance * self.point_value

    def commission_for(self, size: float, sides: int = 2) -> float:
        """Total commission + exchange fees. `sides=2` is a round turn."""
        if sides < 1:
            raise ValueError("sides must be >= 1")
        per_side = self.commission_per_side + self.exchange_fee_per_side
        if self.commission_per_unit:
            return per_side * size * sides
        return per_side * sides
