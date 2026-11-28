"""The fixed research assumptions agreed for Milestone 2.

These are decisions, not measurements. They live in one place so that every
strategy is simulated under the same, visible conditions.
"""

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

#: Each strategy is simulated alone, with its own virtual account of this size (USD).
STARTING_EQUITY = 100_000.0

PERIOD_IN_SAMPLE = "in_sample"
PERIOD_OUT_OF_SAMPLE = "out_of_sample"
PERIOD_CUSTOM = "custom"

#: Parameters may be tuned on this period only.
IN_SAMPLE_START = date(2021, 6, 1)
IN_SAMPLE_END = date(2024, 12, 31)

#: Everything from this date on is reserved. A run that touches it is only
#: allowed with parameters that were frozen beforehand (see ``runner.py``).
OUT_OF_SAMPLE_START = date(2025, 1, 1)


@dataclass(frozen=True)
class Period:
    """The sessions a run may trade in (both ends inclusive, New York dates)."""

    name: str
    start: date
    #: None means "up to the last stored session".
    end: date | None

    def touches_out_of_sample(self) -> bool:
        return self.end is None or self.end >= OUT_OF_SAMPLE_START


IN_SAMPLE = Period(PERIOD_IN_SAMPLE, IN_SAMPLE_START, IN_SAMPLE_END)
OUT_OF_SAMPLE = Period(PERIOD_OUT_OF_SAMPLE, OUT_OF_SAMPLE_START, None)

#: Assumed slippage per side for fills on DAILY bars, in basis points of the
#: price (1 bp = 0.01 %). There is no bid/ask data behind these numbers:
#: "base" is the agreed starting assumption and the others are deliberately
#: worse, to show how much a result depends on it. Commission is assumed zero.
DAILY_COST_SCENARIOS = MappingProxyType({"base": 5.0, "worse": 10.0, "severe": 20.0})

#: The agreed starting assumption for intraday fills (not used by any daily strategy).
INTRADAY_BASE_SLIPPAGE_BPS = 3.0


def buy_fill(reference_price: float, slippage_bps: float) -> float:
    """Price paid when buying at ``reference_price``: slippage makes it worse (higher)."""
    return reference_price * (1.0 + slippage_bps / 10_000.0)


def sell_fill(reference_price: float, slippage_bps: float) -> float:
    """Price received when selling at ``reference_price``: slippage makes it worse (lower)."""
    return reference_price * (1.0 - slippage_bps / 10_000.0)
