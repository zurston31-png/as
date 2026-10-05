"""Enumerations shared across the Flow Model.

All enums are string-valued so they serialize cleanly to JSON/SQLite and
appear readable in trade records and reports.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum with a readable repr (stdlib StrEnum is 3.11+, but we
    keep an explicit base for stable serialization semantics)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Side(StrEnum):
    """Direction of a position."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class SignalAction(StrEnum):
    """What the signal engine decided at a timestamp."""

    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


class Regime(StrEnum):
    """Quantitatively-defined market regimes (see regime/detector.py).

    These are assigned by measurement, never by hand-labelling.
    """

    LOW_VOL = "LOW_VOL"
    HIGH_VOL = "HIGH_VOL"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    CHOP = "CHOP"
    DIRECTIONAL_SHIFT = "DIRECTIONAL_SHIFT"
    UNKNOWN = "UNKNOWN"  # insufficient warmup; never tradable


class SetupType(StrEnum):
    """Trade archetypes. Each has its own target multiple and gate config."""

    SCALP_1R = "SCALP_1R"
    SETUP_2R = "SETUP_2R"
    DIRECTIONAL_3R = "DIRECTIONAL_3R"


class DataQuality(StrEnum):
    """Per-feed / per-feature data quality.

    Ordered worst-to-best via `rank` so a bundle's status is the minimum of
    its parts.
    """

    MISSING = "MISSING"
    STALE = "STALE"
    DEGRADED = "DEGRADED"
    GOOD = "GOOD"

    @property
    def rank(self) -> int:
        return {"MISSING": 0, "STALE": 1, "DEGRADED": 2, "GOOD": 3}[self.value]

    @classmethod
    def worst(cls, *statuses: "DataQuality") -> "DataQuality":
        if not statuses:
            return cls.MISSING
        return min(statuses, key=lambda s: s.rank)

    def is_tradable(self, minimum: "DataQuality") -> bool:
        return self.rank >= minimum.rank


class SplitPhase(StrEnum):
    """Which validation phase a date range belongs to.

    SEALED_OOS is the untouched final holdout. Access requires an explicit,
    logged unseal (see validation/splits.py).
    """

    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    SEALED_OOS = "SEALED_OOS"


class InstrumentType(StrEnum):
    FUTURE = "FUTURE"
    ETF = "ETF"
    INDEX = "INDEX"


class ExitReason(StrEnum):
    """Why a position closed. Recorded on every trade."""

    STOP = "STOP"
    TARGET = "TARGET"
    TIME_STOP = "TIME_STOP"
    SESSION_CLOSE = "SESSION_CLOSE"
    TRAIL_STOP = "TRAIL_STOP"
    RISK_LIMIT = "RISK_LIMIT"
    KILL_SWITCH = "KILL_SWITCH"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    DATA_LOSS = "DATA_LOSS"
    END_OF_BACKTEST = "END_OF_BACKTEST"


class WaitReason(StrEnum):
    """Why no trade was taken. Recorded for every WAIT so that rejection
    analysis (how often each gate fires) is possible."""

    DATA_QUALITY = "data_quality"
    WARMUP = "warmup"
    REGIME_BLOCKED = "regime_blocked"
    LIQUIDITY = "liquidity"
    NO_STRUCTURE = "no_structure"
    SCORE_BELOW_THRESHOLD = "score_below_threshold"
    ORDERFLOW_CONFLICT = "orderflow_conflict"
    OPTIONS_CONTRADICTION = "options_contradiction"
    VOLATILITY_BAND = "volatility_band"
    RR_TOO_LOW = "rr_too_low"
    RISK_LIMIT = "risk_limit"
    SESSION_CLOSED = "session_closed"
    COMPONENT_DISABLED = "component_disabled"
    NO_SETUP_MATCH = "no_setup_match"


class Component(StrEnum):
    """The five Flow Score components."""

    OPTIONS_FLOW = "options_flow"
    ORDER_FLOW = "order_flow"
    STRUCTURE = "structure"
    LIQUIDITY = "liquidity"
    VOL_MOMENTUM = "vol_momentum"


class MonteCarloMethod(StrEnum):
    IID = "iid"
    BLOCK = "block"
    REGIME_AWARE = "regime_aware"


class Session(StrEnum):
    """Intraday session buckets, used for performance breakdowns."""

    ASIA = "ASIA"
    LONDON = "LONDON"
    PRE_RTH = "PRE_RTH"
    RTH_OPEN = "RTH_OPEN"
    RTH_MID = "RTH_MID"
    RTH_CLOSE = "RTH_CLOSE"
    POST_RTH = "POST_RTH"
    CLOSED = "CLOSED"
