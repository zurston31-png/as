"""Short-window volume and buy/sell pressure, with honest provenance.

Every window here is labelled with where it came from, because they are
not equally trustworthy and treating them as if they were is how a
derived estimate ends up quoted as a measurement.

    MEASURED    DexScreener publishes it directly. Four windows only:
                5m, 1h, 6h, 24h.
    DERIVED     Computed by differencing successive stored snapshots.
                Real information, lower resolution, and only available
                once two observations far enough apart exist.
    UNAVAILABLE Cannot be produced from this data source at all.

WHY THERE IS NO 1-MINUTE OR 15-MINUTE WINDOW

DexScreener's `txns` and `volume` objects carry exactly m5, h1, h6 and
h24. There is no m1 and no m15, and they cannot be recovered by
arithmetic: the published windows are ROLLING aggregates, not disjoint
buckets, so subtracting one from another does not isolate the gap between
them. A "15m" figure computed that way would be a number with no defined
meaning that nevertheless looked precise.

What IS obtainable at finer resolution is the RATE OF CHANGE between two
of our own observations - how fast the 5m window is moving - which is
what the early engine's flow features already use. That is reported here
as derived, at whatever spacing the observations actually have.

Anything genuinely unavailable is named and explained rather than
approximated, on the same principle as app/early/features.py.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models

MEASURED = "measured"
DERIVED = "derived"
UNAVAILABLE = "unavailable"

# Windows DexScreener publishes directly.
PUBLISHED_WINDOWS = ("5m", "1h", "6h", "24h")

# Windows people ask for that this provider does not serve.
UNSERVED_WINDOWS: dict[str, str] = {
    "1m": (
        "DexScreener publishes no m1 window. It cannot be subtracted out of m5 either - "
        "the published windows are rolling aggregates, not disjoint buckets, so the "
        "arithmetic has no defined meaning. A per-minute figure needs a trade-level "
        "source (an indexer or an RPC log subscription)."
    ),
    "15m": (
        "DexScreener publishes m5 then jumps to h1; there is no m15. Interpolating "
        "between them would invent a number, and differencing them measures nothing "
        "well-defined because both are rolling."
    ),
}

# Two observations closer together than this describe almost the same
# rolling window, so the difference between them is mostly noise.
MIN_SPACING_MINUTES = 1.0


@dataclass
class Window:
    label: str
    source: str
    volume_usd: float | None = None
    buys: int | None = None
    sells: int | None = None
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.source != UNAVAILABLE

    @property
    def total_txns(self) -> int | None:
        if self.buys is None and self.sells is None:
            return None
        return (self.buys or 0) + (self.sells or 0)

    @property
    def buy_pressure(self) -> float | None:
        """Buys as a share of all transactions. None, never 0.5, when there
        is nothing to divide - a silent market is not a balanced one."""
        total = self.total_txns
        if not total:
            return None
        return (self.buys or 0) / total

    @property
    def avg_trade_usd(self) -> float | None:
        total = self.total_txns
        if not total or self.volume_usd is None:
            return None
        return self.volume_usd / total

    def as_dict(self) -> dict:
        return {
            "window": self.label,
            "source": self.source,
            "available": self.available,
            "volume_usd": self.volume_usd,
            "buys": self.buys,
            "sells": self.sells,
            "total_txns": self.total_txns,
            "buy_pressure": round(self.buy_pressure, 4) if self.buy_pressure is not None else None,
            "avg_trade_usd": round(self.avg_trade_usd, 2) if self.avg_trade_usd is not None else None,
            "detail": self.detail,
        }


@dataclass
class FlowProfile:
    token_address: str
    symbol: str
    windows: list[Window] = field(default_factory=list)
    observations: int = 0

    def get(self, label: str) -> Window | None:
        return next((w for w in self.windows if w.label == label), None)

    @property
    def measured(self) -> list[Window]:
        return [w for w in self.windows if w.source == MEASURED]

    @property
    def accelerating(self) -> bool | None:
        """Is the short window running hotter than the long one?

        Compares 5m volume, annualised to an hourly rate, against the
        actual hourly figure. None when either is missing rather than
        guessing a direction.
        """
        short, long = self.get("5m"), self.get("1h")
        if not short or not long or short.volume_usd is None or not long.volume_usd:
            return None
        return (short.volume_usd * 12) > long.volume_usd

    @property
    def pressure_shift(self) -> float | None:
        """Short-window buy pressure minus the hourly figure.

        Positive means buying has picked up recently relative to the hour.
        This is the closest honest substitute for the sub-5-minute reading
        that is not obtainable.
        """
        short, long = self.get("5m"), self.get("1h")
        if not short or not long:
            return None
        a, b = short.buy_pressure, long.buy_pressure
        return (a - b) if a is not None and b is not None else None

    def summary(self) -> str:
        lines = [f"{self.symbol} ({self.token_address[:10]}...) flow:"]
        for w in self.windows:
            if not w.available:
                lines.append(f"  {w.label:>4}  UNAVAILABLE - {w.detail}")
                continue
            pressure = f"{w.buy_pressure:.0%} buys" if w.buy_pressure is not None else "no txns"
            volume = f"${w.volume_usd:,.0f}" if w.volume_usd is not None else "volume n/a"
            lines.append(f"  {w.label:>4}  {volume:>14}  {pressure:>10}  [{w.source}]")
        if self.accelerating is not None:
            lines.append(
                f"  5m pace is {'ABOVE' if self.accelerating else 'below'} the hourly rate"
            )
        shift = self.pressure_shift
        if shift is not None:
            lines.append(f"  buy pressure vs the hour: {shift:+.0%}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "token_address": self.token_address,
            "symbol": self.symbol,
            "observations": self.observations,
            "accelerating": self.accelerating,
            "pressure_shift": round(self.pressure_shift, 4) if self.pressure_shift is not None else None,
            "windows": [w.as_dict() for w in self.windows],
        }


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def profile_from_snapshot(market, *, observations: list | None = None) -> FlowProfile:
    """Build the flow picture for one token from a live snapshot.

    `observations` are earlier stored snapshots for the same token; when
    two are far enough apart, the rate of change between them is added as
    a derived window.
    """
    profile = FlowProfile(
        token_address=getattr(market, "token_address", "") or "",
        symbol=getattr(market, "token_symbol", None) or "",
        observations=len(observations or []),
    )

    profile.windows.append(Window(
        "5m", MEASURED,
        volume_usd=market.volume_5m_usd,
        buys=market.buys_5m, sells=market.sells_5m,
        detail="shortest window DexScreener publishes",
    ))
    profile.windows.append(Window(
        "1h", MEASURED,
        volume_usd=market.volume_1h_usd,
        buys=market.buys_1h, sells=market.sells_1h,
    ))
    profile.windows.append(Window(
        "6h", MEASURED,
        volume_usd=getattr(market, "volume_6h_usd", None),
        buys=getattr(market, "buys_6h", None), sells=getattr(market, "sells_6h", None),
    ))
    profile.windows.append(Window(
        "24h", MEASURED,
        volume_usd=market.volume_24h_usd,
        buys=market.buys_24h, sells=market.sells_24h,
    ))

    for label, why in UNSERVED_WINDOWS.items():
        profile.windows.append(Window(label, UNAVAILABLE, detail=why))

    derived = _derive_recent(market, observations or [])
    if derived is not None:
        profile.windows.append(derived)

    order = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "6h": 4, "24h": 5}
    profile.windows.sort(key=lambda w: (order.get(w.label, 9), w.source))
    return profile


def _derive_recent(market, observations: list) -> Window | None:
    """Change in the 5m window between our two most recent observations.

    This is the finest resolution actually available: not a 1-minute
    window, but how fast the shortest published window is moving, at
    whatever spacing our own polling produced. Labelled DERIVED and
    carrying that spacing, so nobody reads it as a published figure.
    """
    usable = [o for o in observations if getattr(o, "volume_5m_usd", None) is not None]
    if len(usable) < 1:
        return None

    previous = usable[-1]
    now = _aware(getattr(market, "observed_at", None)) or dt.datetime.now(dt.timezone.utc)
    then = _aware(getattr(previous, "observed_at", None))
    if then is None:
        return None

    spacing = (now - then).total_seconds() / 60
    if spacing < MIN_SPACING_MINUTES:
        return None

    delta = (market.volume_5m_usd or 0.0) - (previous.volume_5m_usd or 0.0)
    return Window(
        f"~{spacing:.0f}m change", DERIVED,
        volume_usd=delta,
        detail=(
            f"change in the 5m volume window across {spacing:.1f} minutes of our own "
            "polling - a rate of change, not a window total"
        ),
    )


def profile_for_token(db: Session, token_address: str, market) -> FlowProfile:
    """Flow for one token, using whatever observations are already stored."""
    rows = (
        db.query(models.TokenObservation)
        .filter(models.TokenObservation.token_address == token_address)
        .order_by(models.TokenObservation.observed_at.desc())
        .limit(12)
        .all()
    )
    return profile_from_snapshot(market, observations=list(reversed(rows)))
