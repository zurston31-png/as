"""Feature computation contracts.

Every feature in the system is produced by a `FeatureComputer`. The contract
is deliberately narrow, and each part of it exists to make a specific
failure impossible:

* **`compute` receives a `MarketView`, never a series or a DataFrame.** The
  view physically cannot hold data past its cutoff, so a computer cannot
  read the future even by accident.
* **`warmup_bars` is declared, not discovered.** A rolling window that is
  only half full produces a number that looks like a feature and is not one.
  The bundle returns `warmup_complete=False` until every computer's window
  is full, and the signal engine turns that into `WAIT("warmup")`.
* **`required_feeds` is declared.** A computer whose feed is absent is
  skipped and reported, never run against empty arrays.
* **`keys` is declared.** The bundle checks that a computer produced exactly
  the keys it promised, so a silently-missing feature is an error rather
  than a `None` that propagates into a score.

A computer must be a pure function of its view. That is what
`validation.lookahead` audits: computing at bar *i* must give the same
answer whether or not bars after *i* exist, and whatever their values are.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from flow_model.core.contracts import FeatureVector
from flow_model.core.enums import DataQuality, Feed
from flow_model.data.market_view import MarketView
from flow_model.utils.logging import get_logger

logger = get_logger("features")


class FeatureError(RuntimeError):
    """Raised when a computer violates its declared contract."""


def percentile_rank(window: np.ndarray, value: float) -> float:
    """Fraction of `window` at or below `value`, in [0, 1].

    Used instead of a z-score wherever a feature feeds a bounded component
    score. A z-score is unbounded, so one extreme observation can dominate a
    weighted sum; a percentile rank cannot.
    """
    if window.size == 0:
        return 0.5
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return 0.5
    return float(np.count_nonzero(finite <= value)) / finite.size


def squash(value: float, scale: float) -> float:
    """Map an unbounded magnitude into [0, 1] via tanh.

    `scale` is the value that maps to roughly 0.76. Bounded by construction,
    so a single outlier cannot dominate a component score.
    """
    if scale <= 0:
        raise FeatureError(f"squash scale must be positive, got {scale}")
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(abs(value) / scale))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` rather than inf/nan on a zero divisor."""
    if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return default
    result = numerator / denominator
    return float(result) if np.isfinite(result) else default


class FeatureComputer(ABC):
    """Produces a set of named features from a point-in-time market view."""

    #: Short identifier used in logs and audit reports.
    name: str = "abstract"

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Bars of history required before any output is meaningful."""

    @property
    @abstractmethod
    def keys(self) -> tuple[str, ...]:
        """Exactly the feature names `compute` will populate."""

    @property
    def required_feeds(self) -> frozenset[Feed]:
        """Feeds without which this computer cannot run at all."""
        return frozenset({Feed.BARS})

    @property
    def optional_feeds(self) -> frozenset[Feed]:
        """Feeds that improve the output but are not required."""
        return frozenset()

    @abstractmethod
    def compute(self, view: MarketView) -> FeatureVector:
        """Features at `view.now`, using only data at or before its cutoff."""

    # --- helpers for implementations ----------------------------------

    def _vector(
        self,
        view: MarketView,
        values: dict[str, float],
        quality: DataQuality = DataQuality.GOOD,
        warmup_complete: bool = True,
        notes: Sequence[str] = (),
    ) -> FeatureVector:
        """Build the vector and check it against the declared keys.

        A computer that promises `atr` and returns `atr_14` would otherwise
        put a `None` into a score; here it is an error at the source.
        """
        promised, produced = set(self.keys), set(values)
        if promised != produced:
            missing, extra = sorted(promised - produced), sorted(produced - promised)
            raise FeatureError(
                f"{self.name}: declared keys do not match output. "
                f"missing={missing} unexpected={extra}"
            )
        for key, value in values.items():
            if not np.isfinite(value):
                raise FeatureError(
                    f"{self.name}: feature {key!r} is not finite ({value}). A "
                    "non-finite feature propagates into every score that reads it; "
                    "return a defined fallback and a note instead."
                )
        return FeatureVector(
            symbol=view.symbol,
            ts=view.now,
            values=values,
            quality_by_key={key: quality for key in values},
            warmup_complete=warmup_complete,
            notes=tuple(notes),
        )

    def _not_ready(
        self, view: MarketView, reason: str, quality: DataQuality = DataQuality.MISSING
    ) -> FeatureVector:
        """A vector marked not-ready, with zeros for every declared key.

        The zeros are placeholders and must never be read: `warmup_complete`
        is False and `quality` is MISSING, and the bundle refuses to emit a
        score from a vector in that state.
        """
        return FeatureVector(
            symbol=view.symbol,
            ts=view.now,
            values={key: 0.0 for key in self.keys},
            quality_by_key={key: quality for key in self.keys},
            warmup_complete=False,
            notes=(f"{self.name}: {reason}",),
        )

    def feeds_available(self, view: MarketView) -> bool:
        return all(view.has_feed(feed) for feed in self.required_feeds)


class FeatureBundle:
    """Runs a set of computers and merges their output.

    Warmup and feed availability are enforced here rather than in each
    computer, so the rule is stated once: the bundle is ready only when
    every enabled computer is ready, and a bundle that is not ready carries
    `warmup_complete=False` for the signal engine to act on.
    """

    def __init__(self, computers: Sequence[FeatureComputer]) -> None:
        self.computers = tuple(computers)
        names = [c.name for c in self.computers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise FeatureError(f"duplicate computer names: {duplicates}")
        seen: dict[str, str] = {}
        for computer in self.computers:
            for key in computer.keys:
                if key in seen:
                    raise FeatureError(
                        f"feature key {key!r} is declared by both {seen[key]!r} and "
                        f"{computer.name!r}; a collision would silently drop one"
                    )
                seen[key] = computer.name
        self.key_owner = seen

    @property
    def warmup_bars(self) -> int:
        """The longest warmup among the computers.

        The bundle is only as ready as its slowest window, and trading on a
        bundle where one window is still filling is the bug this prevents.
        """
        return max((c.warmup_bars for c in self.computers), default=0)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.key_owner))

    def required_feeds(self) -> frozenset[Feed]:
        feeds: set[Feed] = set()
        for computer in self.computers:
            feeds |= computer.required_feeds
        return frozenset(feeds)

    def compute(self, view: MarketView) -> FeatureVector:
        """Merged features at `view.now`."""
        if not self.computers:
            raise FeatureError("FeatureBundle has no computers")

        merged: FeatureVector | None = None
        for computer in self.computers:
            if not computer.feeds_available(view):
                absent = sorted(
                    f.value for f in computer.required_feeds if not view.has_feed(f)
                )
                vector = computer._not_ready(view, f"required feed(s) absent: {absent}")
            elif not view.warmup_ok(computer.warmup_bars):
                vector = computer._not_ready(
                    view,
                    f"warmup incomplete: {view.bar_count()} of {computer.warmup_bars} bars",
                )
            else:
                vector = computer.compute(view)
            merged = vector if merged is None else merged.merge(vector)

        assert merged is not None
        return merged
