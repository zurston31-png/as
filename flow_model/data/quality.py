"""Data quality grading and the feed availability report.

This module makes the system's most important honest statement: **which of
the five Flow Score components can actually be computed, and therefore how
many of the nominal 100 points are real**.

With bar-only data, order flow (25 points) and options flow (20 points) have
no feed. A system that scored the remaining 55 and called it a Flow Score
would produce numbers that look like the documented design but mean
something else entirely, and every threshold calibrated against them would
be calibrated against a different scale. So `AvailabilityReport` states the
shortfall in plain language, and `strict_component_availability` (on by
default) makes the engine refuse rather than quietly rescale.

Two grading rules are worth stating up front because they are not obvious:

* **Staleness beats coverage.** A thin but current feed is usable; a
  complete but frozen one is not. A feed past `max_staleness_seconds` is
  STALE regardless of how well-covered its history is.
* **Severely incomplete is MISSING, not DEGRADED.** Below
  `min_coverage_degraded` a feed is not a degraded feed, it is effectively
  no feed. Grading it DEGRADED would let the engine score from noise.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from pydantic import Field, computed_field

from flow_model.config.schema import DataConfig, FlowScoreConfig
from flow_model.core.enums import Component, DataQuality
from flow_model.core.model import FrozenModel
from flow_model.data.base import Feed, FeedStatus
from flow_model.data.market_view import MarketView
from flow_model.data.series import NS_PER_SECOND
from flow_model.data.store import DataStore, SymbolData
from flow_model.utils.logging import get_logger

logger = get_logger("data.quality")


class QualityThresholds(FrozenModel):
    """Grading cutoffs.

    Defaults are set so that a plausibly-bad dataset actually fails. A
    grader that always returns GOOD is worse than no grader, because it
    supplies false assurance.
    """

    min_coverage_good: float = Field(gt=0.0, le=1.0, default=0.98)
    min_coverage_degraded: float = Field(gt=0.0, le=1.0, default=0.80)
    min_tick_classification_good: float = Field(gt=0.0, le=1.0, default=0.90)
    min_tick_classification_degraded: float = Field(gt=0.0, le=1.0, default=0.60)
    max_crossed_quote_rate: float = Field(ge=0.0, le=1.0, default=0.01)
    coverage_window_bars: int = Field(
        gt=1, default=60, description="Trailing window over which coverage is measured."
    )
    session_break_multiple: float = Field(
        gt=1.0,
        default=12.0,
        description=(
            "An inter-bar step larger than this many intervals is treated as a "
            "session break rather than a gap, and excluded from the coverage "
            "denominator. Without it every overnight break would read as "
            "missing data and coverage would be meaningless."
        ),
    )

    def _check_order(self) -> "QualityThresholds":
        if self.min_coverage_degraded > self.min_coverage_good:
            raise ValueError("min_coverage_degraded exceeds min_coverage_good")
        if self.min_tick_classification_degraded > self.min_tick_classification_good:
            raise ValueError(
                "min_tick_classification_degraded exceeds min_tick_classification_good"
            )
        return self

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._check_order()


class QualityReport(FrozenModel):
    """Per-feed grades at one evaluation timestamp."""

    symbol: str
    ts: datetime
    statuses: dict[Feed, FeedStatus]
    overall: DataQuality
    blocking_feeds: tuple[Feed, ...] = ()
    note: str = ""

    def status_of(self, feed: Feed) -> FeedStatus:
        try:
            return self.statuses[feed]
        except KeyError:
            return FeedStatus(feed=feed, quality=DataQuality.MISSING, note="not graded")

    def is_tradable(self, minimum: DataQuality) -> bool:
        return self.overall.is_tradable(minimum)

    def summary_lines(self) -> tuple[str, ...]:
        lines = [f"{self.symbol} @ {self.ts.isoformat()}  overall={self.overall.value}"]
        for feed in Feed:
            status = self.statuses.get(feed)
            if status is None:
                continue
            age = "n/a" if status.age_seconds is None else f"{status.age_seconds:.0f}s"
            lines.append(
                f"  {feed.value:<18} {status.quality.value:<9} "
                f"rows={status.rows:<7} coverage={status.coverage:.3f} age={age}"
                + (f"  {status.note}" if status.note else "")
            )
        if self.blocking_feeds:
            lines.append(
                "  BLOCKING: " + ", ".join(f.value for f in self.blocking_feeds)
            )
        return tuple(lines)


class ComponentAvailability(FrozenModel):
    """Whether one Flow Score component can be computed from a dataset."""

    component: Component
    computable: bool
    quality: DataQuality
    weight: float = Field(ge=0.0)
    missing_required_feeds: tuple[Feed, ...] = ()
    degraded_feeds: tuple[Feed, ...] = ()
    note: str = ""


class AvailabilityReport(FrozenModel):
    """Dataset-level statement of how much of the Flow Score is real.

    Produced BEFORE a backtest runs, so the shortfall is known in advance
    rather than inferred from odd-looking results afterwards.
    """

    symbol: str
    components: dict[Component, ComponentAvailability]
    total_points: float = Field(gt=0.0)
    available_points: float = Field(ge=0.0)
    strict: bool = True
    feeds_present: tuple[Feed, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unavailable_points(self) -> float:
        return round(self.total_points - self.available_points, 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fraction_available(self) -> float:
        return round(self.available_points / self.total_points, 6)

    @property
    def computable_components(self) -> tuple[Component, ...]:
        return tuple(c for c, a in self.components.items() if a.computable)

    @property
    def incomputable_components(self) -> tuple[Component, ...]:
        return tuple(c for c, a in self.components.items() if not a.computable)

    @property
    def scoring_is_valid(self) -> bool:
        """False when strict mode is on and any component is incomputable.

        The engine must refuse to emit a Flow Score in that case. The
        alternative -- redistributing the missing weight -- inflates every
        score and hides the gap.
        """
        if not self.strict:
            return True
        return not self.incomputable_components

    def summary_lines(self) -> tuple[str, ...]:
        lines = [
            f"Feed availability for {self.symbol}",
            f"  feeds present: {', '.join(f.value for f in self.feeds_present) or 'NONE'}",
            "",
            f"  {'component':<16}{'weight':>8}{'computable':>12}  quality / blocker",
            f"  {'-' * 62}",
        ]
        for component in Component:
            a = self.components.get(component)
            if a is None:
                continue
            blocker = (
                "missing: " + ", ".join(f.value for f in a.missing_required_feeds)
                if a.missing_required_feeds
                else a.quality.value
            )
            lines.append(
                f"  {component.value:<16}{a.weight:>8.1f}"
                f"{('yes' if a.computable else 'NO'):>12}  {blocker}"
            )
        lines += [
            f"  {'-' * 62}",
            f"  available points: {self.available_points:.1f} of {self.total_points:.1f} "
            f"({self.fraction_available * 100:.0f}%)",
        ]
        if self.unavailable_points > 0:
            lines.append(
                f"  UNAVAILABLE: {self.unavailable_points:.1f} points have no data feed."
            )
            if self.strict:
                lines += [
                    "",
                    "  strict_component_availability is ON, so the signal engine will",
                    "  REFUSE to produce a Flow Score from this dataset rather than",
                    "  scoring the remaining components and calling the result a Flow",
                    "  Score. Any score computed here would be on a different scale",
                    "  than the one every threshold was calibrated against.",
                    "  Do not trust a score, win rate, or backtest from this dataset",
                    "  until the missing feeds are supplied.",
                ]
            else:
                lines += [
                    "",
                    "  strict_component_availability is OFF. Scores will be produced",
                    "  from the available components only. They are NOT comparable to",
                    "  scores from a complete dataset, and thresholds calibrated on one",
                    "  do not transfer to the other.",
                ]
        return tuple(lines)

    def as_rows(self) -> list[dict[str, object]]:
        """Flat, JSON-serializable rows for reporting."""
        return [
            {
                "symbol": self.symbol,
                "component": component.value,
                "weight": a.weight,
                "computable": a.computable,
                "quality": a.quality.value,
                "missing_required_feeds": ",".join(f.value for f in a.missing_required_feeds),
                "degraded_feeds": ",".join(f.value for f in a.degraded_feeds),
                "note": a.note,
            }
            for component, a in self.components.items()
        ]


class QualityGrader:
    """Grades feeds at a point in time, and datasets before a run."""

    def __init__(
        self,
        data_config: DataConfig,
        flow_score_config: FlowScoreConfig,
        thresholds: QualityThresholds | None = None,
    ) -> None:
        self.data = data_config
        self.flow = flow_score_config
        self.thresholds = thresholds or QualityThresholds()

    # --- coverage ------------------------------------------------------

    def _bar_coverage(self, view: MarketView) -> float:
        """Fraction of contiguous inter-bar steps in a trailing window.

        Precise definition, because an undocumented "coverage" number is
        worse than none: take the last `coverage_window_bars` visible bar
        close timestamps and the differences between consecutive ones.
        Discard differences larger than `session_break_multiple` intervals
        (those are session breaks, not missing data). Coverage is the share
        of the remaining differences that equal exactly one interval.

        Limitation: a gap longer than the session-break multiple is
        indistinguishable from a session break and is not counted. The
        dataset-level `CleanReport.gaps_detected`, which has the calendar
        available, is the authority on gaps.
        """
        stamps = view.bar_timestamps(self.thresholds.coverage_window_bars)
        if stamps.size < 2:
            return 0.0
        interval_ns = view.primary_interval * NS_PER_SECOND
        if interval_ns <= 0:
            return 0.0
        deltas = np.diff(np.asarray(stamps, dtype=np.int64))
        intraday = deltas[deltas <= interval_ns * self.thresholds.session_break_multiple]
        if intraday.size == 0:
            return 1.0  # every step was a session break; nothing to judge
        contiguous = int(np.count_nonzero(intraday == interval_ns))
        return round(contiguous / intraday.size, 6)

    def _tick_classification(self, view: MarketView) -> float:
        window = self.thresholds.coverage_window_bars
        buys = view.tick_column("buy_volume", window)
        sells = view.tick_column("sell_volume", window)
        unclassified = view.tick_column("unclassified_volume", window)
        classified = float(buys.sum() + sells.sum())
        total = classified + float(unclassified.sum())
        if total <= 0:
            return 0.0
        return round(classified / total, 6)

    def _crossed_quote_rate(self, view: MarketView) -> float:
        window = self.thresholds.coverage_window_bars
        bids = view.quote_column("bid", window)
        asks = view.quote_column("ask", window)
        if bids.size == 0 or bids.size != asks.size:
            return 0.0
        return round(float(np.count_nonzero(asks < bids)) / bids.size, 6)

    # --- per-feed grading ----------------------------------------------

    def grade_feed(self, view: MarketView, feed: Feed) -> FeedStatus:
        if not view.has_feed(feed):
            return FeedStatus(
                feed=feed, quality=DataQuality.MISSING, rows=0, coverage=0.0,
                note="no visible observation",
            )

        age = view.feed_age_seconds(feed)
        rows = self._rows(view, feed)
        last_ts = self._last_ts(view, feed)

        if age is not None and age > self.data.max_staleness_seconds:
            return FeedStatus(
                feed=feed, quality=DataQuality.STALE, rows=rows, coverage=0.0,
                age_seconds=age, last_ts=last_ts,
                note=(
                    f"last observation {age:.0f}s old, limit "
                    f"{self.data.max_staleness_seconds:.0f}s; a frozen feed is "
                    "not usable however complete its history"
                ),
            )

        if feed is Feed.TICK_AGGREGATE:
            return self._grade_ticks(view, rows, age, last_ts)
        if feed is Feed.OPTIONS_SNAPSHOT:
            return self._grade_options(view, rows, age, last_ts)
        if feed is Feed.QUOTES:
            return self._grade_quotes(view, rows, age, last_ts)
        return self._grade_bars(view, rows, age, last_ts)

    def _grade_bars(self, view, rows, age, last_ts) -> FeedStatus:
        coverage = self._bar_coverage(view)
        quality, note = self._from_coverage(
            coverage, self.thresholds.min_coverage_good, self.thresholds.min_coverage_degraded,
            "bar coverage",
        )
        return FeedStatus(feed=Feed.BARS, quality=quality, rows=rows, coverage=coverage,
                          age_seconds=age, last_ts=last_ts, note=note)

    def _grade_quotes(self, view, rows, age, last_ts) -> FeedStatus:
        crossed = self._crossed_quote_rate(view)
        coverage = 1.0 - crossed
        if crossed > self.thresholds.max_crossed_quote_rate:
            return FeedStatus(
                feed=Feed.QUOTES, quality=DataQuality.DEGRADED, rows=rows,
                coverage=coverage, age_seconds=age, last_ts=last_ts,
                note=(
                    f"crossed-quote rate {crossed:.3f} exceeds "
                    f"{self.thresholds.max_crossed_quote_rate:.3f}, which indicates a "
                    "broken or stitched feed"
                ),
            )
        return FeedStatus(feed=Feed.QUOTES, quality=DataQuality.GOOD, rows=rows,
                          coverage=coverage, age_seconds=age, last_ts=last_ts)

    def _grade_ticks(self, view, rows, age, last_ts) -> FeedStatus:
        classification = self._tick_classification(view)
        quality, note = self._from_coverage(
            classification,
            self.thresholds.min_tick_classification_good,
            self.thresholds.min_tick_classification_degraded,
            "aggressor classification",
        )
        return FeedStatus(feed=Feed.TICK_AGGREGATE, quality=quality, rows=rows,
                          coverage=classification, age_seconds=age, last_ts=last_ts, note=note)

    def _grade_options(self, view, rows, age, last_ts) -> FeedStatus:
        if not view.options_are_intraday():
            return FeedStatus(
                feed=Feed.OPTIONS_SNAPSHOT, quality=DataQuality.DEGRADED, rows=rows,
                coverage=1.0, age_seconds=age, last_ts=last_ts,
                note=(
                    "end-of-day chain only: expresses positioning but not flow "
                    "timing, so the options sub-score must be capped"
                ),
            )
        return FeedStatus(feed=Feed.OPTIONS_SNAPSHOT, quality=DataQuality.GOOD, rows=rows,
                          coverage=1.0, age_seconds=age, last_ts=last_ts)

    def _from_coverage(
        self, value: float, good: float, degraded: float, label: str
    ) -> tuple[DataQuality, str]:
        if value < degraded:
            return (
                DataQuality.MISSING,
                f"{label} {value:.3f} below {degraded:.2f}: effectively no feed, "
                "not a degraded one -- scoring from it would be scoring noise",
            )
        if value < good:
            return DataQuality.DEGRADED, f"{label} {value:.3f} below {good:.2f}"
        return DataQuality.GOOD, ""

    @staticmethod
    def _rows(view: MarketView, feed: Feed) -> int:
        if feed is Feed.BARS:
            return view.bar_count()
        if feed is Feed.QUOTES:
            return int(view.quote_column("bid", 10**9).size)
        if feed is Feed.TICK_AGGREGATE:
            return int(view.tick_column("buy_volume", 10**9).size)
        return int(view.options_column("call_volume", 10**9).size)

    @staticmethod
    def _last_ts(view: MarketView, feed: Feed) -> datetime | None:
        age = view.feed_age_seconds(feed)
        if age is None:
            return None
        from datetime import timedelta

        return view.now - timedelta(seconds=age)

    # --- point-in-time report ------------------------------------------

    def required_feeds(self) -> dict[Feed, tuple[Component, ...]]:
        """Feeds that at least one ENABLED component requires."""
        out: dict[Feed, list[Component]] = {}
        for component, requirements in self.flow.feed_requirements.items():
            if not self.flow.enabled_components.get(component, True):
                continue
            for requirement in requirements:
                if requirement.required:
                    out.setdefault(requirement.feed, []).append(component)
        return {feed: tuple(components) for feed, components in out.items()}

    def grade(self, view: MarketView) -> QualityReport:
        statuses = {feed: self.grade_feed(view, feed) for feed in Feed}
        required = self.required_feeds()

        relevant = [statuses[feed].quality for feed in required if feed in statuses]
        overall = DataQuality.worst(*relevant) if relevant else DataQuality.MISSING

        blocking = tuple(
            feed
            for feed in required
            if not statuses[feed].quality.is_tradable(self.data.min_quality_to_trade)
        )
        note = ""
        if blocking:
            note = (
                "required feed(s) below "
                f"{self.data.min_quality_to_trade.value}: "
                + ", ".join(
                    f"{feed.value}={statuses[feed].quality.value}" for feed in blocking
                )
            )
        return QualityReport(
            symbol=view.symbol, ts=view.now, statuses=statuses, overall=overall,
            blocking_feeds=blocking, note=note,
        )

    # --- dataset report ------------------------------------------------

    def availability(self, data: SymbolData) -> AvailabilityReport:
        present = data.available_feeds()
        components: dict[Component, ComponentAvailability] = {}

        for component in Component:
            weight = float(self.flow.weights.get(component, 0.0))
            enabled = self.flow.enabled_components.get(component, True)
            requirements = self.flow.feed_requirements.get(component, ())

            missing = tuple(
                r.feed for r in requirements if r.required and r.feed not in present
            )
            degraded: tuple[Feed, ...] = ()
            if (
                component is Component.OPTIONS_FLOW
                and Feed.OPTIONS_SNAPSHOT in present
                and data.options is not None
                and not data.options.is_intraday
            ):
                degraded = (Feed.OPTIONS_SNAPSHOT,)
            if (
                component is Component.LIQUIDITY
                and Feed.QUOTES not in present
            ):
                degraded = degraded + (Feed.QUOTES,)

            if not enabled:
                components[component] = ComponentAvailability(
                    component=component, computable=False, quality=DataQuality.MISSING,
                    weight=weight, missing_required_feeds=(),
                    note="disabled in configuration",
                )
                continue

            if missing:
                components[component] = ComponentAvailability(
                    component=component, computable=False, quality=DataQuality.MISSING,
                    weight=weight, missing_required_feeds=missing,
                    note=(
                        "no feed for "
                        + ", ".join(f.value for f in missing)
                        + "; not substituted, because a derived estimate would be "
                        "indistinguishable from real data downstream"
                    ),
                )
                continue

            quality = DataQuality.DEGRADED if degraded else DataQuality.GOOD
            note = ""
            if Feed.OPTIONS_SNAPSHOT in degraded:
                note = "end-of-day options only: positioning without flow timing"
            elif Feed.QUOTES in degraded:
                note = "no quote feed: spread and depth terms unavailable, volume only"
            components[component] = ComponentAvailability(
                component=component, computable=True, quality=quality, weight=weight,
                degraded_feeds=degraded, note=note,
            )

        available = sum(a.weight for a in components.values() if a.computable)
        report = AvailabilityReport(
            symbol=data.symbol, components=components,
            total_points=self.flow.total_points, available_points=round(available, 6),
            strict=self.flow.strict_component_availability,
            feeds_present=tuple(sorted(present, key=lambda f: f.value)),
        )
        if report.unavailable_points > 0:
            logger.warning(
                "%s: only %.1f of %.1f Flow Score points are computable (%s have no feed)",
                data.symbol, report.available_points, report.total_points,
                ", ".join(c.value for c in report.incomputable_components),
            )
        return report

    def availability_from_store(self, store: DataStore) -> dict[str, AvailabilityReport]:
        return {symbol: self.availability(store.get(symbol)) for symbol in store.symbols}
