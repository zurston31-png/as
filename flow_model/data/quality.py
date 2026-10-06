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

from datetime import datetime, timedelta

import numpy as np
from pydantic import Field, computed_field

from flow_model.config.schema import DataConfig, FlowScoreConfig
from flow_model.core.enums import Component, DataQuality
from flow_model.core.model import FrozenModel
from flow_model.data.base import Feed, FeedStatus
from flow_model.data.market_view import MarketView
from flow_model.data.series import NS_PER_SECOND, from_ns
from flow_model.data.store import DataStore, SymbolData
from flow_model.utils.logging import get_logger

logger = get_logger("data.quality")


def _median_cadence_seconds(series) -> float | None:
    """Median interval between a series' observations, or None below 2 rows."""
    if series is None or len(series) < 2:
        return None
    return float(np.median(np.diff(series.ts_ns))) / NS_PER_SECOND


def _worst_gap_seconds(series) -> float | None:
    """Longest interval this series ever goes between observations.

    The agreement check between `availability()` and `grade()` uses the WORST
    gap, not the median. A once-per-session feed's median gap is one day but
    its worst is a long weekend, and it is the weekend that decides whether
    the configured staleness limit will start blocking bars every Monday.
    """
    if series is None or len(series) < 2:
        return None
    return float(np.max(np.diff(series.ts_ns))) / NS_PER_SECOND


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
        """Whether trading may proceed on this data.

        A non-empty `blocking_feeds` always means no, independently of
        `overall`. The two can disagree: a feed graded DEGRADED clears a
        DEGRADED dataset-wide minimum while still failing the GOOD floor its
        own component declared, and reporting that as tradable would make
        the per-component floors decorative again.
        """
        if self.blocking_feeds:
            return False
        return self.overall.is_tradable(minimum)

    def summary_lines(self) -> tuple[str, ...]:
        lines = [f"{self.symbol} @ {self.ts.isoformat()}  overall={self.overall.value}"]
        for feed in Feed:
            status = self.statuses.get(feed)
            if status is None:
                continue
            age = "n/a" if status.age_seconds is None else f"{status.age_seconds:.0f}s"
            coverage = (
                "not measured" if status.coverage is None else f"{status.coverage:.3f}"
            )
            lines.append(
                f"  {feed.value:<18} {status.quality.value:<9} "
                f"rows={status.rows:<7} coverage={coverage} age={age}"
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

    def _bar_coverage(self, view: MarketView) -> float | None:
        """Fraction of contiguous inter-bar steps in a trailing window.

        Precise definition, because an undocumented "coverage" number is
        worse than none: take the last `coverage_window_bars` visible bar
        close timestamps and the differences between consecutive ones.
        Discard differences larger than `session_break_multiple` intervals
        (those are session breaks, not missing data). Coverage is the share
        of the remaining differences that equal exactly one interval.

        Returns None when NOTHING is measurable -- every step in the window
        exceeded the break threshold. That case previously returned 1.0, so a
        feed missing 99% of its bars graded GOOD with an empty note, and the
        grade was not even monotonic: going from 8.3% of bars present to 7.7%
        flipped MISSING to GOOD as the steps crossed the threshold. "No
        information" and "perfect coverage" are now different answers.

        At daily granularity and coarser, ANY step longer than one interval is
        a non-session span (a weekend is three daily steps), so the break
        threshold tightens to just above one interval. Otherwise a complete
        daily series graded DEGRADED purely from its weekends.

        Limitation: a gap longer than the break threshold is indistinguishable
        from a session break and is not counted. The dataset-level
        `CleanReport.gaps_detected`, which has the calendar, is the authority
        on gaps.
        """
        stamps = view.bar_timestamps(self.thresholds.coverage_window_bars)
        if stamps.size < 2:
            return None
        interval_ns = view.primary_interval * NS_PER_SECOND
        if interval_ns <= 0:
            return None
        multiple = (
            1.5
            if view.primary_interval >= 86_400
            else self.thresholds.session_break_multiple
        )
        deltas = np.diff(np.asarray(stamps, dtype=np.int64))
        intraday = deltas[deltas <= interval_ns * multiple]
        if intraday.size == 0:
            return None
        contiguous = int(np.count_nonzero(intraday == interval_ns))
        return round(contiguous / intraday.size, 6)

    def _tick_classification(self, view: MarketView) -> tuple[float, str]:
        """Share of traded volume with a known aggressor, and how it was measured.

        Reconciled against BAR volume when a bar feed is available, rather
        than trusting the tick feed's own denominator. `unclassified_volume`
        is an optional column, so a feed that classified half the tape and
        simply omitted the column reported 1.0 and graded GOOD, while the
        same feed honestly declaring its unclassified half graded MISSING.
        Bar volume is the independent quantity the split must reconcile to,
        and it was loaded alongside and never consulted.
        """
        window = self.thresholds.coverage_window_bars
        buys = view.tick_column("buy_volume", window)
        sells = view.tick_column("sell_volume", window)
        classified = float(buys.sum() + sells.sum())

        bar_volume = view.volumes(window)
        if bar_volume.size and buys.size:
            traded = float(bar_volume[-buys.size:].sum()) if buys.size <= bar_volume.size else 0.0
            if traded > 0:
                return round(min(classified / traded, 1.0), 6), "reconciled against bar volume"

        unclassified = view.tick_column("unclassified_volume", window)
        total = classified + float(unclassified.sum())
        if total <= 0:
            return 0.0, "no volume"
        return (
            round(classified / total, 6),
            "self-reported by the feed; no bar volume available to reconcile against",
        )

    def _crossed_quote_rate(self, view: MarketView) -> float:
        window = self.thresholds.coverage_window_bars
        bids = view.quote_column("bid", window)
        asks = view.quote_column("ask", window)
        if bids.size == 0 or bids.size != asks.size:
            return 0.0
        return round(float(np.count_nonzero(asks < bids)) / bids.size, 6)

    # --- per-feed grading ----------------------------------------------

    def staleness_limit(self, view: MarketView, feed: Feed) -> float:
        """How old this feed's newest observation may be before it is STALE.

        The configured limit is a floor; a feed is additionally allowed
        `stale_cadence_multiple` of its OWN observed interval. "Stale" should
        mean missed observations, not elapsed wall-clock. One global bound
        made the documented end-of-day options path unreachable: an EOD chain
        is a session old by construction, so a 120s limit graded it STALE on
        98.7% of bars while `availability()` reported all 100 points real.
        """
        configured = float(
            self.data.max_staleness_seconds_by_feed.get(feed, self.data.max_staleness_seconds)
        )
        cadence = view.feed_cadence_seconds(feed)
        if cadence is None or cadence <= 0:
            return configured
        return max(configured, cadence * self.data.stale_cadence_multiple)

    def grade_feed(self, view: MarketView, feed: Feed) -> FeedStatus:
        if not view.has_feed(feed):
            return FeedStatus(
                feed=feed, quality=DataQuality.MISSING, rows=0, coverage=None,
                note="no visible observation",
            )

        age = view.feed_age_seconds(feed)
        rows = self._rows(view, feed)
        last_ts = self._last_ts(view, feed)
        first_ts = self._first_ts(view, feed)
        limit = self.staleness_limit(view, feed)

        if age is not None and age > limit:
            cadence = view.feed_cadence_seconds(feed)
            cadence_note = "" if cadence is None else f", own cadence {cadence:.0f}s"
            return FeedStatus(
                feed=feed, quality=DataQuality.STALE, rows=rows, coverage=None,
                age_seconds=age, first_ts=first_ts, last_ts=last_ts,
                note=(
                    f"last observation {age:.0f}s old, limit {limit:.0f}s"
                    f"{cadence_note}; a frozen feed is not usable however "
                    "complete its history"
                ),
            )

        if feed is Feed.TICK_AGGREGATE:
            return self._grade_ticks(view, rows, age, first_ts, last_ts)
        if feed is Feed.OPTIONS_SNAPSHOT:
            return self._grade_options(view, rows, age, first_ts, last_ts)
        if feed is Feed.QUOTES:
            return self._grade_quotes(view, rows, age, first_ts, last_ts)
        return self._grade_bars(view, rows, age, first_ts, last_ts)

    def _grade_bars(self, view, rows, age, first_ts, last_ts) -> FeedStatus:
        coverage = self._bar_coverage(view)
        if coverage is None:
            return FeedStatus(
                feed=Feed.BARS, quality=DataQuality.DEGRADED, rows=rows, coverage=None,
                age_seconds=age, first_ts=first_ts, last_ts=last_ts,
                note=(
                    "bar coverage not measurable: every step in the window exceeded "
                    "the session-break threshold, so contiguity cannot be judged. "
                    "Graded DEGRADED rather than GOOD -- absence of information is "
                    "not evidence of completeness."
                ),
            )
        quality, note = self._from_coverage(
            coverage, self.thresholds.min_coverage_good, self.thresholds.min_coverage_degraded,
            "bar coverage",
        )
        return FeedStatus(feed=Feed.BARS, quality=quality, rows=rows, coverage=coverage,
                          age_seconds=age, first_ts=first_ts, last_ts=last_ts, note=note)

    def _grade_quotes(self, view, rows, age, first_ts, last_ts) -> FeedStatus:
        crossed = self._crossed_quote_rate(view)
        coverage = 1.0 - crossed
        if crossed > self.thresholds.max_crossed_quote_rate:
            return FeedStatus(
                feed=Feed.QUOTES, quality=DataQuality.DEGRADED, rows=rows,
                coverage=coverage, age_seconds=age, first_ts=first_ts, last_ts=last_ts,
                note=(
                    f"crossed-quote rate {crossed:.3f} exceeds "
                    f"{self.thresholds.max_crossed_quote_rate:.3f}, which indicates a "
                    "broken or stitched feed"
                ),
            )
        return FeedStatus(feed=Feed.QUOTES, quality=DataQuality.GOOD, rows=rows,
                          coverage=coverage, age_seconds=age, last_ts=last_ts)

    def _grade_ticks(self, view, rows, age, first_ts, last_ts) -> FeedStatus:
        classification, basis = self._tick_classification(view)
        quality, note = self._from_coverage(
            classification,
            self.thresholds.min_tick_classification_good,
            self.thresholds.min_tick_classification_degraded,
            "aggressor classification",
        )
        detail = note or basis
        if note and basis:
            detail = f"{note} ({basis})"
        return FeedStatus(feed=Feed.TICK_AGGREGATE, quality=quality, rows=rows,
                          coverage=classification, age_seconds=age, first_ts=first_ts, last_ts=last_ts,
                          note=detail)

    def _grade_options(self, view, rows, age, first_ts, last_ts) -> FeedStatus:
        if not view.options_are_intraday():
            return FeedStatus(
                feed=Feed.OPTIONS_SNAPSHOT, quality=DataQuality.DEGRADED, rows=rows,
                coverage=1.0, age_seconds=age, first_ts=first_ts, last_ts=last_ts,
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
        return view.now - timedelta(seconds=age)

    @staticmethod
    def _first_ts(view: MarketView, feed: Feed) -> datetime | None:
        """Oldest visible observation for a feed.

        `FeedStatus.first_ts` was declared and never assigned, so every
        report serialized `first_ts: null` -- a field that looks like data
        but is only ever absent.
        """
        series = view._series_for(feed)  # noqa: SLF001 - the grader is part of the data layer
        if series is None or len(series) == 0:
            return None
        return from_ns(int(series.ts_ns[0]))

    # --- point-in-time report ------------------------------------------

    def required_feeds(self) -> dict[Feed, tuple[Component, ...]]:
        """Feeds that at least one ENABLED component requires."""
        return {feed: who for feed, (_, who) in self.feed_floors().items()}

    def feed_floors(self) -> dict[Feed, tuple[DataQuality, tuple[Component, ...]]]:
        """The quality floor each required feed must clear, and who demands it.

        Built from each enabled component's own `FeedRequirement.minimum_quality`
        rather than from one global threshold. Those per-feed floors were
        declared in `defaults.yaml`, loaded, validated -- and then read
        nowhere, so tightening `feed_requirements.structure.minimum_quality`
        to GOOD had no effect at all. The dataset-wide
        `data.min_quality_to_trade` applies on top as a floor under the floor.
        """
        floors: dict[Feed, tuple[DataQuality, list[Component]]] = {}
        for component, requirements in self.flow.feed_requirements.items():
            if not self.flow.enabled_components.get(component, True):
                continue
            for requirement in requirements:
                if not requirement.required:
                    continue
                strictest, claimants = floors.get(
                    requirement.feed, (self.data.min_quality_to_trade, [])
                )
                if requirement.minimum_quality.rank > strictest.rank:
                    strictest = requirement.minimum_quality
                claimants.append(component)
                floors[requirement.feed] = (strictest, claimants)
        return {feed: (floor, tuple(who)) for feed, (floor, who) in floors.items()}

    def grade(self, view: MarketView) -> QualityReport:
        statuses = {feed: self.grade_feed(view, feed) for feed in Feed}
        floors = self.feed_floors()

        relevant = [statuses[feed].quality for feed in floors if feed in statuses]
        overall = DataQuality.worst(*relevant) if relevant else DataQuality.MISSING

        blocking = tuple(
            feed
            for feed, (floor, _) in floors.items()
            if not statuses[feed].quality.is_tradable(floor)
        )
        note = ""
        if blocking:
            note = "; ".join(
                f"{feed.value}={statuses[feed].quality.value} below the "
                f"{floors[feed][0].value} floor required by "
                + ", ".join(c.value for c in floors[feed][1])
                for feed in blocking
            )
        return QualityReport(
            symbol=view.symbol, ts=view.now, statuses=statuses, overall=overall,
            blocking_feeds=blocking, note=note,
        )

    # --- dataset report ------------------------------------------------

    def availability(self, data: SymbolData) -> AvailabilityReport:
        """Dataset-level feed availability. **Deliberately forward-looking.**

        Reads the WHOLE series, including bars after any evaluation instant.
        That is correct for its purpose -- it answers "is this dataset usable
        at all" before a run starts -- but it means this method must NEVER be
        called per bar inside a backtest, where it would leak the existence of
        future observations. `grade(view)` is the point-in-time counterpart
        and the only one a signal engine may call.
        """
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
                # Agreement check. `availability()` is the pre-run promise and
                # `grade()` is what happens per bar; they contradicted each
                # other outright when the staleness limit was shorter than the
                # options cadence -- 100 of 100 points promised, 98.7% of bars
                # blocked. If the limit cannot accommodate this feed's own
                # cadence, say so here rather than at bar 1 of the backtest.
                cadence = _median_cadence_seconds(data.options)
                worst = _worst_gap_seconds(data.options)
                limit = float(
                    self.data.max_staleness_seconds_by_feed.get(
                        Feed.OPTIONS_SNAPSHOT, self.data.max_staleness_seconds
                    )
                )
                allowed = max(limit, (cadence or 0.0) * self.data.stale_cadence_multiple)
                if worst is not None and worst > allowed:
                    components[component] = ComponentAvailability(
                        component=component,
                        computable=False,
                        quality=DataQuality.MISSING,
                        weight=weight,
                        missing_required_feeds=(Feed.OPTIONS_SNAPSHOT,),
                        note=(
                            f"options cadence is incompatible with the staleness "
                            f"limit: the feed goes up to {worst / 3600:.0f}h between "
                            f"snapshots but is allowed {allowed / 3600:.0f}h, so "
                            f"grade() will return STALE and block most bars. Raise "
                            f"data.max_staleness_seconds_by_feed[options_snapshot] "
                            f"above {worst:.0f}s or supply an intraday feed."
                        ),
                    )
                    continue
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
