"""Bar cleaning.

Two entry points, because the file adapters and the cleaner answer the same
question differently on purpose:

* `CsvAdapter`/`ParquetAdapter` **reject** duplicate, out-of-order or
  malformed rows with `SchemaError`. A doubled or interleaved vendor export
  is an upstream bug, and quietly repairing it produces a plausible-looking
  series nobody can reconcile against the source.
* `BarCleaner.clean_raw` **repairs and counts** the same conditions, for a
  caller who has raw vendor arrays and has decided that repair is what they
  want. Every repair is itemized in a `CleanReport`.

`BarCleaner.clean` takes an already-valid `BarSeries`. Because `BarSeries`
validates monotonicity and OHLC ordering at construction, the duplicate,
out-of-order and malformed counts in that path are necessarily zero -- which
is a property of the input, not a claim that nothing was wrong.

## The causality constraint

Cleaning runs once over the whole history at load time, so it is tempting to
use full-sample statistics: "drop any bar whose return exceeds 10 sigma of
all returns". That is a real, if mild, lookahead. Whether bar `t` survives
would depend on bars after `t`, and the surviving dataset would encode
future information — every feature computed on it inherits that.

So outlier detection uses a **strictly trailing** window: the dispersion
estimate for bar `t` comes from bars before `t` and never includes `t`
itself. The estimator is a median absolute deviation rather than a rolling
standard deviation, because a standard deviation is inflated by the very
outlier it is meant to detect, which makes large spikes self-concealing.

Do not "optimize" either property away. `tests/unit/test_clean.py` asserts
both: appending a spike to the end of a series must not change the survival
of any earlier bar.

## What is never done

No price or volume is ever created. Gaps are detected and counted, never
filled; there is no forward-fill, interpolation, or reindex-and-pad anywhere
in this module. A test asserts that the output contains no timestamp that
was absent from the input.
"""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from flow_model.config.schema import DataConfig
from flow_model.core.instruments import InstrumentSpec
from flow_model.data.base import CleanReport, SchemaError
from flow_model.data.series import NS_PER_SECOND, BarSeries, from_ns
from flow_model.utils.logging import get_logger

logger = get_logger("data.clean")

#: On a repeated timestamp, keep the LAST occurrence. A duplicated timestamp
#: in vendor data is normally a corrected revision of the first, so the later
#: row is the authoritative one. Named rather than inlined so the policy is
#: visible to anyone auditing the cleaner.
DUPLICATE_POLICY_KEEP_LAST = True

#: Scale factor making the median absolute deviation a consistent estimator
#: of the standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826

_PRICE_COLUMNS = ("open", "high", "low", "close")


def _modal_phase(ts_ns: np.ndarray, interval_ns: int) -> int:
    """Most common offset of timestamps within the interval grid.

    Bar closes are not generally congruent to 0 mod interval: a session
    opening at 08:20 puts 5-minute closes on a different phase than one
    opening at 09:30. The grid is therefore derived from the series itself
    rather than assumed, so partial-bar detection works for any session.
    """
    if ts_ns.size == 0:
        return 0
    phases = (ts_ns % interval_ns).tolist()
    return int(Counter(phases).most_common(1)[0][0])


def detect_gaps(
    ts_ns: np.ndarray,
    interval_seconds: int,
    spec: InstrumentSpec | None = None,
    calendar=None,
) -> list[tuple[datetime, datetime, int]]:
    """Missing-bar runs as `(after, before, missing_count)`.

    An overnight or weekend break is not a data gap. When `spec` and
    `calendar` are supplied, a jump is only reported when both surrounding
    bars belong to the same trading session, which is what distinguishes a
    genuinely dropped bar from the market being closed. Without them every
    session boundary would be reported, and the gap count would say nothing.
    """
    interval_ns = int(interval_seconds) * NS_PER_SECOND
    if ts_ns.size < 2 or interval_ns <= 0:
        return []

    deltas = np.diff(ts_ns)
    candidates = np.flatnonzero(deltas > interval_ns)
    gaps: list[tuple[datetime, datetime, int]] = []
    for index in candidates:
        before = from_ns(int(ts_ns[index]))
        after = from_ns(int(ts_ns[index + 1]))
        if spec is not None and calendar is not None:
            # No AttributeError guard: `session_date` is part of
            # SessionCalendarProtocol. Swallowing a missing method here meant a
            # conforming-but-incomplete calendar silently produced the
            # no-calendar answer while the report claimed otherwise.
            if calendar.session_date(before, spec) != calendar.session_date(after, spec):
                continue
        missing = int(deltas[index] // interval_ns) - 1
        if missing > 0:
            gaps.append((before, after, missing))
    return gaps


class BarCleaner:
    """Repairs what can be repaired, counts everything, invents nothing."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.outlier_window = int(config.outlier_window_bars)
        self.min_outlier_history = int(config.outlier_min_history_bars)

    # --- public API ----------------------------------------------------

    def clean(
        self,
        series: BarSeries,
        spec: InstrumentSpec | None = None,
        calendar=None,
    ) -> tuple[BarSeries, CleanReport]:
        """Clean an already-validated series.

        Duplicate/out-of-order/malformed counts are zero by construction
        here, because `BarSeries` rejects those at build time.

        The input's `meta` is forwarded. It carries the provenance that
        matters most -- `source`, `path`, and the adapter's
        `timestamp_is_bar_open` convention -- and dropping it at the cleaning
        step destroyed the only record of which convention produced a result.
        """
        return self.clean_raw(
            symbol=series.symbol,
            interval_seconds=series.interval_seconds,
            ts_ns=series.ts_ns,
            columns={name: series.col(name) for name in series.columns},
            spec=spec,
            calendar=calendar,
            meta=series.meta,
        )

    def clean_raw(
        self,
        symbol: str,
        interval_seconds: int,
        ts_ns: Sequence[int] | np.ndarray,
        columns: Mapping[str, Sequence[float] | np.ndarray],
        spec: InstrumentSpec | None = None,
        calendar=None,
        meta: Mapping[str, object] | None = None,
    ) -> tuple[BarSeries, CleanReport]:
        """Clean raw columnar bar data into a validated series."""
        ts = np.asarray(ts_ns, dtype=np.int64)
        cols = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}

        missing = [name for name in BarSeries.REQUIRED if name not in cols]
        if missing:
            raise SchemaError(f"clean_raw: missing required columns {missing}")
        for name, values in cols.items():
            if len(values) != len(ts):
                raise SchemaError(
                    f"clean_raw: column {name!r} has {len(values)} rows but ts has {len(ts)}"
                )

        rows_in = int(len(ts))
        # Gaps are measured on the INPUT. Measuring them on the output made the
        # cleaner report its own outlier removals as defects in the data: a
        # pristine generated dataset came back claiming 60 gaps, every one of
        # them a hole the quarantine had just made.
        input_gaps = detect_gaps(ts, interval_seconds, spec=spec, calendar=calendar)
        counts = {
            "malformed_dropped": 0,
            "duplicates_dropped": 0,
            "out_of_order_dropped": 0,
            "partial_bars_dropped": 0,
            "outliers_quarantined": 0,
            "zero_volume_flagged": 0,
        }
        notes: list[str] = []
        quarantined: list[datetime] = []
        keep = np.ones(rows_in, dtype=bool)

        # 1. malformed -- unrepairable, so dropped first and not considered again
        keep &= self._malformed_mask(cols, counts)

        # 2. duplicates (keep last) and 3. out-of-order
        keep = self._dedupe_and_order(ts, keep, counts)

        # 4. partial final bar
        keep = self._drop_partial_final(ts, keep, interval_seconds, counts, notes)

        # 5. outliers -- strictly trailing estimate
        keep, quarantined = self._quarantine_outliers(ts, cols, keep, counts)

        # 6. zero volume -- flagged, never dropped
        volume = cols["volume"]
        counts["zero_volume_flagged"] = int(np.count_nonzero((volume == 0.0) & keep))

        kept_ts = ts[keep]
        forwarded = {k: v for k, v in dict(meta or {}).items() if k != "interval_seconds"}
        cleaned = BarSeries(
            symbol=symbol,
            ts_ns=kept_ts,
            interval_seconds=int(interval_seconds),
            columns={
                name: values[keep]
                for name, values in cols.items()
                if name in BarSeries.REQUIRED or name in BarSeries.OPTIONAL
            },
            meta={**forwarded, "cleaned": True},
        )

        # 7. gaps -- counted, never filled. The two causes are kept apart:
        # `gaps` is what was wrong with the feed, `introduced` is what this
        # cleaner did to it.
        gaps = input_gaps
        output_gaps = detect_gaps(kept_ts, interval_seconds, spec=spec, calendar=calendar)
        introduced = max(0, len(output_gaps) - len(input_gaps))
        if spec is None or calendar is None:
            notes.append(
                "gap detection ran without a calendar: session breaks are "
                "indistinguishable from dropped bars, so gaps_detected is an upper bound"
            )

        report = CleanReport(
            symbol=symbol,
            interval_seconds=int(interval_seconds),
            rows_in=rows_in,
            rows_out=int(len(cleaned)),
            gaps_detected=len(gaps),
            largest_gap_bars=max((count for _, _, count in gaps), default=0),
            gaps_introduced_by_cleaning=introduced,
            quarantined_timestamps=tuple(quarantined),
            notes=tuple(notes),
            **counts,
        )
        if report.rows_removed:
            logger.info(
                "cleaned %s %ss: %d -> %d rows (%s)",
                symbol, interval_seconds, report.rows_in, report.rows_out,
                ", ".join(f"{k}={v}" for k, v in counts.items() if v),
            )
        return cleaned, report

    # --- steps ---------------------------------------------------------

    @staticmethod
    def _malformed_mask(cols: dict[str, np.ndarray], counts: dict[str, int]) -> np.ndarray:
        o, h, l, c = (cols[k] for k in _PRICE_COLUMNS)
        volume = cols["volume"]
        finite = np.ones(len(o), dtype=bool)
        for values in (o, h, l, c, volume):
            finite &= np.isfinite(values)
        ordered = (h >= l) & (l <= o) & (o <= h) & (l <= c) & (c <= h)
        non_negative = volume >= 0.0
        ok = finite & ordered & non_negative
        counts["malformed_dropped"] = int(np.count_nonzero(~ok))
        return ok

    @staticmethod
    def _dedupe_and_order(
        ts: np.ndarray, keep: np.ndarray, counts: dict[str, int]
    ) -> np.ndarray:
        """Collapse ADJACENT duplicate runs, then drop non-ascending rows.

        Adjacency matters, and a single global de-duplication gets it wrong.
        Two cases must be distinguished:

        * **Adjacent** equal timestamps are a vendor revision -- the same bar
          restated -- so the LAST row of the run wins
          (`DUPLICATE_POLICY_KEEP_LAST`).
        * **Non-adjacent** repetition means the export is interleaved: a row
          from earlier in the session appears late. Here the in-order copy is
          the good one and the late row is the anomaly, so the late row is
          dropped and the original kept.

        Applying keep-last globally would drop the in-order copy and then
        drop the late copy again for breaking the ordering, losing the bar
        entirely. Ordering-first would drop the revision in the adjacent
        case. Doing adjacent-runs first and ordering second handles both.

        Deliberately never sorts: sorting an interleaved export produces a
        plausible-looking series that cannot be reconciled with the source.
        """
        keep = keep.copy()
        indices = np.flatnonzero(keep)
        if indices.size == 0:
            return keep

        if DUPLICATE_POLICY_KEEP_LAST:
            for position in range(indices.size - 1):
                current = int(indices[position])
                following = int(indices[position + 1])
                if int(ts[current]) == int(ts[following]):
                    keep[current] = False
                    counts["duplicates_dropped"] += 1

        last_ts: int | None = None
        for index in np.flatnonzero(keep):
            value = int(ts[index])
            if last_ts is not None and value <= last_ts:
                keep[index] = False
                counts["out_of_order_dropped"] += 1
            else:
                last_ts = value
        return keep

    def _drop_partial_final(
        self,
        ts: np.ndarray,
        keep: np.ndarray,
        interval_seconds: int,
        counts: dict[str, int],
        notes: list[str],
    ) -> np.ndarray:
        """Drop a trailing bar that is off the series' own interval grid.

        A partially-formed bar has a close that is not a close. Only the
        final bar is considered: an off-grid bar in the middle is a vendor
        artifact to report, not a partial bar.
        """
        if not self.config.drop_partial_bars:
            return keep
        indices = np.flatnonzero(keep)
        if indices.size < 2:
            return keep
        interval_ns = int(interval_seconds) * NS_PER_SECOND
        # Establish the grid from the rows BEFORE the final bar, and only act
        # if that phase is genuinely dominant. On mostly off-grid vendor data
        # the modal phase is an artifact, and acting on it dropped a
        # grid-correct final bar.
        body = ts[indices[:-1]]
        phase = _modal_phase(body, interval_ns)
        share = float(np.count_nonzero(body % interval_ns == phase)) / body.size
        if share <= 0.60:
            notes.append(
                f"interval grid not established ({share:.0%} of rows share the modal "
                f"phase); no partial-bar drop attempted"
            )
            return keep
        last = int(indices[-1])
        if int(ts[last]) % interval_ns != phase:
            keep = keep.copy()
            keep[last] = False
            counts["partial_bars_dropped"] = 1
            notes.append(
                f"dropped trailing bar at {from_ns(int(ts[last])).isoformat()}: "
                f"off the interval grid (phase {int(ts[last]) % interval_ns} != {phase})"
            )
        return keep

    def _quarantine_outliers(
        self,
        ts: np.ndarray,
        cols: dict[str, np.ndarray],
        keep: np.ndarray,
        counts: dict[str, int],
    ) -> tuple[np.ndarray, list[datetime]]:
        """Remove bars whose return is extreme versus a STRICTLY TRAILING MAD.

        Two properties this implementation is built around:

        **Causality.** The dispersion estimate for a bar is computed only from
        returns of bars before it, and only from returns that were themselves
        accepted. Nothing after the bar under test is read, so appending data
        to the end of a series cannot change which earlier bars survive.
        Verified by `test_outlier_detection_is_causal`.

        **A bad print costs one bar, not two.** The return of the bar FOLLOWING
        a spike is measured against the last SURVIVING close rather than the
        raw previous row. Measuring against the spike made the reversion look
        equally extreme, so one bad print quarantined two bars and the count
        could not be compared against a known injected count (precision was
        0.52 against the generator's ground truth).

        **Consecutive rejections are capped at one**, and that cap is what
        makes the anchoring safe. Holding the anchor across an unbounded run
        of rejections cascades: if the price genuinely moved, every later bar
        is extreme relative to a stale anchor and the detector rejects the
        rest of the series (measured: precision 0.022, 536 false positives on
        16 injected spikes). With the cap, a transient print costs exactly the
        print, and a genuine level shift costs exactly one bar -- the first
        one, which is indistinguishable from a spike without looking ahead.
        Choosing one bar of false positive over a 1-bar lookahead is
        deliberate.

        The first `min_outlier_history` evaluable bars are never quarantined:
        there is no trailing history to judge them against, and judging them
        on later data is exactly the lookahead this avoids.
        """
        indices = np.flatnonzero(keep)
        quarantined: list[datetime] = []
        if indices.size <= self.min_outlier_history + 1:
            return keep, quarantined

        close = cols["close"][indices]
        if not np.any(np.isfinite(close)) or np.nanmin(close) <= 0:
            return keep, quarantined

        keep = keep.copy()
        threshold = float(self.config.outlier_sigma)
        accepted: deque[float] = deque(maxlen=self.outlier_window)
        last_close = float(close[0])
        consecutive = 0

        for position in range(1, close.size):
            current = float(close[position])
            if not np.isfinite(current) or current <= 0 or last_close <= 0:
                last_close = current if np.isfinite(current) and current > 0 else last_close
                continue
            value = float(np.log(current / last_close))

            if consecutive == 0 and len(accepted) >= self.min_outlier_history:
                window = np.fromiter(accepted, dtype=np.float64, count=len(accepted))
                centre = float(np.median(window))
                mad = float(np.median(np.abs(window - centre)))
                if mad > 0.0 and abs(value - centre) > threshold * MAD_TO_SIGMA * mad:
                    row = int(indices[position])
                    keep[row] = False
                    counts["outliers_quarantined"] += 1
                    quarantined.append(from_ns(int(ts[row])))
                    consecutive = 1
                    # last_close deliberately unchanged: the next bar is judged
                    # against the last price we believe, not against the spike.
                    continue

            consecutive = 0
            accepted.append(value)
            last_close = current

        return keep, quarantined

        close = cols["close"][indices]
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(np.log(np.where(close > 0, close, np.nan)))
        if not np.any(np.isfinite(returns)):
            return keep, quarantined

        keep = keep.copy()
        threshold = float(self.config.outlier_sigma)
        for position in range(self.min_outlier_history, returns.size):
            window = returns[max(0, position - self.outlier_window):position]
            window = window[np.isfinite(window)]
            if window.size < self.min_outlier_history:
                continue
            centre = float(np.median(window))
            mad = float(np.median(np.abs(window - centre)))
            if mad <= 0.0:
                continue  # no dispersion to judge against; do not guess
            sigma = MAD_TO_SIGMA * mad
            value = returns[position]
            if not np.isfinite(value):
                continue
            if abs(value - centre) > threshold * sigma:
                row = int(indices[position + 1])
                keep[row] = False
                counts["outliers_quarantined"] += 1
                quarantined.append(from_ns(int(ts[row])))
        return keep, quarantined
