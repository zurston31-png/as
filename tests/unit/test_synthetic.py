"""The synthetic generator's ground truth has to be trustworthy.

Every later phase is tested against this data, so a defect here is not a
wrong number in one place -- it is a wrong number everywhere, arriving with
a passing test suite. Three groups of tests carry most of that weight:

* **Determinism**, including stream independence: adding a concern to the
  generator must not shift the draws of an existing one, or a dataset
  generated before the change silently stops reproducing.
* **No leaked future**: tick delta correlates with its own bar and not with
  the next one. If it leaked, every lookahead test in the repo would pass
  against data built to let it.
* **Label separability**: a Kaufman efficiency ratio must actually tell
  TRENDING from CHOP. If the generator's own labels are not recoverable
  from its own prices, scoring a detector against them measures nothing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from pydantic import ValidationError

from flow_model.core.determinism import DEFAULT_SEED
from flow_model.core.enums import InstrumentType, Regime, Session
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.base import DataLayerError, Feed
from flow_model.data.series import from_ns
from flow_model.data.store import DataStore
from flow_model.data.synthetic import (
    DEFAULT_REGIME_PARAMS,
    STREAMS,
    RegimeParams,
    SyntheticAdapter,
    SyntheticConfig,
    SyntheticDataset,
    SyntheticMarketGenerator,
)

NY = ZoneInfo("America/New_York")
INTERVAL = 300
RTH_BARS = 78


# --- local fixtures --------------------------------------------------------


@pytest.fixture
def nq_spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="NQ",
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
    )


@pytest.fixture
def sessionless_spec() -> InstrumentSpec:
    """No RTH window, so `bars_per_day` is the only session authority."""
    return InstrumentSpec(
        symbol="XX",
        instrument_type=InstrumentType.ETF,
        tick_size=0.01,
        tick_value=0.01,
    )


def _spec(**overrides) -> InstrumentSpec:
    base = dict(
        symbol="NQ",
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
    )
    base.update(overrides)
    return InstrumentSpec(**base)


def _generate(config: SyntheticConfig, seed: int = 11, **kwargs) -> SyntheticDataset:
    spec = kwargs.pop("spec", None) or _spec()
    start = kwargs.pop("start", date(2021, 1, 4))
    end = kwargs.pop("end", date(2021, 4, 1))
    symbol = kwargs.pop("symbol", "NQ")
    return SyntheticMarketGenerator(config, seed=seed).generate(
        symbol, spec, start, end, kwargs.pop("interval_seconds", INTERVAL), **kwargs
    )


@pytest.fixture(scope="module")
def long_dataset() -> SyntheticDataset:
    """Three years of bars: dwell and regime statistics need the sample."""
    return SyntheticMarketGenerator(SyntheticConfig(), seed=17).generate(
        "NQ", _spec(), date(2019, 1, 2), date(2022, 1, 1), INTERVAL
    )


@pytest.fixture(scope="module")
def year_dataset() -> SyntheticDataset:
    return SyntheticMarketGenerator(SyntheticConfig(), seed=5).generate(
        "NQ", _spec(), date(2021, 1, 4), date(2022, 1, 1), INTERVAL
    )


@pytest.fixture(scope="module")
def flat_dataset() -> SyntheticDataset:
    """One regime with zero persistence.

    This isolates the no-lookahead claim: with no return autocorrelation
    there is no honest channel by which this bar's delta could relate to the
    next bar's return, so any correlation found is a leak.
    """
    config = SyntheticConfig(
        regimes=(
            RegimeParams(
                regime=Regime.CHOP,
                drift_per_bar=0.0,
                vol_per_bar=0.001,
                mean_persistence=0.0,
                expected_bars=50.0,
            ),
        ),
        include_options=False,
    )
    return SyntheticMarketGenerator(config, seed=3).generate(
        "NQ", _spec(), date(2021, 1, 4), date(2022, 1, 1), INTERVAL
    )


# --- local helpers ---------------------------------------------------------


def _bar_returns(dataset: SyntheticDataset) -> np.ndarray:
    """Open-to-close log return of each bar."""
    return np.log(dataset.bars.col("close") / dataset.bars.col("open"))


def _delta_ratio(dataset: SyntheticDataset) -> np.ndarray:
    buys = dataset.ticks.col("buy_volume")
    sells = dataset.ticks.col("sell_volume")
    classified = buys + sells
    return np.where(classified > 0, (buys - sells) / np.maximum(classified, 1.0), 0.0)


def _efficiency_ratio(closes: np.ndarray, period: int) -> np.ndarray:
    """Kaufman efficiency ratio: net move over summed absolute moves.

    Defined here rather than imported so this test measures the generator
    and not Phase 3's implementation of the same formula.
    """
    out = np.full(len(closes), np.nan)
    steps = np.abs(np.diff(closes))
    for i in range(period, len(closes)):
        travelled = float(steps[i - period : i].sum())
        if travelled > 0:
            out[i] = abs(closes[i] - closes[i - period]) / travelled
    return out


def _pure_windows(labels: np.ndarray, period: int) -> np.ndarray:
    """Bars whose whole `period` lookback sat in one regime."""
    pure = np.zeros(len(labels), dtype=bool)
    for i in range(period, len(labels)):
        window = labels[i - period : i + 1]
        pure[i] = bool(np.all(window == window[0]))
    return pure


def _local_times(dataset: SyntheticDataset) -> list[datetime]:
    return [ts.astimezone(NY) for ts in dataset.bars.timestamps()]


class _StubCalendar:
    """Minimal SessionCalendarProtocol: one holiday, one half day."""

    def __init__(self, holiday: date | None = None, half_day: date | None = None) -> None:
        self.holiday = holiday
        self.half_day = half_day

    def is_trading_day(self, day: date, spec: InstrumentSpec) -> bool:
        return day.weekday() < 5 and day != self.holiday

    def is_half_day(self, day: date, spec: InstrumentSpec) -> bool:
        return day == self.half_day

    def session_of(self, ts: datetime, spec: InstrumentSpec) -> Session:
        return Session.RTH_MID

    def is_rth(self, ts: datetime, spec: InstrumentSpec) -> bool:
        return True


# --- regime parameters and config -----------------------------------------


def test_default_regimes_cover_every_tradable_regime():
    """A missing regime is a label the detector can never be scored on."""
    covered = {params.regime for params in DEFAULT_REGIME_PARAMS}
    assert covered == {regime for regime in Regime if regime is not Regime.UNKNOWN}


def test_regime_params_reject_unknown_regime():
    with pytest.raises(ValidationError, match="UNKNOWN"):
        RegimeParams(regime=Regime.UNKNOWN, vol_per_bar=0.001)


@pytest.mark.parametrize("phi", [1.0, -1.0, 1.5])
def test_regime_params_reject_divergent_persistence(phi):
    with pytest.raises(ValidationError):
        RegimeParams(regime=Regime.CHOP, vol_per_bar=0.001, mean_persistence=phi)


def test_regime_params_reject_expected_bars_of_one():
    """expected_bars == 1 means a regime that can never persist."""
    with pytest.raises(ValidationError):
        RegimeParams(regime=Regime.CHOP, vol_per_bar=0.001, expected_bars=1.0)


def test_regime_params_reject_zero_volatility():
    with pytest.raises(ValidationError):
        RegimeParams(regime=Regime.CHOP, vol_per_bar=0.0)


def test_config_rejects_duplicate_regime_params():
    duplicate = (
        RegimeParams(regime=Regime.CHOP, vol_per_bar=0.001),
        RegimeParams(regime=Regime.CHOP, vol_per_bar=0.002),
    )
    with pytest.raises(ValidationError, match="duplicate regime"):
        SyntheticConfig(regimes=duplicate)


def test_config_rejects_an_empty_regime_tuple():
    with pytest.raises(ValidationError, match="at least one regime"):
        SyntheticConfig(regimes=())


def test_config_is_frozen_and_forbids_unknown_fields():
    config = SyntheticConfig()
    with pytest.raises(ValidationError):
        SyntheticConfig(start_price=1.0, no_such_field=True)
    with pytest.raises(ValidationError):
        config.start_price = 1.0


def test_config_feeds_follow_the_include_flags():
    assert SyntheticConfig().feeds() == frozenset(Feed)
    bars_only = SyntheticConfig(
        include_quotes=False, include_ticks=False, include_options=False
    )
    assert bars_only.feeds() == frozenset({Feed.BARS})


def test_stream_names_are_unique():
    """Two concerns sharing a stream name would share draws."""
    assert len(set(STREAMS)) == len(STREAMS)


# --- the Markov chain ------------------------------------------------------


def test_transition_matrix_rows_sum_to_one():
    matrix = SyntheticMarketGenerator(SyntheticConfig()).transition_matrix()
    assert matrix.shape == (len(DEFAULT_REGIME_PARAMS),) * 2
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.all(matrix >= 0.0)


def test_self_transition_probability_matches_expected_bars():
    config = SyntheticConfig()
    matrix = SyntheticMarketGenerator(config).transition_matrix()
    for i, params in enumerate(config.regimes):
        assert matrix[i, i] == pytest.approx(1.0 - 1.0 / params.expected_bars)


def test_directional_shift_prefers_a_trending_successor():
    """A break resolving into chop is not what the term means."""
    config = SyntheticConfig()
    matrix = SyntheticMarketGenerator(config).transition_matrix()
    order = [params.regime for params in config.regimes]
    row = matrix[order.index(Regime.DIRECTIONAL_SHIFT)]
    trending = row[order.index(Regime.TRENDING_UP)] + row[order.index(Regime.TRENDING_DOWN)]
    others = row.sum() - row[order.index(Regime.DIRECTIONAL_SHIFT)] - trending
    assert trending > others


def test_single_regime_chain_never_leaves():
    config = SyntheticConfig(
        regimes=(RegimeParams(regime=Regime.LOW_VOL, vol_per_bar=0.001),)
    )
    generator = SyntheticMarketGenerator(config)
    assert generator.transition_matrix().tolist() == [[1.0]]
    assert np.all(generator.regime_path("NQ", 500) == 0)


def test_regime_path_is_reproducible_and_within_range():
    generator = SyntheticMarketGenerator(SyntheticConfig(), seed=4)
    first = generator.regime_path("NQ", 2_000)
    assert np.array_equal(first, generator.regime_path("NQ", 2_000))
    assert first.min() >= 0 and first.max() < len(DEFAULT_REGIME_PARAMS)
    assert len(np.unique(first)) > 1


def test_regime_path_of_zero_bars_is_empty():
    assert len(SyntheticMarketGenerator(SyntheticConfig()).regime_path("NQ", 0)) == 0


# --- determinism -----------------------------------------------------------


def _signature(dataset: SyntheticDataset) -> bytes:
    parts = [dataset.bars.ts_ns.tobytes(), dataset.regime_labels.tobytes()]
    for name in ("open", "high", "low", "close", "volume"):
        parts.append(dataset.bars.col(name).tobytes())
    for series, columns in (
        (dataset.ticks, ("buy_volume", "sell_volume", "unclassified_volume")),
        (dataset.quotes, ("bid", "ask", "bid_size", "ask_size")),
        (dataset.options, ("call_premium", "put_premium", "call_oi", "atm_iv")),
    ):
        if series is not None:
            parts.extend(series.col(name).tobytes() for name in columns)
    return b"".join(parts)


def test_same_seed_produces_byte_identical_arrays():
    config = SyntheticConfig()
    assert _signature(_generate(config, seed=7)) == _signature(_generate(config, seed=7))


def test_different_seed_produces_a_different_path():
    config = SyntheticConfig()
    assert _signature(_generate(config, seed=7)) != _signature(_generate(config, seed=8))


def test_different_symbol_produces_a_different_path():
    """Two symbols sharing one path would make every correlation study junk."""
    config = SyntheticConfig()
    a = _generate(config, seed=7, symbol="NQ")
    b = _generate(config, seed=7, symbol="ES", spec=_spec(symbol="ES"))
    assert len(a) == len(b)
    assert not np.array_equal(a.bars.col("close"), b.bars.col("close"))


def test_disabling_ticks_does_not_shift_the_bar_stream():
    config = SyntheticConfig()
    full = _generate(config, seed=9)
    without = _generate(config.replace(include_ticks=False), seed=9)
    assert without.ticks is None
    assert full.bars.col("close").tobytes() == without.bars.col("close").tobytes()
    assert full.bars.col("volume").tobytes() == without.bars.col("volume").tobytes()


def test_disabling_quotes_does_not_shift_the_tick_stream():
    config = SyntheticConfig()
    full = _generate(config, seed=9)
    without = _generate(config.replace(include_quotes=False), seed=9)
    assert without.quotes is None
    assert full.ticks.col("buy_volume").tobytes() == without.ticks.col("buy_volume").tobytes()


def test_disabling_options_does_not_shift_the_bar_stream():
    config = SyntheticConfig()
    full = _generate(config, seed=9)
    without = _generate(config.replace(include_options=False), seed=9)
    assert without.options is None
    assert full.bars.col("close").tobytes() == without.bars.col("close").tobytes()


def test_changing_the_tick_signal_does_not_shift_the_bar_stream():
    config = SyntheticConfig()
    strong = _generate(config.replace(delta_signal_strength=0.9), seed=9)
    weak = _generate(config.replace(delta_signal_strength=0.0), seed=9)
    assert strong.bars.col("close").tobytes() == weak.bars.col("close").tobytes()
    assert not np.array_equal(
        strong.ticks.col("buy_volume"), weak.ticks.col("buy_volume")
    )


def test_a_longer_range_extends_the_shorter_one():
    """Extending a backtest window must not rewrite its history."""
    config = SyntheticConfig()
    short = _generate(config, seed=9, end=date(2021, 3, 1))
    long = _generate(config, seed=9, end=date(2021, 6, 1))
    n = len(short)
    assert n < len(long)
    assert np.array_equal(long.bars.ts_ns[:n], short.bars.ts_ns)
    assert np.array_equal(long.bars.col("close")[:n], short.bars.col("close"))
    assert np.array_equal(long.regime_labels[:n], short.regime_labels)


# --- bars and timestamps ---------------------------------------------------


def test_ohlc_ordering_holds_over_a_long_run(long_dataset):
    bars = long_dataset.bars
    o, h, l, c = (bars.col(name) for name in ("open", "high", "low", "close"))
    assert len(bars) > 50_000
    assert np.all(l <= np.minimum(o, c))
    assert np.all(h >= np.maximum(o, c))
    assert np.all(h >= l)
    assert np.all(l > 0.0)
    assert np.all(np.isfinite(c))


def test_open_equals_the_previous_close(year_dataset):
    opens = year_dataset.bars.col("open")
    closes = year_dataset.bars.col("close")
    assert np.array_equal(opens[1:], closes[:-1])


def test_prices_sit_on_the_tick_grid(year_dataset):
    tick = 0.25
    for name in ("open", "high", "low", "close"):
        values = year_dataset.bars.col(name) / tick
        assert np.allclose(values, np.round(values), atol=1e-9)


def test_timestamps_are_strictly_ascending(long_dataset):
    assert np.all(np.diff(long_dataset.bars.ts_ns) > 0)


def test_timestamps_are_timezone_aware_utc(year_dataset):
    first = year_dataset.bars.first_ts
    assert first.tzinfo is not None
    assert first.utcoffset() == timedelta(0)


def test_no_bars_on_weekends(long_dataset):
    weekdays = {ts.weekday() for ts in _local_times(long_dataset)}
    assert weekdays <= {0, 1, 2, 3, 4}


def test_bars_fall_inside_the_rth_session(year_dataset):
    """A bar outside the session is data the exchange never produced."""
    minutes = {ts.hour * 60 + ts.minute for ts in _local_times(year_dataset)}
    assert min(minutes) == 9 * 60 + 35       # first close is one interval in
    assert max(minutes) == 16 * 60           # last close is the session close
    assert all(9 * 60 + 30 < m <= 16 * 60 for m in minutes)


def test_session_bar_count_comes_from_the_spec_window(year_dataset):
    local = _local_times(year_dataset)
    per_day: dict[date, int] = {}
    for ts in local:
        per_day[ts.date()] = per_day.get(ts.date(), 0) + 1
    assert set(per_day.values()) == {RTH_BARS}


def test_a_spec_with_no_rth_window_is_refused(sessionless_spec):
    """The generator will not invent a session the calendar reports CLOSED.

    With no declared RTH it used to assume a 09:30 open, while
    `calendar.session_of` returns Session.CLOSED for every timestamp of such
    an instrument -- so every generated bar was simultaneously a valid bar
    and outside any session. Two modules disagreeing about the same data is
    now impossible rather than resolved twice.
    """
    with pytest.raises(DataLayerError, match="no RTH window declared"):
        _generate(SyntheticConfig(bars_per_day=10), seed=2, spec=sessionless_spec,
                  symbol="XX", end=date(2021, 1, 16))


def test_interval_longer_than_the_session_is_rejected():
    with pytest.raises(DataLayerError, match="does not fit"):
        _generate(SyntheticConfig(), interval_seconds=86_400)


def test_nonpositive_interval_is_rejected():
    with pytest.raises(DataLayerError, match="interval_seconds"):
        _generate(SyntheticConfig(), interval_seconds=0)


def test_empty_date_range_is_rejected():
    with pytest.raises(DataLayerError, match="empty range"):
        _generate(SyntheticConfig(), start=date(2021, 1, 4), end=date(2021, 1, 4))


def test_a_range_with_no_sessions_is_rejected():
    with pytest.raises(DataLayerError, match="no trading sessions"):
        _generate(SyntheticConfig(), start=date(2021, 1, 9), end=date(2021, 1, 11))


def test_a_session_window_wrapping_midnight_is_rejected():
    spec = _spec(rth=SessionWindow(name="ETH", start="18:00", end="17:00"))
    with pytest.raises(DataLayerError, match="wraps"):
        _generate(SyntheticConfig(), spec=spec)


def test_holiday_calendar_removes_a_session():
    holiday = date(2021, 1, 18)
    calendar = _StubCalendar(holiday=holiday)
    with_cal = _generate(SyntheticConfig(), seed=4, end=date(2021, 2, 1), calendar=calendar)
    without = _generate(SyntheticConfig(), seed=4, end=date(2021, 2, 1))
    days = {ts.date() for ts in _local_times(with_cal)}
    assert holiday not in days
    assert holiday in {ts.date() for ts in _local_times(without)}
    assert len(with_cal) == len(without) - RTH_BARS


def test_half_day_calendar_shortens_a_session():
    half = date(2021, 1, 19)
    dataset = _generate(
        SyntheticConfig(), seed=4, end=date(2021, 2, 1),
        calendar=_StubCalendar(half_day=half),
    )
    local = _local_times(dataset)
    on_half = [ts for ts in local if ts.date() == half]
    assert len(on_half) == 42                      # 09:30 -> 13:00 at 5 minutes
    assert max(on_half).hour == 13 and max(on_half).minute == 0


def test_dst_change_does_not_shift_the_local_session():
    """Session times are exchange-local; UTC offsets move, the session does not."""
    dataset = _generate(
        SyntheticConfig(), seed=4, start=date(2021, 3, 8), end=date(2021, 3, 20)
    )
    local = _local_times(dataset)
    per_day: dict[date, list[datetime]] = {}
    for ts in local:
        per_day.setdefault(ts.date(), []).append(ts)
    assert len(per_day) == 10
    for day, stamps in per_day.items():
        assert min(stamps).strftime("%H:%M") == "09:35", day
        assert max(stamps).strftime("%H:%M") == "16:00", day
    offsets = {ts.utcoffset() for ts in local}
    assert len(offsets) == 2                       # the clock really did change


# --- volume ----------------------------------------------------------------


def test_volume_is_u_shaped_across_the_session(year_dataset):
    volume = year_dataset.bars.col("volume")
    position = np.arange(len(volume)) % RTH_BARS
    opening = volume[position < 8].mean()
    midday = volume[(position >= 35) & (position < 43)].mean()
    closing = volume[position >= 70].mean()
    assert opening > midday * 1.4
    assert closing > midday * 1.2
    assert opening > closing


def test_volume_is_never_negative(long_dataset):
    assert np.all(long_dataset.bars.col("volume") >= 0.0)


def test_regime_volume_multiplier_scales_volume():
    def mean_volume(multiplier: float) -> float:
        config = SyntheticConfig(
            regimes=(
                RegimeParams(
                    regime=Regime.HIGH_VOL,
                    vol_per_bar=0.001,
                    volume_multiplier=multiplier,
                ),
            ),
            include_ticks=False,
            include_quotes=False,
            include_options=False,
        )
        return float(_generate(config, seed=6).bars.col("volume").mean())

    assert mean_volume(2.0) == pytest.approx(2.0 * mean_volume(1.0), rel=0.02)


# --- tick aggregates -------------------------------------------------------


def test_tick_split_conserves_bar_volume_exactly(long_dataset):
    """Not approximately: an order-flow feature must not see volume the bar
    never traded, and a rounding residue would accumulate into CVD."""
    ticks = long_dataset.ticks
    total = (
        ticks.col("buy_volume") + ticks.col("sell_volume") + ticks.col("unclassified_volume")
    )
    assert np.array_equal(total, long_dataset.bars.col("volume"))


def test_tick_volumes_are_non_negative(long_dataset):
    for name in ("buy_volume", "sell_volume", "unclassified_volume"):
        assert np.all(long_dataset.ticks.col(name) >= 0.0)


def test_tick_classification_coverage_matches_config(year_dataset):
    assert year_dataset.ticks.coverage() == pytest.approx(0.95, abs=0.005)


def test_full_coverage_leaves_nothing_unclassified():
    dataset = _generate(SyntheticConfig(tick_classification_coverage=1.0), seed=8)
    assert np.all(dataset.ticks.col("unclassified_volume") == 0.0)
    assert dataset.ticks.coverage() == pytest.approx(1.0)


def test_partial_coverage_is_reported_not_hidden():
    dataset = _generate(SyntheticConfig(tick_classification_coverage=0.6), seed=8)
    assert dataset.ticks.coverage() == pytest.approx(0.6, abs=0.01)
    assert np.any(dataset.ticks.col("unclassified_volume") > 0.0)


def test_tick_delta_correlates_with_the_same_bar_return(year_dataset):
    correlation = float(np.corrcoef(_delta_ratio(year_dataset), _bar_returns(year_dataset))[0, 1])
    assert correlation > 0.15


def test_raw_tick_delta_also_correlates_with_the_same_bar_return(year_dataset):
    delta = year_dataset.ticks.col("buy_volume") - year_dataset.ticks.col("sell_volume")
    assert float(np.corrcoef(delta, _bar_returns(year_dataset))[0, 1]) > 0.15


def test_tick_delta_does_not_lead_the_next_bar(flat_dataset):
    """With zero persistence there is no honest channel to the next bar, so
    any correlation here is a leak and makes every lookahead test vacuous."""
    ratio = _delta_ratio(flat_dataset)
    returns = _bar_returns(flat_dataset)
    assert float(np.corrcoef(ratio, returns)[0, 1]) > 0.2        # the signal is there
    assert abs(float(np.corrcoef(ratio[:-1], returns[1:])[0, 1])) < 0.03


def test_tick_delta_next_bar_correlation_stays_small_with_mixed_regimes(year_dataset):
    """The residual is the regimes' own autocorrelation, bounded by
    delta_signal_strength * mean persistence -- not a hand-placed leak."""
    ratio = _delta_ratio(year_dataset)
    returns = _bar_returns(year_dataset)
    assert abs(float(np.corrcoef(ratio[:-1], returns[1:])[0, 1])) < 0.06


def test_zero_delta_signal_strength_is_pure_noise():
    config = SyntheticConfig(delta_signal_strength=0.0, include_options=False)
    dataset = _generate(config, seed=3, end=date(2022, 1, 1))
    correlation = float(np.corrcoef(_delta_ratio(dataset), _bar_returns(dataset))[0, 1])
    assert abs(correlation) < 0.03


def test_no_tick_series_when_ticks_are_excluded():
    """The absent feed is absent, not estimated from bar volume."""
    dataset = _generate(SyntheticConfig(include_ticks=False), seed=3)
    assert dataset.ticks is None
    assert Feed.TICK_AGGREGATE not in dataset.feeds()


def test_tick_series_declares_its_classification_method(year_dataset):
    assert year_dataset.ticks.classification_method == "synthetic_aggressor"


# --- quotes ----------------------------------------------------------------


def test_quote_ask_always_exceeds_bid(long_dataset):
    quotes = long_dataset.quotes
    assert np.all(quotes.col("ask") > quotes.col("bid"))
    assert quotes.crossed_count() == 0


def test_quote_mid_tracks_the_bar_close_within_one_tick(year_dataset):
    quotes = year_dataset.quotes
    mid = (quotes.col("bid") + quotes.col("ask")) / 2.0
    assert np.all(np.abs(mid - year_dataset.bars.col("close")) <= 0.25)


def test_one_quote_per_bar_at_the_bar_close(year_dataset):
    assert len(year_dataset.quotes) == len(year_dataset.bars)
    assert np.array_equal(year_dataset.quotes.ts_ns, year_dataset.bars.ts_ns)


def test_quote_spread_is_a_whole_number_of_ticks(year_dataset):
    spread = (year_dataset.quotes.col("ask") - year_dataset.quotes.col("bid")) / 0.25
    assert np.all(spread >= 1.0)
    assert np.allclose(spread, np.round(spread), atol=1e-9)


def test_quote_spread_widens_with_the_configured_mean():
    def mean_spread(ticks: float) -> float:
        dataset = _generate(SyntheticConfig(spread_ticks_mean=ticks), seed=12)
        return float((dataset.quotes.col("ask") - dataset.quotes.col("bid")).mean())

    assert mean_spread(4.0) > mean_spread(1.0) * 2.5


def test_quote_sizes_are_positive(year_dataset):
    for name in ("bid_size", "ask_size"):
        assert np.all(year_dataset.quotes.col(name) > 0.0)


def test_no_quote_series_when_quotes_are_excluded():
    dataset = _generate(SyntheticConfig(include_quotes=False), seed=3)
    assert dataset.quotes is None
    assert Feed.QUOTES not in dataset.feeds()


# --- options ---------------------------------------------------------------


def test_eod_options_are_one_per_trading_day_at_the_session_close(year_dataset):
    options = year_dataset.options
    sessions = {ts.astimezone(NY).date() for ts in year_dataset.bars.timestamps()}
    assert len(options) == len(sessions)
    local = [ts.astimezone(NY) for ts in options.timestamps()]
    assert {ts.strftime("%H:%M") for ts in local} == {"16:00"}
    assert {ts.date() for ts in local} == sessions


def test_eod_options_are_flagged_not_intraday(year_dataset):
    """EOD data expresses positioning but not flow timing; the quality layer
    caps the sub-score on this flag, so it must not lie."""
    assert year_dataset.options.is_intraday is False
    assert year_dataset.options.source == "synthetic_eod"
    assert year_dataset.options.snapshot_at(0).is_intraday is False


def test_intraday_options_are_one_per_bar():
    dataset = _generate(SyntheticConfig(options_intraday=True), seed=3)
    assert len(dataset.options) == len(dataset.bars)
    assert dataset.options.is_intraday is True
    assert dataset.options.source == "synthetic_intraday"
    assert np.array_equal(dataset.options.ts_ns, dataset.bars.ts_ns)


def test_options_oi_change_is_the_first_difference_of_oi(year_dataset):
    for level, change in (("call_oi", "call_oi_change"), ("put_oi", "put_oi_change")):
        values = year_dataset.options.col(level)
        assert np.allclose(year_dataset.options.col(change), np.diff(values, prepend=values[:1]))


def test_options_levels_are_non_negative_and_populated(year_dataset):
    options = year_dataset.options
    for name in ("call_volume", "put_volume", "call_premium", "put_premium",
                 "call_oi", "put_oi"):
        assert np.all(options.col(name) >= 0.0), name
        assert np.any(options.col(name) > 0.0), name
    snapshot = options.snapshot_at(3)
    assert snapshot.atm_iv is not None and snapshot.atm_iv > 0.0
    assert snapshot.iv_25d_call is not None and snapshot.iv_25d_call > 0.0
    assert snapshot.iv_25d_put is not None and snapshot.iv_25d_put > 0.0
    assert snapshot.underlying_price is not None
    assert snapshot.gamma_exposure_proxy is not None


def test_options_premium_imbalance_is_bounded_and_varies(year_dataset):
    options = year_dataset.options
    calls = options.col("call_premium")
    puts = options.col("put_premium")
    imbalance = (calls - puts) / (calls + puts)
    assert np.all(np.abs(imbalance) < 1.0)
    assert imbalance.std() > 0.05
    assert np.any(imbalance > 0) and np.any(imbalance < 0)


def test_options_skew_is_positive_on_average(year_dataset):
    """25-delta put IV above call IV: downside demand is the normal state."""
    options = year_dataset.options
    skew = options.col("iv_25d_put") - options.col("iv_25d_call")
    assert float(skew.mean()) > 0.0


def test_options_premium_imbalance_tracks_the_session_that_closed(year_dataset):
    """Built from the day already closed, so next-day predictability can only
    arrive through regime persistence."""
    options = year_dataset.options
    calls, puts = options.col("call_premium"), options.col("put_premium")
    imbalance = (calls - puts) / (calls + puts)
    closes = year_dataset.bars.col("close")
    anchors = np.searchsorted(year_dataset.bars.ts_ns, options.ts_ns)
    day_return = np.diff(np.log(closes[anchors]), prepend=0.0)
    assert float(np.corrcoef(imbalance[1:], day_return[1:])[0, 1]) > 0.1


def test_no_options_series_when_options_are_excluded():
    dataset = _generate(SyntheticConfig(include_options=False), seed=3)
    assert dataset.options is None
    assert Feed.OPTIONS_SNAPSHOT not in dataset.feeds()


# --- regime ground truth ---------------------------------------------------


def test_regime_dwell_times_match_expected_bars(long_dataset):
    expected = {p.regime.value: p.expected_bars for p in long_dataset.config.regimes}
    checked = 0
    for name, runs in long_dataset.dwell_bars().items():
        if len(runs) < 40:
            continue
        ratio = float(np.mean(runs)) / expected[name]
        assert 0.75 < ratio < 1.3, f"{name}: mean dwell {np.mean(runs):.1f}"
        checked += 1
    assert checked == len(expected)


def test_efficiency_ratio_separates_trending_from_chop(long_dataset):
    """If the generator's own labels are not recoverable from its own prices,
    scoring a detector against them measures nothing."""
    period = 20
    closes = long_dataset.bars.col("close")
    ratios = _efficiency_ratio(closes, period)
    pure = _pure_windows(long_dataset.regime_labels, period)
    names = long_dataset.regime_names

    def mean_ratio(regime: Regime) -> float:
        code = names.index(regime.value)
        mask = pure & (long_dataset.regime_labels == code) & ~np.isnan(ratios)
        assert mask.sum() > 200, regime
        return float(np.mean(ratios[mask]))

    chop = mean_ratio(Regime.CHOP)
    assert mean_ratio(Regime.TRENDING_UP) > chop * 1.5
    assert mean_ratio(Regime.TRENDING_DOWN) > chop * 1.5


def test_trending_regimes_carry_their_signed_drift(long_dataset):
    returns = np.diff(np.log(long_dataset.bars.col("close")))
    labels = long_dataset.regime_labels[1:]
    names = long_dataset.regime_names
    up = returns[labels == names.index(Regime.TRENDING_UP.value)]
    down = returns[labels == names.index(Regime.TRENDING_DOWN.value)]
    assert up.mean() > 0.0
    assert down.mean() < 0.0


def test_realized_volatility_matches_the_regime_parameter(long_dataset):
    """vol_per_bar means the stdev of the log return, including under
    persistence -- otherwise a trending regime would also read as high-vol
    and the two labels could not be told apart."""
    returns = np.diff(np.log(long_dataset.bars.col("close")))
    labels = long_dataset.regime_labels[1:]
    for code, params in enumerate(long_dataset.config.regimes):
        sample = returns[labels == code]
        if len(sample) < 500:
            continue
        assert float(np.std(sample)) == pytest.approx(params.vol_per_bar, rel=0.12)


def test_directional_shift_episodes_take_both_directions(long_dataset):
    """A fixed sign would make every shift in every backtest a long."""
    names = long_dataset.regime_names
    code = names.index(Regime.DIRECTIONAL_SHIFT.value)
    labels = long_dataset.regime_labels
    closes = long_dataset.bars.col("close")
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(labels)]))
    moves = [
        float(np.log(closes[stop - 1] / closes[start]))
        for start, stop in zip(starts.tolist(), stops.tolist())
        if labels[start] == code and stop - start > 3
    ]
    assert len(moves) > 50
    assert sum(1 for m in moves if m > 0) > 10
    assert sum(1 for m in moves if m < 0) > 10


def test_regime_at_and_label_series_agree(year_dataset):
    labels = year_dataset.label_series()
    assert len(labels) == len(year_dataset.bars)
    assert all(isinstance(label, Regime) for label in labels[:50])
    for index in (0, 17, len(year_dataset) - 1):
        assert year_dataset.regime_at(index) is labels[index]


def test_regime_at_rejects_an_out_of_range_index(year_dataset):
    with pytest.raises(IndexError):
        year_dataset.regime_at(len(year_dataset))
    with pytest.raises(IndexError):
        year_dataset.regime_at(-1)


def test_regime_counts_sum_to_the_bar_count(year_dataset):
    assert sum(year_dataset.regime_counts().values()) == len(year_dataset.bars)


def test_dataset_rejects_misaligned_labels(year_dataset):
    with pytest.raises(DataLayerError, match="regime labels"):
        SyntheticDataset(
            symbol="NQ",
            interval_seconds=INTERVAL,
            bars=year_dataset.bars,
            regime_labels=year_dataset.regime_labels[:-1],
            regime_names=year_dataset.regime_names,
        )


def test_describe_marks_the_data_synthetic_and_disclaims_edge(year_dataset):
    described = year_dataset.describe()
    assert described["synthetic"] is True
    assert described["source"] == "synthetic"
    assert "edge" in str(described["edge_claim"])
    assert described["bars"] == len(year_dataset.bars)
    assert described["options_intraday"] is False
    assert described["feeds"] == sorted(f.value for f in year_dataset.feeds())
    assert described["config_hash"]


# --- gaps and outliers -----------------------------------------------------


def test_gap_injection_removes_bars_and_leaves_survivors_untouched():
    """Bars are dropped, never filled: a fabricated bar is fabricated data."""
    config = SyntheticConfig()
    clean = _generate(config, seed=9)
    gapped = _generate(config.replace(gap_probability=0.05), seed=9)
    assert 0 < len(gapped) < len(clean)
    assert len(gapped.dropped_timestamps) == len(clean) - len(gapped)

    kept = np.isin(clean.bars.ts_ns, gapped.bars.ts_ns)
    assert np.array_equal(gapped.bars.ts_ns, clean.bars.ts_ns[kept])
    assert np.array_equal(gapped.bars.col("close"), clean.bars.col("close")[kept])
    assert np.array_equal(gapped.regime_labels, clean.regime_labels[kept])


def test_gap_injection_keeps_every_per_bar_feed_aligned():
    gapped = _generate(SyntheticConfig(gap_probability=0.08), seed=9)
    assert len(gapped.ticks) == len(gapped.bars)
    assert len(gapped.quotes) == len(gapped.bars)
    assert np.array_equal(gapped.ticks.ts_ns, gapped.bars.ts_ns)
    assert np.array_equal(gapped.quotes.ts_ns, gapped.bars.ts_ns)
    total = (
        gapped.ticks.col("buy_volume")
        + gapped.ticks.col("sell_volume")
        + gapped.ticks.col("unclassified_volume")
    )
    assert np.array_equal(total, gapped.bars.col("volume"))


def test_gap_injection_creates_missing_session_timestamps():
    gapped = _generate(SyntheticConfig(gap_probability=0.08), seed=9)
    deltas = np.diff(gapped.bars.ts_ns)
    assert np.any(deltas == 2 * INTERVAL * 1_000_000_000)
    assert all(ts.tzinfo is not None for ts in gapped.dropped_timestamps)


def test_no_bars_dropped_when_gap_probability_is_zero():
    dataset = _generate(SyntheticConfig(), seed=9)
    assert dataset.dropped_timestamps == ()
    sessions = len({ts.astimezone(NY).date() for ts in dataset.bars.timestamps()})
    assert len(dataset) == sessions * RTH_BARS


def test_outlier_injection_creates_extreme_returns():
    config = SyntheticConfig()
    clean = _generate(config, seed=9)
    spiked = _generate(config.replace(outlier_probability=0.01), seed=9)
    baseline = float(np.std(np.diff(np.log(clean.bars.col("close")))))
    assert len(spiked.outlier_indices) > 10
    for index in spiked.outlier_indices:
        move = abs(
            float(
                np.log(spiked.bars.col("close")[index] / spiked.bars.col("open")[index])
            )
        )
        assert move > 3.0 * baseline


def test_outlier_prints_revert_instead_of_shifting_the_series():
    """A spike that propagated would be a level change, a different defect."""
    config = SyntheticConfig(outlier_probability=0.01)
    spiked = _generate(config, seed=9)
    opens = spiked.bars.col("open")
    closes = spiked.bars.col("close")
    interior = [i for i in spiked.outlier_indices if i + 1 < len(spiked)]
    assert interior
    assert all(opens[i + 1] != closes[i] for i in interior)


def test_no_outliers_when_probability_is_zero(year_dataset):
    assert year_dataset.outlier_indices == ()


def test_outlier_bars_are_still_structurally_valid():
    spiked = _generate(SyntheticConfig(outlier_probability=0.01), seed=9)
    for index in spiked.outlier_indices:
        assert spiked.bars.bar_at(index).is_valid


def test_gaps_and_outliers_together_report_consistent_truth():
    dataset = _generate(
        SyntheticConfig(gap_probability=0.03, outlier_probability=0.01), seed=9
    )
    assert dataset.outlier_indices
    assert all(0 <= index < len(dataset) for index in dataset.outlier_indices)
    assert dataset.describe()["bars_dropped"] == len(dataset.dropped_timestamps)
    assert dataset.describe()["outliers_injected"] == len(dataset.outlier_indices)


# --- adapter ---------------------------------------------------------------


def _adapter(spec: InstrumentSpec, **kwargs) -> SyntheticAdapter:
    config = kwargs.pop("config", SyntheticConfig())
    return SyntheticAdapter(
        config, seed=kwargs.pop("seed", 4), spec_by_symbol={spec.symbol: spec}, **kwargs
    )


def test_adapter_feeds_default_to_the_config(nq_spec):
    assert _adapter(nq_spec).available_feeds("NQ") == frozenset(Feed)
    bars_only_config = SyntheticConfig(
        include_quotes=False, include_ticks=False, include_options=False
    )
    assert _adapter(nq_spec, config=bars_only_config).available_feeds("NQ") == frozenset(
        {Feed.BARS}
    )


def test_adapter_restricted_to_bars_returns_none_for_other_feeds(nq_spec):
    adapter = _adapter(nq_spec, feeds=frozenset({Feed.BARS}))
    start, end = date(2021, 1, 4), date(2021, 2, 1)
    assert adapter.available_feeds("NQ") == frozenset({Feed.BARS})
    assert adapter.load_quotes("NQ", start, end) is None
    assert adapter.load_tick_aggregates("NQ", INTERVAL, start, end) is None
    assert adapter.load_options("NQ", start, end) is None
    assert len(adapter.load_bars("NQ", INTERVAL, start, end)) > 0


def test_adapter_cannot_report_a_feed_the_config_never_generates(nq_spec):
    """Requesting ticks from a tick-free config must not conjure them."""
    config = SyntheticConfig(include_ticks=False)
    adapter = _adapter(nq_spec, config=config, feeds=frozenset(Feed))
    assert Feed.TICK_AGGREGATE not in adapter.available_feeds("NQ")
    assert adapter.load_tick_aggregates("NQ", INTERVAL, date(2021, 1, 4), date(2021, 2, 1)) is None


def test_adapter_rejects_an_unknown_symbol(nq_spec):
    with pytest.raises(DataLayerError, match="no InstrumentSpec"):
        _adapter(nq_spec).available_feeds("ES")


def test_adapter_rejects_a_foreign_interval(nq_spec):
    """A second independent path would give one symbol two histories."""
    with pytest.raises(DataLayerError, match="generates 300s bars"):
        _adapter(nq_spec).load_bars("NQ", 900, date(2021, 1, 4), date(2021, 2, 1))


def test_adapter_caches_without_changing_output(nq_spec):
    adapter = _adapter(nq_spec)
    start, end = date(2021, 1, 4), date(2021, 2, 1)
    first = adapter.load_bars("NQ", INTERVAL, start, end)
    second = adapter.load_bars("NQ", INTERVAL, start, end)
    assert first is second
    fresh = _adapter(nq_spec).load_bars("NQ", INTERVAL, start, end)
    assert first.col("close").tobytes() == fresh.col("close").tobytes()


def test_adapter_fingerprint_depends_on_seed_and_config(nq_spec):
    """The base fingerprint hashes only the range, which would attribute two
    different synthetic datasets to the same data_hash."""
    start, end = date(2021, 1, 4), date(2021, 2, 1)
    base = _adapter(nq_spec).fingerprint("NQ", INTERVAL, start, end, 100)
    other_seed = _adapter(nq_spec, seed=5).fingerprint("NQ", INTERVAL, start, end, 100)
    other_config = _adapter(
        nq_spec, config=SyntheticConfig(start_price=5000.0)
    ).fingerprint("NQ", INTERVAL, start, end, 100)
    assert base.source == "synthetic"
    assert base.data_hash != other_seed.data_hash
    assert base.data_hash != other_config.data_hash
    assert base.data_hash == _adapter(nq_spec).fingerprint(
        "NQ", INTERVAL, start, end, 100
    ).data_hash


def test_adapter_serves_datastore_with_every_feed(nq_spec):
    store = DataStore()
    data = store.load(_adapter(nq_spec), "NQ", date(2021, 1, 4), date(2021, 2, 1), INTERVAL)
    assert data.available_feeds() == frozenset(Feed)
    assert len(data.primary_bars) == 20 * RTH_BARS
    assert len(data.ticks[INTERVAL]) == len(data.primary_bars)
    assert data.fingerprint is not None and data.fingerprint.data_hash


def test_datastore_bars_only_adapter_loads_no_other_feed(nq_spec):
    adapter = _adapter(nq_spec, feeds=frozenset({Feed.BARS}))
    data = DataStore().load(adapter, "NQ", date(2021, 1, 4), date(2021, 2, 1), INTERVAL)
    assert data.quotes is None
    assert data.options is None
    assert data.ticks == {}
    assert data.available_feeds() == frozenset({Feed.BARS})


def test_datastore_skips_higher_intervals_rather_than_faking_them(nq_spec):
    data = DataStore().load(
        _adapter(nq_spec),
        "NQ",
        date(2021, 1, 4),
        date(2021, 2, 1),
        INTERVAL,
        higher_intervals=(900, 3600),
    )
    assert sorted(data.bars) == [INTERVAL]


def test_market_view_sees_a_sane_prefix_and_no_future(nq_spec):
    store = DataStore()
    store.load(_adapter(nq_spec), "NQ", date(2021, 1, 4), date(2021, 2, 1), INTERVAL)
    timeline = store.timeline("NQ")
    index = 100
    view = store.view("NQ", from_ns(int(timeline[index])))
    view.assert_no_future_leak()

    assert view.bar_count() == index + 1
    assert view.available_feeds() == frozenset(Feed)
    assert view.last_bar().close_ts == from_ns(int(timeline[index]))
    assert view.closes(5).size == 5
    assert view.quote() is not None and view.quote().ask > view.quote().bid
    assert view.last_tick_aggregate() is not None
    assert view.options_snapshot() is not None
    assert view.options_are_intraday() is False
    assert view.options_snapshot().ts <= view.cutoff


def test_market_view_at_the_first_bar_has_no_history(nq_spec):
    store = DataStore()
    store.load(_adapter(nq_spec), "NQ", date(2021, 1, 4), date(2021, 2, 1), INTERVAL)
    timeline = store.timeline("NQ")
    view = store.view("NQ", from_ns(int(timeline[0])))
    view.assert_no_future_leak()
    assert view.bar_count() == 1
    assert not view.warmup_ok(2)


def test_every_view_over_the_whole_timeline_is_leak_free(nq_spec):
    """The firewall is checked bar by bar, not just at a sampled index."""
    store = DataStore()
    store.load(_adapter(nq_spec), "NQ", date(2021, 1, 4), date(2021, 1, 15), INTERVAL)
    count = 0
    for view in store.iter_views("NQ"):
        view.assert_no_future_leak()
        count += 1
    assert count == 9 * RTH_BARS          # Mon 4th through Thu 14th, end-exclusive


def test_adapter_dataset_exposes_the_ground_truth_labels(nq_spec):
    adapter = _adapter(nq_spec)
    dataset = adapter.dataset("NQ", date(2021, 1, 4), date(2021, 2, 1))
    assert len(dataset.regime_labels) == len(dataset.bars)
    assert dataset.seed == 4
    assert dataset.regime_at(0) in set(Regime)


def test_generator_default_seed_is_the_project_seed():
    generator = SyntheticMarketGenerator(SyntheticConfig())
    assert generator.seed == DEFAULT_SEED


def test_unregistered_stream_name_is_rejected():
    """A typo'd stream name would silently share draws with another concern."""
    generator = SyntheticMarketGenerator(SyntheticConfig())
    with pytest.raises(ValueError, match="unregistered stream"):
        generator._stream("NQ", "not_a_stream")
