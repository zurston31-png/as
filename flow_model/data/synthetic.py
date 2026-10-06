"""Deterministic synthetic market generator.

Everything this module emits is MADE UP. It exists to exercise plumbing and
to let estimators be scored against known truth. No performance number
measured on this data says anything whatsoever about real-market edge: a
backtest that looks excellent here has demonstrated that the code runs, and
nothing more. Every report built from a synthetic dataset carries the
source tag `synthetic` for exactly that reason.

What it is actually for: the regime sequence is drawn FIRST, from a Markov
chain, and bars are drawn conditioned on it. Every bar therefore ships with
the label that produced it, so Phase 3's regime detector can be scored
against truth instead of against somebody's reading of a chart.

Two properties are load-bearing, and both are tested:

1. **No fabricated feeds.** With `include_ticks=False` the dataset carries
   `ticks=None`; nothing here derives a delta from bar volume. The degraded
   cases this module produces on purpose -- EOD-only options, partial tick
   classification, dropped bars, outlier prints -- are the cases the quality
   and cleaning layers exist to find, so they are produced honestly and
   their ground truth is returned alongside.

2. **No leaked future.** Tick delta is correlated with the *same* bar's
   return innovation and with nothing later. Synthetic data that leaked
   tomorrow into today would make every lookahead test in the repo pass
   vacuously, which is the one failure mode that would be invisible from
   above.

The residual next-bar correlation is not exactly zero and cannot be: in a
regime with positive return persistence, consecutive returns are
correlated, so a quantity correlated with this bar's return is necessarily
correlated with the next bar's return by roughly `delta_signal_strength *
mean_persistence`. That is the regime's own autocorrelation showing
through -- a property real data has too -- and it is bounded by
construction rather than hand-placed. The same reasoning governs the
options premium imbalance: it is computed from the session that has already
closed, and whatever next-day predictability it carries arrives through
regime persistence alone.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import Field, model_validator

from flow_model.core.determinism import DEFAULT_SEED, rng, stable_hash
from flow_model.core.enums import Regime
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel
from flow_model.data.base import (
    DataLayerError,
    DatasetFingerprint,
    DataSourceAdapter,
    Feed,
    SessionCalendarProtocol,
)
from flow_model.data.series import (
    NS_PER_SECOND,
    BarSeries,
    OptionsSeries,
    QuoteSeries,
    TickSeries,
    to_ns,
)

# Named randomness streams. Distinct names are not cosmetic: `rng()` derives
# an independent generator per name, so adding a concern here cannot shift
# the draws of an existing one and a dataset generated last month stays
# reproducible after this module grows.
STREAMS: tuple[str, ...] = (
    "regime",
    "shift_sign",
    "returns",
    "ranges",
    "volume",
    "ticks",
    "quotes",
    "options",
    "gaps",
    "outliers",
)

SOURCE_NAME = "synthetic"

# Session geometry used only when an InstrumentSpec declares no RTH window.
DEFAULT_SESSION_START_MINUTES = 9 * 60 + 30
HALF_DAY_CLOSE_MINUTES = 13 * 60

# Intraday volume curve: volume = base * u(x), x = position in session in
# [0, 1]. Two decaying exponentials rather than a parabola because the real
# shape is asymmetric -- the open is busier than the close, and the midday
# trough is flat rather than pointed.
U_OPEN_WEIGHT = 1.6
U_OPEN_DECAY = 0.12
U_CLOSE_WEIGHT = 0.9
U_CLOSE_DECAY = 0.18

# Gain on the tanh that maps the tick imbalance signal into (-1, 1). Below
# ~1.0 the map is near-linear, so `delta_signal_strength` survives it as
# very nearly the realized correlation.
TICK_IMBALANCE_GAIN = 0.8
TICK_AVG_TRADE_SIZE = 4.0
TICK_MAX_TRADE_SHARE = 0.04

QUOTE_SPREAD_LOG_SIGMA = 0.30
QUOTE_SIZE_LOG_SIGMA = 0.55
QUOTE_BASE_SIZE = 40.0

# How much of the options premium imbalance is explained by the session that
# has already closed. The rest is noise. Nothing here reads a future bar.
OPTIONS_SIGNAL_RHO = 0.35
OPTIONS_IMBALANCE_GAIN = 0.7
OPTIONS_VOLUME_SHARE_GAIN = 0.6
OPTIONS_OI_LOG_SIGMA = 0.04
OPTIONS_IV_LOG_SIGMA = 0.12
OPTIONS_SKEW_BASE = 0.12
OPTIONS_SKEW_NOISE = 0.04
OPTIONS_SKEW_IMBALANCE = 0.10
OPTIONS_IV_FLOOR = 0.03
OPTIONS_IV_CEILING = 2.5
GAMMA_PROXY_SCALE = 1.0e-4

# Successor preference for the Markov chain. A regime not listed here
# spreads its exit probability uniformly over the others. DIRECTIONAL_SHIFT
# is listed because a break in the mean that resolves into chop is not what
# the term means: a shift is followed by a trend or it was not a shift.
TRANSITION_PREFERENCE: Mapping[Regime, Mapping[Regime, float]] = {
    Regime.DIRECTIONAL_SHIFT: {
        Regime.TRENDING_UP: 4.0,
        Regime.TRENDING_DOWN: 4.0,
    },
}


class RegimeParams(FrozenModel):
    """Per-regime generating parameters.

    These are the ground truth a detector is scored against, so each one is
    a named quantity with one meaning and no hidden interaction.
    """

    regime: Regime
    drift_per_bar: float = Field(
        default=0.0,
        description=(
            "Mean log return per bar. For DIRECTIONAL_SHIFT this is a "
            "MAGNITUDE: the sign is drawn once per episode, because a break "
            "in the mean has a direction but no fixed one, and a fixed sign "
            "would make every shift in a backtest a long."
        ),
    )
    vol_per_bar: float = Field(
        gt=0.0,
        description="Stdev of the log return. Holds exactly: the AR(1) "
        "innovation is rescaled by sqrt(1 - phi^2) so persistence changes "
        "the shape of the path without changing its volatility.",
    )
    mean_persistence: float = Field(
        default=0.0,
        gt=-1.0,
        lt=1.0,
        description="AR(1) coefficient on the return deviation. > 0 trends, "
        "< 0 mean-reverts. |phi| < 1 or the path diverges.",
    )
    volume_multiplier: float = Field(gt=0.0, default=1.0)
    expected_bars: float = Field(
        gt=1.0,
        default=40.0,
        description="Mean dwell time in bars; sets the self-transition "
        "probability to 1 - 1/expected_bars. Must exceed 1, or the regime "
        "could never persist for two consecutive bars.",
    )

    @model_validator(mode="after")
    def _check(self) -> "RegimeParams":
        if self.regime is Regime.UNKNOWN:
            raise ValueError(
                "Regime.UNKNOWN means 'insufficient warmup' and is never "
                "tradable; generating bars labelled UNKNOWN would score a "
                "detector against a label it is not allowed to emit."
            )
        return self


DEFAULT_REGIME_PARAMS: tuple[RegimeParams, ...] = (
    RegimeParams(
        regime=Regime.LOW_VOL,
        drift_per_bar=0.00001,
        vol_per_bar=0.0004,
        mean_persistence=0.05,
        volume_multiplier=0.8,
        expected_bars=60.0,
    ),
    RegimeParams(
        regime=Regime.HIGH_VOL,
        drift_per_bar=0.0,
        vol_per_bar=0.0022,
        mean_persistence=0.0,
        volume_multiplier=1.8,
        expected_bars=40.0,
    ),
    RegimeParams(
        regime=Regime.TRENDING_UP,
        drift_per_bar=0.00014,
        vol_per_bar=0.0009,
        mean_persistence=0.35,
        volume_multiplier=1.2,
        expected_bars=50.0,
    ),
    RegimeParams(
        regime=Regime.TRENDING_DOWN,
        drift_per_bar=-0.00014,
        vol_per_bar=0.0011,
        mean_persistence=0.35,
        volume_multiplier=1.3,
        expected_bars=45.0,
    ),
    RegimeParams(
        regime=Regime.CHOP,
        drift_per_bar=0.0,
        vol_per_bar=0.0007,
        mean_persistence=-0.35,
        volume_multiplier=0.9,
        expected_bars=55.0,
    ),
    RegimeParams(
        regime=Regime.DIRECTIONAL_SHIFT,
        drift_per_bar=0.0005,
        vol_per_bar=0.0018,
        mean_persistence=0.25,
        volume_multiplier=2.2,
        expected_bars=6.0,
    ),
)


class SyntheticConfig(FrozenModel):
    """What to generate. Defaults describe a liquid index future."""

    start_price: float = Field(gt=0.0, default=18000.0)
    bars_per_day: int = Field(
        gt=0,
        default=78,
        description=(
            "Bars per session, used ONLY when the InstrumentSpec declares no "
            "RTH window. When it does, the window and the interval decide, "
            "because two sources of truth for the session length is how bars "
            "end up outside the session."
        ),
    )
    regimes: tuple[RegimeParams, ...] = DEFAULT_REGIME_PARAMS

    include_quotes: bool = True
    include_ticks: bool = True
    include_options: bool = True
    options_intraday: bool = Field(
        default=False,
        description="False emits one EOD chain snapshot per session, which is "
        "the degraded case real research actually faces.",
    )

    tick_classification_coverage: float = Field(ge=0.0, le=1.0, default=0.95)
    delta_signal_strength: float = Field(
        ge=0.0,
        le=1.0,
        default=0.3,
        description="Correlation between tick delta and the SAME bar's return "
        "innovation. 0 makes delta pure noise, which is the right control for "
        "an order-flow feature test.",
    )
    spread_ticks_mean: float = Field(gt=0.0, default=1.0)

    gap_probability: float = Field(
        ge=0.0,
        le=0.5,
        default=0.0,
        description="Probability of dropping an interior bar. Bounded at 0.5: "
        "beyond that a session is more hole than data and 'gap detection' "
        "stops meaning anything.",
    )
    outlier_probability: float = Field(
        ge=0.0,
        le=0.1,
        default=0.0,
        description="Probability of a bad-print spike on a bar.",
    )

    base_volume: float = Field(gt=0.0, default=2500.0)
    volume_noise_sigma: float = Field(ge=0.0, default=0.35)
    bar_range_factor: float = Field(
        ge=0.0,
        default=1.2,
        description="Scales the high/low extension beyond the open-close body, "
        "in units of the regime's per-bar volatility.",
    )
    outlier_sigma_multiple: float = Field(gt=0.0, default=12.0)

    base_atm_iv: float = Field(gt=0.0, default=0.18)
    base_options_volume: float = Field(gt=0.0, default=50_000.0)
    base_option_premium: float = Field(gt=0.0, default=4.0e6)
    base_open_interest: float = Field(gt=0.0, default=250_000.0)

    @model_validator(mode="after")
    def _check(self) -> "SyntheticConfig":
        if not self.regimes:
            raise ValueError("at least one regime is required to generate bars")
        seen = [params.regime for params in self.regimes]
        duplicates = sorted({r.value for r in seen if seen.count(r) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate regime parameters for {duplicates}: a regime with two "
                "parameter sets has no ground truth"
            )
        return self

    @property
    def regime_names(self) -> tuple[str, ...]:
        """Label for each integer code, in code order."""
        return tuple(params.regime.value for params in self.regimes)

    def feeds(self) -> frozenset[Feed]:
        """Feeds this config actually produces."""
        feeds = {Feed.BARS}
        if self.include_quotes:
            feeds.add(Feed.QUOTES)
        if self.include_ticks:
            feeds.add(Feed.TICK_AGGREGATE)
        if self.include_options:
            feeds.add(Feed.OPTIONS_SNAPSHOT)
        return frozenset(feeds)


# ---------------------------------------------------------------------------
# tick-grid helpers
# ---------------------------------------------------------------------------


def _round_to_tick(values: np.ndarray, tick: float) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64) / tick) * tick


def _ceil_to_tick(values: np.ndarray, tick: float) -> np.ndarray:
    # The inner round absorbs float noise, so a price already on the grid is
    # not pushed up a whole tick by a 1e-16 residue.
    return np.ceil(np.round(np.asarray(values, dtype=np.float64) / tick, 9)) * tick


def _floor_to_tick(values: np.ndarray, tick: float) -> np.ndarray:
    return np.floor(np.round(np.asarray(values, dtype=np.float64) / tick, 9)) * tick


def _lognormal_unit_mean(stream: np.random.Generator, sigma: float, size: int) -> np.ndarray:
    """Lognormal multiplicative noise with mean 1.

    Centering matters: an uncentered lognormal would inflate mean volume by
    exp(sigma^2 / 2), so `base_volume` would not mean what it says.
    """
    if sigma <= 0.0:
        return np.ones(size, dtype=np.float64)
    return np.exp(stream.normal(0.0, sigma, size=size) - 0.5 * sigma * sigma)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class SyntheticDataset:
    """Generated series plus the ground truth that produced them.

    `regime_labels` holds integer codes rather than strings: the codes are
    what a determinism test can compare byte-for-byte and what a confusion
    matrix indexes. `label_series()` is the readable form.

    `outlier_indices` and `dropped_timestamps` are ground truth for the
    cleaning layer -- a cleaner is scored against what was actually injected
    rather than against what it happens to find.
    """

    __slots__ = (
        "symbol",
        "interval_seconds",
        "bars",
        "quotes",
        "ticks",
        "options",
        "regime_labels",
        "regime_names",
        "outlier_indices",
        "dropped_timestamps",
        "config",
        "seed",
    )

    def __init__(
        self,
        symbol: str,
        interval_seconds: int,
        bars: BarSeries,
        regime_labels: np.ndarray,
        regime_names: tuple[str, ...],
        quotes: QuoteSeries | None = None,
        ticks: TickSeries | None = None,
        options: OptionsSeries | None = None,
        outlier_indices: tuple[int, ...] = (),
        dropped_timestamps: tuple[datetime, ...] = (),
        config: SyntheticConfig | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if len(regime_labels) != len(bars):
            raise DataLayerError(
                f"{symbol}: {len(regime_labels)} regime labels for {len(bars)} bars; "
                "a label array that does not align with the bars would score a "
                "detector against the wrong bar"
            )
        self.symbol = symbol
        self.interval_seconds = int(interval_seconds)
        self.bars = bars
        self.quotes = quotes
        self.ticks = ticks
        self.options = options
        self.regime_labels = regime_labels
        self.regime_names = regime_names
        self.outlier_indices = outlier_indices
        self.dropped_timestamps = dropped_timestamps
        self.config = config
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.bars)

    def regime_at(self, index: int) -> Regime:
        """Ground-truth regime of bar `index`."""
        if not 0 <= index < len(self.regime_labels):
            raise IndexError(
                f"regime index {index} out of range for {len(self.regime_labels)} bars"
            )
        return Regime(self.regime_names[int(self.regime_labels[index])])

    def label_series(self) -> np.ndarray:
        """One `Regime` per bar, as an object array."""
        lookup = np.array([Regime(name) for name in self.regime_names], dtype=object)
        if not len(self.regime_labels):
            return np.empty(0, dtype=object)
        return lookup[self.regime_labels]

    def regime_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.regime_names}
        if len(self.regime_labels):
            codes, totals = np.unique(self.regime_labels, return_counts=True)
            for code, total in zip(codes.tolist(), totals.tolist()):
                counts[self.regime_names[int(code)]] = int(total)
        return counts

    def dwell_bars(self) -> dict[str, list[int]]:
        """Observed run lengths per regime, for scoring against expected_bars."""
        runs: dict[str, list[int]] = {name: [] for name in self.regime_names}
        codes = self.regime_labels
        if not len(codes):
            return runs
        boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(codes)]))
        for start, stop in zip(starts.tolist(), stops.tolist()):
            runs[self.regime_names[int(codes[start])]].append(int(stop - start))
        return runs

    def feeds(self) -> frozenset[Feed]:
        feeds = {Feed.BARS} if len(self.bars) else set()
        if self.quotes is not None and len(self.quotes):
            feeds.add(Feed.QUOTES)
        if self.ticks is not None and len(self.ticks):
            feeds.add(Feed.TICK_AGGREGATE)
        if self.options is not None and len(self.options):
            feeds.add(Feed.OPTIONS_SNAPSHOT)
        return frozenset(feeds)

    def describe(self) -> dict[str, object]:
        dwell = self.dwell_bars()
        return {
            "source": SOURCE_NAME,
            "synthetic": True,
            "edge_claim": "none: synthetic data cannot evidence real-market edge",
            "symbol": self.symbol,
            "interval_seconds": self.interval_seconds,
            "seed": self.seed,
            "bars": len(self.bars),
            "first_ts": self.bars.first_ts.isoformat() if len(self.bars) else None,
            "last_ts": self.bars.last_ts.isoformat() if len(self.bars) else None,
            "quotes": len(self.quotes) if self.quotes is not None else 0,
            "ticks": len(self.ticks) if self.ticks is not None else 0,
            "options": len(self.options) if self.options is not None else 0,
            "options_intraday": (
                bool(self.options.is_intraday) if self.options is not None else False
            ),
            "feeds": sorted(feed.value for feed in self.feeds()),
            "regime_counts": self.regime_counts(),
            "regime_mean_dwell": {
                name: (round(float(np.mean(runs)), 3) if runs else None)
                for name, runs in dwell.items()
            },
            "outliers_injected": len(self.outlier_indices),
            "bars_dropped": len(self.dropped_timestamps),
            "config_hash": stable_hash(self.config) if self.config is not None else "",
        }


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


class SyntheticMarketGenerator:
    """Draws a regime path, then bars conditioned on it.

    The order matters and is the whole point: labels are causes here, not
    post-hoc annotations, so a detector scored against them is being scored
    against the thing that actually produced the prices.
    """

    def __init__(self, config: SyntheticConfig, seed: int = DEFAULT_SEED) -> None:
        self.config = config
        self.seed = int(seed)

    # --- randomness ----------------------------------------------------

    def _stream(self, symbol: str, name: str) -> np.random.Generator:
        """Independent generator for one named concern of one symbol.

        The symbol is part of the name so a two-symbol dataset does not get
        two copies of the same price path, which would make every
        correlation study on it meaningless.
        """
        if name not in STREAMS:
            raise ValueError(f"unregistered stream {name!r}; known streams: {list(STREAMS)}")
        return rng(self.seed, SOURCE_NAME, symbol, name)

    # --- regime chain --------------------------------------------------

    def transition_matrix(self) -> np.ndarray:
        """Row-stochastic matrix derived from each regime's `expected_bars`.

        Self-transition is 1 - 1/expected_bars, which makes the dwell time
        geometric with mean `expected_bars`. The exit mass is split by
        `TRANSITION_PREFERENCE`, so a successor preference is a declared
        number rather than an emergent accident.
        """
        regimes = self.config.regimes
        m = len(regimes)
        matrix = np.zeros((m, m), dtype=np.float64)
        for i, params in enumerate(regimes):
            if m == 1:
                matrix[i, i] = 1.0
                continue
            self_prob = 1.0 - 1.0 / params.expected_bars
            preference = TRANSITION_PREFERENCE.get(params.regime, {})
            weights = np.array(
                [
                    0.0 if j == i else float(preference.get(other.regime, 1.0))
                    for j, other in enumerate(regimes)
                ],
                dtype=np.float64,
            )
            total = float(weights.sum())
            if total <= 0.0:
                # Every preferred successor is absent from this config; fall
                # back to uniform rather than making the regime absorbing.
                weights = np.ones(m, dtype=np.float64)
                weights[i] = 0.0
                total = float(weights.sum())
            matrix[i] = (1.0 - self_prob) * weights / total
            matrix[i, i] = self_prob
        return matrix

    def regime_path(self, symbol: str, n_bars: int) -> np.ndarray:
        """Integer regime codes, one per bar.

        The initial state is drawn uniformly rather than from the stationary
        distribution: a stationary draw needs an eigen solve whose last bits
        depend on the BLAS build, and this generator promises byte-identical
        output on any machine. The chain mixes to stationary on its own.
        """
        matrix = self.transition_matrix()
        m = matrix.shape[0]
        draws = self._stream(symbol, "regime").random(n_bars if n_bars > 0 else 0)
        codes = np.zeros(n_bars, dtype=np.int16)
        if n_bars == 0:
            return codes
        cumulative = np.cumsum(matrix, axis=1)
        state = min(int(draws[0] * m), m - 1)
        codes[0] = state
        for i in range(1, n_bars):
            state = int(np.searchsorted(cumulative[state], draws[i], side="right"))
            state = min(state, m - 1)
            codes[i] = state
        return codes

    # --- session grid --------------------------------------------------

    def _session_bar_count(self, spec: InstrumentSpec, interval_seconds: int) -> int:
        window = spec.rth
        if window is None:
            return int(self.config.bars_per_day)
        span_minutes = window.end_minutes - window.start_minutes
        return int(span_minutes * 60 // interval_seconds)

    def _session_start_minutes(self, spec: InstrumentSpec) -> int:
        return (
            spec.rth.start_minutes if spec.rth is not None else DEFAULT_SESSION_START_MINUTES
        )

    def _grid(
        self,
        spec: InstrumentSpec,
        start: date,
        end: date,
        interval_seconds: int,
        calendar: SessionCalendarProtocol | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bar-close timestamps for [start, end), plus position-in-session.

        Returns (ts_ns, pos_in_session, bars_in_session, session_id). Only
        timestamps inside the session are emitted: a bar outside the session
        is not a bar, and a feature that averages one has been handed data
        the exchange never produced.
        """
        if end <= start:
            raise DataLayerError(f"empty range: end {end} must be after start {start}")
        if spec.rth is not None and spec.rth.wraps_midnight:
            raise DataLayerError(
                f"{spec.symbol}: RTH window {spec.rth.start}-{spec.rth.end} wraps "
                "midnight; this generator models a single within-day session and "
                "would otherwise assign bars to the wrong trading day"
            )

        full_count = self._session_bar_count(spec, interval_seconds)
        if full_count <= 0:
            raise DataLayerError(
                f"{spec.symbol}: interval {interval_seconds}s does not fit in the "
                "session window; no bars could be emitted"
            )
        start_minutes = self._session_start_minutes(spec)
        tz = ZoneInfo(spec.timezone)
        interval_ns = int(interval_seconds) * NS_PER_SECOND

        ts_chunks: list[np.ndarray] = []
        pos_chunks: list[np.ndarray] = []
        size_chunks: list[np.ndarray] = []
        id_chunks: list[np.ndarray] = []

        day = start
        session_id = 0
        while day < end:
            if self._is_trading_day(day, spec, calendar):
                count = full_count
                if self._is_half_day(day, spec, calendar):
                    count = int((HALF_DAY_CLOSE_MINUTES - start_minutes) * 60 // interval_seconds)
                if count > 0:
                    local_open = datetime(
                        day.year,
                        day.month,
                        day.day,
                        start_minutes // 60,
                        start_minutes % 60,
                        tzinfo=tz,
                    )
                    # Convert to UTC first and then advance in absolute
                    # nanoseconds: adding a timedelta to a zone-aware local
                    # datetime is wall-clock arithmetic, which silently
                    # shifts a session by an hour across a DST boundary.
                    base_ns = to_ns(local_open.astimezone(timezone.utc))
                    offsets = np.arange(1, count + 1, dtype=np.int64) * interval_ns
                    ts_chunks.append(base_ns + offsets)
                    pos_chunks.append(np.arange(count, dtype=np.int32))
                    size_chunks.append(np.full(count, count, dtype=np.int32))
                    id_chunks.append(np.full(count, session_id, dtype=np.int32))
                    session_id += 1
            day = day + timedelta(days=1)

        if not ts_chunks:
            raise DataLayerError(
                f"{spec.symbol}: no trading sessions in [{start}, {end}); nothing to generate"
            )
        return (
            np.concatenate(ts_chunks),
            np.concatenate(pos_chunks),
            np.concatenate(size_chunks),
            np.concatenate(id_chunks),
        )

    @staticmethod
    def _is_trading_day(
        day: date, spec: InstrumentSpec, calendar: SessionCalendarProtocol | None
    ) -> bool:
        if calendar is not None:
            return bool(calendar.is_trading_day(day, spec))
        return day.weekday() < 5

    @staticmethod
    def _is_half_day(
        day: date, spec: InstrumentSpec, calendar: SessionCalendarProtocol | None
    ) -> bool:
        checker = getattr(calendar, "is_half_day", None) if calendar is not None else None
        return bool(checker(day, spec)) if checker is not None else False

    # --- the generate entry point --------------------------------------

    def generate(
        self,
        symbol: str,
        spec: InstrumentSpec,
        start: date,
        end: date,
        interval_seconds: int,
        calendar: SessionCalendarProtocol | None = None,
    ) -> SyntheticDataset:
        """Generate every configured feed for [start, end), with labels."""
        if interval_seconds <= 0:
            raise DataLayerError(f"interval_seconds must be positive, got {interval_seconds}")
        config = self.config
        tick = float(spec.tick_size)

        ts_ns, pos, session_size, session_id = self._grid(
            spec, start, end, interval_seconds, calendar
        )
        n = int(len(ts_ns))

        codes = self.regime_path(symbol, n)
        drift, sigma, phi, volume_multiplier = self._regime_arrays(symbol, codes)

        returns, innovations = self._returns(symbol, drift, sigma, phi)
        opens, highs, lows, closes = self._ohlc(symbol, returns, sigma, tick, config)
        volume = self._volume(symbol, pos, session_size, volume_multiplier)

        outlier_mask = self._outlier_mask(symbol, n)
        if outlier_mask.any():
            highs, lows, closes = self._inject_outliers(
                symbol, outlier_mask, opens, highs, lows, closes, sigma, tick
            )

        buy, sell, unclassified, buy_trades, sell_trades, max_trade = (
            self._ticks(symbol, volume, innovations) if config.include_ticks else (None,) * 6
        )
        bid, ask, bid_size, ask_size = (
            self._quotes(symbol, closes, spec) if config.include_quotes else (None,) * 4
        )

        keep = self._keep_mask(symbol, n)
        dropped_ts = tuple(
            datetime.fromtimestamp(int(value) / NS_PER_SECOND, tz=timezone.utc)
            for value in ts_ns[~keep]
        )
        if int(keep.sum()) == 0:
            raise DataLayerError(
                f"{symbol}: gap injection removed every bar; lower gap_probability"
            )

        outlier_indices = tuple(np.flatnonzero(outlier_mask[keep]).tolist())
        ts_ns, pos, session_size, session_id = (
            ts_ns[keep],
            pos[keep],
            session_size[keep],
            session_id[keep],
        )
        codes = codes[keep]
        sigma_kept = sigma[keep]
        opens, highs, lows, closes = opens[keep], highs[keep], lows[keep], closes[keep]
        volume = volume[keep]

        bars = self._bar_series(symbol, ts_ns, opens, highs, lows, closes, volume, interval_seconds)

        ticks = None
        if config.include_ticks:
            ticks = self._tick_series(
                symbol,
                ts_ns,
                buy[keep],
                sell[keep],
                unclassified[keep],
                buy_trades[keep],
                sell_trades[keep],
                max_trade[keep],
                volume,
            )
        quotes = None
        if config.include_quotes:
            quotes = QuoteSeries(
                symbol=symbol,
                ts_ns=ts_ns,
                columns={
                    "bid": bid[keep],
                    "ask": ask[keep],
                    "bid_size": bid_size[keep],
                    "ask_size": ask_size[keep],
                },
                meta={"source": SOURCE_NAME},
            )
        options = None
        if config.include_options:
            options = self._options_series(
                symbol, ts_ns, opens, closes, sigma_kept, session_id
            )

        return SyntheticDataset(
            symbol=symbol,
            interval_seconds=int(interval_seconds),
            bars=bars,
            regime_labels=codes,
            regime_names=config.regime_names,
            quotes=quotes,
            ticks=ticks,
            options=options,
            outlier_indices=outlier_indices,
            dropped_timestamps=dropped_ts,
            config=config,
            seed=self.seed,
        )

    # --- generation stages ---------------------------------------------

    def _regime_arrays(
        self, symbol: str, codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-bar parameter arrays, with the DIRECTIONAL_SHIFT sign applied."""
        regimes = self.config.regimes
        drift_by_code = np.array([p.drift_per_bar for p in regimes], dtype=np.float64)
        vol_by_code = np.array([p.vol_per_bar for p in regimes], dtype=np.float64)
        phi_by_code = np.array([p.mean_persistence for p in regimes], dtype=np.float64)
        mult_by_code = np.array([p.volume_multiplier for p in regimes], dtype=np.float64)

        drift = drift_by_code[codes]
        n = len(codes)
        shift_code = next(
            (i for i, p in enumerate(regimes) if p.regime is Regime.DIRECTIONAL_SHIFT), None
        )
        if shift_code is not None and n:
            # One sign per episode, held for its whole run: a shift whose sign
            # flipped bar to bar would be high volatility, not a break.
            signs = self._stream(symbol, "shift_sign").integers(0, 2, size=n) * 2.0 - 1.0
            is_shift = codes == shift_code
            run_start = np.concatenate(([True], codes[1:] != codes[:-1])) & is_shift
            last_start = np.maximum.accumulate(np.where(run_start, np.arange(n), -1))
            held = np.ones(n, dtype=np.float64)
            seen = last_start >= 0
            held[seen] = signs[last_start[seen]]
            drift = np.where(is_shift, drift * held, drift)

        return drift, vol_by_code[codes], phi_by_code[codes], mult_by_code[codes]

    def _returns(
        self, symbol: str, drift: np.ndarray, sigma: np.ndarray, phi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """AR(1) log returns, and the standardized innovation per bar.

        The innovation is returned because the tick delta is built from it:
        correlating delta with the bar's own surprise rather than with its
        realized return keeps the delta free of any dependence on earlier
        bars that a feature could mistake for predictive content.
        """
        n = len(drift)
        innovations = self._stream(symbol, "returns").standard_normal(n)
        # sqrt(1 - phi^2) keeps the unconditional stdev equal to vol_per_bar,
        # so persistence changes the shape of the path and not its size --
        # otherwise a trending regime would also be a high-vol regime and the
        # two labels could not be told apart.
        scale = sigma * np.sqrt(1.0 - phi * phi)
        returns = np.empty(n, dtype=np.float64)
        deviation = 0.0
        for i in range(n):
            deviation = phi[i] * deviation + scale[i] * innovations[i]
            returns[i] = drift[i] + deviation
        return returns, innovations

    def _ohlc(
        self,
        symbol: str,
        returns: np.ndarray,
        sigma: np.ndarray,
        tick: float,
        config: SyntheticConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(returns)
        closes = _round_to_tick(config.start_price * np.exp(np.cumsum(returns)), tick)
        opens = np.empty(n, dtype=np.float64)
        opens[0] = _round_to_tick(np.array([config.start_price]), tick)[0]
        if n > 1:
            opens[1:] = closes[:-1]

        stream = self._stream(symbol, "ranges")
        up = np.abs(stream.standard_normal(n)) * sigma * closes * config.bar_range_factor
        down = np.abs(stream.standard_normal(n)) * sigma * closes * config.bar_range_factor
        body_high = np.maximum(opens, closes)
        body_low = np.minimum(opens, closes)
        highs = _ceil_to_tick(body_high + up, tick)
        lows = _floor_to_tick(body_low - down, tick)
        # Clamp rather than trust the arithmetic: BarSeries would raise on a
        # violation, but a generator that trips its consumer's validator is a
        # generator bug, and a float residue is not a reason to lose a dataset.
        highs = np.maximum(highs, body_high)
        lows = np.minimum(np.maximum(lows, tick), body_low)
        return opens, highs, lows, closes

    def _volume(
        self,
        symbol: str,
        pos: np.ndarray,
        session_size: np.ndarray,
        volume_multiplier: np.ndarray,
    ) -> np.ndarray:
        """Intraday U-shape times the regime multiplier times lognormal noise.

        Volumes are whole units. That is not cosmetic: the tick split has to
        sum to the bar's volume exactly, and exact integer arithmetic is the
        only way that identity survives.
        """
        config = self.config
        span = np.maximum(session_size - 1, 1).astype(np.float64)
        x = pos.astype(np.float64) / span
        shape = (
            1.0
            + U_OPEN_WEIGHT * np.exp(-x / U_OPEN_DECAY)
            + U_CLOSE_WEIGHT * np.exp(-(1.0 - x) / U_CLOSE_DECAY)
        )
        noise = _lognormal_unit_mean(
            self._stream(symbol, "volume"), config.volume_noise_sigma, len(pos)
        )
        volume = np.round(config.base_volume * shape * volume_multiplier * noise)
        return np.maximum(volume, 0.0)

    def _outlier_mask(self, symbol: str, n: int) -> np.ndarray:
        """Which bars get a bad print. Drawn even at probability 0.

        Always consuming the draw keeps the stream aligned, so turning
        outliers on does not shift any other series.
        """
        draws = self._stream(symbol, "outliers").random(n)
        if self.config.outlier_probability <= 0.0 or n == 0:
            return np.zeros(n, dtype=bool)
        mask = draws < self.config.outlier_probability
        mask[0] = False  # the first bar's open anchors the series
        return mask

    def _inject_outliers(
        self,
        symbol: str,
        mask: np.ndarray,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        sigma: np.ndarray,
        tick: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Spike a bar's close and its matching extreme, and let it revert.

        The next bar's open is left at the pre-spike level, so the print
        reverts instead of shifting the series permanently. That deliberately
        breaks `open == previous close` at the seam, which is what makes a
        bad print findable; a spike that propagated would be a level change,
        which is a different defect with a different fix.
        """
        highs, lows, closes = highs.copy(), lows.copy(), closes.copy()
        stream = self._stream(symbol, "outliers")
        indices = np.flatnonzero(mask)
        signs = stream.integers(0, 2, size=len(indices)) * 2.0 - 1.0
        shock = signs * self.config.outlier_sigma_multiple * sigma[indices]
        spiked = _round_to_tick(closes[indices] * np.exp(shock), tick)
        spiked = np.maximum(spiked, tick)
        closes[indices] = spiked
        body_high = np.maximum(opens[indices], spiked)
        body_low = np.minimum(opens[indices], spiked)
        highs[indices] = np.maximum(highs[indices], body_high)
        lows[indices] = np.minimum(lows[indices], body_low)
        return highs, lows, closes

    def _ticks(
        self, symbol: str, volume: np.ndarray, innovations: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Aggressor split whose delta carries `delta_signal_strength`.

        The split is exact by construction: `unclassified = volume -
        classified` and `sell = classified - buy`, all integers, so
        buy + sell + unclassified == volume with no float residue. Volume
        that cannot be classified stays unclassified; it is never given a
        side, because an invented side is indistinguishable from a real one
        to every order-flow feature above.
        """
        config = self.config
        n = len(volume)
        stream = self._stream(symbol, "ticks")
        noise = stream.standard_normal(n)
        rho = float(config.delta_signal_strength)
        signal = rho * innovations + math.sqrt(max(0.0, 1.0 - rho * rho)) * noise
        buy_fraction = 0.5 * (1.0 + np.tanh(TICK_IMBALANCE_GAIN * signal))

        classified = np.round(volume * config.tick_classification_coverage)
        classified = np.minimum(classified, volume)
        unclassified = volume - classified
        buy = np.round(classified * buy_fraction)
        buy = np.clip(buy, 0.0, classified)
        sell = classified - buy

        trades = np.maximum(np.round(volume / TICK_AVG_TRADE_SIZE), 0.0)
        buy_trades = np.round(trades * buy_fraction)
        sell_trades = trades - buy_trades
        max_trade = np.maximum(
            np.round(volume * TICK_MAX_TRADE_SHARE * (1.0 + np.abs(noise) * 0.25)), 0.0
        )
        max_trade = np.minimum(max_trade, volume)
        return buy, sell, unclassified, buy_trades, sell_trades, max_trade

    def _quotes(
        self, symbol: str, closes: np.ndarray, spec: InstrumentSpec
    ) -> tuple[np.ndarray, ...]:
        """One top-of-book snapshot per bar close.

        The spread is a whole number of ticks with a one-tick floor, because
        a sub-tick spread cannot exist and a crossed book from rounding would
        be read downstream as a broken feed.
        """
        config = self.config
        n = len(closes)
        tick = float(spec.tick_size)
        stream = self._stream(symbol, "quotes")
        draw = config.spread_ticks_mean * _lognormal_unit_mean(
            stream, QUOTE_SPREAD_LOG_SIGMA, n
        )
        spread_ticks = np.maximum(np.round(draw), 1.0)
        spread = spread_ticks * tick
        bid = _round_to_tick(closes - spread / 2.0, tick)
        bid = np.maximum(bid, tick)
        ask = bid + spread
        sizes = QUOTE_BASE_SIZE * _lognormal_unit_mean(stream, QUOTE_SIZE_LOG_SIGMA, n)
        bid_size = np.maximum(np.round(sizes), 1.0)
        sizes = QUOTE_BASE_SIZE * _lognormal_unit_mean(stream, QUOTE_SIZE_LOG_SIGMA, n)
        ask_size = np.maximum(np.round(sizes), 1.0)
        return bid, ask, bid_size, ask_size

    def _keep_mask(self, symbol: str, n: int) -> np.ndarray:
        """Which bars survive gap injection.

        The draw is always consumed so enabling gaps cannot shift another
        stream. Bars are DROPPED, never filled: a gap is counted by the
        cleaning layer, and a fabricated bar is fabricated data.
        """
        draws = self._stream(symbol, "gaps").random(n)
        keep = np.ones(n, dtype=bool)
        if self.config.gap_probability <= 0.0 or n == 0:
            return keep
        keep = draws >= self.config.gap_probability
        keep[0] = True  # the first bar carries the series' opening price
        return keep

    # --- series assembly ------------------------------------------------

    def _bar_series(
        self,
        symbol: str,
        ts_ns: np.ndarray,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volume: np.ndarray,
        interval_seconds: int,
    ) -> BarSeries:
        body_high = np.maximum(opens, closes)
        body_low = np.minimum(opens, closes)
        ok = (lows <= body_low) & (highs >= body_high) & (highs >= lows)
        if not np.all(ok):
            bad = int(np.argmax(~ok))
            raise DataLayerError(
                f"{symbol}: generated bar {bad} violates OHLC ordering "
                f"(o={opens[bad]} h={highs[bad]} l={lows[bad]} c={closes[bad]}); "
                "this is a generator bug, not bad input"
            )
        if np.any(volume < 0):
            raise DataLayerError(f"{symbol}: generated negative volume")
        return BarSeries(
            symbol=symbol,
            ts_ns=ts_ns,
            columns={
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volume,
            },
            interval_seconds=int(interval_seconds),
            meta={"source": SOURCE_NAME, "synthetic": True},
        )

    def _tick_series(
        self,
        symbol: str,
        ts_ns: np.ndarray,
        buy: np.ndarray,
        sell: np.ndarray,
        unclassified: np.ndarray,
        buy_trades: np.ndarray,
        sell_trades: np.ndarray,
        max_trade: np.ndarray,
        volume: np.ndarray,
    ) -> TickSeries:
        total = buy + sell + unclassified
        if not np.array_equal(total, volume):
            bad = int(np.argmax(total != volume))
            raise DataLayerError(
                f"{symbol}: tick split does not conserve volume at bar {bad} "
                f"({total[bad]} != {volume[bad]}); an order-flow feature would be "
                "reading volume the bar never traded"
            )
        return TickSeries(
            symbol=symbol,
            ts_ns=ts_ns,
            columns={
                "buy_volume": buy,
                "sell_volume": sell,
                "unclassified_volume": unclassified,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "max_trade_size": max_trade,
            },
            meta={
                "classification_method": "synthetic_aggressor",
                "source": SOURCE_NAME,
            },
        )

    def _day_slices(self, session_id: np.ndarray) -> list[tuple[int, int]]:
        if not len(session_id):
            return []
        boundaries = np.flatnonzero(session_id[1:] != session_id[:-1]) + 1
        starts = np.concatenate(([0], boundaries)).tolist()
        stops = np.concatenate((boundaries, [len(session_id)])).tolist()
        return list(zip(starts, stops))

    def _options_series(
        self,
        symbol: str,
        ts_ns: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        sigma: np.ndarray,
        session_id: np.ndarray,
    ) -> OptionsSeries:
        """Chain-level aggregates, EOD by default.

        The premium imbalance is computed from the session that has ALREADY
        closed. Any next-day predictability it carries therefore arrives
        through the regime's own persistence; nothing here reads a future
        bar. Hand-coding that relationship would make every options-flow
        test in Phase 5 pass against data built to satisfy it.
        """
        config = self.config
        intraday = config.options_intraday
        if intraday:
            anchors = np.arange(len(ts_ns), dtype=np.int64)
            reference = np.log(closes / opens)
            per_bar_vol = np.maximum(sigma, 1e-12)
            scale = per_bar_vol
        else:
            slices = self._day_slices(session_id)
            anchors = np.array([stop - 1 for _, stop in slices], dtype=np.int64)
            reference = np.array(
                [math.log(closes[stop - 1] / opens[start]) for start, stop in slices],
                dtype=np.float64,
            )
            per_bar_vol = np.array(
                [float(np.mean(sigma[start:stop])) for start, stop in slices],
                dtype=np.float64,
            )
            lengths = np.array([stop - start for start, stop in slices], dtype=np.float64)
            # A session's return scales with sqrt(bars), so standardizing by the
            # per-bar vol alone would make every day look like an outlier.
            scale = np.maximum(per_bar_vol * np.sqrt(lengths), 1e-12)
            per_bar_vol = np.maximum(per_bar_vol, 1e-12)

        m = len(anchors)
        stream = self._stream(symbol, "options")
        noise = stream.standard_normal(m)
        rho = OPTIONS_SIGNAL_RHO
        standardized = np.clip(reference / scale, -8.0, 8.0)
        signal = rho * standardized + math.sqrt(1.0 - rho * rho) * noise
        imbalance = np.tanh(OPTIONS_IMBALANCE_GAIN * signal)

        total_volume = config.base_options_volume * _lognormal_unit_mean(stream, 0.30, m)
        call_share = 0.5 * (1.0 + OPTIONS_VOLUME_SHARE_GAIN * imbalance)
        call_volume = np.round(total_volume * call_share)
        put_volume = np.round(total_volume * (1.0 - call_share))

        total_premium = config.base_option_premium * _lognormal_unit_mean(stream, 0.35, m)
        call_premium = total_premium * (1.0 + imbalance) / 2.0
        put_premium = total_premium * (1.0 - imbalance) / 2.0

        call_oi = np.round(
            config.base_open_interest
            * np.exp(np.cumsum(stream.normal(0.0, OPTIONS_OI_LOG_SIGMA, m)))
        )
        put_oi = np.round(
            config.base_open_interest
            * np.exp(np.cumsum(stream.normal(0.0, OPTIONS_OI_LOG_SIGMA, m)))
        )
        # OI change is the first difference of the level, so the two columns
        # cannot disagree. The first row's change is 0 because the prior
        # session is outside the requested range and guessing it would be
        # fabricating an observation.
        call_oi_change = np.diff(call_oi, prepend=call_oi[:1]) if m else call_oi
        put_oi_change = np.diff(put_oi, prepend=put_oi[:1]) if m else put_oi

        delta_call = np.clip(0.5 + 0.25 * imbalance, 0.05, 0.95)
        delta_put = np.clip(0.5 - 0.25 * imbalance, 0.05, 0.95)

        median_vol = float(np.median([p.vol_per_bar for p in config.regimes]))
        atm_iv = np.clip(
            config.base_atm_iv
            * (per_bar_vol / max(median_vol, 1e-12))
            * _lognormal_unit_mean(stream, OPTIONS_IV_LOG_SIGMA, m),
            OPTIONS_IV_FLOOR,
            OPTIONS_IV_CEILING,
        )
        skew = np.clip(
            OPTIONS_SKEW_BASE
            + OPTIONS_SKEW_NOISE * stream.standard_normal(m)
            - OPTIONS_SKEW_IMBALANCE * imbalance,
            0.0,
            0.6,
        )
        underlying = closes[anchors]
        return OptionsSeries(
            symbol=symbol,
            ts_ns=ts_ns[anchors],
            columns={
                "call_volume": call_volume,
                "put_volume": put_volume,
                "call_premium": call_premium,
                "put_premium": put_premium,
                "call_oi": call_oi,
                "put_oi": put_oi,
                "call_oi_change": call_oi_change,
                "put_oi_change": put_oi_change,
                "delta_weighted_call_volume": call_volume * delta_call,
                "delta_weighted_put_volume": put_volume * delta_put,
                "atm_iv": atm_iv,
                "iv_25d_put": atm_iv * (1.0 + skew),
                "iv_25d_call": atm_iv * (1.0 - 0.5 * skew),
                "gamma_exposure_proxy": (call_oi - put_oi) * underlying * GAMMA_PROXY_SCALE,
                "underlying_price": underlying,
            },
            meta={
                "is_intraday": bool(intraday),
                "source": "synthetic_intraday" if intraday else "synthetic_eod",
            },
        )


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------


class SyntheticAdapter(DataSourceAdapter):
    """Serves generated data through the normal adapter interface.

    This is what lets the whole stack above the data layer be exercised with
    no files on disk. `feeds` narrows what the adapter admits to having, so a
    bars-only run is a real bars-only run: the restricted feeds return None
    rather than a quietly-degraded substitute, which is the degradation path
    the signal engine is supposed to take.
    """

    name = SOURCE_NAME

    def __init__(
        self,
        config: SyntheticConfig,
        seed: int = DEFAULT_SEED,
        spec_by_symbol: Mapping[str, InstrumentSpec] | None = None,
        feeds: frozenset[Feed] | None = None,
        interval_seconds: int = 300,
        calendar: SessionCalendarProtocol | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.spec_by_symbol = dict(spec_by_symbol or {})
        self.interval_seconds = int(interval_seconds)
        self.calendar = calendar
        produced = config.feeds()
        self._feeds = produced if feeds is None else frozenset(feeds) & produced
        self.generator = SyntheticMarketGenerator(config, seed=self.seed)
        self._cache: dict[tuple[str, int, date, date], SyntheticDataset] = {}

    # --- interface ------------------------------------------------------

    def available_feeds(self, symbol: str) -> frozenset[Feed]:
        self._spec(symbol)
        return self._feeds

    def load_bars(
        self, symbol: str, interval_seconds: int, start: date, end: date
    ) -> BarSeries:
        return self._dataset(symbol, int(interval_seconds), start, end).bars

    def load_quotes(self, symbol: str, start: date, end: date) -> QuoteSeries | None:
        if Feed.QUOTES not in self._feeds:
            return None
        return self._dataset(symbol, self.interval_seconds, start, end).quotes

    def load_tick_aggregates(
        self, symbol: str, interval_seconds: int, start: date, end: date
    ) -> TickSeries | None:
        if Feed.TICK_AGGREGATE not in self._feeds:
            return None
        return self._dataset(symbol, int(interval_seconds), start, end).ticks

    def load_options(self, symbol: str, start: date, end: date) -> OptionsSeries | None:
        if Feed.OPTIONS_SNAPSHOT not in self._feeds:
            return None
        return self._dataset(symbol, self.interval_seconds, start, end).options

    def fingerprint(
        self, symbol: str, interval_seconds: int, start: date, end: date, rows: int
    ) -> DatasetFingerprint:
        """Provenance that includes the config and the seed.

        The base implementation hashes only (symbol, range, rows, source),
        which for generated data would give two different configurations the
        same `data_hash` and attribute their results to the same dataset.
        """
        return DatasetFingerprint(
            symbol=symbol,
            source=self.name,
            interval_seconds=int(interval_seconds),
            start=start,
            end=end,
            rows=rows,
            data_hash=stable_hash(
                {
                    "source": self.name,
                    "symbol": symbol,
                    "interval_seconds": int(interval_seconds),
                    "start": start,
                    "end": end,
                    "rows": rows,
                    "seed": self.seed,
                    "config": self.config,
                    "feeds": sorted(f.value for f in self._feeds),
                }
            ),
            feeds=tuple(sorted(self._feeds, key=lambda f: f.value)),
        )

    # --- datasets -------------------------------------------------------

    def dataset(self, symbol: str, start: date, end: date) -> SyntheticDataset:
        """The generated dataset at the adapter's primary interval."""
        return self._dataset(symbol, self.interval_seconds, start, end)

    def _spec(self, symbol: str) -> InstrumentSpec:
        try:
            return self.spec_by_symbol[symbol]
        except KeyError:
            raise DataLayerError(
                f"no InstrumentSpec for {symbol!r}; known symbols: "
                f"{sorted(self.spec_by_symbol)}. The generator needs the tick size "
                "and session window to emit prices that could exist."
            ) from None

    def _dataset(
        self, symbol: str, interval_seconds: int, start: date, end: date
    ) -> SyntheticDataset:
        if interval_seconds != self.interval_seconds:
            raise DataLayerError(
                f"{symbol}: this adapter generates {self.interval_seconds}s bars, not "
                f"{interval_seconds}s. Drawing a second independent path would give the "
                "symbol two contradictory price histories; aggregate the primary series "
                "instead, or construct a second adapter."
            )
        key = (symbol, int(interval_seconds), start, end)
        cached = self._cache.get(key)
        if cached is None:
            cached = self.generator.generate(
                symbol=symbol,
                spec=self._spec(symbol),
                start=start,
                end=end,
                interval_seconds=int(interval_seconds),
                calendar=self.calendar,
            )
            self._cache[key] = cached
        return cached
