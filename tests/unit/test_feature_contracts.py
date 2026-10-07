"""Architectural guards on the feature layer.

`validation/lookahead.py` audits the mapping from a `MarketView` to a
feature value, and since Phase 2 a view physically cannot hold future data.
That leaves exactly one way for a computer to see the future: capture a
full-sample statistic at construction time, before any view exists.

    class Leaky(FeatureComputer):
        def __init__(self, dataset):
            self.scale = dataset.primary_bars.col("close").mean()   # the future

The audit harness cannot see through that when handed an instance, which is
verified below. So it is banned at the constructor instead: these tests
discover every `FeatureComputer` subclass in the package and assert that
none of them accepts market data. A computer takes configuration.

The rest of this file tests the enforcement machinery itself. A guard that
cannot fail is worth nothing, so each cheat class is constructed and the
harness is required to catch it.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from flow_model.core.contracts import FeatureVector
from flow_model.core.enums import DataQuality, Feed
from flow_model.data.series import BarSeries, ColumnSeries, to_ns_array
from flow_model.data.store import DataStore, SymbolData
from flow_model.features.base import (
    FeatureBundle,
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
    squash,
)
from flow_model.validation.lookahead import (
    LookaheadAuditResult,
    assert_no_lookahead,
    audit_computer,
    mutate_after,
    truncate,
)

TS0 = datetime(2020, 1, 2, 14, 35, tzinfo=timezone.utc)
INTERVAL = 300

#: Types that are market DATA. A computer may take configuration (a
#: FeatureConfig, a StructureLevelConfig, an InstrumentSpec, a calendar) but
#: never one of these.
FORBIDDEN_PARAMETER_TYPES = (
    SymbolData, DataStore, BarSeries, ColumnSeries, FeatureVector, np.ndarray,
)
FORBIDDEN_PARAMETER_NAMES = frozenset(
    {"data", "dataset", "store", "series", "bars", "view", "prices", "closes", "array"}
)


def _dataset(n=400, seed=3):
    ts = to_ns_array(TS0 + timedelta(seconds=INTERVAL * (i + 1)) for i in range(n))
    close = 18000.0 + np.cumsum(np.random.default_rng(seed).normal(0, 5.0, n))
    bars = BarSeries(symbol="NQ", ts_ns=ts, interval_seconds=INTERVAL, columns={
        "open": close - 1.0, "high": close + 5.0, "low": close - 5.0,
        "close": close, "volume": np.full(n, 1000.0)})
    return SymbolData(symbol="NQ", primary_interval=INTERVAL, bars={INTERVAL: bars})


def _discover_computers() -> list[type[FeatureComputer]]:
    """Every concrete FeatureComputer subclass in the package.

    Discovered rather than listed, so a module added later is covered
    automatically instead of silently escaping the guard.
    """
    for package in ("flow_model.features", "flow_model.regime", "flow_model.signals"):
        try:
            module = importlib.import_module(package)
        except ImportError:
            continue
        for info in pkgutil.iter_modules(module.__path__):
            try:
                importlib.import_module(f"{package}.{info.name}")
            except ImportError:
                continue

    found: list[type[FeatureComputer]] = []
    stack = [FeatureComputer]
    while stack:
        parent = stack.pop()
        for child in parent.__subclasses__():
            stack.append(child)
            # Production computers only. The deliberate cheaters in this file
            # exist to prove the guard can fail, and the guard does catch them
            # -- verified by test_the_ban_catches_a_capturing_constructor.
            if not inspect.isabstract(child) and child.__module__.startswith("flow_model"):
                found.append(child)
    return sorted(set(found), key=lambda c: c.__name__)


# --- the construction-capture ban ----------------------------------------


def test_the_ban_catches_a_capturing_constructor():
    """Guards the guard. `_CapturedAtInit` takes a dataset, which is exactly
    what the ban exists to stop, so running the check against it must
    produce a finding. Without this, a discovery bug would make
    `test_no_computer_constructor_accepts_market_data` pass vacuously."""
    offenders = _offending_parameters(_CapturedAtInit)
    assert offenders, "the ban did not flag a constructor that takes a dataset"
    assert "dataset" in offenders[0]


def test_the_ban_accepts_a_config_only_constructor():
    assert _offending_parameters(_Honest) == []


def _offending_parameters(computer: type) -> list[str]:
    """Constructor parameters that look like market data."""
    offenders: list[str] = []
    signature = inspect.signature(computer.__init__)
    hints = getattr(computer.__init__, "__annotations__", {})
    for name, parameter in signature.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        annotation = hints.get(name, parameter.annotation)
        if isinstance(annotation, type) and issubclass(annotation, FORBIDDEN_PARAMETER_TYPES):
            offenders.append(
                f"{computer.__module__}.{computer.__name__}.__init__({name}: "
                f"{annotation.__name__})"
            )
        elif name in FORBIDDEN_PARAMETER_NAMES:
            offenders.append(
                f"{computer.__module__}.{computer.__name__}.__init__({name}) -- "
                f"a parameter named {name!r} suggests market data"
            )
    return offenders


def test_no_computer_constructor_accepts_market_data():
    """The one leak the lookahead audit cannot see through."""
    offenders = [o for c in _discover_computers() for o in _offending_parameters(c)]
    assert not offenders, (
        "a FeatureComputer constructor accepts market data, which lets it capture "
        "a full-sample statistic the lookahead audit cannot detect:\n  "
        + "\n  ".join(offenders)
    )


def test_every_computer_declares_its_contract():
    for computer in _discover_computers():
        assert isinstance(computer.name, str) and computer.name != "abstract", (
            f"{computer.__name__} did not override `name`"
        )
        assert "keys" in dir(computer), f"{computer.__name__} has no `keys`"
        assert "warmup_bars" in dir(computer), f"{computer.__name__} has no `warmup_bars`"


# --- the harness catches each cheat class --------------------------------


class _Honest(FeatureComputer):
    name = "honest"
    warmup_bars = 20
    keys = ("mean20",)

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def compute(self, view):
        if not view.warmup_ok(self.warmup_bars):
            return self._not_ready(view, "warmup")
        return self._vector(view, {"mean20": float(view.closes(20).mean()) * self.scale})


class _CapturedAtInit(FeatureComputer):
    """Normalizes by a constant captured from the whole dataset."""

    name = "captured"
    warmup_bars = 20
    keys = ("norm",)

    def __init__(self, dataset) -> None:
        self.scale = float(dataset.primary_bars.col("close").mean())

    def compute(self, view):
        if not view.warmup_ok(self.warmup_bars):
            return self._not_ready(view, "warmup")
        return self._vector(view, {"norm": float(view.closes(20).mean()) / self.scale})


class _NonDeterministic(FeatureComputer):
    name = "nondeterministic"
    warmup_bars = 20
    keys = ("x",)

    def compute(self, view):
        if not view.warmup_ok(self.warmup_bars):
            return self._not_ready(view, "warmup")
        return self._vector(view, {"x": float(np.random.default_rng().random())})


class _LiesAboutWarmup(FeatureComputer):
    name = "liar"
    warmup_bars = 250
    keys = ("y",)

    def compute(self, view):
        return FeatureVector(
            symbol=view.symbol, ts=view.now, values={"y": 1.0},
            quality_by_key={"y": DataQuality.GOOD}, warmup_complete=True,
        )


def test_audit_passes_an_honest_computer():
    result = audit_computer(_Honest(), _dataset(), sample=12)
    assert result.passed, result.summary()
    assert result.bars_checked > 0
    assert result.keys_checked == ("mean20",)


def test_audit_catches_a_non_deterministic_computer():
    result = audit_computer(_NonDeterministic(), _dataset(), sample=12)
    assert not result.passed
    assert "determinism" in {f.kind for f in result.findings}


def test_audit_catches_a_computer_that_lies_about_warmup():
    """The sample must include pre-warmup bars or this claim is never tested.
    An earlier version of the harness sampled only from bars past warmup and
    passed this computer."""
    result = audit_computer(_LiesAboutWarmup(), _dataset(), sample=12)
    assert not result.passed
    assert {f.kind for f in result.findings} == {"warmup"}


def test_audit_catches_construction_capture_only_with_a_factory():
    """Documents the harness's boundary honestly, in a test.

    Handed an instance, the captured constant never changes, so the audit
    cannot see the leak. Handed a factory, the computer is rebuilt against
    the mutated dataset and the constant moves with it.
    """
    data = _dataset()
    assert audit_computer(_CapturedAtInit(data), data, sample=12).passed

    caught = audit_computer(data=data, factory=_CapturedAtInit, sample=12)
    assert not caught.passed
    assert {f.kind for f in caught.findings} <= {"truncation", "mutation"}


def test_audit_reports_nothing_audited_rather_than_passing():
    """A dataset too short for the computer's warmup is not a pass."""
    result = audit_computer(_Honest(), _dataset(n=10), sample=12)
    assert result.bars_checked == 0
    assert not result.passed or result.notes
    assert any("nothing audited" in n or "not a pass" in n for n in result.notes)
    with pytest.raises(AssertionError, match="NOTHING AUDITED"):
        assert_no_lookahead([result])


def test_assert_no_lookahead_lists_every_finding():
    result = audit_computer(_NonDeterministic(), _dataset(), sample=6)
    with pytest.raises(AssertionError) as excinfo:
        assert_no_lookahead([result])
    assert "lookahead audit failed" in str(excinfo.value)


def test_assert_no_lookahead_accepts_a_clean_result():
    assert_no_lookahead([audit_computer(_Honest(), _dataset(), sample=6)])


# --- truncate / mutate helpers -------------------------------------------


def test_truncate_cuts_every_feed_by_timestamp():
    data = _dataset(n=100)
    cutoff = int(data.primary_bars.ts_ns[49])
    cut = truncate(data, cutoff)
    assert len(cut.primary_bars) == 50
    assert int(cut.primary_bars.ts_ns[-1]) == cutoff


def test_mutate_after_leaves_the_past_identical():
    """The whole point: if the past moved too, an invariance test proves
    nothing. The integrator's first attempt at this got it wrong."""
    data = _dataset(n=100)
    cutoff = int(data.primary_bars.ts_ns[49])
    mutated = mutate_after(data, cutoff, factor=7.0)
    before = data.primary_bars.col("close")[:50]
    after = mutated.primary_bars.col("close")[:50]
    assert np.array_equal(before, after)
    assert not np.array_equal(
        data.primary_bars.col("close")[50:], mutated.primary_bars.col("close")[50:]
    )


def test_mutated_series_is_still_structurally_valid():
    """Scaling rather than randomizing, so the mutation trips the audit
    rather than a BarSeries constructor."""
    data = _dataset(n=100)
    mutated = mutate_after(data, int(data.primary_bars.ts_ns[49]), factor=7.0)
    assert mutated.primary_bars.bar_at(80).is_valid


# --- base.py contract ----------------------------------------------------


def test_vector_rejects_a_key_mismatch():
    class Mismatched(_Honest):
        name = "mismatched"
        keys = ("promised",)

        def compute(self, view):
            return self._vector(view, {"delivered": 1.0})

    data = _dataset()
    view = DataStore().add(data).view("NQ", data.primary_bars.last_ts)
    with pytest.raises(FeatureError, match="declared keys do not match"):
        Mismatched().compute(view)


def test_vector_rejects_a_non_finite_value():
    class Broken(_Honest):
        name = "broken"
        keys = ("x",)

        def compute(self, view):
            return self._vector(view, {"x": float("nan")})

    data = _dataset()
    view = DataStore().add(data).view("NQ", data.primary_bars.last_ts)
    with pytest.raises(FeatureError, match="not finite"):
        Broken().compute(view)


def test_not_ready_is_marked_unusable():
    data = _dataset()
    view = DataStore().add(data).view("NQ", data.primary_bars.last_ts)
    vector = _Honest()._not_ready(view, "because")
    assert not vector.warmup_complete
    assert vector.quality is DataQuality.MISSING
    assert set(vector.values) == {"mean20"}
    assert any("because" in note for note in vector.notes)


def test_bundle_rejects_a_key_collision():
    class Other(_Honest):
        name = "other"

    with pytest.raises(FeatureError, match="declared by both"):
        FeatureBundle([_Honest(), Other()])


def test_bundle_rejects_duplicate_names():
    with pytest.raises(FeatureError, match="duplicate computer names"):
        FeatureBundle([_Honest(), _Honest()])


def test_bundle_warmup_is_the_slowest_computer():
    class Slow(FeatureComputer):
        name = "slow"
        warmup_bars = 200
        keys = ("slow_value",)

        def compute(self, view):
            return self._vector(view, {"slow_value": 1.0})

    bundle = FeatureBundle([_Honest(), Slow()])
    assert bundle.warmup_bars == 200
    assert bundle.keys == ("mean20", "slow_value")


def test_bundle_marks_not_ready_until_the_slowest_window_is_full():
    class Slow(FeatureComputer):
        name = "slow"
        warmup_bars = 200
        keys = ("slow_value",)

        def compute(self, view):
            return self._vector(view, {"slow_value": 1.0})

    data = _dataset(n=400)
    store = DataStore().add(data)
    bundle = FeatureBundle([_Honest(), Slow()])

    early = bundle.compute(store.view("NQ", data.primary_bars.bar_at(100).close_ts))
    assert not early.warmup_complete

    late = bundle.compute(store.view("NQ", data.primary_bars.bar_at(399).close_ts))
    assert late.warmup_complete
    assert set(late.values) == {"mean20", "slow_value"}


def test_bundle_reports_an_absent_required_feed_rather_than_running():
    class NeedsTicks(FeatureComputer):
        name = "needs_ticks"
        warmup_bars = 5
        keys = ("delta_sum",)
        required_feeds = frozenset({Feed.TICK_AGGREGATE})

        def compute(self, view):
            return self._vector(view, {"delta_sum": float(view.deltas(5).sum())})

    data = _dataset()
    store = DataStore().add(data)
    vector = FeatureBundle([NeedsTicks()]).compute(
        store.view("NQ", data.primary_bars.last_ts)
    )
    assert not vector.warmup_complete
    assert any("tick_aggregate" in note for note in vector.notes)


def test_bundle_rejects_an_empty_computer_list():
    data = _dataset()
    view = DataStore().add(data).view("NQ", data.primary_bars.last_ts)
    with pytest.raises(FeatureError, match="no computers"):
        FeatureBundle([]).compute(view)


# --- bounded transforms --------------------------------------------------


def test_percentile_rank_is_bounded_and_monotone():
    window = np.arange(100.0)
    assert percentile_rank(window, -1.0) == 0.0
    assert percentile_rank(window, 1000.0) == 1.0
    assert percentile_rank(window, 50.0) == pytest.approx(0.51)
    assert percentile_rank(window, 25.0) < percentile_rank(window, 75.0)


def test_percentile_rank_handles_empty_and_non_finite():
    assert percentile_rank(np.zeros(0), 1.0) == 0.5
    assert percentile_rank(np.array([np.nan, np.nan]), 1.0) == 0.5


def test_squash_is_bounded():
    """A z-score is unbounded, so one extreme observation would dominate a
    weighted component score. Every magnitude feature uses this instead."""
    assert squash(0.0, 1.0) == 0.0
    assert 0.0 <= squash(1e12, 1.0) <= 1.0
    assert squash(1.0, 1.0) == pytest.approx(0.76159, abs=1e-4)
    assert squash(-5.0, 1.0) == squash(5.0, 1.0)      # magnitude only
    assert squash(float("nan"), 1.0) == 0.0
    with pytest.raises(FeatureError):
        squash(1.0, 0.0)


def test_safe_divide():
    assert safe_divide(1.0, 2.0) == 0.5
    assert safe_divide(1.0, 0.0) == 0.0
    assert safe_divide(1.0, 0.0, default=-1.0) == -1.0
    assert safe_divide(float("inf"), 2.0) == 0.0
    assert safe_divide(1.0, float("nan")) == 0.0
