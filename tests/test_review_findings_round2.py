"""Regression tests for the second review round.

One of these covers a fail-open in code written earlier in the same
session: the paper-only guards added to the operator scripts recognised
only the literal string "true", while the application accepts several
other spellings. A guard that misses `LIVE_TRADING=1` is worse than no
guard, because it reports safety it is not providing.
"""
import datetime as dt
import pathlib

import pytest
from pydantic import ValidationError

from app import models

NOW = dt.datetime.now(dt.timezone.utc)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# the operator guards must reject every spelling the application accepts
# ---------------------------------------------------------------------------

ENABLED_SPELLINGS = ["true", "True", "TRUE", "1", "yes", "on", "y", "t", ' "true" ']
DISABLED_SPELLINGS = ["false", "False", "0", "no", "off", "n", "f", ""]


def _guard_modules():
    """Load both stdlib-only scripts without executing their main()."""
    import importlib.util
    import sys

    loaded = {}
    for name, path in (
        ("guard_setup", "scripts/setup_and_run.py"),
        ("guard_signal", "scripts/send_test_signal.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


@pytest.mark.parametrize("value", ENABLED_SPELLINGS)
def test_every_enabled_spelling_is_recognised(value):
    """pydantic-settings coerces all of these to True, so a bot configured
    with any of them is live. The guard read only "true", which meant
    LIVE_TRADING=1 sailed straight through the check added to stop exactly
    that."""
    for module in _guard_modules().values():
        assert module._env_flag_is_enabled(value) is True, (
            f"{module.__name__} does not treat {value!r} as enabled"
        )


@pytest.mark.parametrize("value", DISABLED_SPELLINGS)
def test_the_disabled_spellings_are_recognised(value):
    """The guard has to be passable, or the launcher never starts."""
    for module in _guard_modules().values():
        assert module._env_flag_is_enabled(value) is False


def test_an_unparseable_value_counts_as_enabled():
    """A refusal path: a value that cannot be read means the configuration
    cannot be confirmed safe, and assuming it is safe is the failure this
    guard exists to prevent."""
    for module in _guard_modules().values():
        assert module._env_flag_is_enabled("maybe") is True


def test_the_guard_matches_what_pydantic_actually_accepts():
    """Pins the guard against the real contract rather than against a list
    someone wrote from memory."""
    from pydantic import BaseModel

    class _M(BaseModel):
        flag: bool = False

    guard = next(iter(_guard_modules().values()))
    for value in ENABLED_SPELLINGS + DISABLED_SPELLINGS:
        try:
            pydantic_says = _M(flag=value.strip().strip('"')).flag
        except ValidationError:
            continue          # pydantic rejects it; the guard's own default applies
        assert guard._env_flag_is_enabled(value) is pydantic_says, (
            f"{value!r}: guard says {guard._env_flag_is_enabled(value)}, "
            f"pydantic says {pydantic_says}"
        )


def test_a_commented_out_flag_is_not_treated_as_enabled(tmp_path):
    """`# LIVE_TRADING=true` is a note, not a setting. Refusing on it would
    make the guard cry wolf on a perfectly safe file."""
    env = tmp_path / ".env"
    env.write_text("# LIVE_TRADING=true\nLIVE_TRADING=false\n")
    for module in _guard_modules().values():
        assert module._live_flags_enabled_in_env(env) == []


def test_an_export_prefixed_flag_is_still_read(tmp_path):
    """`export FOO=bar` is valid in a hand-edited .env."""
    env = tmp_path / ".env"
    env.write_text("export LIVE_TRADING=1\n")
    for module in _guard_modules().values():
        assert module._live_flags_enabled_in_env(env) == ["LIVE_TRADING"]


def test_a_missing_env_file_is_not_a_refusal(tmp_path):
    """Nothing configured yet is the first-run case, not a live one."""
    for module in _guard_modules().values():
        assert module._live_flags_enabled_in_env(tmp_path / "absent.env") == []


# ---------------------------------------------------------------------------
# a measurement that cannot be taken is never recorded as zero
# ---------------------------------------------------------------------------

class _Obs:
    """A stand-in TokenObservation, matching the shape flow_features reads.

    Mirrors the `Obs` helper in tests/test_early_signal.py rather than
    inventing a new one - the real model carries far more than these
    features touch.
    """

    def __init__(self, minutes_ago, *, buys=None, sells=None, liquidity=None):
        self.observed_at = NOW - dt.timedelta(minutes=minutes_ago)
        self.buys_1h, self.sells_1h, self.liquidity_usd = buys, sells, liquidity


def _snapshot(**overrides):
    """A complete MarketSnapshot with the fields under test overridden."""
    from app.services.price_feed import MarketSnapshot

    defaults = dict(
        price_usd=0.004, liquidity_usd=180_000.0, volume_24h_usd=250_000.0,
        buys_24h=600, sells_24h=420, price_change_1h_pct=5.0,
        price_change_24h_pct=20.0, pair_created_at=NOW - dt.timedelta(days=4),
        fdv_usd=900_000.0, token_address="MintRound2",
        volume_1h_usd=14_000.0, market_cap_usd=800_000.0,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


def _feature(features, name):
    return next(f for f in features if f.name == name)


def test_a_half_reported_transaction_count_is_unmeasurable():
    """buys=100, sells=None is not "100 transactions". The old guard
    checked only buys_1h and then folded the missing sells to zero with
    `or 0`, publishing an incomplete observation as a complete one."""
    from app.early.features import flow_features

    obs = [
        _Obs(30, buys=50, sells=50, liquidity=100_000.0),
        _Obs(0, buys=100, sells=None, liquidity=100_000.0),
    ]
    feature = _feature(flow_features(obs), "txn_rate_change")
    assert not feature.available
    assert feature.value is None


def test_a_fully_reported_transaction_count_is_measured():
    """The complement - the guard must not reject complete data."""
    from app.early.features import flow_features

    obs = [
        _Obs(30, buys=50, sells=50, liquidity=100_000.0),
        _Obs(0, buys=100, sells=100, liquidity=100_000.0),
    ]
    feature = _feature(flow_features(obs), "txn_rate_change")
    assert feature.available
    assert feature.value == pytest.approx(2.0)


def test_a_drained_pool_does_not_look_stable():
    """The dangerous one. `if o.liquidity_usd` dropped every zero reading,
    so a pool that emptied had its swing computed over only the surviving
    non-zero depths - and came out looking steady."""
    from app.early.features import flow_features

    depths = [50_000.0, 50_000.0, 50_000.0, 0.0, 0.0]
    obs = [
        _Obs((len(depths) - 1 - i) * 10, buys=100, sells=100, liquidity=d)
        for i, d in enumerate(depths)
    ]
    feature = _feature(flow_features(obs), "liquidity_stability")
    assert feature.available
    assert feature.value == pytest.approx(0.0), (
        "a pool that went from $50k to zero swung 100%, so stability is 0"
    )


def test_an_all_zero_pool_is_unmeasurable_not_perfectly_stable():
    """Every reading zero would divide by zero once the readings are no
    longer filtered out. Reporting 1.0 would claim perfect stability for a
    pool that does not exist."""
    from app.early.features import flow_features

    obs = [_Obs(n * 10, buys=10, sells=10, liquidity=0.0) for n in range(4)][::-1]
    feature = _feature(flow_features(obs), "liquidity_stability")
    assert not feature.available
    assert feature.value is None


@pytest.mark.parametrize("name,overrides", [
    ("volume_to_liquidity", {"volume_24h_usd": 0.0, "liquidity_usd": 50_000.0}),
    ("liquidity_to_marketcap", {"liquidity_usd": 0.0, "market_cap_usd": 1_000_000.0}),
])
def test_a_zero_ratio_is_described_as_a_measurement(name, overrides):
    """A genuine 0.0 - no volume at all against a real pool - is a
    measurement. `if vtl` is false for it, so the detail string said the
    data was missing when it was not."""
    from app.early.features import snapshot_features

    feature = _feature(snapshot_features(_snapshot(**overrides)), name)
    assert feature.available
    assert feature.value == pytest.approx(0.0)
    assert "missing" not in feature.detail


# ---------------------------------------------------------------------------
# fill provenance survives to the database
# ---------------------------------------------------------------------------

def test_the_trade_row_can_record_that_a_fill_was_estimated():
    """The flag is useless if it stops at SwapResult: a quote-derived fill
    would be indistinguishable from a measured one in every later P&L
    query."""
    assert "fill_estimated_from_quote" in models.Trade.__table__.columns


def test_every_place_that_persists_execution_costs_persists_provenance():
    """Three sites copy the execution-cost fields onto the Trade row. All
    three must carry this one too, or provenance is recorded for some
    fills and silently absent for others."""
    source = pathlib.Path("app/services/trading_service.py").read_text()
    assert source.count("trade.fill_delay_seconds = result.fill_delay_seconds") == 3
    assert source.count(
        "trade.fill_estimated_from_quote = result.fill_estimated_from_quote"
    ) == 3
