"""Shared fixtures."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from flow_model.config.loader import default_config, load_config
from flow_model.core.contracts import Bar, CostBreakdown, TradeRecord
from flow_model.core.enums import ExitReason, Regime, SetupType, Side, SplitPhase
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.core.enums import InstrumentType
from flow_model.utils.logging import reset_logging


@pytest.fixture(autouse=True)
def _clean_logging():
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def config():
    return default_config()


@pytest.fixture
def nq() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="NQ",
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        commission_per_side=2.25,
        exchange_fee_per_side=0.37,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
    )


@pytest.fixture
def qqq() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="QQQ",
        instrument_type=InstrumentType.ETF,
        tick_size=0.01,
        tick_value=0.01,
        commission_per_side=0.005,
        commission_per_unit=True,
    )


@pytest.fixture
def make_trade():
    """Factory for a self-consistent TradeRecord.

    Derives gross_pnl and pnl from the price path so a test cannot
    accidentally construct an internally inconsistent record -- the model's
    own validators would reject it anyway.
    """

    def _make(
        *,
        side: Side = Side.LONG,
        entry: float = 18000.0,
        stop: float = 17980.0,
        exit_price: float | None = None,
        rr: float = 1.0,
        size: float = 1.0,
        point_value: float = 20.0,
        costs: CostBreakdown | None = None,
        setup: SetupType = SetupType.SCALP_1R,
        regime: Regime = Regime.LOW_VOL,
        flow_score: float = 75.0,
        signal_ts: datetime | None = None,
        bars_held: int = 5,
        hold_minutes: float = 25.0,
        exit_reason: ExitReason = ExitReason.TARGET,
        split_label: SplitPhase | None = SplitPhase.TRAIN,
        trade_id: str = "t1",
        symbol: str = "NQ",
        mae_r: float = -0.3,
        mfe_r: float = 1.0,
    ) -> TradeRecord:
        signal_ts = signal_ts or datetime(2020, 6, 1, 10, 0, tzinfo=timezone.utc)
        stop_distance = abs(entry - stop)
        target = entry + side.sign * stop_distance * rr
        if exit_price is None:
            exit_price = target
        costs = costs if costs is not None else CostBreakdown()
        gross = side.sign * (exit_price - entry) * point_value * size
        risk = stop_distance * point_value * size
        return TradeRecord(
            trade_id=trade_id,
            run_id="test_run",
            symbol=symbol,
            config_hash="testhash",
            signal_ts=signal_ts,
            entry_ts=signal_ts + timedelta(minutes=1),
            exit_ts=signal_ts + timedelta(minutes=1 + hold_minutes),
            bars_held=bars_held,
            side=side,
            setup=setup,
            planned_entry=entry,
            stop_price=stop,
            target_price=target,
            entry_price=entry,
            exit_price=exit_price,
            size=size,
            filled_size=size,
            point_value=point_value,
            exit_reason=exit_reason,
            risk_dollars=risk,
            gross_pnl=gross,
            costs=costs,
            pnl=gross - costs.total,
            mae_r=mae_r,
            mfe_r=mfe_r,
            flow_score=flow_score,
            regime=regime,
            split_label=split_label,
        )

    return _make


@pytest.fixture
def synthetic_bars():
    """Deterministic synthetic bars for feature/engine tests.

    Not a market simulator -- just a well-formed, reproducible OHLCV series
    with valid OHLC ordering, used to test plumbing rather than edge.
    """

    def _make(
        n: int = 300,
        start_price: float = 18000.0,
        drift: float = 0.0,
        vol: float = 0.0008,
        seed: int = 7,
        interval_seconds: int = 300,
        symbol: str = "NQ",
        start_ts: datetime | None = None,
    ) -> list[Bar]:
        import numpy as np

        rng = np.random.default_rng(seed)
        ts0 = start_ts or datetime(2020, 1, 2, 9, 30, tzinfo=timezone.utc)
        returns = rng.normal(drift, vol, size=n)
        closes = start_price * np.exp(np.cumsum(returns))
        bars: list[Bar] = []
        prev_close = start_price
        for i, close in enumerate(closes):
            wiggle = abs(rng.normal(0.0, vol)) * close
            high = max(prev_close, close) + wiggle
            low = min(prev_close, close) - wiggle
            bars.append(
                Bar(
                    symbol=symbol,
                    close_ts=ts0 + timedelta(seconds=interval_seconds * (i + 1)),
                    open=prev_close,
                    high=high,
                    low=low,
                    close=float(close),
                    volume=float(rng.integers(500, 5000)),
                    interval_seconds=interval_seconds,
                )
            )
            prev_close = float(close)
        return bars

    return _make


@pytest.fixture
def tmp_config(tmp_path):
    """Config whose artifact paths all live under tmp_path."""
    return load_config(
        overrides=[
            f"paths.reports_dir={tmp_path}",
            f"paths.database_path={tmp_path}/flow.db",
            f"paths.experiment_db_path={tmp_path}/experiments.db",
            f"paths.log_dir={tmp_path}/logs",
            f"seal.audit_path={tmp_path}/seal_audit.jsonl",
        ]
    )


@pytest.fixture
def sealed_dates() -> tuple[date, date]:
    return date(2023, 1, 1), date(2025, 1, 1)
