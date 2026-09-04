"""Shared dataclasses for the backtesting engine: configuration, a single
completed trade, and the overall result. Kept separate from engine.py and
stats.py so neither has to import the other to get at these.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app.data.candles import Timeframe


@dataclass
class BacktestConfig:
    """Every knob the backtester needs, independent of the live `.env` -
    a backtest must be reproducible from its config alone, not from
    whatever the deployment environment happens to have set.
    """

    starting_balance_usd: float = 1000.0

    # --- realistic execution ---
    # Applied per side (once on entry, once on exit), not round-trip.
    fee_pct: float = 0.0025
    slippage_pct: float = 0.004
    spread_pct: float = 0.0015
    # A decision made on the close of bar i fills at the OPEN of bar
    # i + execution_delay_bars, not at the price that produced the signal -
    # you cannot react to your own close in zero time, and filling at the
    # same bar's close is a classic look-ahead bug.
    execution_delay_bars: int = 1

    # --- entry ---
    min_score_to_enter: float = 75.0
    weights: dict[str, float] | None = None
    higher_timeframe: Timeframe | None = None
    breakout_period: int = 20
    require_regime_tradeable: bool = True
    allowed_trend_regimes: tuple[str, ...] | None = ("bull_trend", "sideways")
    min_reward_risk: float = 1.5
    warmup_bars: int = 210

    # --- position sizing / stop (mirrors app/risk/manager.py's formula) ---
    risk_pct_per_trade: float = 0.02
    max_position_pct_of_portfolio: float = 0.25
    max_trade_size_usd: float = 1e12          # effectively unlimited unless set
    use_atr_stop: bool = True
    atr_period: int = 14
    atr_multiple: float = 2.5
    fallback_stop_loss_pct: float = 0.15
    min_stop_loss_pct: float = 0.03
    reward_risk_multiple: float = 2.0         # take-profit distance = this * stop distance

    # --- portfolio protection (mirrors app/risk/manager.py's gates) ---
    cooldown_bars: int = 3
    max_trades_per_day: int = 8
    max_consecutive_losses: int = 4
    daily_loss_limit_pct: float = 0.05

    # --- exits (passed through to app/exits/manager.py's ExitManager) ---
    exit_overrides: dict | None = None


@dataclass
class BacktestTrade:
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_price: float          # fill price, after slippage/spread
    exit_price: float           # fill price, after slippage/spread
    qty: float
    size_usd: float             # entry notional
    fees_usd: float             # entry + exit fees combined
    pnl_usd: float              # net of fees
    pnl_pct: float
    r_multiple: float           # pnl_usd / (size_usd * stop_distance_pct) at entry
    exit_reason: str
    signal_score: float
    market_regime: str
    bars_held: int

    @property
    def is_win(self) -> bool:
        return self.pnl_usd > 0


@dataclass
class BacktestStats:
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    total_return_pct: float
    total_return_usd: float
    final_balance_usd: float
    avg_win_usd: float | None
    avg_loss_usd: float | None
    profit_factor: float | None
    expectancy_usd: float
    expectancy_r: float | None
    avg_r_multiple: float | None
    max_drawdown_pct: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    longest_winning_streak: int
    longest_losing_streak: int


@dataclass
class BacktestResult:
    symbol: str
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[dt.datetime, float]] = field(default_factory=list)
    stats: BacktestStats | None = None
    rejections: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
