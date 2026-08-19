"""Assembles the full performance picture from the database.

One entry point, `build_performance_report`, so the dashboard, the CLI
script and any future consumer all see the same numbers computed the same
way. Everything below is read-only: this module never writes a row and
never changes a setting.

The report deliberately leads with what is NOT yet known. A performance
page that opens with "+34% return" and buries "on 12 trades" three
sections down is telling the truth in the order most likely to be
misread.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.analysis import trade_analytics as ta
from app.analysis.monte_carlo import MonteCarloResult, run_monte_carlo
from app.analysis.validation import ValidationInputs, ValidationReport, evaluate
from app.config import settings
from app.dashboard.analytics import PortfolioStats, compute_portfolio_stats


@dataclass
class PerformanceReport:
    strategy_version: str | None
    stats: PortfolioStats
    costs: ta.CostSummary
    holding: ta.HoldingTimeSummary
    extremes: ta.Extremes
    breakdowns: list[ta.Breakdown] = field(default_factory=list)
    rejections: ta.RejectionSummary | None = None
    monte_carlo: MonteCarloResult | None = None
    validation: ValidationReport | None = None
    version_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def net_pnl_usd(self) -> float:
        return self.stats.expectancy_usd * self.stats.trade_count

    @property
    def gross_pnl_usd(self) -> float:
        """P&L before execution costs, for the "where did it go?" question.

        Only meaningful when cost data is complete; when it isn't, this
        understates costs and therefore overstates the gross figure, which
        is why the caller is told about the gap rather than left to assume.
        """
        return self.net_pnl_usd + self.costs.total_execution_cost_usd

    def _profit_factor_state(self) -> str:
        pf = self.stats.profit_factor
        if pf is None:
            return "no closed trades yet"
        if not math.isfinite(pf):
            return "no losing trades yet - undefined, not excellent"
        return "computed"

    def as_dict(self) -> dict:
        return {
            "strategy_version": self.strategy_version,
            "trade_count": self.stats.trade_count,
            "net_pnl_usd": round(self.net_pnl_usd, 2),
            "gross_pnl_usd": round(self.gross_pnl_usd, 2),
            "win_rate": round(self.stats.win_rate, 1),
            # An all-winners record gives an infinite profit factor, which
            # is not valid JSON - serialising it raw made this endpoint 500
            # in exactly the early state it most needs to describe. Emitted
            # as null with the reason alongside, never as a large number
            # that would read as a real measurement.
            "profit_factor": (
                self.stats.profit_factor
                if self.stats.profit_factor is not None and math.isfinite(self.stats.profit_factor)
                else None
            ),
            "profit_factor_state": self._profit_factor_state(),
            "expectancy_usd": round(self.stats.expectancy_usd, 2),
            "max_drawdown_pct": round(self.stats.max_drawdown_pct, 2),
            "costs": {
                "total_fees_usd": round(self.costs.total_fees_usd, 2),
                "total_slippage_usd": round(self.costs.total_slippage_usd, 2),
                "total_execution_cost_usd": round(self.costs.total_execution_cost_usd, 2),
                "coverage_pct": round(self.costs.coverage_pct, 1),
                "complete": self.costs.cost_data_complete,
            },
            "breakdowns": [b.as_dict() for b in self.breakdowns],
            "rejections": self.rejections.as_dict() if self.rejections else None,
            "monte_carlo": self.monte_carlo.as_dict() if self.monte_carlo else None,
            "validation": self.validation.as_dict() if self.validation else None,
            "version_counts": dict(self.version_counts),
            "warnings": list(self.warnings),
        }


def build_performance_report(
    db: Session,
    *,
    strategy_version: str | None = None,
    monte_carlo_simulations: int = 2_000,
    rng: random.Random | None = None,
) -> PerformanceReport:
    """Everything known about the record, in one object.

    `strategy_version` restricts the report to one configuration. Passing
    None reports across ALL versions, which is the honest default only when
    a single version exists - so the report says so in its warnings rather
    than presenting a pooled number as if it described one strategy.
    """
    query = db.query(models.Trade)
    if strategy_version is not None:
        query = query.filter(models.Trade.strategy_version == strategy_version)
    trades = query.all()

    warnings: list[str] = []
    version_counts = {
        label: len(rows) for label, rows in ta.split_by_strategy_version(trades).items()
    }
    if strategy_version is None and len(version_counts) > 1:
        warnings.append(
            f"these numbers pool {len(version_counts)} strategy versions "
            f"({', '.join(sorted(version_counts))}) - they describe no single strategy. "
            "Filter to one version before drawing a conclusion."
        )

    stats = compute_portfolio_stats(trades, settings.PORTFOLIO_STARTING_BALANCE_USD)
    costs = ta.summarize_costs(trades)
    if not costs.cost_data_complete:
        warnings.append(
            f"{costs.legs_missing_cost_data} filled leg(s) have no recorded execution cost "
            f"(coverage {costs.coverage_pct:.0f}%) - fees and slippage below are understated, "
            "not complete. Trades written before cost recording existed are the usual cause."
        )

    closed = ta.closed_trades(trades)
    signal_ids = {t.signal_id for t in closed if t.signal_id is not None}
    signals = {
        s.id: s for s in db.query(models.Signal).filter(models.Signal.id.in_(signal_ids)).all()
    } if signal_ids else {}
    liquidity_by_signal = {
        r.signal_id: r.liquidity_usd
        for r in db.query(models.RugCheckResult)
        .filter(models.RugCheckResult.signal_id.in_(signal_ids))
        .all()
    } if signal_ids else {}

    breakdowns = [
        ta.breakdown_by_signal_score(trades, signals),
        ta.breakdown_by_market_quality(trades, signals),
        ta.breakdown_by_liquidity(trades, liquidity_by_signal),
        ta.breakdown_by_holding_time(trades),
        ta.breakdown_by_exit_reason(trades),
    ]

    rejections = ta.summarize_rejections(db.query(models.RiskEvent).all())

    pnls = [t.pnl_usd for t in closed if t.pnl_usd is not None]
    monte_carlo = None
    if pnls:
        monte_carlo = run_monte_carlo(
            pnls,
            starting_equity=settings.PORTFOLIO_STARTING_BALANCE_USD,
            mode="bootstrap",
            simulations=monte_carlo_simulations,
            rng=rng,
        )

    extremes = ta.find_extremes(trades)
    validation = evaluate(
        ValidationInputs(
            closed_trades=stats.trade_count,
            expectancy_usd=stats.expectancy_usd if stats.trade_count else None,
            profit_factor=stats.profit_factor,
            max_drawdown_pct=stats.max_drawdown_pct if stats.trade_count else None,
            best_trade_share_of_profit=extremes.best_trade_share_of_profit,
            winning_trades=stats.win_count,
            monte_carlo_p95_drawdown_pct=monte_carlo.p95_max_drawdown_pct if monte_carlo else None,
            monte_carlo_sample_size=monte_carlo.sample_size if monte_carlo else None,
            # Out-of-sample and walk-forward come from the backtester, not
            # from live rows. Left unmeasured here rather than guessed -
            # scripts/run_backtest.py and scripts/walk_forward.py produce
            # them, and an operator supplies them deliberately.
        )
    )

    return PerformanceReport(
        strategy_version=strategy_version,
        stats=stats,
        costs=costs,
        holding=ta.summarize_holding_time(trades),
        extremes=extremes,
        breakdowns=breakdowns,
        rejections=rejections,
        monte_carlo=monte_carlo,
        validation=validation,
        version_counts=version_counts,
        warnings=warnings,
    )
