"""Tests for app/analysis/monte_carlo.py.

The property that makes the module worth having: the realized drawdown of
one path understates the risk, and resampling exposes that. Several tests
below assert exactly that relationship rather than a specific number, so
they stay meaningful if the implementation changes.
"""
import random

import pytest

from app.analysis.monte_carlo import (
    MIN_TRADES_FOR_MONTE_CARLO,
    _percentile,
    _path_stats,
    run_monte_carlo,
)


def _seeded() -> random.Random:
    return random.Random(20260819)


def _sample(n: int = 60, edge: float = 6.0) -> list[float]:
    """A profitable but realistic run: frequent small losses, occasional
    larger wins, positive expectancy."""
    rng = random.Random(7)
    return [rng.choice([-10.0, -8.0, -5.0, 3.0, 12.0, 40.0]) + edge for _ in range(n)]


# ---------------------------------------------------------------------------
# path statistics
# ---------------------------------------------------------------------------

def test_path_stats_on_a_known_sequence():
    final, dd, streak = _path_stats([100.0, -50.0, -50.0, 200.0], starting_equity=1_000.0)
    assert final == pytest.approx(200.0)
    # Peak 1100 -> trough 1000 = 9.09%
    assert dd == pytest.approx(100 / 1100 * 100)
    assert streak == 2


def test_drawdown_is_measured_from_the_peak_not_the_start():
    _, dd, _ = _path_stats([500.0, -500.0], starting_equity=1_000.0)
    assert dd == pytest.approx(500 / 1500 * 100)


def test_a_flat_trade_counts_toward_a_losing_streak():
    """Break-even after costs is not a win, and a run of them is exactly
    the drought that breaks a trader's discipline."""
    _, _, streak = _path_stats([0.0, 0.0, 0.0, 5.0], starting_equity=1_000.0)
    assert streak == 3


def test_percentile_interpolates():
    assert _percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)
    assert _percentile([0.0, 10.0, 20.0], 0.0) == 0.0
    assert _percentile([0.0, 10.0, 20.0], 1.0) == 20.0
    assert _percentile([], 0.5) == 0.0
    assert _percentile([42.0], 0.9) == 42.0


# ---------------------------------------------------------------------------
# shuffle mode: path risk
# ---------------------------------------------------------------------------

def test_shuffle_holds_the_total_fixed():
    """Reordering the same trades cannot change where they end up. If this
    ever fails, the resampler is inventing or dropping trades."""
    pnls = _sample()
    result = run_monte_carlo(
        pnls, starting_equity=1_000.0, mode="shuffle", simulations=200, rng=_seeded()
    )
    expected = sum(pnls)
    assert result.median_final_pnl == pytest.approx(expected)
    assert result.p05_final_pnl == pytest.approx(expected)
    assert result.p95_final_pnl == pytest.approx(expected)


def test_shuffle_says_so_in_its_warnings():
    result = run_monte_carlo(
        _sample(), starting_equity=1_000.0, mode="shuffle", simulations=100, rng=_seeded()
    )
    assert any("same P&L" in w for w in result.warnings)


def test_the_realized_drawdown_understates_the_risk():
    """The single realized path is one draw. Resampling the same trades
    routinely produces a worse ride, and the 95th percentile - not the
    realized figure - is what a position-sizing decision should use."""
    pnls = _sample()
    _, realized_dd, _ = _path_stats(pnls, starting_equity=1_000.0)
    result = run_monte_carlo(
        pnls, starting_equity=1_000.0, mode="shuffle", simulations=1_000, rng=_seeded()
    )
    assert result.p95_max_drawdown_pct > realized_dd
    assert result.worst_max_drawdown_pct >= result.p95_max_drawdown_pct


def test_shuffle_refuses_a_different_path_length():
    with pytest.raises(ValueError, match="shuffle mode reorders"):
        run_monte_carlo(_sample(), starting_equity=1_000.0, mode="shuffle", path_length=10)


# ---------------------------------------------------------------------------
# bootstrap mode: outcome risk
# ---------------------------------------------------------------------------

def test_bootstrap_produces_a_spread_of_outcomes():
    result = run_monte_carlo(
        _sample(), starting_equity=1_000.0, mode="bootstrap", simulations=1_000, rng=_seeded()
    )
    assert result.p05_final_pnl < result.median_final_pnl < result.p95_final_pnl


def test_a_profitable_edge_can_still_lose_over_a_short_run():
    """The number that matters before risking anything: a positive
    expectancy does NOT mean the next 60 trades are profitable."""
    result = run_monte_carlo(
        _sample(edge=2.0), starting_equity=1_000.0, mode="bootstrap",
        simulations=2_000, rng=_seeded(),
    )
    assert result.median_final_pnl > 0
    assert result.probability_of_loss > 0.0


def test_a_losing_edge_almost_always_loses():
    losing = [-20.0, -15.0, -10.0, 5.0, 8.0] * 12
    result = run_monte_carlo(
        losing, starting_equity=1_000.0, mode="bootstrap", simulations=1_000, rng=_seeded()
    )
    assert result.median_final_pnl < 0
    assert result.probability_of_loss > 0.9


def test_bootstrap_can_project_a_longer_run():
    short = run_monte_carlo(
        _sample(), starting_equity=1_000.0, mode="bootstrap",
        simulations=500, path_length=60, rng=_seeded(),
    )
    long = run_monte_carlo(
        _sample(), starting_equity=1_000.0, mode="bootstrap",
        simulations=500, path_length=240, rng=_seeded(),
    )
    assert long.median_final_pnl > short.median_final_pnl


# ---------------------------------------------------------------------------
# honesty about the sample
# ---------------------------------------------------------------------------

def test_a_small_sample_is_flagged_unreliable():
    result = run_monte_carlo(
        [10.0, -5.0, 8.0], starting_equity=1_000.0, simulations=100, rng=_seeded()
    )
    assert result.reliable is False
    assert any("not evidence" in w for w in result.warnings)
    assert result.sample_size == 3


def test_a_large_enough_sample_is_not_flagged():
    result = run_monte_carlo(
        _sample(MIN_TRADES_FOR_MONTE_CARLO), starting_equity=1_000.0,
        mode="bootstrap", simulations=100, rng=_seeded(),
    )
    assert result.reliable is True
    assert not any("not evidence" in w for w in result.warnings)


def test_no_trades_returns_an_empty_result_rather_than_crashing():
    result = run_monte_carlo([], starting_equity=1_000.0)
    assert result.sample_size == 0
    assert result.simulations == 0
    assert result.reliable is False
    assert result.warnings == ["no closed trades to resample"]


def test_seeding_makes_the_run_reproducible():
    a = run_monte_carlo(_sample(), starting_equity=1_000.0, mode="bootstrap",
                        simulations=300, rng=random.Random(1))
    b = run_monte_carlo(_sample(), starting_equity=1_000.0, mode="bootstrap",
                        simulations=300, rng=random.Random(1))
    assert a.as_dict() == b.as_dict()


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown Monte Carlo mode"):
        run_monte_carlo([1.0], starting_equity=1_000.0, mode="wishful")


def test_a_non_positive_starting_equity_is_rejected():
    # Drawdown is a percentage of equity; zero equity makes it meaningless
    # rather than infinite, so refuse rather than emit a nonsense number.
    with pytest.raises(ValueError, match="starting_equity must be positive"):
        run_monte_carlo([1.0], starting_equity=0.0)


def test_as_dict_is_json_safe():
    import json

    result = run_monte_carlo(_sample(), starting_equity=1_000.0, simulations=100, rng=_seeded())
    json.dumps(result.as_dict())
    assert "Monte Carlo" in result.summary()
