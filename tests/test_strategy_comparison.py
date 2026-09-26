from app.backtesting.strategy_comparison import STRATEGY_PROFILES, compare_strategies
from app.backtesting.types import BacktestConfig
from app.data.candles import Timeframe
from app.data.providers import SyntheticCandleProvider
from app.signals.scoring import DEFAULT_WEIGHTS


def _series(regime: str = "bull", seed: int = 1, limit: int = 650):
    return SyntheticCandleProvider(regime=regime, seed=seed).fetch("TESTCOIN", Timeframe.M15, limit=limit)


def test_every_profile_weights_sum_to_the_same_total_as_default():
    default_total = sum(DEFAULT_WEIGHTS.values())
    for name, weights in STRATEGY_PROFILES.items():
        assert sum(weights.values()) == default_total or abs(sum(weights.values()) - default_total) < 1e-9, name


def test_every_profile_has_the_same_factor_names_as_default():
    default_names = set(DEFAULT_WEIGHTS)
    for name, weights in STRATEGY_PROFILES.items():
        assert set(weights) == default_names, name


def test_compare_strategies_ranks_all_profiles():
    series = _series()
    rankings = compare_strategies(series, base_config=BacktestConfig(warmup_bars=210))
    assert {r.label for r in rankings} == set(STRATEGY_PROFILES)


def test_rankings_are_sorted_best_first():
    series = _series()
    rankings = compare_strategies(series, base_config=BacktestConfig(warmup_bars=210))
    scores = [r.rank_score for r in rankings]
    assert scores == sorted(scores, reverse=True)


def test_a_subset_of_profiles_can_be_compared():
    series = _series()
    subset = {"momentum": STRATEGY_PROFILES["momentum"], "trend_following": STRATEGY_PROFILES["trend_following"]}
    rankings = compare_strategies(series, profiles=subset, base_config=BacktestConfig(warmup_bars=210))
    assert {r.label for r in rankings} == {"momentum", "trend_following"}


def test_zero_trade_strategies_rank_last_not_crash():
    """In a bear regime the long-only regime filter should block every
    strategy's entries - all of them should rank at -inf, not raise from
    comparing None-based stats."""
    series = _series(regime="bear")
    rankings = compare_strategies(series, base_config=BacktestConfig(warmup_bars=210))
    assert all(r.rank_score == float("-inf") for r in rankings)


def test_ranking_does_not_depend_on_win_rate_alone():
    """Construct two synthetic stats-equivalent scenarios via the ranking
    function directly: a high-win-rate strategy with a terrible profit
    factor must not automatically outrank a lower-win-rate one with a
    strong profit factor and small drawdown - this proves the ranking
    formula is expectancy/profit-factor/drawdown driven, not win-rate
    driven, matching the spec's explicit "not win rate alone" requirement.
    """
    from app.backtesting.strategy_comparison import _rank_key
    from app.backtesting.types import BacktestResult, BacktestStats
    from app.backtesting.walk_forward import WalkForwardResult

    def _fake_wf(stats: BacktestStats) -> WalkForwardResult:
        empty = BacktestResult(symbol="X")
        oos = BacktestResult(symbol="X", stats=stats)
        return WalkForwardResult(
            symbol="X", selection_metric="expectancy_r", chosen_label="x",
            chosen_config=BacktestConfig(), candidates=[], train=empty, validation=empty,
            out_of_sample=oos,
        )

    high_win_rate_bad_pf = BacktestStats(
        trade_count=10, win_count=9, loss_count=1, win_rate=90.0,
        total_return_pct=-5.0, total_return_usd=-50.0, final_balance_usd=950.0,
        avg_win_usd=5.0, avg_loss_usd=-95.0, profit_factor=0.47,
        expectancy_usd=-5.0, expectancy_r=-0.3, avg_r_multiple=-0.3,
        max_drawdown_pct=40.0, sharpe_ratio=None, sortino_ratio=None,
        longest_winning_streak=9, longest_losing_streak=1,
    )
    lower_win_rate_good_pf = BacktestStats(
        trade_count=10, win_count=4, loss_count=6, win_rate=40.0,
        total_return_pct=20.0, total_return_usd=200.0, final_balance_usd=1200.0,
        avg_win_usd=80.0, avg_loss_usd=-13.3, profit_factor=4.0,
        expectancy_usd=20.0, expectancy_r=1.0, avg_r_multiple=1.0,
        max_drawdown_pct=5.0, sharpe_ratio=None, sortino_ratio=None,
        longest_winning_streak=2, longest_losing_streak=3,
    )
    assert _rank_key(_fake_wf(lower_win_rate_good_pf)) > _rank_key(_fake_wf(high_win_rate_bad_pf))
