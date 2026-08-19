"""Tests for app/analysis/validation.py.

The gate exists to stop one specific mistake: reading an early promising
sample as proof. Most of these tests are therefore about what it REFUSES
to call validated, not about what it accepts.
"""
from app.analysis.validation import (
    MAX_ACCEPTABLE_DRAWDOWN_PCT,
    MIN_WINNERS_FOR_CONCENTRATION,
    MAX_MONTE_CARLO_P95_DRAWDOWN_PCT,
    MAX_SINGLE_TRADE_PROFIT_SHARE,
    MIN_CLOSED_TRADES,
    MIN_PROFIT_FACTOR,
    ValidationInputs,
    ValidationStatus,
    evaluate,
)


def _passing(**overrides) -> ValidationInputs:
    """Inputs that clear every criterion, so a test can break exactly one."""
    defaults = dict(
        closed_trades=250,
        expectancy_usd=4.20,
        profit_factor=1.75,
        max_drawdown_pct=14.0,
        best_trade_share_of_profit=0.18,
        winning_trades=140,
        monte_carlo_p95_drawdown_pct=22.0,
        monte_carlo_sample_size=250,
        out_of_sample_trades=60,
        out_of_sample_profitable=True,
        walk_forward_windows=5,
        walk_forward_profitable_windows=4,
    )
    defaults.update(overrides)
    return ValidationInputs(**defaults)


def _named(report, name):
    return next(c for c in report.criteria if c.name == name)


# ---------------------------------------------------------------------------
# the default is not "fine"
# ---------------------------------------------------------------------------

def test_a_brand_new_strategy_is_experimental():
    report = evaluate(ValidationInputs(
        closed_trades=0, expectancy_usd=None, profit_factor=None, max_drawdown_pct=None,
    ))
    assert report.status is ValidationStatus.EXPERIMENTAL
    assert report.insufficient_evidence


def test_a_tiny_but_perfect_sample_is_not_validated():
    """The exact trap: eleven trades, every one a winner, no losses at all.
    Both the sample size and the infinite profit factor must block it."""
    report = evaluate(_passing(
        closed_trades=11, profit_factor=float("inf"), monte_carlo_sample_size=11,
    ))
    assert report.status is not ValidationStatus.VALIDATED
    assert _named(report, "sample size").evidence_sufficient is False
    assert not _named(report, "profit factor").passed
    assert "small sample" in _named(report, "profit factor").detail


def test_sample_size_alone_blocks_an_otherwise_perfect_record():
    """99 trades with flawless numbers is still EXPERIMENTAL, not FAILING.

    Too few trades means the test hasn't been taken, not that it was
    failed - so the strategy is blocked from VALIDATED without being
    branded a bad one."""
    report = evaluate(_passing(closed_trades=MIN_CLOSED_TRADES - 1))
    assert report.status is ValidationStatus.EXPERIMENTAL
    assert report.failures == []
    assert [c.name for c in report.insufficient_evidence] == ["sample size"]


def test_a_full_record_that_clears_everything_is_validated():
    report = evaluate(_passing())
    assert report.status is ValidationStatus.VALIDATED
    assert report.failures == []
    assert report.insufficient_evidence == []


# ---------------------------------------------------------------------------
# individual criteria
# ---------------------------------------------------------------------------

def test_a_high_win_rate_with_negative_expectancy_fails():
    """Winning often and losing money is a real and common shape - many
    small wins wiped out by a few large losses."""
    report = evaluate(_passing(expectancy_usd=-0.75))
    assert report.status is ValidationStatus.FAILING
    assert not _named(report, "expectancy").passed


def test_barely_profitable_is_not_good_enough():
    """A profit factor just above 1.0 is inside the noise, and real
    execution is worse than simulated."""
    report = evaluate(_passing(profit_factor=1.05))
    assert report.status is ValidationStatus.FAILING
    assert MIN_PROFIT_FACTOR > 1.05


def test_an_unsurvivable_drawdown_fails_however_good_the_endpoint():
    report = evaluate(_passing(max_drawdown_pct=MAX_ACCEPTABLE_DRAWDOWN_PCT + 5))
    assert report.status is ValidationStatus.FAILING
    assert not _named(report, "max drawdown").passed


def test_profit_resting_on_one_trade_fails():
    report = evaluate(_passing(best_trade_share_of_profit=MAX_SINGLE_TRADE_PROFIT_SHARE + 0.2))
    assert report.status is ValidationStatus.FAILING
    assert "rests on one trade" in _named(report, "profit concentration").detail


def test_concentration_says_nothing_with_only_a_few_winners():
    """With three winners the best is at least a third of the profit by
    definition. Reading that as concentration would fail every young
    strategy for doing arithmetic."""
    report = evaluate(_passing(
        best_trade_share_of_profit=1.0,
        winning_trades=MIN_WINNERS_FOR_CONCENTRATION - 1,
    ))
    criterion = _named(report, "profit concentration")
    assert criterion.evidence_sufficient is False
    assert "arithmetic rather than concentration" in criterion.detail


def test_the_resampled_drawdown_can_fail_a_strategy_whose_realized_one_passed():
    """The realized path was lucky in its ordering. That is the whole
    reason the Monte Carlo criterion exists separately."""
    report = evaluate(_passing(
        max_drawdown_pct=12.0,
        monte_carlo_p95_drawdown_pct=MAX_MONTE_CARLO_P95_DRAWDOWN_PCT + 10,
    ))
    assert _named(report, "max drawdown").passed
    assert not _named(report, "monte carlo drawdown").passed
    assert report.status is ValidationStatus.FAILING


def test_an_unprofitable_out_of_sample_result_fails():
    report = evaluate(_passing(out_of_sample_profitable=False))
    assert report.status is ValidationStatus.FAILING
    assert "curve fit" in _named(report, "out-of-sample").detail


def test_walk_forward_needs_most_windows_profitable():
    report = evaluate(_passing(walk_forward_windows=5, walk_forward_profitable_windows=2))
    assert report.status is ValidationStatus.FAILING
    assert not _named(report, "walk-forward").passed


# ---------------------------------------------------------------------------
# unmeasured is not passed
# ---------------------------------------------------------------------------

def test_an_unrun_walk_forward_keeps_the_strategy_experimental():
    report = evaluate(_passing(walk_forward_windows=None, walk_forward_profitable_windows=None))
    assert report.status is ValidationStatus.EXPERIMENTAL
    criterion = _named(report, "walk-forward")
    assert criterion.evidence_sufficient is False
    assert criterion.passed is False       # unmeasured must never read as a pass


def test_too_few_walk_forward_windows_is_unmeasured_not_failed():
    report = evaluate(_passing(walk_forward_windows=1, walk_forward_profitable_windows=1))
    assert _named(report, "walk-forward").evidence_sufficient is False
    assert report.status is ValidationStatus.EXPERIMENTAL


def test_a_monte_carlo_on_too_few_trades_does_not_count():
    report = evaluate(_passing(monte_carlo_sample_size=8, monte_carlo_p95_drawdown_pct=5.0))
    criterion = _named(report, "monte carlo drawdown")
    assert criterion.evidence_sufficient is False
    assert "arithmetic, not evidence" in criterion.detail


def test_a_measured_failure_outranks_a_missing_measurement():
    """Losing money over 300 trades is a real answer. Calling that
    'experimental' because a walk-forward hasn't been run would let a
    known-bad strategy hide behind incomplete testing."""
    report = evaluate(_passing(
        expectancy_usd=-2.0, walk_forward_windows=None, walk_forward_profitable_windows=None,
    ))
    assert report.status is ValidationStatus.FAILING


# ---------------------------------------------------------------------------
# the paper/real boundary
# ---------------------------------------------------------------------------

def test_validated_never_means_cleared_for_real_money():
    """Passing every statistical test means the paper record is strong. It
    is still a paper record - the fills were simulated - so this property
    is False even for a fully validated strategy, by construction."""
    report = evaluate(_passing())
    assert report.status is ValidationStatus.VALIDATED
    assert report.cleared_for_real_money is False
    assert "not about real execution" in report.headline


def test_thresholds_are_not_configurable_from_the_environment():
    """The bar for calling a strategy proven must not be lowerable from
    .env when the strategy fails to clear it."""
    import inspect

    from app.analysis import validation

    source = inspect.getsource(validation)
    assert "settings." not in source


def test_the_report_serialises_and_reads():
    import json

    report = evaluate(_passing(expectancy_usd=-1.0))
    payload = report.as_dict()
    json.dumps(payload)
    assert payload["status"] == "failing"
    assert payload["failure_count"] >= 1

    blank = evaluate(ValidationInputs(
        closed_trades=0, expectancy_usd=None, profit_factor=None, max_drawdown_pct=None,
    ))
    assert "EXPERIMENTAL" in blank.summary()
    assert blank.as_dict()["insufficient_evidence_count"] == len(blank.criteria)
