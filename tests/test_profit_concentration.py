"""How few trades the record's profit rests on.

The gate's existing concentration check asks whether one TRADE dominates.
At 51 closed trades this record passed it comfortably - best trade 12% of
gross profit - while four partial profit-takes produced +$43.74 and the
other 47 trades produced -$12.40 between them. No single trade dominated;
four of them did, and nothing measured that.

These tests pin the measure that closes the gap, and - just as important -
pin the two obvious measures that were rejected, so nobody "improves" the
module by adding one back.
"""
import datetime as dt

import pytest

from app import models
from app.analysis import concentration as conc
from app.analysis.validation import ValidationInputs, ValidationStatus, evaluate

START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _exit(pnl: float, reason: str = "trend reversal: lower highs after peak $0.01", n: int = 0):
    return models.Trade(
        symbol="COIN", side="sell", status=models.TradeStatus.FILLED.value,
        pnl_usd=pnl, close_reason=reason,
        closed_at=START + dt.timedelta(minutes=n),
    )


def _book(pnls_and_reasons):
    return [_exit(pnl, reason, n) for n, (pnl, reason) in enumerate(pnls_and_reasons)]


# --- fragility --------------------------------------------------------------

def test_the_number_of_trades_that_carry_the_profit(): 
    """Net is +$10. The best two trades are +$8 and +$7; removing them
    leaves -$5. Removing only the best leaves +$2, still profitable. So
    the answer is 2."""
    book = _book([(8.0, "a"), (7.0, "a"), (-2.0, "b"), (-3.0, "b")])
    report = conc.build(book)
    assert report.net_pnl_usd == pytest.approx(10.0)
    assert report.trades_to_flip == 2
    assert report.flip_share == pytest.approx(0.5)


def test_a_record_carried_by_one_trade():
    book = _book([(50.0, "a"), (-1.0, "b"), (-1.0, "b"), (-1.0, "b")])
    assert conc.build(book).trades_to_flip == 1


def test_an_evenly_spread_record_takes_many_trades_to_flip():
    """The shape the measure exists to distinguish from the one above:
    same net, spread over the whole sample."""
    book = _book([(2.0, "a")] * 30 + [(-1.0, "b")] * 20)
    report = conc.build(book)
    assert report.trades_to_flip == 20
    assert report.survives_bar is True


def test_an_unprofitable_record_is_not_applicable_rather_than_maximally_fragile():
    """0 would read as "one trade carries everything", which is the
    opposite of what a losing record means. None says "wrong question"."""
    report = conc.build(_book([(1.0, "a"), (-5.0, "b")]))
    assert report.net_pnl_usd < 0
    assert report.trades_to_flip is None
    assert report.flip_share is None
    assert report.survives_bar is None


def test_an_empty_record_measures_nothing():
    report = conc.build([])
    assert report.closed_trades == 0
    assert report.trades_to_flip is None
    assert report.load_bearing is None


def test_open_trades_are_excluded():
    book = _book([(5.0, "a"), (-1.0, "b")])
    book.append(models.Trade(symbol="COIN", side="buy", status="filled", opened_at=START))
    assert conc.build(book).closed_trades == 2


# --- the load-bearing exit rule --------------------------------------------

def test_the_load_bearing_rule_and_the_rest_of_the_book():
    """The shape of the real record: a rare, highly profitable exit and a
    common one that roughly breaks even."""
    book = _book(
        [(12.0, "partial profit-take at +N%")] * 4
        + [(0.2, "trend reversal")] * 40
        + [(-4.0, "momentum loss")] * 5
    )
    report = conc.build(book)
    assert report.load_bearing.rule == "partial profit-take at +N%"
    assert report.load_bearing.trade_count == 4
    assert report.load_bearing.total_pnl_usd == pytest.approx(48.0)
    assert report.remainder_trades == 45
    assert report.remainder_pnl_usd == pytest.approx(8.0 - 20.0)


def test_the_load_bearing_rule_is_chosen_by_pnl_not_by_frequency():
    """The common rule is not the load-bearing one. Sorting by trade count
    - which is how the exit-reason breakdown orders its buckets - would
    pick the wrong one."""
    book = _book([(30.0, "rare winner")] + [(0.1, "common")] * 40)
    assert conc.build(book).load_bearing.rule == "rare winner"


def test_trades_with_no_exit_reason_still_count_in_the_remainder():
    """The breakdown skips them; the remainder must not, or "the rest of
    the book" would silently exclude trades that are part of it."""
    book = _book([(10.0, "a")])
    book.append(_exit(-3.0, "", n=9))
    book[-1].close_reason = None
    report = conc.build(book)
    assert report.closed_trades == 2
    assert report.remainder_trades == 1
    assert report.remainder_pnl_usd == pytest.approx(-3.0)


# --- the measures deliberately NOT built ------------------------------------

def test_a_healthy_book_is_not_flagged_merely_for_having_a_profitable_exit():
    """Guards the rejected design. "Is the remainder positive after
    removing the best rule?" fails EVERY strategy: strip the profitable
    exit from any book and what is left is the stop losses. This book is
    healthy - profit spread over 30 trades across two exits - and must
    read as robust even though its remainder is negative."""
    book = _book(
        [(6.0, "take profit")] * 30
        + [(3.0, "trailing stop")] * 20
        + [(-3.0, "stop loss")] * 50
    )
    report = conc.build(book)
    assert report.net_pnl_usd == pytest.approx(90.0)
    assert report.remainder_pnl_usd < 0          # the rejected measure would fail it
    assert report.survives_bar is True           # the measure actually used does not


# --- the gate ---------------------------------------------------------------

def _inputs(**kw):
    base = dict(
        closed_trades=51, expectancy_usd=0.61, profit_factor=1.30,
        max_drawdown_pct=3.1, best_trade_share_of_profit=0.12, winning_trades=29,
        monte_carlo_p95_drawdown_pct=6.6, monte_carlo_sample_size=51,
        out_of_sample_trades=40, out_of_sample_profitable=True,
        walk_forward_windows=5, walk_forward_profitable_windows=4,
    )
    base.update(kw)
    return ValidationInputs(**base)


def _criterion(report, name):
    return next(c for c in report.criteria if c.name == name)


def test_fragility_is_reported_by_the_gate():
    c = _criterion(evaluate(_inputs(trades_to_flip=3)), "profit fragility")
    assert c.state == "FAIL"
    assert "3 of 51" in c.detail


def test_a_failing_fragility_check_never_changes_the_verdict():
    """The point of the whole design. MIN_FRAGILITY_SHARE is a judgement,
    not a derived bound, so it must not silently decide whether a strategy
    reads as VALIDATED. With every blocking criterion cleared and 100
    trades, a fragile record is still VALIDATED - and says FAIL on the
    fragility line so a person can see it and decide."""
    report = evaluate(_inputs(closed_trades=100, trades_to_flip=3))
    assert _criterion(report, "profit fragility").state == "FAIL"
    assert report.status is ValidationStatus.VALIDATED
    assert "profit fragility" not in report.headline


def test_an_advisory_failure_is_visible_but_not_blamed_for_the_verdict():
    """It appears in `failures` - nothing that failed is hidden - but not
    in `blocking_failures`, and `failure_count` counts the latter. The
    JSON would otherwise carry `status: validated` and `failure_count: 1`
    in the same payload."""
    report = evaluate(_inputs(closed_trades=100, trades_to_flip=3))
    assert [c.name for c in report.failures] == ["profit fragility"]
    assert report.blocking_failures == []
    payload = report.as_dict()
    assert payload["failure_count"] == 0
    assert payload["advisory_failure_count"] == 1


def test_an_unprofitable_record_reports_insufficient_evidence_not_failure():
    c = _criterion(evaluate(_inputs(trades_to_flip=None)), "profit fragility")
    assert c.state == "insufficient data"
    assert "expectancy" in c.detail


def test_too_few_winners_to_say_anything():
    """Same guard as the single-trade criterion: with a handful of
    winners, a handful carrying the profit is arithmetic."""
    c = _criterion(evaluate(_inputs(winning_trades=4, trades_to_flip=2)), "profit fragility")
    assert c.state == "insufficient data"


def test_a_robust_record_passes():
    c = _criterion(evaluate(_inputs(closed_trades=100, trades_to_flip=15)), "profit fragility")
    assert c.state == "pass"
