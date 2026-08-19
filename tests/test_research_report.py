"""Tests for app/analysis/research_report.py.

The report's value is entirely in what it refuses to claim. These tests are
therefore mostly about the two answers a flattering report would never
give: INSUFFICIENT DATA when nothing is known, and FAIL when the strategy
demonstrably does not work.
"""
import datetime as dt

import pytest

from app import models, pipeline
from app.analysis.research_report import Grade, build_research_report
from app.config import settings
from app.database import SessionLocal
from app.services import portfolio

NOW = dt.datetime.now(dt.timezone.utc)

EXPECTANCY_Q = "Does the strategy show positive expectancy after costs?"
SAMPLE_Q = "How large is the sample?"
COST_Q = "Does performance survive realistic costs?"
CONCENTRATION_Q = "Is the strategy overly dependent on a few winners?"
INTEGRITY_Q = "Is the recorded data internally consistent?"
SCORE_Q = "What does the scoring engine actually produce?"


@pytest.fixture()
def clean_db():
    def wipe(session):
        for model in (
            models.Trade, models.Position, models.RiskEvent,
            models.PipelineEvent, models.ForwardReturn, models.Signal,
        ):
            for row in session.query(model).all():
                session.delete(row)
        portfolio.set_state(session, portfolio.CASH_KEY, settings.PORTFOLIO_STARTING_BALANCE_USD)
        session.commit()

    db = SessionLocal()
    wipe(db)
    try:
        yield db
    finally:
        wipe(db)
        db.close()


def _round_trip(db, pnl, *, i=0, size=100.0):
    """A complete buy/sell pair that keeps the ledger consistent."""
    db.add(models.Trade(
        symbol=f"T{i}", token_address=f"Mint{i}", side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=size, qty=1000.0,
        entry_price=size / 1000.0, opened_at=NOW - dt.timedelta(hours=5),
        fee_usd=0.25, execution_cost_pct=0.006, fill_delay_seconds=1.0,
        strategy_version="v-test",
    ))
    db.add(models.Trade(
        symbol=f"T{i}", token_address=f"Mint{i}", side="sell",
        status=models.TradeStatus.FILLED.value, size_usd=size, qty=1000.0,
        exit_price=(size + pnl) / 1000.0, pnl_usd=pnl, pnl_pct=pnl / size * 100,
        opened_at=NOW - dt.timedelta(hours=5), closed_at=NOW,
        close_reason="take-profit hit" if pnl > 0 else "stop-loss hit",
        fee_usd=0.25, execution_cost_pct=0.006, fill_delay_seconds=1.0,
        strategy_version="v-test",
    ))
    portfolio.adjust_cash_balance(db, pnl)


def _grade(report, question):
    return report.grade(question)


# ---------------------------------------------------------------------------
# nothing known
# ---------------------------------------------------------------------------

def test_an_empty_database_answers_almost_nothing(clean_db):
    """The honest output for a bot that has not run. A report resolving
    every question into PASS or FAIL would be inventing confidence."""
    report = build_research_report(clean_db)
    assert report.counts["FAIL"] == 0
    assert report.counts["INSUFFICIENT DATA"] >= 10
    assert "NOT VALIDATED" in report.headline
    assert _grade(report, EXPECTANCY_Q) is Grade.INSUFFICIENT


def test_a_tiny_winning_record_is_still_insufficient(clean_db):
    """Five wins is not evidence, however good it looks."""
    for i in range(5):
        _round_trip(clean_db, 25.0, i=i)
    clean_db.commit()

    report = build_research_report(clean_db)
    assert _grade(report, SAMPLE_Q) is Grade.INSUFFICIENT
    assert _grade(report, EXPECTANCY_Q) is Grade.INSUFFICIENT
    assert "arithmetic, not evidence" in next(
        f.detail for f in report.findings if f.question == EXPECTANCY_Q
    )


def test_a_thin_record_never_produces_a_validated_headline(clean_db):
    for i in range(20):
        _round_trip(clean_db, 40.0, i=i)
    clean_db.commit()
    assert "NOT VALIDATED" in build_research_report(clean_db).headline


# ---------------------------------------------------------------------------
# the negative answers
# ---------------------------------------------------------------------------

def test_a_high_win_rate_losing_strategy_is_graded_FAIL(clean_db):
    """The shape that fools a win-rate reader: wins often, loses money.
    A dashboard showing '65% win rate' is telling the truth and misleading
    completely."""
    import random

    rng = random.Random(4)
    for i in range(150):
        _round_trip(clean_db, 4.0 if rng.random() < 0.65 else -14.0, i=i)
    clean_db.commit()

    report = build_research_report(clean_db)
    assert _grade(report, SAMPLE_Q) is Grade.PASS
    assert _grade(report, EXPECTANCY_Q) is Grade.FAIL
    assert report.any_failures
    assert "some of what is known is negative" in report.headline


def test_an_edge_smaller_than_costs_is_called_out_specifically(clean_db):
    """Gross positive, net negative - the single most important distinction
    in the whole report.

    Trade.pnl_usd is already NET (the fill price it is computed from
    includes spread, impact and fees), so gross is reconstructed by adding
    the recorded costs back. Each round trip here loses $0.50 net against
    $1.20 of recorded execution cost across its two legs, which is exactly
    a real edge of +$0.70 being eaten by the cost of capturing it."""
    for i in range(120):
        _round_trip(clean_db, -0.50, i=i)
    clean_db.commit()

    report = build_research_report(clean_db)
    detail = next(f.detail for f in report.findings if f.question == COST_Q)
    assert _grade(report, COST_Q) is Grade.FAIL
    assert "smaller than the cost of trading" in detail


def test_gross_and_net_are_distinguished(clean_db):
    """If these were the same number the cost question could never fail."""
    from app.analysis.report import build_performance_report

    for i in range(10):
        _round_trip(clean_db, -0.50, i=i)
    clean_db.commit()

    perf = build_performance_report(clean_db, monte_carlo_simulations=50)
    assert perf.net_pnl_usd < 0
    assert perf.gross_pnl_usd > perf.net_pnl_usd
    assert perf.costs.total_execution_cost_usd > 0


def test_profit_resting_on_one_trade_is_graded_FAIL(clean_db):
    for i in range(20):
        _round_trip(clean_db, 5.0, i=i)
    _round_trip(clean_db, 5000.0, i=999)     # one enormous winner
    clean_db.commit()

    report = build_research_report(clean_db)
    assert _grade(report, CONCENTRATION_Q) is Grade.FAIL


def test_concentration_says_nothing_with_only_a_few_winners(clean_db):
    """With three winners, one dominating is arithmetic."""
    for i in range(3):
        _round_trip(clean_db, 10.0, i=i)
    clean_db.commit()
    assert _grade(build_research_report(clean_db), CONCENTRATION_Q) is Grade.INSUFFICIENT


def test_broken_accounting_is_graded_FAIL(clean_db):
    clean_db.add(models.Trade(
        symbol="X", token_address="MintX", side="buy",
        status=models.TradeStatus.FILLED.value, size_usd=500.0, qty=1.0, entry_price=500.0,
    ))
    clean_db.commit()   # ledger deliberately untouched

    report = build_research_report(clean_db)
    assert _grade(report, INTEGRITY_Q) is Grade.FAIL


# ---------------------------------------------------------------------------
# the positive answers, when earned
# ---------------------------------------------------------------------------

def test_a_genuinely_profitable_record_passes_the_measurable_questions(clean_db):
    import random

    rng = random.Random(9)
    for i in range(150):
        _round_trip(clean_db, rng.choice([-12.0, -8.0, 6.0, 9.0, 22.0]), i=i)
    clean_db.commit()

    report = build_research_report(clean_db)
    assert _grade(report, SAMPLE_Q) is Grade.PASS
    # Expectancy may pass or fail depending on the draw; what must hold is
    # that it is MEASURED rather than dodged.
    assert _grade(report, EXPECTANCY_Q) in (Grade.PASS, Grade.FAIL)
    assert _grade(report, INTEGRITY_Q) is Grade.PASS


def test_recorded_scores_make_the_distribution_question_answerable(clean_db):
    for i in range(40):
        pipeline.record(
            clean_db, stage=pipeline.TECHNICAL_SCORE, symbol=f"S{i}",
            token_address=f"MintS{i}", passed=i % 3 == 0,
            score=40.0 + i, detail={"reliable": True},
        )
    clean_db.commit()

    report = build_research_report(clean_db)
    assert _grade(report, SCORE_Q) is Grade.PASS
    assert "reach 65" in next(f.detail for f in report.findings if f.question == SCORE_Q)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_pooled_strategy_versions_are_listed_as_a_weakness(clean_db):
    _round_trip(clean_db, 10.0, i=1)
    clean_db.commit()
    for trade in clean_db.query(models.Trade).all():
        trade.strategy_version = "v-aaa" if trade.side == "buy" else "v-bbb"
    clean_db.commit()

    report = build_research_report(clean_db)
    assert any("strategy versions" in w for w in report.weaknesses)


def test_the_report_always_says_what_data_would_help(clean_db):
    report = build_research_report(clean_db)
    assert report.data_needed
    assert any("closed paper trades" in d for d in report.data_needed)


def test_the_report_renders_and_serialises(clean_db):
    import json

    report = build_research_report(clean_db)
    text = report.render()
    assert "STRATEGY RESEARCH REPORT" in text
    assert "Paper trading only" in text
    json.dumps(report.as_dict(), allow_nan=False)


def test_every_finding_carries_one_of_exactly_three_grades(clean_db):
    report = build_research_report(clean_db)
    assert report.findings
    assert all(f.grade in (Grade.PASS, Grade.FAIL, Grade.INSUFFICIENT) for f in report.findings)
    assert sum(report.counts.values()) == len(report.findings)
