"""The independent completeness audit passes on a complete stage and fails on an incomplete one."""

import json
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from app.universe.database import open_research_database

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_completeness.py"


def _weekdays(first: date, last: date):
    day = first
    while day <= last:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def _build(path: Path, *, bbb_volume: int, stored_median: float = 10_000_000.0) -> None:
    open_research_database(f"sqlite:///{path.as_posix()}").dispose()
    db = sqlite3.connect(path)
    db.execute(
        "insert into securities (security_id, first_session, last_trade_session, last_symbol, "
        "status, exit_type, declared_type, real_trade_sessions) values "
        "(1, '2019-08-01', '2020-02-28', 'AAA', 'trading', 'none', 'unknown', 150)"
    )
    for day in _weekdays(date(2019, 8, 1), date(2020, 2, 28)):
        db.execute(
            "insert into security_bars (security_id, session, symbol, open, high, low, close, "
            "volume, vwap, trade_count, split_factor) values (1, ?, 'AAA', 10, 10, 10, 10, "
            "1000000, 10, 100, 1)",
            (day.isoformat(),),
        )
    db.execute(
        "insert into membership (decision_session, security_id, rule_fingerprint, eligible, "
        "reason, raw_close, median_dollar_volume, rank, member) values "
        "('2020-01-31', 1, 'x', 1, null, 10, ?, 1, 1)",
        (stored_median,),
    )
    # One month of the screen: ranks AAA 1, C2 2, C3 3 (within 3 x N = 3), C4 4, BBB 5.
    bars = {
        "AAA": (10, 10, 21_000_000),
        "C2": (10, 10, 20_000_000),
        "C3": (10, 10, 19_000_000),
        "C4": (1, 1, 100_000_000),  # high below $5: cannot pass the price filter
        "BBB": (20, 0.01, bbb_volume),  # tiny traded value, but a high bound when volume is big
    }
    for rank, (symbol, (high, vwap, volume)) in enumerate(bars.items(), start=1):
        db.execute(
            "insert into monthly_bars (symbol, month, high, close, volume, vwap, trade_count) "
            "values (?, '2020-01-01', ?, ?, ?, ?, 100)",
            (symbol, high, high, volume, vwap),
        )
        selection = "screen" if rank <= 3 else "not_selected"
        db.execute(
            "insert into candidates (symbol, sources, months_with_trades, best_rank, best_month, "
            "selection, reason) values (?, 'sec_ticker_list', 1, ?, '2020-01-01', ?, 'test')",
            (symbol, rank, selection),
        )
        db.execute(
            "insert into external_responses (source, request_key, response, fetched_at) values "
            "('monthly_fetched', ?, ?, '2026-01-01 00:00:00')",
            (symbol, json.dumps(1)),
        )
        if selection == "screen":
            db.execute(
                "insert into batch_symbols (symbol, category, selection, reason, declared_type) "
                "values (?, 'stage', 'stage:screen', 'test', 'unknown')",
                (symbol,),
            )
    db.commit()
    db.close()


def _audit(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(path)], capture_output=True, text=True
    )


def test_a_complete_stage_passes(tmp_path):
    path = tmp_path / "complete.db"
    # BBB's bound: 2 x 20 x 1,000,000 / 63 = 634,921 < N-th member 10,000,000.
    _build(path, bbb_volume=1_000_000)
    result = _audit(path)
    assert result.returncode == 0, result.stdout
    assert "RESULT: PASS" in result.stdout
    assert "at some decision: 0 (of 2 checked at 1 decisions)" in result.stdout


def test_a_left_out_ticker_that_could_have_been_a_member_fails(tmp_path):
    path = tmp_path / "incomplete.db"
    # BBB's bound: 2 x 20 x 20,000,000 / 63 = 12.7 million >= 10 million.
    _build(path, bbb_volume=20_000_000)
    result = _audit(path)
    assert result.returncode == 1
    assert "FAIL tickers without daily data whose bound reaches the N-th member" in result.stdout
    assert "BBB on 2020-01-31" in result.stdout


def test_a_stored_median_that_does_not_recompute_fails(tmp_path):
    path = tmp_path / "wrong_median.db"
    _build(path, bbb_volume=1_000_000, stored_median=9_000_000.0)
    result = _audit(path)
    assert result.returncode == 1
    assert "FAIL member medians recomputed" in result.stdout
