"""The auction-only bar audit (proposed G1): what it flags, what it must not flag, pass/fail.

Synthetic bars only (TEST FIXTURE); no real market data.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_auction_bars.py"
SESSIONS = [f"2021-{m:02d}-{d:02d}" for m in (1, 2) for d in range(1, 29)]  # 56 sessions


def _build(path: Path) -> None:
    db = sqlite3.connect(path)
    db.execute(
        "create table security_bars (security_id int, session text, symbol text, open real,"
        " high real, low real, close real, volume int, trade_count int)"
    )
    rows = []
    for i, s in enumerate(SESSIONS):
        # 1: liquid, then one auction-only bar at session 40 -> flagged
        rows.append((1, s, "LIQ", 10, 11, 9, 10.5, 100000, 3 if i == 40 else 5000))
        # 2: liquid ticker OLD until session 29; 3: a new thin security reuses the ticker
        if i < 30:
            rows.append((2, s, "OLD", 10, 11, 9, 10.5, 100000, 5000))
        elif i >= 32:
            rows.append((3, s, "OLD", 20, 20, 20, 20, 10, 2))
        # 4: liquid, suspended for 25 sessions, then one thin bar -> not liquid any more
        if i < 25:
            rows.append((4, s, "SUS", 10, 11, 9, 10.5, 100000, 5000))
        elif i == 50:
            rows.append((4, s, "SUS", 5, 5, 5, 5, 1, 1))
    db.executemany("insert into security_bars values (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_flags_only_the_auction_only_bar_of_a_liquid_security(tmp_path):
    path = tmp_path / "bars.db"
    _build(path)
    out = tmp_path / "flagged.tsv"
    result = _run("--db", str(path), "--out", str(out))
    assert result.returncode == 1, result.stdout  # unexplained -> FAIL
    lines = out.read_text().splitlines()[1:]
    assert [line.split("\t")[:3] for line in lines] == [[SESSIONS[40], "1", "LIQ"]]
    # the reused ticker (security 3) and the suspended security (4) are not flagged
    assert "RESULT: FAIL" in result.stdout


def test_passes_when_every_flagged_bar_has_an_explanation_and_treatment(tmp_path):
    path = tmp_path / "bars.db"
    _build(path)
    explained = tmp_path / "explained.tsv"
    explained.write_text(
        "session\tsecurity_id\texplanation\ttreatment\n"
        f"{SESSIONS[40]}\t1\tprovider defect (fixture)\tkept and flagged\n"
    )
    result = _run("--db", str(path), "--explained", str(explained))
    assert result.returncode == 0, result.stdout
    assert "RESULT: PASS" in result.stdout


def test_an_explanation_without_a_treatment_does_not_count(tmp_path):
    path = tmp_path / "bars.db"
    _build(path)
    explained = tmp_path / "explained.tsv"
    explained.write_text(
        f"session\tsecurity_id\texplanation\ttreatment\n{SESSIONS[40]}\t1\tdefect\t\n"
    )
    assert _run("--db", str(path), "--explained", str(explained)).returncode == 1
