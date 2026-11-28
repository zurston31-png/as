"""The partial-day audit (proposed): systemic, isolated, after-spike labels; nothing else alerts.

Synthetic bars only (TEST FIXTURE); no real market data.
"""

import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_partial_days.py"


def _sessions(n):
    day, out = date(2019, 1, 7), []
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


SESSIONS = _sessions(120)
SYSTEMIC = SESSIONS[100]


def _build(path):
    db = sqlite3.connect(path)
    db.execute(
        "create table security_bars (security_id int, session text, symbol text, trade_count int)"
    )
    rows = []
    for sid in range(1, 21):  # 20 liquid securities; 10 of them lose their trades on SYSTEMIC
        for i, s in enumerate(SESSIONS):
            n = 5000
            if s == SYSTEMIC and sid <= 10:
                n = 40
            if sid == 11 and i == 90:
                n = 200  # one isolated collapse (4% of normal)
            if sid == 12:  # a spike from session 60 to 79, then a quiet day at 85
                n = 50000 if 60 <= i < 80 else (3000 if i == 85 else 5000)
            rows.append((sid, s, f"S{sid}", n))
    db.executemany("insert into security_bars values (?,?,?,?)", rows)
    db.commit()
    db.close()


def test_labels(tmp_path):
    path, out = tmp_path / "bars.db", tmp_path / "alerts.tsv"
    _build(path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(path), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    got = {
        (r.split("\t")[0], r.split("\t")[2]): r.split("\t")[6]
        for r in out.read_text().splitlines()[1:]
    }
    assert {k for k, v in got.items() if v == "systemic"} == {
        (SYSTEMIC, f"S{sid}") for sid in range(1, 11)
    }
    assert got[(SESSIONS[90], "S11")] == "isolated"
    # 3,000 trades against an inflated 50,000 baseline: low, but only after a spike
    assert got[(SESSIONS[85], "S12")] == "after_volume_spike"
    assert len(got) == 12
