"""A backtest may satisfy the gate only if its candles were real.

WHY THIS FILE EXISTS

Two of the eight validation criteria - out-of-sample and walk-forward -
cannot come from live rows. They come from app/backtesting/, and until now
nothing carried them across: the report said "no walk-forward analysis run
yet" no matter how many times one was run.

Connecting them creates a hazard worth more care than the feature. The
backtester DEFAULTS to synthetic candles. A real run of
`scripts/run_backtest.py --walk-forward` on generated history produced, on
this machine, 3 of 3 profitable windows and a profitable 12-trade
out-of-sample window - enough to flip both criteria green on a market that
never existed. Every other honesty rule in this project exists to keep the
record clean; this would poison it at the top.

So the producer records where its candles came from and the reader refuses
anything that is not real market history. These tests pin the refusal,
because the failure mode is silent and flattering.
"""
import json

import pytest

from app.analysis import backtest_evidence as be


def _write(tmp_path, **overrides):
    payload = be.as_payload(
        out_of_sample_trades=40,
        out_of_sample_profitable=True,
        walk_forward_windows=3,
        walk_forward_profitable_windows=3,
        data_source="csv",
        symbol="WIF",
        timeframe="15m",
        candles=2400,
    )
    payload.update(overrides)
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(payload))
    return p


def test_synthetic_candles_are_refused(tmp_path):
    """The exact shape a default --walk-forward run produces."""
    path = _write(tmp_path, data_source="synthetic")
    evidence, message = be.load(path)
    assert evidence is None
    assert "SYNTHETIC" in message
    assert "--csv" in message, "the refusal must say how to produce an admissible run"


def test_real_history_is_accepted(tmp_path):
    evidence, message = be.load(_write(tmp_path))
    assert evidence is not None
    assert evidence.out_of_sample_trades == 40
    assert evidence.walk_forward_profitable_windows == 3
    assert "WIF" in message


def test_an_unknown_source_is_refused_rather_than_assumed_real(tmp_path):
    """Fail closed: a source this build does not recognise is not evidence.

    The alternative - treating anything that is not the string "synthetic"
    as real - would admit a future provider nobody has vetted, and admit a
    typo.
    """
    evidence, message = be.load(_write(tmp_path, data_source="backtest_v2"))
    assert evidence is None
    assert "does not recognise" in message


def test_a_stale_schema_is_refused(tmp_path):
    evidence, message = be.load(_write(tmp_path, schema_version=99))
    assert evidence is None
    assert "schema_version" in message


def test_missing_fields_are_named(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text())
    del payload["out_of_sample_trades"]
    payload["walk_forward_windows"] = None
    path.write_text(json.dumps(payload))
    evidence, message = be.load(path)
    assert evidence is None
    assert "out_of_sample_trades" in message and "walk_forward_windows" in message


def test_more_profitable_windows_than_windows_is_refused(tmp_path):
    """Arithmetic that cannot be true means the file is not trustworthy."""
    evidence, message = be.load(
        _write(tmp_path, walk_forward_windows=3, walk_forward_profitable_windows=4)
    )
    assert evidence is None
    assert "impossible" in message


def test_a_missing_file_reports_rather_than_raises(tmp_path):
    """The caller's right response is to carry on with the criteria
    unmeasured, so this must not be an exception."""
    evidence, message = be.load(tmp_path / "nope.json")
    assert evidence is None
    assert "no such backtest result file" in message


def test_unparseable_json_raises(tmp_path):
    """A corrupt file is an operator error, not an unmeasured criterion -
    silently reporting 'not run' would hide it."""
    p = tmp_path / "wf.json"
    p.write_text("{not json")
    with pytest.raises(be.EvidenceRejected):
        be.load(p)
