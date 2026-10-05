"""Structured logging."""

from __future__ import annotations

import json

from flow_model.utils.logging import get_logger, new_run_id, reset_logging, setup_logging


def test_run_id_reaches_child_logger_records(tmp_path):
    """Logger-level filters are skipped for propagated records, so the filter
    has to live on the handlers. This caught a real bug."""
    run_id = new_run_id("test")
    setup_logging(level="DEBUG", run_id=run_id, log_dir=tmp_path,
                  json_format=True, console=False)
    get_logger("signals").info("evaluated")
    get_logger().info("root line")
    reset_logging()
    lines = (tmp_path / f"{run_id}.log").read_text().strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["run_id"] == run_id for line in lines)


def test_extra_fields_are_preserved(tmp_path):
    run_id = new_run_id("test")
    setup_logging(level="INFO", run_id=run_id, log_dir=tmp_path,
                  json_format=True, console=False)
    get_logger("signals").info("signal", extra={"symbol": "NQ", "flow_score": 74.0})
    reset_logging()
    record = json.loads((tmp_path / f"{run_id}.log").read_text().strip())
    assert record["symbol"] == "NQ"
    assert record["flow_score"] == 74.0
    assert record["logger"] == "flow_model.signals"


def test_setup_is_idempotent(tmp_path):
    """A sweep that configures logging per fold must not duplicate lines."""
    run_id = new_run_id("test")
    for _ in range(3):
        setup_logging(level="INFO", run_id=run_id, log_dir=tmp_path,
                      json_format=True, console=False)
    get_logger("x").info("once")
    reset_logging()
    assert len((tmp_path / f"{run_id}.log").read_text().strip().splitlines()) == 1


def test_force_reconfigures(tmp_path):
    first = new_run_id("a")
    setup_logging(level="INFO", run_id=first, log_dir=tmp_path, console=False)
    second = new_run_id("b")
    setup_logging(level="INFO", run_id=second, log_dir=tmp_path, console=False, force=True)
    get_logger("x").info("line")
    reset_logging()
    assert (tmp_path / f"{second}.log").exists()


def test_level_is_respected(tmp_path):
    run_id = new_run_id("test")
    setup_logging(level="WARNING", run_id=run_id, log_dir=tmp_path,
                  json_format=True, console=False)
    log = get_logger("x")
    log.debug("hidden")
    log.info("hidden")
    log.warning("shown")
    reset_logging()
    lines = (tmp_path / f"{run_id}.log").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["level"] == "WARNING"


def test_run_ids_are_unique_and_sortable():
    ids = [new_run_id() for _ in range(5)]
    assert len(set(ids)) == 5
    assert all(i.startswith("run_") for i in ids)


def test_exceptions_are_captured(tmp_path):
    run_id = new_run_id("test")
    setup_logging(level="ERROR", run_id=run_id, log_dir=tmp_path,
                  json_format=True, console=False)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("x").exception("failed")
    reset_logging()
    record = json.loads((tmp_path / f"{run_id}.log").read_text().strip())
    assert "ValueError: boom" in record["exception"]
