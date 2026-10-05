"""CLI behaviour, including the refusal to fake unbuilt phases."""

from __future__ import annotations

import pytest

from flow_model.cli import main


def test_phases_reports_build_status(capsys):
    assert main(["phases"]) == 0
    out = capsys.readouterr().out
    assert "[x] Phase  1" in out
    assert "[ ] Phase  7" in out
    assert "No backtest has been run" in out


@pytest.mark.parametrize("command", ["backtest", "walkforward", "montecarlo", "dashboard", "paper", "data", "features"])
def test_unbuilt_commands_refuse_rather_than_fabricate(command, capsys):
    """A command that printed a placeholder equity curve would be worse than
    one that exits non-zero."""
    assert main([command]) == 2
    captured = capsys.readouterr()
    assert "not built yet" in captured.err
    assert "Phase" in captured.err
    assert "No placeholder results" in captured.err
    assert captured.out == ""


def test_config_hash_command(capsys):
    assert main(["config", "hash"]) == 0
    assert len(capsys.readouterr().out.strip()) == 16


def test_config_validate(capsys):
    assert main(["config", "validate"]) == 0
    assert "config valid" in capsys.readouterr().out


def test_config_show_section(capsys):
    assert main(["config", "show", "--section", "risk"]) == 0
    assert "risk_per_trade_pct" in capsys.readouterr().out


def test_config_save(tmp_path, capsys):
    out = tmp_path / "c.yaml"
    assert main(["config", "save", "-o", str(out)]) == 0
    assert out.exists()
    assert "config_hash" in capsys.readouterr().out


def test_config_override_from_cli(capsys):
    assert main(["config", "show", "--section", "risk", "-s", "risk.risk_per_trade_pct=0.0025"]) == 0
    assert "0.0025" in capsys.readouterr().out


def test_bad_override_exits_one(capsys):
    assert main(["config", "validate", "-s", "risk.risk_per_trade_pct=0.9"]) == 1
    assert "configuration error" in capsys.readouterr().err


def test_instruments_listing_marks_untradable(capsys):
    assert main(["instruments"]) == 0
    out = capsys.readouterr().out
    assert "NQ" in out and "20" in out
    assert "reference only" in out      # SPX


def test_splits_shows_seal_status(tmp_path, capsys):
    assert main(["splits", "-s", f"seal.audit_path={tmp_path}/a.jsonl"]) == 0
    out = capsys.readouterr().out
    assert "sealed OOS window" in out
    assert "research-usable range" in out
    assert "2023-01-01" in out


def test_experiments_list_empty(tmp_path, capsys):
    assert main(["experiments", "-s", f"paths.experiment_db_path={tmp_path}/e.db"]) == 0
    assert "no experiments logged yet" in capsys.readouterr().out


def test_experiments_summary(tmp_path, capsys):
    assert main(["experiments", "summary", "-s", f"paths.experiment_db_path={tmp_path}/e.db"]) == 0
    assert "effective_trials_on_evaluation_data" in capsys.readouterr().out
