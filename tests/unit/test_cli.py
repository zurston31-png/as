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


@pytest.mark.parametrize("command", ["backtest", "walkforward", "montecarlo", "dashboard", "paper", "features"])
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


# --- the data command (Phase 2) ------------------------------------------

DEMO = [
    "-s", "backtest.symbols=[NQ]",
    "-s", "backtest.start=2016-01-01",
    "-s", "backtest.end=2016-02-01",
    "-s", "seal.enabled=false",
]


def test_data_command_reports_55_points_for_bars_only(capsys):
    """The honest headline must survive refactors of the CLI."""
    assert main(["data", "--feeds", "bars", *DEMO]) == 0
    out = capsys.readouterr().out
    assert "available points: 55.0 of 100.0" in out
    assert "UNAVAILABLE: 45.0 points have no data feed" in out
    assert "missing: tick_aggregate" in out
    assert "missing: options_snapshot" in out


def test_data_command_warns_that_synthetic_proves_nothing(capsys):
    assert main(["data", "--feeds", "bars", *DEMO]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out
    assert "real edge" in out


def test_data_command_reports_full_availability_with_every_feed(capsys):
    assert main(["data", *DEMO]) == 0
    out = capsys.readouterr().out
    assert "available points: 100.0 of 100.0" in out
    assert "UNAVAILABLE" not in out


def test_data_command_reports_cleaning_and_provenance(capsys):
    assert main(["data", "--feeds", "bars", *DEMO]) == 0
    out = capsys.readouterr().out
    assert "retention" in out
    assert "data_hash" in out
    assert "combined data_hash" in out


def test_data_command_rejects_an_unknown_feed(capsys):
    assert main(["data", "--feeds", "nonsense", *DEMO]) == 1
    assert "nonsense" in capsys.readouterr().err


def test_data_command_declines_an_unwired_provider(capsys):
    assert main(["data", "--provider", "databento", *DEMO]) == 2
    err = capsys.readouterr().err
    assert "synthetic" in err
    assert "CsvAdapter" in err      # points at what does exist


def test_shortening_the_backtest_without_moving_the_seal_is_rejected(capsys):
    """The seal must stay consistent with the backtest period; silently
    clamping it would be the wrong failure mode."""
    assert main(["data", "-s", "backtest.end=2016-02-01"]) == 1
    assert "sealed window" in capsys.readouterr().err


def test_phases_marks_phase_two_done(capsys):
    assert main(["phases"]) == 0
    out = capsys.readouterr().out
    assert "[x] Phase  2" in out
    assert "[ ] Phase  3" in out
