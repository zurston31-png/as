"""Repairing the shadow returns that are arithmetic, not outcomes.

One unbounded `(exit / entry - 1) * 100` set the champion's mean
per-opportunity return to 10,205,008%, and the promotion gate reported
effect sizes off it. ee6d52a stopped new rows being written that way;
this repairs the ones already on disk. The tests below pin what it
touches and - more importantly - what it must leave alone.
"""
import datetime as dt

import pytest

from app import models
from app.shadow.resolver import MAX_PLAUSIBLE_RETURN_PCT
from scripts.repair_shadow_returns import is_corrupt

NOW = dt.datetime(2026, 9, 27, 12, 0, tzinfo=dt.timezone.utc)


def _row(*, entry_price=1.0, exit_price=1.1, gross=10.0, net=9.5, strategy="champion"):
    return models.ShadowPosition(
        decision_id=1, opportunity_id="op1", strategy_id=strategy,
        token_address="mint", symbol="SYM",
        opened_at=NOW, closed_at=NOW, entry_price=entry_price,
        exit_price=exit_price, gross_return_pct=gross, return_pct=net,
    )


# ---------------------------------------------------------------------------
# what it repairs
# ---------------------------------------------------------------------------

def test_an_impossible_return_is_flagged():
    row = _row(entry_price=0.000001, exit_price=1.0, gross=99_999_900.0,
               net=99_999_899.0)
    why = is_corrupt(row)

    assert why is not None
    assert "recorded" in why
    # the original value is quoted so the repair loses nothing
    assert "99,999,900" in why


def test_a_zero_entry_price_is_flagged_even_with_a_small_return():
    """An entry price of zero could not have produced any return, so
    whatever is recorded against it was not computed from a real trade."""
    row = _row(entry_price=0.0, gross=5.0, net=4.5)
    why = is_corrupt(row)

    assert why is not None
    assert "could never produce a return" in why


def test_the_check_reads_the_gross_figure_where_there_is_one():
    """gross_return_pct is the raw (exit/entry - 1) * 100, so it is where
    the fault actually lands; return_pct is the same value less a cost of
    well under a percent."""
    row = _row(gross=50_000.0, net=49_999.0)
    assert is_corrupt(row) is not None


def test_it_falls_back_to_the_net_figure_when_gross_is_missing():
    """Older rows predate gross_return_pct. They must still be checkable."""
    row = _row(gross=None, net=50_000.0)
    assert is_corrupt(row) is not None


# ---------------------------------------------------------------------------
# what it must NOT repair
# ---------------------------------------------------------------------------

def test_an_ordinary_loss_is_left_alone():
    assert is_corrupt(_row(gross=-10.0, net=-10.5)) is None


def test_an_ordinary_win_is_left_alone():
    assert is_corrupt(_row(gross=33.3, net=32.2)) is None


def test_a_genuine_outlier_is_left_alone():
    """The bound is 100x precisely so a real memecoin move survives it. A
    repair that quietly deleted the biggest winners would bias the record
    in the most damaging direction available."""
    row = _row(entry_price=1.0, exit_price=50.0, gross=4_900.0, net=4_899.0)
    assert is_corrupt(row) is None
    assert row.gross_return_pct < MAX_PLAUSIBLE_RETURN_PCT


def test_an_already_unresolved_row_is_left_alone():
    """A row with no recorded return is already saying the right thing."""
    row = _row(gross=None, net=None)
    assert is_corrupt(row) is None


def test_the_bound_is_applied_symmetrically():
    """A nonsensical LOSS is as much a scale fault as a nonsensical gain,
    and leaving it would drag the mean the other way."""
    row = _row(gross=-99_000.0, net=-99_001.0)
    assert is_corrupt(row) is not None


def test_a_row_exactly_at_the_bound_survives():
    """Strictly greater-than, so the documented threshold is inclusive and
    a row is repaired only when it is past the line, not on it."""
    row = _row(gross=MAX_PLAUSIBLE_RETURN_PCT, net=MAX_PLAUSIBLE_RETURN_PCT - 1)
    assert is_corrupt(row) is None
