"""Position state and reconciliation."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import BASE

from tradebot.execution.book import PositionBook
from tradebot.execution.paper import PaperBroker
from tradebot.models import Candle, Side, TradeSignal


def signal(side=Side.LONG, sid=None, entry=100.0, stop=98.0, tp1=104.0, tp2=108.0):
    s = TradeSignal("TEST", "5m", side, entry, stop, tp1, tp2, ts=BASE)
    if sid:
        s.id = sid
    return s


def bar(high, low, close, offset=1):
    return Candle(BASE + timedelta(minutes=5 * offset), low, high, low, close, 100)


# ------------------------------------------------------------------ state

def test_flat_is_the_default():
    state = PositionBook.state_of([])
    assert state.flat and state.state == "flat" and state.side is None


def test_a_long_position_reads_as_long(config):
    broker = PaperBroker(config)
    broker.open(signal(), 10)
    state = PositionBook.state_of(broker.positions)
    assert state.state == "long" and state.side is Side.LONG
    assert state.qty == 10 and state.count == 1 and not state.awaiting_exit


def test_a_runner_past_tp1_is_flagged_as_awaiting_exit(config):
    config.execution.partial_at_tp1 = 0.5
    broker = PaperBroker(config)
    broker.open(signal(), 10)
    broker.on_candle(bar(high=104.5, low=100.5, close=104))

    state = PositionBook.state_of(broker.positions)
    assert state.awaiting_exit                     # half out, half still exposed
    assert state.qty == pytest.approx(5.0)
    assert state.side is Side.LONG                 # still directional


def test_opposing_positions_read_as_mixed(config):
    broker = PaperBroker(config)
    broker.open(signal(Side.LONG, "a"), 5)
    broker.open(signal(Side.SHORT, "b", entry=100, stop=102, tp1=96, tp2=92), 5)
    state = PositionBook.state_of(broker.positions)
    assert state.state == "mixed" and state.side is None and state.count == 2


# ---------------------------------------------------------- reconciliation

def test_a_matching_book_reconciles(config):
    broker, book = PaperBroker(config), PositionBook()
    sig = signal(sid="abc")
    broker.open(sig, 10)
    book.record_entry(sig, 10)

    report = book.reconcile(broker.positions, "now")
    assert report.ok and "1 position(s) match" in report.message


def test_a_position_the_broker_does_not_have_is_caught(config):
    broker, book = PaperBroker(config), PositionBook()
    book.record_entry(signal(sid="ghost"), 10)

    report = book.reconcile(broker.positions, "now")
    assert not report.ok
    assert report.missing == ["ghost"]
    assert "booked but not held" in report.message


def test_a_position_the_book_does_not_know_about_is_caught(config):
    broker, book = PaperBroker(config), PositionBook()
    broker.open(signal(sid="surprise"), 10)

    report = book.reconcile(broker.positions, "now")
    assert not report.ok
    assert report.unexpected == ["surprise"]
    assert "held but not booked" in report.message


def test_a_size_mismatch_is_caught(config):
    broker, book = PaperBroker(config), PositionBook()
    sig = signal(sid="drift")
    broker.open(sig, 10)
    book.record_entry(sig, 7)                      # wrong size booked

    report = book.reconcile(broker.positions, "now")
    assert not report.ok
    assert report.qty_drift == {"drift": [7, 10]}
    assert "size mismatch" in report.message


def test_a_partial_keeps_the_book_in_step(config):
    config.execution.partial_at_tp1 = 0.5
    broker, book = PaperBroker(config), PositionBook()
    sig = signal(sid="runner")
    broker.open(sig, 10)
    book.record_entry(sig, 10)

    broker.on_candle(bar(high=104.5, low=100.5, close=104))
    assert not book.reconcile(broker.positions, "now").ok   # book still says 10

    book.record_partial("runner", broker.positions[0].remaining)
    assert book.reconcile(broker.positions, "now").ok


def test_an_exit_clears_the_booking(config):
    broker, book = PaperBroker(config), PositionBook()
    sig = signal(sid="gone")
    broker.open(sig, 10)
    book.record_entry(sig, 10)

    trades = broker.on_candle(bar(high=101, low=97, close=97.5))
    book.record_exit(trades[0].signal_id)
    assert book.reconcile(broker.positions, "now").ok
    assert book.expected_ids == []


def test_adopt_takes_the_brokers_view(config):
    broker, book = PaperBroker(config), PositionBook()
    broker.open(signal(sid="real"), 10)
    book.record_entry(signal(sid="imaginary"), 5)
    assert not book.reconcile(broker.positions, "now").ok

    book.adopt(broker.positions)
    assert book.reconcile(broker.positions, "now").ok
    assert book.expected_ids == ["real"]
