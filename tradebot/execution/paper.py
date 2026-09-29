"""Paper execution.

Fills are deliberately pessimistic:

* entry takes configured slippage against you;
* when a candle's range contains both the stop and a target, the stop is
  assumed to have been hit first.

A paper curve that flatters you is worse than no paper curve at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from ..config import Config
from ..models import Candle, ClosedTrade, Fill, Position, Side, TradeSignal, utcnow


class PaperBroker:
    def __init__(self, config: Config, on_close: Optional[Callable[[ClosedTrade], None]] = None) -> None:
        self.cfg = config
        self._positions: list[Position] = []
        self.closed: list[ClosedTrade] = []
        self.on_close = on_close
        self.trades_path = Path(config.execution.trades_path)

    @property
    def positions(self) -> list[Position]:
        return list(self._positions)

    @property
    def open_count(self) -> int:
        return len(self._positions)

    # ---------------------------------------------------------------- open

    def open(self, signal: TradeSignal, qty: float) -> Position:
        slip = self.cfg.execution.slippage_ticks * self.cfg.market.tick_size
        fill_price = signal.entry + slip * signal.side.sign
        pos = Position(
            signal=signal,
            qty=qty,
            entry_price=fill_price,
            opened_at=signal.ts or utcnow(),
            stop=signal.stop,
            tp1=signal.tp1,
            tp2=signal.tp2,
            remaining=qty,
        )
        pos.fills.append(Fill(pos.opened_at, fill_price, qty, "entry"))
        self._positions.append(pos)
        return pos

    # --------------------------------------------------------------- price

    def on_candle(self, candle: Candle) -> list[ClosedTrade]:
        """Advance every open position through one bar. Returns closed trades."""
        closed: list[ClosedTrade] = []
        for pos in list(self._positions):
            result = self._process(pos, candle)
            closed.extend(result)
        return closed

    def _process(self, pos: Position, candle: Candle) -> list[ClosedTrade]:
        closed: list[ClosedTrade] = []
        ex = self.cfg.execution
        long = pos.side is Side.LONG
        hit_stop = candle.low <= pos.stop if long else candle.high >= pos.stop
        hit_tp1 = (candle.high >= pos.tp1 if long else candle.low <= pos.tp1) and not pos.tp1_hit
        hit_tp2 = candle.high >= pos.tp2 if long else candle.low <= pos.tp2

        # Pessimistic ordering: stop wins any bar where both could have printed.
        if hit_stop:
            closed.append(self._close(pos, pos.stop, candle, "stop" if not pos.tp1_hit else "stop_after_tp1"))
            return closed

        if hit_tp1:
            scale = max(0.0, min(1.0, ex.partial_at_tp1))
            qty_out = _round_qty(pos.remaining * scale, self.cfg.market.qty_step)
            if qty_out > 0 and qty_out < pos.remaining:
                self._partial(pos, pos.tp1, qty_out, candle, "tp1")
                pos.tp1_hit = True
                if ex.move_stop_to_breakeven_after_tp1:
                    pos.stop = pos.entry_price
            else:
                closed.append(self._close(pos, pos.tp1, candle, "tp1"))
                return closed

        if hit_tp2 and pos.remaining > 0:
            closed.append(self._close(pos, pos.tp2, candle, "tp2"))
        return closed

    # -------------------------------------------------------------- closes

    def _partial(self, pos: Position, price: float, qty: float, candle: Candle, kind: str) -> None:
        pnl = (price - pos.entry_price) * qty * pos.side.sign * self.cfg.market.point_value
        pnl -= qty * self.cfg.execution.commission_per_unit
        pos.realized += pnl
        pos.remaining = round(pos.remaining - qty, 10)
        pos.fills.append(Fill(candle.ts, price, qty, kind))

    def _close(self, pos: Position, price: float, candle: Candle, reason: str) -> ClosedTrade:
        qty = pos.remaining
        pnl = (price - pos.entry_price) * qty * pos.side.sign * self.cfg.market.point_value
        pnl -= qty * self.cfg.execution.commission_per_unit
        pos.fills.append(Fill(candle.ts, price, qty, reason))
        total_pnl = pos.realized + pnl
        pos.remaining = 0.0

        risk_per_unit = abs(pos.signal.entry - pos.signal.stop) * self.cfg.market.point_value
        r_multiple = total_pnl / (risk_per_unit * pos.qty) if risk_per_unit and pos.qty else 0.0

        trade = ClosedTrade(
            signal_id=pos.signal.id,
            symbol=pos.signal.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            qty=pos.qty,
            pnl=total_pnl,
            r_multiple=r_multiple,
            opened_at=pos.opened_at,
            closed_at=candle.ts,
            exit_reason=reason,
        )
        if pos in self._positions:
            self._positions.remove(pos)
        self.closed.append(trade)
        self._append_log(trade)
        if self.on_close:
            self.on_close(trade)
        return trade

    def close_all(self, price: float, candle: Candle, reason: str = "manual") -> list[ClosedTrade]:
        return [self._close(pos, price, candle, reason) for pos in list(self._positions)]

    def _append_log(self, trade: ClosedTrade) -> None:
        try:
            self.trades_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trades_path.open("a") as fh:
                fh.write(json.dumps(trade.to_dict()) + "\n")
        except OSError:
            pass  # a logging failure must never take the trading loop down


def _round_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return round(round(qty / step) * step, 10)
