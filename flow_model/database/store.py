"""Reading and writing runs, trades, signals, orders and equity.

Stdlib `sqlite3` only, parameterized SQL only, one statement per table built
from the column tuples in `schema.py` so a column cannot drift away from its
placeholder.

## Storage is not hashing

Every JSON column here goes through `utils/serialization.dumps`, never
`core/determinism.canonical_json`. That distinction is a committed bug fix,
and the bug is worth restating because it is invisible in a diff: canonical
JSON renders floats as fixed-width strings so that a digest is stable across
platforms, which means a win rate of 0.72 read back out of storage as the
STRING "7.200000000000e-01". Hashing and storage are different jobs.
`canonical_json` is for identity; `dumps` is for anything that will be read
back or displayed.

## Round-tripping

`write_run` then `read_trades` returns `TradeRecord`s equal to the ones
written, nested `CostBreakdown` included. Three things make that work:

* floats are stored in `DOUBLE PRECISION` columns, which SQLite keeps as
  8-byte IEEE doubles, so the value read back is bit-identical;
* timestamps are stored as ISO-8601 with an explicit offset, which Pydantic
  parses back to the same aware instant;
* the derived columns (`r_multiple`, `cost_drag_r`, ...) are written for the
  benefit of SQL breakdowns and DROPPED on read, because
  `FrozenModel.from_mapping` keeps only constructor inputs. A derived column
  that was read back in could disagree with the fields it is derived from.

## What this module will not do

It does not compute a statistic. A database that returns a win rate is a
database that has to decide what a win rate over zero trades is, and the
honest answer there -- "undefined, n=0" -- belongs to `analytics/` where it
can be said in a type. Here, an empty run writes one `runs` row, zero trade
rows, and reads back as exactly that.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from flow_model.backtest.journal import OrderRecord, SignalRecord
from flow_model.core.contracts import EquityPoint, TradeRecord
from flow_model.database.schema import (
    EQUITY_COLUMNS,
    ORDER_COLUMNS,
    RUN_COLUMNS,
    SCHEMA_VERSION,
    SEAL_AUDIT_COLUMNS,
    SIGNAL_COLUMNS,
    TRADE_COLUMNS,
    TRADE_DERIVED_COLUMNS,
    create_statements,
)
from flow_model.utils.logging import get_logger
from flow_model.utils.serialization import dumps, loads

__all__ = [
    "Database",
    "DatabaseError",
    "equity_row",
    "order_row",
    "signal_row",
    "trade_row",
]

logger = get_logger("database.store")

_COST_FIELDS = (
    "commission",
    "exchange_fees",
    "entry_slippage",
    "exit_slippage",
    "spread_cost",
)


class DatabaseError(RuntimeError):
    """Raised for a storage problem that must not be papered over."""


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _enum(value) -> str | None:
    return None if value is None else value.value


def trade_row(record: TradeRecord) -> tuple[Any, ...]:
    """One `trades` row, in `TRADE_COLUMNS + TRADE_DERIVED_COLUMNS` order."""
    costs = record.costs
    values: dict[str, Any] = {
        "trade_id": record.trade_id,
        "run_id": record.run_id,
        "symbol": record.symbol,
        "config_hash": record.config_hash,
        "data_hash": record.data_hash,
        "split_label": _enum(record.split_label),
        "signal_ts": _iso(record.signal_ts),
        "entry_ts": _iso(record.entry_ts),
        "exit_ts": _iso(record.exit_ts),
        "bars_held": record.bars_held,
        "session": record.session.value,
        "side": record.side.value,
        "setup": record.setup.value,
        "planned_entry": record.planned_entry,
        "stop_price": record.stop_price,
        "target_price": record.target_price,
        "entry_price": record.entry_price,
        "exit_price": record.exit_price,
        "size": record.size,
        "filled_size": record.filled_size,
        "point_value": record.point_value,
        "exit_reason": record.exit_reason.value,
        "ambiguous_fill": int(record.ambiguous_fill),
        "risk_dollars": record.risk_dollars,
        "gross_pnl": record.gross_pnl,
        "cost_total": costs.total,
        "pnl": record.pnl,
        "mae_r": record.mae_r,
        "mfe_r": record.mfe_r,
        "flow_score": record.flow_score,
        "options_flow_score": record.options_flow_score,
        "order_flow_score": record.order_flow_score,
        "structure_score": record.structure_score,
        "liquidity_score": record.liquidity_score,
        "momentum_score": record.momentum_score,
        "regime": record.regime.value,
        "volatility": record.volatility,
        "vol_percentile": record.vol_percentile,
        "data_quality": record.data_quality.value,
        "entry_reason": record.entry_reason,
        "exit_reason_detail": record.exit_reason_detail,
        "r_multiple": record.r_multiple,
        "planned_r_multiple": record.planned_r_multiple,
        "cost_drag_r": record.cost_drag_r,
        "holding_time_seconds": record.holding_time_seconds,
    }
    for field in _COST_FIELDS:
        values[f"cost_{field}"] = getattr(costs, field)
    return tuple(values[name] for name in TRADE_COLUMNS + TRADE_DERIVED_COLUMNS)


def _trade_from_row(row: sqlite3.Row) -> TradeRecord:
    data = dict(row)
    costs = {field: data.pop(f"cost_{field}") for field in _COST_FIELDS}
    data.pop("cost_total", None)
    for name in TRADE_DERIVED_COLUMNS:
        data.pop(name, None)
    data["costs"] = costs
    return TradeRecord.from_mapping(data)


def signal_row(record: SignalRecord) -> tuple[Any, ...]:
    values: dict[str, Any] = {
        "run_id": record.run_id,
        "symbol": record.symbol,
        "ts": _iso(record.ts),
        "action": record.action.value,
        "setup": _enum(record.setup),
        "flow_score": record.flow_score,
        "confidence": record.confidence,
        "regime": record.regime.value,
        "data_quality": record.data_quality.value,
        "wait_reason": _enum(record.wait_reason),
        "wait_detail": record.wait_detail,
        "blocking_stage": record.blocking_stage,
        "stages_run_json": dumps(record.stages_run),
        "reasons_json": dumps(record.reasons),
        "component_points_json": dumps(record.component_points),
        "reference_price": record.reference_price,
        "atr": record.atr,
        "side": _enum(record.side),
        "intent_built": int(record.intent_built),
        "order_submitted": int(record.order_submitted),
        "equity_at_signal": record.equity_at_signal,
        "sizing_reason": record.sizing_reason,
        "limit_breach": record.limit_breach,
        "split_label": _enum(record.split_label),
    }
    return tuple(values[name] for name in SIGNAL_COLUMNS)


def _signal_from_row(row: sqlite3.Row) -> SignalRecord:
    data = dict(row)
    data["stages_run"] = loads(data.pop("stages_run_json")) or []
    data["reasons"] = loads(data.pop("reasons_json")) or []
    data["component_points"] = loads(data.pop("component_points_json")) or {}
    return SignalRecord.from_mapping(data)


def order_row(record: OrderRecord) -> tuple[Any, ...]:
    values: dict[str, Any] = {
        "run_id": record.run_id,
        "symbol": record.symbol,
        "signal_ts": _iso(record.signal_ts),
        "submitted_ts": _iso(record.submitted_ts),
        "eligible_ts": _iso(record.eligible_ts),
        "order_type": record.order_type,
        "side": record.side.value,
        "status": record.status,
        "requested_size": record.requested_size,
        "filled_size": record.filled_size,
        "fill_ts": _iso(record.fill_ts),
        "fill_bar_offset": record.fill_bar_offset,
        "reference_price": record.reference_price,
        "planned_price": record.planned_price,
        "participation_cap": record.participation_cap,
        "bar_volume": record.bar_volume,
        "eligibility_lag_seconds": record.eligibility_lag_seconds,
        "ambiguous_fill": int(record.ambiguous_fill),
        "miss_reason": record.miss_reason,
        "bars_short_of_horizon": record.bars_short_of_horizon,
        "notes_json": dumps(record.notes),
    }
    for field in _COST_FIELDS:
        values[f"cost_{field}"] = getattr(record.costs, field)
    return tuple(values[name] for name in ORDER_COLUMNS)


def _order_from_row(row: sqlite3.Row) -> OrderRecord:
    data = dict(row)
    costs = {field: data.pop(f"cost_{field}") for field in _COST_FIELDS}
    data["costs"] = costs
    data["notes"] = loads(data.pop("notes_json")) or []
    return OrderRecord.from_mapping(data)


def equity_row(run_id: str, point: EquityPoint) -> tuple[Any, ...]:
    values: dict[str, Any] = {
        "run_id": run_id,
        "ts": _iso(point.ts),
        "equity": point.equity,
        "realized_equity": point.realized_equity,
        "open_positions": point.open_positions,
        "open_risk_dollars": point.open_risk_dollars,
        "high_water_mark": point.high_water_mark,
        "drawdown_pct": point.drawdown_pct,
    }
    return tuple(values[name] for name in EQUITY_COLUMNS)


def _equity_from_row(row: sqlite3.Row) -> EquityPoint:
    data = dict(row)
    data.pop("run_id", None)
    data.pop("drawdown_pct", None)  # computed on the contract
    return EquityPoint.from_mapping(data)


def _run_row(result) -> tuple[Any, ...]:
    provenance = result.provenance
    values: dict[str, Any] = {
        "run_id": provenance.run_id,
        "created_at": _iso(provenance.created_at) or "",
        "symbol": provenance.symbol,
        "config_hash": provenance.config_hash,
        "data_hash": provenance.data_hash,
        "seed": provenance.seed,
        "git_commit": provenance.git_commit,
        "git_dirty": int(provenance.git_dirty),
        "code_version": provenance.code_version,
        "split_label": _enum(provenance.split_label),
        "split_name": provenance.split_name,
        "starting_equity": provenance.starting_equity,
        "compound_equity": int(provenance.compound_equity),
        "record_wait_signals": int(provenance.record_wait_signals),
        "record_equity_every_bar": int(provenance.record_equity_every_bar),
        "bars_in_clock": provenance.bars_in_clock,
        "bars_processed": provenance.bars_processed,
        "max_bars": provenance.max_bars,
        "first_ts": _iso(provenance.first_ts),
        "last_ts": _iso(provenance.last_ts),
        "interval_seconds": provenance.interval_seconds,
        "trade_count": len(result.trades),
        "bars_evaluated": result.bars_evaluated,
        "waits_dropped": result.waits_dropped,
        "open_at_end": result.open_at_end,
        "unresolved_orders": result.unresolved_orders,
        "inputs_fingerprint": provenance.inputs_fingerprint(),
        "result_fingerprint": result.result_fingerprint(),
        "dataset_json": dumps(provenance.dataset),
        "rejection_json": dumps(result.rejection),
        "cost_totals_json": dumps(result.cost_totals),
        "cost_drag_json": dumps(result.cost_drag_by_exit_reason),
        "notes_json": dumps(result.notes),
    }
    return tuple(values[name] for name in RUN_COLUMNS)


def _insert_sql(table: str, columns: Sequence[str]) -> str:
    placeholders = ", ".join("?" for _ in columns)
    return (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    )


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------


class Database:
    """A SQLite file holding one or many runs.

    Natural primary keys everywhere, so writing the same run twice raises
    instead of doubling its trade count. `write_run(..., replace=True)` is the
    explicit way to overwrite, and it deletes the old rows first rather than
    merging two runs into one id.
    """

    def __init__(self, path: str | Path, *, initialize: bool = True) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        if initialize:
            self.initialize()

    # --- lifecycle -----------------------------------------------------

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def initialize(self) -> None:
        with self._conn:
            for statement in create_statements():
                self._conn.execute(statement)
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = ?", ("schema_version",)
        ).fetchone()
        return int(row["value"]) if row else 0

    # --- writing -------------------------------------------------------

    def write_run(self, result, *, replace: bool = False) -> str:
        """Persist a whole run: the run row and all of its rows.

        One transaction, so a failed write leaves no half-run behind. A
        half-written run is the worst of the three outcomes: it looks like a
        complete run with fewer trades.
        """
        run_id = result.provenance.run_id
        if replace:
            self.delete_run(run_id)
        try:
            with self._conn:
                self._conn.execute(_insert_sql("runs", RUN_COLUMNS), _run_row(result))
                self._conn.executemany(
                    _insert_sql("trades", TRADE_COLUMNS + TRADE_DERIVED_COLUMNS),
                    [trade_row(t) for t in result.trades],
                )
                self._conn.executemany(
                    _insert_sql("signals", SIGNAL_COLUMNS),
                    [signal_row(s) for s in result.signals],
                )
                self._conn.executemany(
                    _insert_sql("orders", ORDER_COLUMNS),
                    [order_row(o) for o in result.orders],
                )
                self._conn.executemany(
                    _insert_sql("equity", EQUITY_COLUMNS),
                    [equity_row(run_id, p) for p in result.equity],
                )
        except sqlite3.IntegrityError as exc:
            raise DatabaseError(
                f"run {run_id!r} could not be written: {exc}. The run id is derived "
                "from the config hash, the data hash, the seed and the split, so a "
                "collision means this exact run is already stored. Pass replace=True "
                "to overwrite it deliberately"
            ) from exc
        logger.info(
            "wrote run %s: %d trades, %d signals, %d orders, %d equity points",
            run_id,
            len(result.trades),
            len(result.signals),
            len(result.orders),
            len(result.equity),
        )
        return run_id

    def write_trades(self, trades: Iterable[TradeRecord]) -> int:
        """Insert trade rows on their own, for a caller holding records only."""
        rows = [trade_row(t) for t in trades]
        with self._conn:
            self._conn.executemany(
                _insert_sql("trades", TRADE_COLUMNS + TRADE_DERIVED_COLUMNS), rows
            )
        return len(rows)

    def delete_run(self, run_id: str) -> int:
        """Remove a run and everything belonging to it. Returns rows deleted."""
        deleted = 0
        with self._conn:
            for table in ("trades", "signals", "orders", "equity", "runs"):
                cursor = self._conn.execute(
                    f"DELETE FROM {table} WHERE run_id = ?", (run_id,)
                )
                deleted += cursor.rowcount if cursor.rowcount > 0 else 0
        return deleted

    # --- reading -------------------------------------------------------

    def _select(self, table: str, run_id: str | None, order_by: str) -> list[sqlite3.Row]:
        if run_id is None:
            sql = f"SELECT * FROM {table} ORDER BY {order_by}"
            return list(self._conn.execute(sql))
        sql = f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {order_by}"
        return list(self._conn.execute(sql, (run_id,)))

    def read_trades(self, run_id: str | None = None) -> tuple[TradeRecord, ...]:
        rows = self._select("trades", run_id, "entry_ts, trade_id")
        return tuple(_trade_from_row(row) for row in rows)

    def read_signals(self, run_id: str | None = None) -> tuple[SignalRecord, ...]:
        rows = self._select("signals", run_id, "ts, symbol")
        return tuple(_signal_from_row(row) for row in rows)

    def read_orders(self, run_id: str | None = None) -> tuple[OrderRecord, ...]:
        rows = self._select("orders", run_id, "signal_ts, symbol")
        return tuple(_order_from_row(row) for row in rows)

    def read_equity(self, run_id: str | None = None) -> tuple[EquityPoint, ...]:
        rows = self._select("equity", run_id, "ts")
        return tuple(_equity_from_row(row) for row in rows)

    def read_run(self, run_id: str) -> dict[str, Any] | None:
        """One run row, with its JSON columns decoded. None when absent."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return _decode_run(row)

    def runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs ORDER BY created_at DESC, run_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [_decode_run(row) for row in self._conn.execute(sql)]

    def counts(self) -> dict[str, int]:
        """Row counts per table. Counts, which are defined at zero."""
        out: dict[str, int] = {}
        for table in ("runs", "trades", "signals", "orders", "equity", "experiments", "seal_audit"):
            row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            out[table] = int(row["n"])
        return out

    # --- the seal audit mirror -----------------------------------------

    def import_seal_audit(self, path: str | Path) -> int:
        """Mirror the append-only seal audit JSONL into the `seal_audit` table.

        A MIRROR, one direction only. The JSONL in `validation/splits.py` stays
        the source of truth for the one-open budget: a budget enforced by a
        table that any caller can UPDATE is not a budget. Nothing here opens
        the seal, requests sealed data, or so much as constructs a guard --
        this reads a log file about openings that already happened.
        """
        from flow_model.validation.splits import SealAudit

        entries = SealAudit(path).entries()
        rows = []
        for entry in entries:
            requested = entry.requested
            rows.append(
                tuple(
                    {
                        "sequence": entry.sequence,
                        "ts": _iso(entry.ts),
                        "reason": entry.reason,
                        "experiment_id": entry.experiment_id,
                        "approved_by": entry.approved_by,
                        "config_hash": entry.config_hash,
                        "requested_start": (
                            None if requested is None else requested.start.isoformat()
                        ),
                        "requested_end": (
                            None if requested is None else requested.end.isoformat()
                        ),
                        "pid": entry.pid,
                    }[name]
                    for name in SEAL_AUDIT_COLUMNS
                )
            )
        if not rows:
            return 0
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO seal_audit "
                f"({', '.join(SEAL_AUDIT_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in SEAL_AUDIT_COLUMNS)})",
                rows,
            )
        return len(rows)


def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for column in ("dataset", "rejection", "cost_totals", "cost_drag", "notes"):
        key = f"{column}_json"
        if key in data:
            data[column] = loads(data.pop(key))
    for flag in (
        "git_dirty",
        "compound_equity",
        "record_wait_signals",
        "record_equity_every_bar",
    ):
        if flag in data and data[flag] is not None:
            data[flag] = bool(data[flag])
    return data
