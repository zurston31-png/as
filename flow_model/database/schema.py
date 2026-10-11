"""SQLite DDL for the run artifacts, written to be Postgres-compatible.

Section 12 names the tables: `runs`, `trades`, `signals`, `equity`,
`experiments`, `seal_audit`. All six are declared here, plus `orders` --
see below.

## Why the types look like this

SQLite's column types are advisory and Postgres's are not, so the DDL is
written for the stricter one and SQLite's affinity rules are left to do what
they do:

* `DOUBLE PRECISION` rather than `REAL`. In SQLite both mean an 8-byte IEEE
  double; in Postgres `REAL` is 4 bytes, which would silently round every
  price and every PnL on a migration.
* `TEXT` for timestamps, holding ISO-8601 with an explicit UTC offset. A
  `TIMESTAMP` column in SQLite has NUMERIC affinity, and an ISO string stored
  there stays text anyway -- so the column type would promise a conversion
  that does not happen. ISO text sorts chronologically and round-trips to an
  aware `datetime` exactly, and Postgres casts it to `timestamptz` on demand.
* `BOOLEAN`, which SQLite stores as 0/1 and Pydantic reads back as a bool.
* No `AUTOINCREMENT`, no `WITHOUT ROWID`, no SQLite-only pragmas in the DDL.
  Every key is a natural key, which also makes the write idempotent to
  detect: inserting a run twice violates a primary key instead of quietly
  doubling the trade count.

## `orders`, which section 12 does not name

An entry attempt that was MISSED is neither a signal nor a trade. With only
the six named tables it disappears, and "the signal fired and the limit never
traded through" becomes indistinguishable from "no signal fired" -- two
findings with the same empty trade list, only one of which is about the
strategy. The table is an addition to section 12's list, recorded here as
such.

## `experiments` and `seal_audit`

`experiments` mirrors `validation/experiment_log.py`'s own DDL column for
column, so one database file can hold both and `ExperimentLog` remains the
only writer. `seal_audit` is a queryable MIRROR of the append-only JSONL log
in `validation/splits.py`; that file stays the source of truth, because a
seal-opening budget that can be edited with an UPDATE is not a budget.
"""

from __future__ import annotations

__all__ = [
    "EQUITY_COLUMNS",
    "INDEXES",
    "ORDER_COLUMNS",
    "RUN_COLUMNS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SEAL_AUDIT_COLUMNS",
    "SIGNAL_COLUMNS",
    "TABLES",
    "TRADE_COLUMNS",
    "TRADE_DERIVED_COLUMNS",
    "create_statements",
]

#: Bumped when a column is added, removed or retyped.
SCHEMA_VERSION = 1

TABLES = (
    "runs",
    "trades",
    "signals",
    "orders",
    "equity",
    "experiments",
    "seal_audit",
    "schema_meta",
)


# ---------------------------------------------------------------------------
# Column orders. One definition, used to build both the DDL and the INSERTs,
# so a column can never drift out of step with its placeholder.
# ---------------------------------------------------------------------------

RUN_COLUMNS = (
    "run_id",
    "created_at",
    "symbol",
    "config_hash",
    "data_hash",
    "seed",
    "git_commit",
    "git_dirty",
    "code_version",
    "split_label",
    "split_name",
    "starting_equity",
    "compound_equity",
    "record_wait_signals",
    "record_equity_every_bar",
    "bars_in_clock",
    "bars_processed",
    "max_bars",
    "first_ts",
    "last_ts",
    "interval_seconds",
    "trade_count",
    "bars_evaluated",
    "waits_dropped",
    "open_at_end",
    "unresolved_orders",
    "inputs_fingerprint",
    "result_fingerprint",
    "dataset_json",
    "rejection_json",
    "cost_totals_json",
    "cost_drag_json",
    "notes_json",
)

#: `TradeRecord`'s constructor inputs, flattened. `costs` becomes five
#: `cost_*` columns; everything else is one field to one column.
TRADE_COLUMNS = (
    "trade_id",
    "run_id",
    "symbol",
    "config_hash",
    "data_hash",
    "split_label",
    "signal_ts",
    "entry_ts",
    "exit_ts",
    "bars_held",
    "session",
    "side",
    "setup",
    "planned_entry",
    "stop_price",
    "target_price",
    "entry_price",
    "exit_price",
    "size",
    "filled_size",
    "point_value",
    "exit_reason",
    "ambiguous_fill",
    "risk_dollars",
    "gross_pnl",
    "cost_commission",
    "cost_exchange_fees",
    "cost_entry_slippage",
    "cost_exit_slippage",
    "cost_spread_cost",
    "cost_total",
    "pnl",
    "mae_r",
    "mfe_r",
    "flow_score",
    "options_flow_score",
    "order_flow_score",
    "structure_score",
    "liquidity_score",
    "momentum_score",
    "regime",
    "volatility",
    "vol_percentile",
    "data_quality",
    "entry_reason",
    "exit_reason_detail",
)

#: Computed on the record and stored so the breakdowns in section 12 can be
#: written as SQL. Dropped on read (`FrozenModel.from_mapping` keeps only
#: constructor inputs), so they cannot diverge from the record that produced
#: them by being read back in.
TRADE_DERIVED_COLUMNS = (
    "r_multiple",
    "planned_r_multiple",
    "cost_drag_r",
    "holding_time_seconds",
)

SIGNAL_COLUMNS = (
    "run_id",
    "symbol",
    "ts",
    "action",
    "setup",
    "flow_score",
    "confidence",
    "regime",
    "data_quality",
    "wait_reason",
    "wait_detail",
    "blocking_stage",
    "stages_run_json",
    "reasons_json",
    "component_points_json",
    "reference_price",
    "atr",
    "side",
    "intent_built",
    "order_submitted",
    "equity_at_signal",
    "sizing_reason",
    "limit_breach",
    "split_label",
)

ORDER_COLUMNS = (
    "run_id",
    "symbol",
    "signal_ts",
    "submitted_ts",
    "eligible_ts",
    "order_type",
    "side",
    "status",
    "requested_size",
    "filled_size",
    "fill_ts",
    "fill_bar_offset",
    "reference_price",
    "planned_price",
    "participation_cap",
    "bar_volume",
    "eligibility_lag_seconds",
    "ambiguous_fill",
    "miss_reason",
    "cost_commission",
    "cost_exchange_fees",
    "cost_entry_slippage",
    "cost_exit_slippage",
    "cost_spread_cost",
    "bars_short_of_horizon",
    "notes_json",
)

EQUITY_COLUMNS = (
    "run_id",
    "ts",
    "equity",
    "realized_equity",
    "open_positions",
    "open_risk_dollars",
    "high_water_mark",
    "drawdown_pct",
)

SEAL_AUDIT_COLUMNS = (
    "sequence",
    "ts",
    "reason",
    "experiment_id",
    "approved_by",
    "config_hash",
    "requested_start",
    "requested_end",
    "pid",
)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key                     TEXT PRIMARY KEY,
    value                   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT PRIMARY KEY,
    created_at              TEXT NOT NULL DEFAULT '',
    symbol                  TEXT NOT NULL,
    config_hash             TEXT NOT NULL,
    data_hash               TEXT NOT NULL DEFAULT '',
    seed                    BIGINT NOT NULL,
    git_commit              TEXT NOT NULL DEFAULT '',
    git_dirty               BOOLEAN NOT NULL DEFAULT 0,
    code_version            TEXT NOT NULL DEFAULT '',
    split_label             TEXT,
    split_name              TEXT NOT NULL DEFAULT '',
    starting_equity         DOUBLE PRECISION NOT NULL,
    compound_equity         BOOLEAN NOT NULL DEFAULT 1,
    record_wait_signals     BOOLEAN NOT NULL DEFAULT 1,
    record_equity_every_bar BOOLEAN NOT NULL DEFAULT 1,
    bars_in_clock           BIGINT NOT NULL DEFAULT 0,
    bars_processed          BIGINT NOT NULL DEFAULT 0,
    max_bars                BIGINT,
    first_ts                TEXT,
    last_ts                 TEXT,
    interval_seconds        BIGINT NOT NULL DEFAULT 0,
    trade_count             BIGINT NOT NULL DEFAULT 0,
    bars_evaluated          BIGINT NOT NULL DEFAULT 0,
    waits_dropped           BIGINT NOT NULL DEFAULT 0,
    open_at_end             BIGINT NOT NULL DEFAULT 0,
    unresolved_orders       BIGINT NOT NULL DEFAULT 0,
    inputs_fingerprint      TEXT NOT NULL DEFAULT '',
    result_fingerprint      TEXT NOT NULL DEFAULT '',
    dataset_json            TEXT NOT NULL DEFAULT '{}',
    rejection_json          TEXT NOT NULL DEFAULT '{}',
    cost_totals_json        TEXT NOT NULL DEFAULT '{}',
    cost_drag_json          TEXT NOT NULL DEFAULT '{}',
    notes_json              TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id                TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    config_hash             TEXT NOT NULL DEFAULT '',
    data_hash               TEXT NOT NULL DEFAULT '',
    split_label             TEXT,
    signal_ts               TEXT NOT NULL,
    entry_ts                TEXT NOT NULL,
    exit_ts                 TEXT NOT NULL,
    bars_held               BIGINT NOT NULL DEFAULT 0,
    session                 TEXT NOT NULL DEFAULT 'CLOSED',
    side                    TEXT NOT NULL,
    setup                   TEXT NOT NULL,
    planned_entry           DOUBLE PRECISION NOT NULL,
    stop_price              DOUBLE PRECISION NOT NULL,
    target_price            DOUBLE PRECISION NOT NULL,
    entry_price             DOUBLE PRECISION NOT NULL,
    exit_price              DOUBLE PRECISION NOT NULL,
    size                    DOUBLE PRECISION NOT NULL,
    filled_size             DOUBLE PRECISION NOT NULL,
    point_value             DOUBLE PRECISION NOT NULL,
    exit_reason             TEXT NOT NULL,
    ambiguous_fill          BOOLEAN NOT NULL DEFAULT 0,
    risk_dollars            DOUBLE PRECISION NOT NULL,
    gross_pnl               DOUBLE PRECISION NOT NULL,
    cost_commission         DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_exchange_fees      DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_entry_slippage     DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_exit_slippage      DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_spread_cost        DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_total              DOUBLE PRECISION NOT NULL DEFAULT 0,
    pnl                     DOUBLE PRECISION NOT NULL,
    mae_r                   DOUBLE PRECISION NOT NULL DEFAULT 0,
    mfe_r                   DOUBLE PRECISION NOT NULL DEFAULT 0,
    flow_score              DOUBLE PRECISION NOT NULL,
    options_flow_score      DOUBLE PRECISION NOT NULL DEFAULT 0,
    order_flow_score        DOUBLE PRECISION NOT NULL DEFAULT 0,
    structure_score         DOUBLE PRECISION NOT NULL DEFAULT 0,
    liquidity_score         DOUBLE PRECISION NOT NULL DEFAULT 0,
    momentum_score          DOUBLE PRECISION NOT NULL DEFAULT 0,
    regime                  TEXT NOT NULL,
    volatility              DOUBLE PRECISION,
    vol_percentile          DOUBLE PRECISION,
    data_quality            TEXT NOT NULL DEFAULT 'GOOD',
    entry_reason            TEXT NOT NULL DEFAULT '',
    exit_reason_detail      TEXT NOT NULL DEFAULT '',
    r_multiple              DOUBLE PRECISION,
    planned_r_multiple      DOUBLE PRECISION,
    cost_drag_r             DOUBLE PRECISION,
    holding_time_seconds    DOUBLE PRECISION,
    PRIMARY KEY (run_id, trade_id)
);

CREATE TABLE IF NOT EXISTS signals (
    run_id                  TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    ts                      TEXT NOT NULL,
    action                  TEXT NOT NULL,
    setup                   TEXT,
    flow_score              DOUBLE PRECISION,
    confidence              DOUBLE PRECISION NOT NULL DEFAULT 0,
    regime                  TEXT NOT NULL DEFAULT 'UNKNOWN',
    data_quality            TEXT NOT NULL DEFAULT 'MISSING',
    wait_reason             TEXT,
    wait_detail             TEXT NOT NULL DEFAULT '',
    blocking_stage          TEXT NOT NULL DEFAULT '',
    stages_run_json         TEXT NOT NULL DEFAULT '[]',
    reasons_json            TEXT NOT NULL DEFAULT '[]',
    component_points_json   TEXT NOT NULL DEFAULT '{}',
    reference_price         DOUBLE PRECISION,
    atr                     DOUBLE PRECISION,
    side                    TEXT,
    intent_built            BOOLEAN NOT NULL DEFAULT 0,
    order_submitted         BOOLEAN NOT NULL DEFAULT 0,
    equity_at_signal        DOUBLE PRECISION,
    sizing_reason           TEXT NOT NULL DEFAULT '',
    limit_breach            TEXT NOT NULL DEFAULT '',
    split_label             TEXT,
    PRIMARY KEY (run_id, symbol, ts)
);

CREATE TABLE IF NOT EXISTS orders (
    run_id                  TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    signal_ts               TEXT NOT NULL,
    submitted_ts            TEXT NOT NULL,
    eligible_ts             TEXT NOT NULL,
    order_type              TEXT NOT NULL,
    side                    TEXT NOT NULL,
    status                  TEXT NOT NULL,
    requested_size          DOUBLE PRECISION NOT NULL,
    filled_size             DOUBLE PRECISION NOT NULL DEFAULT 0,
    fill_ts                 TEXT,
    fill_bar_offset         BIGINT NOT NULL DEFAULT 0,
    reference_price         DOUBLE PRECISION,
    planned_price           DOUBLE PRECISION NOT NULL DEFAULT 0,
    participation_cap       DOUBLE PRECISION,
    bar_volume              DOUBLE PRECISION,
    eligibility_lag_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
    ambiguous_fill          BOOLEAN NOT NULL DEFAULT 0,
    miss_reason             TEXT NOT NULL DEFAULT '',
    cost_commission         DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_exchange_fees      DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_entry_slippage     DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_exit_slippage      DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_spread_cost        DOUBLE PRECISION NOT NULL DEFAULT 0,
    bars_short_of_horizon   BIGINT NOT NULL DEFAULT 0,
    notes_json              TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (run_id, symbol, signal_ts)
);

CREATE TABLE IF NOT EXISTS equity (
    run_id                  TEXT NOT NULL,
    ts                      TEXT NOT NULL,
    equity                  DOUBLE PRECISION NOT NULL,
    realized_equity         DOUBLE PRECISION NOT NULL,
    open_positions          BIGINT NOT NULL DEFAULT 0,
    open_risk_dollars       DOUBLE PRECISION NOT NULL DEFAULT 0,
    high_water_mark         DOUBLE PRECISION NOT NULL DEFAULT 0,
    drawdown_pct            DOUBLE PRECISION,
    PRIMARY KEY (run_id, ts)
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id        TEXT PRIMARY KEY,
    sequence             INTEGER NOT NULL,
    created_at           TEXT    NOT NULL,
    name                 TEXT    NOT NULL,
    git_commit           TEXT    NOT NULL DEFAULT '',
    git_dirty            INTEGER NOT NULL DEFAULT 0,
    config_hash          TEXT    NOT NULL,
    config_json          TEXT    NOT NULL DEFAULT '',
    dataset_id           TEXT    NOT NULL DEFAULT '',
    data_hash            TEXT    NOT NULL DEFAULT '',
    split_id             TEXT    NOT NULL DEFAULT '',
    phase                TEXT    NOT NULL,
    params_changed_json  TEXT    NOT NULL DEFAULT '{}',
    reason               TEXT    NOT NULL DEFAULT '',
    results_json         TEXT    NOT NULL DEFAULT '{}',
    notes                TEXT    NOT NULL DEFAULT '',
    parent_experiment_id TEXT,
    seal_opened          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS seal_audit (
    sequence                BIGINT PRIMARY KEY,
    ts                      TEXT NOT NULL,
    reason                  TEXT NOT NULL,
    experiment_id           TEXT NOT NULL DEFAULT '',
    approved_by             TEXT NOT NULL DEFAULT '',
    config_hash             TEXT NOT NULL DEFAULT '',
    requested_start         TEXT,
    requested_end           TEXT,
    pid                     BIGINT NOT NULL DEFAULT 0
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_trades_run        ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_exit       ON trades(exit_reason);
CREATE INDEX IF NOT EXISTS idx_trades_regime     ON trades(regime);
CREATE INDEX IF NOT EXISTS idx_trades_setup      ON trades(setup);
CREATE INDEX IF NOT EXISTS idx_trades_split      ON trades(split_label);
CREATE INDEX IF NOT EXISTS idx_trades_entry_ts   ON trades(entry_ts);
CREATE INDEX IF NOT EXISTS idx_signals_run       ON signals(run_id);
CREATE INDEX IF NOT EXISTS idx_signals_action    ON signals(action);
CREATE INDEX IF NOT EXISTS idx_signals_wait      ON signals(wait_reason);
CREATE INDEX IF NOT EXISTS idx_signals_stage     ON signals(blocking_stage);
CREATE INDEX IF NOT EXISTS idx_orders_run        ON orders(run_id);
CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders(status);
CREATE INDEX IF NOT EXISTS idx_equity_run        ON equity(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_config       ON runs(config_hash);
CREATE INDEX IF NOT EXISTS idx_runs_split        ON runs(split_label);
CREATE INDEX IF NOT EXISTS idx_experiments_config ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_experiments_phase  ON experiments(phase);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
"""


def create_statements() -> tuple[str, ...]:
    """The DDL as individual statements, tables before indexes."""
    statements = [s.strip() for s in SCHEMA.split(";")]
    statements += [s.strip() for s in INDEXES.split(";")]
    return tuple(s for s in statements if s)
