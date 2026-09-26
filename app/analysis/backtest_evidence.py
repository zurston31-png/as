"""Carrying a backtest's out-of-sample result into the validation gate.

Two of the gate's eight criteria - out-of-sample and walk-forward - cannot
be computed from live trade rows. They describe a test performed against
HISTORY, which is what app/backtesting/ does. Until now nothing connected
the two: `scripts/run_backtest.py --walk-forward` printed a result and the
report said "no walk-forward analysis run yet" forever, no matter how many
times it was run. This module is the missing link, and it is deliberately
narrow.

THE REASON THIS FILE IS SO SUSPICIOUS OF ITS OWN INPUT

`scripts/run_backtest.py` defaults to a SYNTHETIC candle provider. Run it
with no arguments and it will happily produce a walk-forward result with
three profitable windows - about a market that never existed. Feeding that
into the gate would turn two "NO DATA" rows green on the strength of
invented history, which is the single most damaging thing this codebase
could do to itself: every other honesty rule exists to keep the record
clean, and this would poison it at the top.

So a run records where its candles came from, and `load()` refuses any
evidence not derived from real market history. The refusal is loud and
returns None rather than raising, because the caller's correct response is
to carry on reporting the criteria as unmeasured - not to crash, and not
to quietly substitute a pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# A run whose candles were generated rather than observed. Never admissible.
SYNTHETIC_SOURCE = "synthetic"

# Sources that describe real observed market history.
REAL_SOURCES = frozenset({"csv"})

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BacktestEvidence:
    """Out-of-sample and walk-forward figures from one real backtest run."""

    out_of_sample_trades: int
    out_of_sample_profitable: bool
    walk_forward_windows: int
    walk_forward_profitable_windows: int
    data_source: str
    symbol: str
    timeframe: str
    candles: int

    def provenance(self) -> str:
        return (
            f"{self.walk_forward_windows} walk-forward window(s) over "
            f"{self.candles} {self.timeframe} candles of {self.symbol} "
            f"({self.data_source})"
        )


def as_payload(
    *,
    out_of_sample_trades: int,
    out_of_sample_profitable: bool,
    walk_forward_windows: int,
    walk_forward_profitable_windows: int,
    data_source: str,
    symbol: str,
    timeframe: str,
    candles: int,
) -> dict:
    """The on-disk form a backtest run writes.

    `data_source` is recorded by the producer, not inferred by the reader,
    so a synthetic run cannot be laundered into a real one by editing the
    numbers alone - the field says plainly what the candles were.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "data_source": data_source,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "out_of_sample_trades": out_of_sample_trades,
        "out_of_sample_profitable": out_of_sample_profitable,
        "walk_forward_windows": walk_forward_windows,
        "walk_forward_profitable_windows": walk_forward_profitable_windows,
    }


class EvidenceRejected(Exception):
    """Raised by `load` only for a file that cannot be parsed at all."""


def load(path: str | Path) -> tuple[BacktestEvidence | None, str]:
    """Read a saved backtest result, or explain why it is not admissible.

    Returns `(evidence, message)`. `evidence` is None whenever the file
    cannot be counted toward the gate, and `message` always says why in
    terms an operator can act on.
    """
    p = Path(path)
    if not p.exists():
        return None, f"no such backtest result file: {p}"

    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceRejected(f"could not read {p}: {exc}") from exc

    if not isinstance(raw, dict):
        return None, f"{p} is not a backtest result object"

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        return None, (
            f"{p} has schema_version {version!r}, this build reads "
            f"{SCHEMA_VERSION} - regenerate it with the current "
            f"scripts/run_backtest.py"
        )

    source = raw.get("data_source")
    if source == SYNTHETIC_SOURCE:
        return None, (
            f"{p} was produced from SYNTHETIC candles. A walk-forward over "
            f"generated history says nothing about this strategy, so it "
            f"cannot satisfy the out-of-sample or walk-forward criteria. "
            f"Re-run with --csv against real OHLCV history."
        )
    if source not in REAL_SOURCES:
        return None, (
            f"{p} records data_source {source!r}, which this build does not "
            f"recognise as real market history"
        )

    required = (
        "out_of_sample_trades",
        "out_of_sample_profitable",
        "walk_forward_windows",
        "walk_forward_profitable_windows",
    )
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        return None, f"{p} is missing required field(s): {', '.join(missing)}"

    try:
        evidence = BacktestEvidence(
            out_of_sample_trades=int(raw["out_of_sample_trades"]),
            out_of_sample_profitable=bool(raw["out_of_sample_profitable"]),
            walk_forward_windows=int(raw["walk_forward_windows"]),
            walk_forward_profitable_windows=int(raw["walk_forward_profitable_windows"]),
            data_source=str(source),
            symbol=str(raw.get("symbol", "unknown")),
            timeframe=str(raw.get("timeframe", "unknown")),
            candles=int(raw.get("candles", 0)),
        )
    except (TypeError, ValueError) as exc:
        return None, f"{p} has a malformed field: {exc}"

    if evidence.walk_forward_profitable_windows > evidence.walk_forward_windows:
        return None, (
            f"{p} claims {evidence.walk_forward_profitable_windows} profitable "
            f"windows out of {evidence.walk_forward_windows} - impossible"
        )

    return evidence, f"loaded {evidence.provenance()}"
