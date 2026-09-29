"""The audit log.

Every decision the bot makes gets one append-only JSONL record carrying the
whole chain: the candle it fired on, the decision timestamp, every rule's
output, the AI verdict, the full sizing arithmetic, the risk decision, and the
final state. If you cannot reconstruct why a trade happened six weeks later,
the log is not doing its job.

"Immutable" here means tamper-evident rather than write-protected: each record
carries a monotonic sequence number and the SHA-256 of the previous record, so
editing or deleting anything in the middle breaks the chain from that point on
and `verify()` says where. The file itself is still an ordinary file - put it
somewhere append-only if you need more than evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .models import utcnow

log = logging.getLogger(__name__)

GENESIS = "0" * 64


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class VerifyResult:
    ok: bool
    records: int
    broken_at: Optional[int] = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "records": self.records,
                "broken_at": self.broken_at, "message": self.message}


class AuditLog:
    def __init__(self, path: str | Path, config_fingerprint: str = "") -> None:
        self.path = Path(path)
        self.config_fingerprint = config_fingerprint
        self._lock = threading.Lock()
        self._sequence, self._prev_hash = self._resume()

    # ------------------------------------------------------------- resume

    def _resume(self) -> tuple[int, str]:
        """Pick up the chain where a previous run left off."""
        if not self.path.exists():
            return 0, GENESIS
        last = None
        try:
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        last = line
        except OSError as exc:
            log.warning("could not read the audit log: %s", exc)
            return 0, GENESIS
        if not last:
            return 0, GENESIS
        try:
            record = json.loads(last)
            return int(record.get("sequence", 0)), str(record.get("hash", GENESIS))
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("audit log ends with an unreadable record - starting a new chain")
            return 0, GENESIS

    @property
    def sequence(self) -> int:
        return self._sequence

    # -------------------------------------------------------------- write

    def append(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one record. Never raises - a logging failure must not stop trading."""
        with self._lock:
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "logged_at": utcnow().isoformat(),
                "event": event,
                "config_fingerprint": self.config_fingerprint,
                "prev_hash": self._prev_hash,
                "payload": payload,
            }
            body = json.dumps(record, sort_keys=True, default=str)
            record["hash"] = _digest(self._prev_hash + body)
            self._prev_hash = record["hash"]
            line = json.dumps(record, sort_keys=True, default=str)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            log.error("could not write to the audit log: %s", exc)
        return record

    def signal(self, signal, context_dict: dict[str, Any], extra: Optional[dict] = None):
        """The full record for one decision."""
        payload = {
            "signal_id": signal.id,
            "decision_ts": (signal.decision_ts or signal.ts).isoformat(),
            "bar_ts": signal.ts.isoformat(),
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "side": signal.side.value,
            "actionable": signal.actionable,
            "blocked_by": signal.blocked_by,
            "levels": {"entry": signal.entry, "stop": signal.stop,
                       "tp1": signal.tp1, "tp2": signal.tp2},
            "candle": signal.trigger_candle,
            "context": context_dict,
            "strategy": signal.strategy_verdict.to_dict() if signal.strategy_verdict else None,
            "ai": signal.ai.to_dict() if signal.ai else None,
            "risk": signal.risk.to_dict() if signal.risk else None,
            **(extra or {}),
        }
        return self.append("signal", payload)

    # --------------------------------------------------------------- read

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def verify(self) -> VerifyResult:
        """Walk the hash chain and report the first record that doesn't check out."""
        prev = GENESIS
        count = 0
        expected_sequence = 0
        for record in self.records():
            count += 1
            expected_sequence += 1
            sequence = record.get("sequence")
            if sequence != expected_sequence:
                return VerifyResult(False, count, sequence,
                                    f"sequence jumped: expected {expected_sequence}, got {sequence}")
            if record.get("prev_hash") != prev:
                return VerifyResult(False, count, sequence,
                                    f"record {sequence} does not follow the previous hash")
            body = {k: v for k, v in record.items() if k != "hash"}
            recomputed = _digest(prev + json.dumps(body, sort_keys=True, default=str))
            if recomputed != record.get("hash"):
                return VerifyResult(False, count, sequence,
                                    f"record {sequence} has been modified since it was written")
            prev = record["hash"]
        return VerifyResult(True, count, None, f"{count} record(s), chain intact")


class NullAuditLog(AuditLog):
    """Used by backtests and tests, where writing a chain to disk is noise."""

    def __init__(self) -> None:  # noqa: D107
        self.path = Path(os.devnull)
        self.config_fingerprint = ""
        self._lock = threading.Lock()
        self._sequence, self._prev_hash = 0, GENESIS

    def append(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        return {"sequence": self._sequence, "event": event, "payload": payload}


def fingerprint(config_dict: dict[str, Any]) -> str:
    """A short hash of the effective config, stamped on every record."""
    return _digest(json.dumps(config_dict, sort_keys=True, default=str))[:16]
