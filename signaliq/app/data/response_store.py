"""Raw provider responses, kept exactly as received, so a run can be reproduced from them.

Providers revise history: between Stage 1.1 and Phase 1, Alpaca changed its split-adjusted
closes for NEE and FTV. Re-requesting the same history therefore does not reproduce a run.
This store keeps the body of every successful response, byte for byte, in its own SQLite file:

- ``record``: a request already in the store is answered from it; any other request goes to
  the provider, and the response body is stored before it is used. A record run can be
  resumed, and a request repeated within a run gets the same answer both times.
- ``replay``: every request is answered from the store. The client is built without
  credentials and with a transport that refuses to connect, so nothing can come from the
  network. A request that is not in the store raises :class:`ResponseNotStored`.

Off by default. Switched on with two environment variables::

    SIGNALIQ_ALPACA_RESPONSES=/path/to/alpaca_responses.db
    SIGNALIQ_ALPACA_RESPONSES_MODE=record   # or replay

Each body is stored zlib-compressed together with the sha256 of the uncompressed bytes. The
hash is checked on every read, so a damaged store fails loudly instead of returning altered
data. Only status-200 responses are stored; errors are retried as before and never stored.
Credentials are sent in headers and are never part of a key or of anything stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import zlib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx

from app.database import protected_label

PATH_VARIABLE = "SIGNALIQ_ALPACA_RESPONSES"
MODE_VARIABLE = "SIGNALIQ_ALPACA_RESPONSES_MODE"
MODES = ("record", "replay")
SCHEMA_VERSION = "1"
#: Minimum free disk before a new response is written (as for the SEC cache).
MIN_FREE_BYTES = 2 * 1024**3

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS responses ("
    " key TEXT PRIMARY KEY,"
    " path TEXT NOT NULL,"
    " params TEXT NOT NULL,"
    " status INTEGER NOT NULL,"
    " content_type TEXT,"
    " body BLOB NOT NULL,"
    " body_sha256 TEXT NOT NULL,"
    " body_bytes INTEGER NOT NULL,"
    " request_id TEXT,"
    " fetched_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS store_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
)
_OWN_TABLES = {"responses", "store_meta"}


class ResponseStoreError(RuntimeError):
    """The response store is misconfigured, damaged or incomplete."""


class ResponseNotStored(ResponseStoreError):
    """Replay asked for a response the store does not hold."""


class StoreDiskLow(ResponseStoreError):
    """Less than :data:`MIN_FREE_BYTES` free: stop before writing."""


def request_key(path: str, params: dict) -> str:
    """Canonical key: the path and the query parameters sorted by name."""
    return path + "?" + urlencode(sorted((str(k), str(v)) for k, v in params.items()))


class ResponseStore:
    """One SQLite file of raw responses. Thread-safe."""

    def __init__(self, path: str | Path, mode: str) -> None:
        if mode not in MODES:
            raise ResponseStoreError(f"mode must be one of {MODES}, got {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self.served_from_store = 0
        self.fetched_live = 0
        self._lock = threading.Lock()
        if mode == "replay":
            if not self.path.is_file():
                raise ResponseStoreError(f"replay needs an existing response store: {self.path}")
            self._db = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
            )
            self._db.execute("PRAGMA query_only=ON")
            self._check_tables(require=True)
        else:
            label = protected_label(self.path)
            if label:
                raise ResponseStoreError(
                    f"{self.path} is {label}, archived and protected: "
                    "replay it, never record into it"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._check_tables(require=False)
            for statement in _SCHEMA:
                self._db.execute(statement)
            self._db.execute(
                "INSERT OR IGNORE INTO store_meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,)
            )
            self._db.execute(
                "INSERT OR IGNORE INTO store_meta VALUES ('created_at', ?)",
                (datetime.now(UTC).isoformat(timespec="seconds"),),
            )
            self._db.commit()

    def _check_tables(self, *, require: bool) -> None:
        names = {
            row[0]
            for row in self._db.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
            )
        }
        foreign = names - _OWN_TABLES
        if foreign:
            raise ResponseStoreError(
                f"{self.path} is not a response store (it has tables {sorted(foreign)}); "
                "refusing to use it"
            )
        if require and names != _OWN_TABLES:
            raise ResponseStoreError(f"{self.path} is not a complete response store")

    @property
    def replaying(self) -> bool:
        return self.mode == "replay"

    def get(self, path: str, params: dict) -> httpx.Response | None:
        """The stored response for this request, or None. Checks the body's sha256."""
        key = request_key(path, params)
        with self._lock:
            row = self._db.execute(
                "select status, content_type, body, body_sha256, request_id "
                "from responses where key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        status, content_type, packed, digest, request_id = row
        body = zlib.decompress(packed)
        if hashlib.sha256(body).hexdigest() != digest:
            raise ResponseStoreError(f"stored response is damaged (sha256 mismatch): {key}")
        self.served_from_store += 1
        headers = {"content-type": content_type or "application/json"}
        if request_id:
            headers["x-request-id"] = request_id
        return httpx.Response(
            status,
            headers=headers,
            content=body,
            request=httpx.Request("GET", "https://replay.invalid" + key),
        )

    def put(self, path: str, params: dict, response: httpx.Response) -> None:
        """Store a status-200 response body exactly as received."""
        if self.replaying:
            raise ResponseStoreError("a replay store is read-only")
        if response.status_code != 200:
            return
        free = shutil.disk_usage(self.path.parent).free
        if free < MIN_FREE_BYTES:
            raise StoreDiskLow(
                f"only {free / 1024**3:.1f} GB free; stopping before writing to {self.path}"
            )
        body = response.content
        key = request_key(path, params)
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    path,
                    json.dumps({str(k): str(v) for k, v in params.items()}, sort_keys=True),
                    response.status_code,
                    response.headers.get("content-type"),
                    zlib.compress(body, 6),
                    hashlib.sha256(body).hexdigest(),
                    len(body),
                    response.headers.get("x-request-id"),
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            self._db.commit()
            self.fetched_live += 1

    def summary(self) -> dict:
        """Row count, total bytes and a fingerprint over (key, body sha256) in key order."""
        digest = hashlib.sha256()
        count = total = 0
        with self._lock:
            for key, sha, size in self._db.execute(
                "select key, body_sha256, body_bytes from responses order by key"
            ):
                digest.update(f"{key}\t{sha}\n".encode())
                count += 1
                total += size
        return {"responses": count, "body_bytes": total, "fingerprint": digest.hexdigest()[:16]}

    def verify(self) -> list[str]:
        """Decompress every body and check its sha256 and length. Returns the problems."""
        problems = []
        with self._lock:
            rows = self._db.execute(
                "select key, body, body_sha256, body_bytes from responses order by key"
            ).fetchall()
        for key, packed, sha, size in rows:
            try:
                body = zlib.decompress(packed)
            except zlib.error as exc:
                problems.append(f"{key}: cannot decompress ({exc})")
                continue
            if len(body) != size or hashlib.sha256(body).hexdigest() != sha:
                problems.append(f"{key}: body does not match its recorded sha256 or length")
        return problems

    def close(self) -> None:
        with self._lock:
            self._db.close()


class NoNetworkTransport(httpx.BaseTransport):
    """Transport used in replay: any attempt to reach the network is an error."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise ResponseStoreError(
            f"replay mode: a network request was attempted and refused ({request.url.path})"
        )


_shared: dict[tuple[str, str], ResponseStore] = {}
_shared_lock = threading.Lock()


def from_environment() -> ResponseStore | None:
    """The store named by the environment, shared within the process; None when off."""
    path = os.environ.get(PATH_VARIABLE, "").strip()
    mode = os.environ.get(MODE_VARIABLE, "").strip()
    if not path and not mode:
        return None
    if not path or mode not in MODES:
        raise ResponseStoreError(
            f"set both {PATH_VARIABLE} (a file path) and {MODE_VARIABLE} (one of {MODES})"
        )
    resolved = str(Path(path).resolve())
    with _shared_lock:
        store = _shared.get((resolved, mode))
        if store is None:
            store = _shared[(resolved, mode)] = ResponseStore(resolved, mode)
        return store


def reset_shared() -> None:
    """Close and forget the shared stores (for tests)."""
    with _shared_lock:
        for store in _shared.values():
            store.close()
        _shared.clear()
