"""SEC answers: this database's own cache first, then a read-only parent cache.

A later stage (Stage 2) must not modify the Stage 1.1 database, and that database (20 GB,
almost all of it cached SEC documents) cannot be copied within the disk allowance. So a new
database keeps its own cache, and SEC answers it does not hold are read from a *parent*
database opened read-only at the SQLite level (``mode=ro`` and ``PRAGMA query_only``). New
answers are written only to the new database. Nothing is ever written to the parent.

* ``SIGNALIQ_SIP_RESEARCH_PARENT_DB``: path of the parent database (optional).
* ``SIGNALIQ_SEC_OFFLINE=1``: any SEC answer that is not cached stops the run
  (``SecUnavailable``) instead of being requested, so a run can be proved to fetch nothing.

New answers longer than ``COMPRESS_OVER`` characters are stored zlib-compressed; readers
decode them transparently. A write stops the run (``DiskLow``) when the database's disk has
less than ``MIN_FREE_BYTES`` free; the answers already stored are kept, and a rerun resumes.
"""

import base64
import json
import os
import shutil
import sqlite3
import threading
import zlib
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.universe.models import ExternalResponse

PARENT_VARIABLE = "SIGNALIQ_SIP_RESEARCH_PARENT_DB"
OFFLINE_VARIABLE = "SIGNALIQ_SEC_OFFLINE"
COMPRESS_OVER = 64 * 1024
MIN_FREE_BYTES = 2 * 1024**3
_ZLIB = "zlib_b64"

_parent: sqlite3.Connection | None = None
_parent_path: Path | None = None
_parent_lock = threading.Lock()


class DiskLow(RuntimeError):
    """Too little free disk to store another answer safely."""


def offline() -> bool:
    return os.environ.get(OFFLINE_VARIABLE, "").strip() == "1"


def open_parent(path: str | Path) -> sqlite3.Connection:
    """A read-only connection to a research database (writes raise ``sqlite3.OperationalError``)."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"parent database not found: {resolved}")
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro", uri=True, check_same_thread=False
    )
    connection.execute("PRAGMA query_only=ON")
    connection.execute("select 1 from external_responses limit 1")
    return connection


def configure_parent(path: str | Path | None) -> None:
    """Use ``path`` as the read-only parent cache (None: no parent)."""
    global _parent, _parent_path
    if _parent is not None:
        _parent.close()
    _parent, _parent_path = (
        (None, None) if path is None else (open_parent(path), Path(path).resolve())
    )


def configure_from_environment() -> None:
    value = os.environ.get(PARENT_VARIABLE, "").strip()
    configure_parent(value or None)


def parent_path() -> Path | None:
    return _parent_path


def encode(text):
    if isinstance(text, str) and len(text) > COMPRESS_OVER:
        return {_ZLIB: base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")}
    return text


def decode(value):
    if isinstance(value, dict) and set(value) == {_ZLIB}:
        return zlib.decompress(base64.b64decode(value[_ZLIB])).decode("utf-8")
    return value


def lookup(session: Session, key: str):
    """The cached SEC answer for ``key``: this database first, then the parent; else None."""
    row = session.get(ExternalResponse, ("sec", key))
    if row is not None:
        return decode(row.response)
    if _parent is not None:
        with _parent_lock:
            found = _parent.execute(
                "select response from external_responses where source = 'sec' and request_key = ?",
                (key,),
            ).fetchone()
        if found is not None:
            return decode(json.loads(found[0]))
    return None


def _database_dir(session: Session) -> Path | None:
    database = session.get_bind().url.database
    if not database or database == ":memory:":
        return None
    return Path(database.removeprefix("file:").split("?")[0]).resolve().parent


def check_disk(session: Session) -> None:
    folder = _database_dir(session)
    if folder is None:
        return
    free = shutil.disk_usage(folder).free
    if free < MIN_FREE_BYTES:
        raise DiskLow(
            f"only {free / 1024**3:.2f} GB free at {folder} (minimum "
            f"{MIN_FREE_BYTES / 1024**3:.0f} GB): stopping before another write; "
            "answers stored so far are kept and a rerun resumes from them"
        )


def store(session: Session, key: str, text) -> None:
    """Store an SEC answer in this database (never in the parent)."""
    check_disk(session)
    session.merge(ExternalResponse(source="sec", request_key=key, response=encode(text)))


def count(session: Session) -> int:
    """Distinct cached SEC answers: this database's plus the parent's it does not override."""
    own = session.scalar(
        select(func.count()).select_from(ExternalResponse).where(ExternalResponse.source == "sec")
    )
    if _parent is None:
        return own
    parent_total = _parent.execute(
        "select count(*) from external_responses where source = 'sec'"
    ).fetchone()[0]
    overlap = 0
    if own:
        keys = session.execute(
            select(ExternalResponse.request_key).where(ExternalResponse.source == "sec")
        ).scalars()
        for key in keys:
            overlap += (
                _parent.execute(
                    "select 1 from external_responses where source = 'sec' and request_key = ?",
                    (key,),
                ).fetchone()
                is not None
            )
    return own + parent_total - overlap
