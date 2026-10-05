"""Deterministic randomness and content hashing.

Two guarantees this module exists to provide:

1. **Reproducibility.** Every stochastic component (Monte Carlo, execution
   jitter, bootstrap sampling) draws from a *named* generator derived from
   one master seed. Adding a new stochastic component therefore cannot
   change the draws of an existing one -- which it would if everything
   shared a single global generator.

2. **Identity.** `stable_hash` produces the same digest for the same logical
   content across processes and Python versions. Config hashes and data
   hashes are how a performance number is attributed to a specific
   experiment; if the hash were unstable the experiment log would be
   worthless.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_SEED = 20240101
_HASH_LENGTH = 16


def _canonical(obj: Any) -> Any:
    """Convert to a form whose JSON encoding is canonical and portable.

    Floats are the subtle case: `repr` differs across platforms for some
    values, and -0.0 != 0.0 textually but is equal numerically. We format
    floats to 12 significant digits and normalize zero and non-finites.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if obj == 0:
            return "0.000000000000e+00"
        return f"{obj:.12e}"
    if isinstance(obj, Decimal):
        return _canonical(float(obj))
    if isinstance(obj, Enum):
        return _canonical(obj.value)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return _canonical(obj.item())
    if isinstance(obj, np.ndarray):
        return [_canonical(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        # sort by canonical key text so ordering never affects the digest
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonical(v) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if hasattr(obj, "model_dump"):  # pydantic model
        return _canonical(obj.model_dump(mode="python"))
    if hasattr(obj, "__dict__"):
        return _canonical(vars(obj))
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Canonical JSON text for hashing and for storage in the experiment log."""
    return json.dumps(
        _canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def stable_hash(obj: Any, length: int = _HASH_LENGTH) -> str:
    """Short, stable, content-addressed digest (hex)."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:length]


def full_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Digest of a data file's bytes, for `data_hash` provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:_HASH_LENGTH]


def hash_dataset(
    symbol: str, start: datetime | date, end: datetime | date, rows: int, source: str
) -> str:
    """Digest describing a logical dataset slice.

    Used when data comes from a database or API rather than a file, so that
    a result can still be tied to the exact slice it was computed from.
    """
    return stable_hash(
        {"symbol": symbol, "start": start, "end": end, "rows": rows, "source": source}
    )


def derive_seed(master_seed: int, *names: str) -> int:
    """Derive an independent, reproducible seed for a named stream.

    `derive_seed(42, "monte_carlo", "block")` always yields the same value
    and is independent of `derive_seed(42, "execution_jitter")`.
    """
    key = canonical_json([master_seed, *names]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def rng(master_seed: int, *names: str) -> np.random.Generator:
    """A NumPy Generator for a named stream."""
    return np.random.default_rng(derive_seed(master_seed, *names))


def seed_global(master_seed: int = DEFAULT_SEED) -> None:
    """Seed the global `random` and legacy NumPy generators.

    Prefer `rng()`. This exists only to make third-party code that reaches
    for global state behave reproducibly.
    """
    random.seed(master_seed)
    np.random.seed(master_seed % (2**32))


def spawn_streams(master_seed: int, names: Iterable[str]) -> dict[str, np.random.Generator]:
    return {name: rng(master_seed, name) for name in names}
