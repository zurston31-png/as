"""JSON serialization for storage and reports.

Distinct from `core.determinism.canonical_json`, which exists for *hashing*
and deliberately renders floats as fixed-width strings so a digest is
platform-stable. That form must never be used for storage: a float written
with it reads back as a string, silently turning `win_rate: 0.72` into
`"7.200000000000e-01"`.

Rule: `canonical_json` for identity, `dumps` (here) for anything that will
be read back or displayed.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(obj: Any) -> Any:
    """Convert to JSON-native types, preserving numeric types as numbers."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # JSON has no NaN/Infinity; represent them as null so a reader does
        # not crash, and so a NaN is visibly absent rather than silently 0.
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Enum):
        return jsonable(obj.value)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        # model_dump INCLUDES computed fields, which is what a report wants:
        # a CleanReport's `retention` or a TradeRecord's `r_multiple` is the
        # part a reader looks at. Round-tripping still works, because
        # FrozenModel.from_mapping drops keys that are not constructor
        # inputs. Using to_init_dict here would silently omit every derived
        # value from stored reports.
        return jsonable(obj.model_dump(mode="python"))
    return str(obj)


def dumps(obj: Any, indent: int | None = None) -> str:
    """Stable-key JSON text suitable for storage and round-tripping."""
    return json.dumps(jsonable(obj), sort_keys=True, indent=indent, allow_nan=False)


def loads(text: str) -> Any:
    return json.loads(text) if text else None


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(obj, indent=indent), encoding="utf-8")
    return p
