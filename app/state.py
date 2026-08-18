"""Tiny JSON-backed key/value helper on top of the `bot_state` table.

Used for the trading-halt flag and the paper-trading cash ledger — anything
that's a single mutable value rather than an auditable row.
"""
import json

from sqlalchemy.orm import Session

from app import models


def get_state(db: Session, key: str, default=None):
    row = db.get(models.BotState, key)
    if row is None:
        return default
    try:
        return json.loads(row.value)
    except (TypeError, ValueError):
        return default


def set_state(db: Session, key: str, value) -> None:
    row = db.get(models.BotState, key)
    payload = json.dumps(value)
    if row is None:
        db.add(models.BotState(key=key, value=payload))
    else:
        row.value = payload
    db.flush()
