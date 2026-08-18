"""SQLAlchemy engine/session setup. Synchronous on purpose: alert volume for
a memecoin watchlist is low (signals per minute, not per millisecond), so a
plain blocking session is simpler and more robust than async SQLAlchemy.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def _connect_args() -> dict:
    if settings.DATABASE_URL.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


if settings.DATABASE_URL.startswith("sqlite:///./"):
    db_path = settings.DATABASE_URL.replace("sqlite:///./", "", 1)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db():
    """FastAPI dependency for read-only routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  (registers tables on Base.metadata)

    Base.metadata.create_all(bind=engine)
    _seed_state()


def _seed_state() -> None:
    from app.state import get_state, set_state

    db = SessionLocal()
    try:
        if get_state(db, "cash_balance_usd") is None:
            set_state(db, "cash_balance_usd", settings.PORTFOLIO_STARTING_BALANCE_USD)
        if get_state(db, "trading_halted") is None:
            set_state(db, "trading_halted", False)
        db.commit()
    finally:
        db.close()
