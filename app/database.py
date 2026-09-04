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


if settings.DATABASE_URL.startswith("sqlite:///"):
    # Absolute paths are created too, not only relative ones.
    #
    # The default URL is relative, which means the database lives beside
    # whatever directory the bot was started from - so a deployment that
    # unpacks each new build into its own folder silently starts a fresh
    # history every upgrade, and the dashboard resets to zero trades. The
    # fix is to point DATABASE_URL at one absolute path shared by every
    # build, and that only works if the directory is created for it: the
    # relative-only branch this replaces left an absolute path to fail at
    # connect time with "unable to open database file", which reads as a
    # permissions problem rather than a missing folder.
    db_path = settings.DATABASE_URL.replace("sqlite:///", "", 1)
    if db_path.startswith("./"):
        db_path = db_path[2:]
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

    # create_all() only creates tables that don't exist - it never adds a
    # column to a table that does. Without this, upgrading a deployment that
    # was running an earlier build leaves its tables missing every column
    # added since, and the first query dies on "no such column".
    from app.migrations import apply_additive_migrations

    apply_additive_migrations(engine)
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
