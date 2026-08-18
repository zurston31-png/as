import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point at a throwaway sqlite file and set required env vars *before* any
# `app.*` module is imported, since Settings() is instantiated at import time.
_tmp_db = Path(tempfile.gettempdir()) / "memecoin_bot_test.db"
if _tmp_db.exists():
    _tmp_db.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db}"
os.environ.setdefault("WEBHOOK_SECRET", "test-webhook-secret-please-ignore")
# Must not be the shipped placeholder: the dashboard refuses that outright,
# the same way the webhook refuses its placeholder secret.
os.environ.setdefault("DASHBOARD_PASSWORD", "test-dashboard-password")
os.environ.setdefault("LIVE_TRADING", "false")
os.environ.setdefault("RUGCHECK_ENABLED", "true")

import pytest  # noqa: E402

from app.database import Base, SessionLocal, engine, init_db  # noqa: E402
from app.risk.manager import resume_trading  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _reset_halt_state():
    db = SessionLocal()
    try:
        resume_trading(db)
        db.commit()
    finally:
        db.close()
    yield


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
