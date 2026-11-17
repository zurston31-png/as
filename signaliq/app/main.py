"""FastAPI application for SignalIQ.

Start it locally from the project folder with::

    uvicorn app.main:app --reload

then open http://127.0.0.1:8000/docs for the interactive documentation.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import router
from app.config import Settings, get_settings
from app.database import create_db_engine, create_session_factory, init_db

DESCRIPTION = (
    "Read-only access to validated historical US stock bars stored by SignalIQ.\n\n"
    "Milestone 1 is a data foundation only: it provides no trading strategies, "
    "predictions or buy/sell recommendations, and the symbol universe is a small "
    "curated list rather than the whole market."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Tests pass their own ``settings``."""
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The database is opened when the server starts, not at import time.
        engine = create_db_engine(app_settings.database_url)
        init_db(engine)
        app.state.settings = app_settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="SignalIQ", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
