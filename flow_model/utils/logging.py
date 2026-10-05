"""Structured logging.

Every log line carries the `run_id`, so output from concurrent walk-forward
folds or robustness sweeps stays attributable. Configuration is explicit and
idempotent: calling `setup_logging` twice does not duplicate handlers (which
would silently double every line in a long sweep).
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOGGER_NAME = "flow_model"
_CONFIGURED: dict[str, bool] = {}


class RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for machine-readable run logs."""

    _SKIP = frozenset(
        vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", ""),
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in self._SKIP and key != "run_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def new_run_id(prefix: str = "run") -> str:
    """Sortable, unique run identifier."""
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def setup_logging(
    level: str = "INFO",
    run_id: str | None = None,
    log_dir: str | Path | None = None,
    json_format: bool = False,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the package logger. Idempotent unless `force=True`."""
    logger = logging.getLogger(LOGGER_NAME)
    run_id = run_id or new_run_id()

    if _CONFIGURED.get(LOGGER_NAME) and not force:
        return logger

    logger.handlers.clear()
    logger.filters.clear()
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # The filter must live on the HANDLERS, not the logger. A logger's own
    # filters are only consulted for records logged directly to it -- records
    # propagated up from a child (`flow_model.signals`) bypass them, which
    # would leave run_id empty on exactly the lines we care about.
    run_filter = RunIdFilter(run_id)

    text_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter: logging.Formatter = JsonFormatter() if json_format else text_fmt

    if console:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(formatter)
        handler.addFilter(run_filter)
        logger.addHandler(handler)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / f"{run_id}.log", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter() if json_format else text_fmt)
        file_handler.addFilter(run_filter)
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    _CONFIGURED[LOGGER_NAME] = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Child logger under the package root."""
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def reset_logging() -> None:
    """Tear down handlers. For tests."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.filters.clear()
    _CONFIGURED.pop(LOGGER_NAME, None)
