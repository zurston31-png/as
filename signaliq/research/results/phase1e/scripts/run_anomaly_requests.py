#!/usr/bin/env python3
"""Run exactly the four approved Alpaca requests (Phase 1d) into a new, empty response store.

Read-only market data; nothing is written to any research database. Every HTTP attempt is
logged in the store. A failed request is reported as it failed; nothing is substituted.

    python research/results/phase1e/scripts/run_anomaly_requests.py
"""

import hashlib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.data.alpaca_client import AlpacaAPIError, AlpacaClient  # noqa: E402
from app.data.response_store import ResponseStore, request_key  # noqa: E402

STORE = ROOT / "data" / "phase1d_anomaly" / "alpaca_responses.db"
LOG = ROOT / "research" / "results" / "phase1e" / "request_log.txt"
APPROVED = [
    "https://data.alpaca.markets/v2/stocks/bars?symbols=SPGI,AFL&timeframe=1Min&start=2021-10-25T13:30:00Z&end=2021-10-25T20:00:00Z&limit=10000&adjustment=raw&feed=sip&currency=USD&sort=asc",
    "https://data.alpaca.markets/v2/stocks/bars?symbols=AFL,CL,INFY,SPGI&timeframe=1Day&start=2021-10-25T00:00:00Z&end=2021-10-25T23:59:59Z&limit=10000&adjustment=raw&feed=iex&currency=USD&sort=asc",
    "https://data.alpaca.markets/v2/stocks/bars?symbols=AJG,AME&timeframe=1Min&start=2022-03-08T14:30:00Z&end=2022-03-08T21:00:00Z&limit=10000&adjustment=raw&feed=sip&currency=USD&sort=asc",
    "https://data.alpaca.markets/v2/stocks/bars?symbols=AJG,AME,AVB&timeframe=1Day&start=2022-03-08T00:00:00Z&end=2022-03-08T23:59:59Z&limit=10000&adjustment=raw&feed=iex&currency=USD&sort=asc",
]  # fmt: skip


def when(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def main() -> int:
    if STORE.exists():
        print(f"refusing to start: {STORE} exists (a new, empty store is required)")
        return 2
    STORE.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log = logging.getLogger("app.data.alpaca_client")
    log.setLevel(logging.INFO)
    log.addHandler(handler)

    store = ResponseStore(STORE, "record")
    base = get_settings()
    status = 0
    for n, url in enumerate(APPROVED, 1):
        q = dict(parse_qsl(urlparse(url).query))
        settings = base.model_copy(update={"alpaca_feed": q["feed"], "alpaca_adjustment": "raw"})
        pages = bars = 0
        try:
            with AlpacaClient(settings, store=store, backoff_jitter=0.0) as client:
                for page in client.iter_bar_pages(
                    q["symbols"].split(","),
                    timeframe=q["timeframe"],
                    start=when(q["start"]),
                    end=when(q["end"]),
                    limit=int(q["limit"]),
                ):
                    pages += 1
                    bars += page.bar_count
            print(f"request {n}: ok, {pages} page(s), {bars} bars")
        except AlpacaAPIError as exc:
            status = 1
            print(f"request {n}: FAILED {type(exc).__name__} status={exc.status_code}: {exc}")
    keys = {r[0] for r in store._db.execute("select distinct key from attempts")}
    store.close()
    approved_keys = {
        request_key(urlparse(u).path, dict(parse_qsl(urlparse(u).query))) for u in APPROVED
    }
    extra = keys - approved_keys
    print(f"distinct requests sent: {len(keys)}; all equal to an approved URL: {not extra}")
    if extra:
        print(f"NOT APPROVED: {sorted(extra)}")
        status = 1
    print(f"store sha256 {hashlib.sha256(STORE.read_bytes()).hexdigest()}")
    return status


if __name__ == "__main__":
    sys.exit(main())
