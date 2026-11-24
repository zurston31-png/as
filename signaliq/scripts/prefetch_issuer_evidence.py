#!/usr/bin/env python3
"""Fetch the SEC answers the issuer step needs, several securities at a time. Read-only on SEC.

    python scripts/prefetch_issuer_evidence.py [--workers N]

Runs exactly the pipeline's issuer step (``evidence.issuer_points`` and the SIC history of
every company found) for each security in the research database, with N securities in
flight at once, and stores every SEC answer in the same cache the pipeline reads. It
decides nothing and writes nothing else: the stage run then reads these answers from the
cache, so its result is the same as if it had fetched them itself, one at a time.

Requests stay under SEC's limit of 10 per second (one every ``sec.PAUSE_SECONDS`` across
all workers), carry the same User-Agent, and are retried like the pipeline's
(``sec.MAX_ATTEMPTS``). A request SEC never answers stops the script
(``SecUnavailable``); answers already received are kept, and a rerun continues from them.
"""

import argparse
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from sqlalchemy import select

from app.database import create_session_factory, session_scope
from app.universe import evidence, sec
from app.universe import sec_history as sh
from app.universe.database import open_research_database
from app.universe.models import ExternalResponse, Security


class Pacer:
    """One request every ``interval`` seconds across all threads."""

    def __init__(self, interval: float):
        self.interval, self.lock, self.next = interval, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next - now)
            self.next = max(now, self.next) + self.interval
        if delay:
            time.sleep(delay)


class SharedFetcher:
    def __init__(self):
        self.http = httpx.Client(headers={"User-Agent": sec.USER_AGENT}, timeout=60)
        self.pacer = Pacer(sec.PAUSE_SECONDS)
        self.lock = threading.Lock()
        self.pending: dict[str, str] = {}
        self.sink: queue.Queue = queue.Queue()
        self.requests = 0

    def fetch(self, key: str, url: str, params: dict | None) -> str:
        response = None
        for attempt in range(sec.MAX_ATTEMPTS):
            self.pacer.wait()
            try:
                response = self.http.get(url, params=params)
            except httpx.TransportError:
                response = None
            else:
                with self.lock:
                    self.requests += 1
                if response.status_code in (200, 404):
                    break
            if attempt < sec.MAX_ATTEMPTS - 1:
                time.sleep(sec.RETRY_WAITS[min(attempt, len(sec.RETRY_WAITS) - 1)])
        else:
            status = "connection failed" if response is None else f"HTTP {response.status_code}"
            raise sec.SecUnavailable(
                f"SEC did not answer after {sec.MAX_ATTEMPTS} tries ({status}): {key}"
            )
        text = response.text if response.status_code == 200 else ""
        with self.lock:
            self.pending[key] = text
        self.sink.put((key, text))
        return text


class PrefetchSec(sec.CachedSec):
    """The pipeline's client, reading the cache through its own session and fetching what
    is missing through the shared fetcher (stored by the writer, not by this session)."""

    def __init__(self, session, fetcher: SharedFetcher):
        super().__init__(session, network=False)
        self.fetcher = fetcher

    def get(self, url, params=None):
        key = sec.cache_key(url, params)
        row = self.session.get(ExternalResponse, ("sec", key))
        if row is not None:
            return row.response
        with self.fetcher.lock:
            if key in self.fetcher.pending:
                return self.fetcher.pending[key]
        return self.fetcher.fetch(key, url, params)


def one_security(factory, fetcher, sid: int, found) -> int:
    with session_scope(factory) as session:
        client = PrefetchSec(session, fetcher)
        points, _ = evidence.issuer_points(session, client, found, only={sid})
        security = session.get(Security, sid)
        for cik in sorted({p.cik for p in points.get(sid, [])}):
            sh.sic_history(
                client,
                cik,
                security.first_session - timedelta(days=365),
                security.last_trade_session,
            )
        session.rollback()  # nothing of this session is kept; the writer stores the answers
    return sid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    engine = open_research_database()
    factory = create_session_factory(engine)
    fetcher = SharedFetcher()
    with session_scope(factory) as session:
        found = evidence.sec_findings(session, sec.CachedSec(session, network=False))
        ids = [
            x.security_id
            for x in session.execute(select(Security).order_by(Security.security_id)).scalars()
        ]
    print(f"securities: {len(ids)}; workers: {args.workers}", flush=True)
    stored = done = 0
    started = time.monotonic()
    writer = factory()

    def drain() -> None:
        nonlocal stored
        while True:
            try:
                key, text = fetcher.sink.get_nowait()
            except queue.Empty:
                break
            writer.merge(ExternalResponse(source="sec", request_key=key, response=text))
            stored += 1
        writer.commit()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_security, factory, fetcher, sid, found): sid for sid in ids}
            for future in as_completed(futures):
                future.result()  # an unanswered request stops the script here
                done += 1
                drain()
                if done % 25 == 0:
                    print(
                        f"{done}/{len(ids)} securities; {fetcher.requests} requests; "
                        f"{stored} answers stored; {time.monotonic() - started:.0f}s",
                        flush=True,
                    )
    finally:
        drain()
        writer.close()
        fetcher.http.close()
        engine.dispose()
    print(
        f"done: {done} securities, {fetcher.requests} requests, {stored} answers stored", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
