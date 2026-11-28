#!/usr/bin/env python3
"""Controlled record/replay test of HTTP attempt logging, against a local fake provider.

Nothing reaches Alpaca: a local HTTPS server on 127.0.0.1 (throwaway self-signed certificate)
plays scripted failures, and the credentials are fake. It runs the real AlpacaClient through
the real HTTP stack, in record mode, then shuts the server down and replays the store with the
network blocked (proxy pointed at 127.0.0.1:9, no credentials, NoNetworkTransport).

Scenarios (fake tickers, synthetic bars):
  TESTA  page 1: 500, 429 (Retry-After 1), 200 with a next page;
         page 2: connection dropped, read timeout, 200            -> data, 6 attempts
  TESTB  422                                                      -> AlpacaRequestError, 1
  TESTC  503 on every attempt                                     -> AlpacaServerError, 4
  TESTD  connection dropped on every attempt                      -> AlpacaNetworkError, 4

Checks: every attempt the server received is logged, in order, with the expected status or
error; failed bodies are stored intact; replay returns the same data and raises the same
failures (class and status) with 0 requests; the store is unchanged by the replay.

    python scripts/attempt_log_controlled_test.py [--report FILE]
"""

import argparse
import hashlib
import json
import logging
import os
import socket
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import SecretStr

from app.config import Settings
from app.data import response_store
from app.data.alpaca_client import AlpacaAPIError, AlpacaClient
from app.data.response_store import ResponseStore

TIMEOUT = 1.0
START = datetime(2024, 1, 2, 5, 0, tzinfo=UTC)
END = datetime(2024, 1, 13, 4, 59, 59, tzinfo=UTC)


def bar(day: date, close: float) -> dict:
    return {"t": f"{day.isoformat()}T05:00:00Z", "o": 10.0, "h": 11.0, "l": 9.0, "c": close,
            "v": 1000, "n": 25, "vw": 10.2}  # fmt: skip


def page(symbol: str, bars: list, token: str | None) -> tuple:
    return ("json", 200, {"bars": {symbol: bars}, "next_page_token": token}, {})


#: (symbols, page_token) -> actions served in order, one per attempt.
SCRIPT = {
    ("TESTA", None): [
        ("json", 500, {"message": "fixture server error"}, {}),
        ("json", 429, {"message": "fixture rate limit"}, {"Retry-After": "1"}),
        page("TESTA", [bar(date(2024, 1, 2), 10.5)], "p2"),
    ],
    ("TESTA", "p2"): [("drop",), ("sleep",), page("TESTA", [bar(date(2024, 1, 3), 10.75)], None)],
    ("TESTB", None): [("json", 422, {"message": "fixture invalid request"}, {})],
    ("TESTC", None): [("json", 503, {"message": "fixture unavailable"}, {})] * 4,
    ("TESTD", None): [("drop",)] * 4,
}
EXPECTED_ATTEMPTS = {
    "TESTA": [500, 429, 200, "RemoteProtocolError", "ReadTimeout", 200],
    "TESTB": [422],
    "TESTC": [503] * 4,
    "TESTD": ["RemoteProtocolError"] * 4,
}
EXPECTED_OUTCOME = {
    "TESTA": "data",
    "TESTB": "AlpacaRequestError 422",
    "TESTC": "AlpacaServerError 503",
    "TESTD": "AlpacaNetworkError",
}


class Provider(BaseHTTPRequestHandler):
    received: ClassVar[list[str]] = []
    lock = threading.Lock()

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        key = (query["symbols"][0], query.get("page_token", [None])[0])
        with self.lock:
            Provider.received.append(f"{key[0]} {key[1] or '-'}")
            action = SCRIPT[key].pop(0)
        if action[0] == "drop":
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        if action[0] == "sleep":
            time.sleep(TIMEOUT * 2)
            return
        _, status, payload, headers = action
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-ID", f"fixture-{len(Provider.received)}")
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def certificate(directory: Path) -> tuple[Path, Path]:
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def run_all(settings: Settings, store: ResponseStore) -> dict:
    outcomes = {}
    with AlpacaClient(settings, store=store, backoff_jitter=0.0) as client:
        for symbol in SCRIPT_SYMBOLS:
            try:
                outcomes[symbol] = client.fetch_bars(
                    [symbol], timeframe="1Day", start=START, end=END
                )
            except AlpacaAPIError as exc:
                outcomes[symbol] = f"{type(exc).__name__} {exc.status_code or ''}".strip()
    return outcomes


SCRIPT_SYMBOLS = ("TESTA", "TESTB", "TESTC", "TESTD")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    lines: list[str] = []
    failures = 0

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    def check(ok, text):
        nonlocal failures
        failures += not ok
        say(f"  {'ok  ' if ok else 'FAIL'} {text}")

    work = Path(tempfile.mkdtemp(prefix="attempt_log_test_"))
    cert, key = certificate(work)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    requests_log = work / "client.log"
    handler = logging.FileHandler(requests_log)
    client_logger = logging.getLogger("app.data.alpaca_client")
    client_logger.setLevel(logging.INFO)
    client_logger.addHandler(handler)

    settings = Settings(
        alpaca_api_key_id=SecretStr("FAKE-KEY-ID-FOR-LOCAL-TEST"),
        alpaca_api_secret_key=SecretStr("FAKE-SECRET-FOR-LOCAL-TEST"),
        symbols=["TESTA"],
        history_start=date(2024, 1, 2),
        database_url=f"sqlite:///{(work / 'unused.db').as_posix()}",
        alpaca_data_base_url=f"https://127.0.0.1:{port}",
        http_timeout_seconds=TIMEOUT,
        http_max_retries=3,
        http_backoff_base_seconds=0.05,
        http_backoff_max_seconds=0.2,
    )
    store_path = work / "alpaca_responses.db"
    say("Controlled record/replay test of HTTP attempt logging (local fake provider only)")
    say(f"server https://127.0.0.1:{port}; credentials fake; store {store_path.name}")

    os.environ.update({"SSL_CERT_FILE": str(cert), "NO_PROXY": "127.0.0.1"})
    say("\n1. Record")
    recorder = ResponseStore(store_path, "record")
    recorded = run_all(settings, recorder)
    recorder.close()
    server.shutdown()
    server.server_close()
    received = list(Provider.received)
    logged = requests_log.read_text().count("Alpaca request:")
    db = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    rows = db.execute(
        "select key, attempt, status, error, body is not null from attempts order by id"
    ).fetchall()
    db.close()
    check(
        len(rows) == len(received) == logged,
        f"attempts logged {len(rows)} = requests the server received {len(received)} "
        f"= client request log lines {logged}",
    )
    for symbol in SCRIPT_SYMBOLS:
        got = [
            status if status is not None else error.split(":")[0]
            for key, _, status, error, _ in rows
            if f"symbols={symbol}&" in key
        ]
        check(got == EXPECTED_ATTEMPTS[symbol], f"{symbol} attempts in order: {got}")
        outcome = "data" if isinstance(recorded[symbol], dict) else recorded[symbol]
        check(outcome == EXPECTED_OUTCOME[symbol], f"{symbol} recorded outcome: {outcome}")
    failed_bodies = sum(
        1 for *_, status, _, has_body in rows if status not in (None, 200) and has_body
    )
    check(
        failed_bodies == sum(1 for *_, s, _, _ in rows if s not in (None, 200)),
        f"every non-200 attempt has its body stored ({failed_bodies})",
    )
    problems = ResponseStore(store_path, "replay").verify()
    check(problems == [], f"store verify: {len(problems)} damaged bodies (responses and attempts)")
    sha_before = hashlib.sha256(store_path.read_bytes()).hexdigest()

    say("\n2. Replay (server stopped; proxy -> 127.0.0.1:9; no credentials; NoNetworkTransport)")
    os.environ.pop("NO_PROXY")
    os.environ.update({"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"})
    response_store.reset_shared()
    handler.close()
    requests_log.write_text("")
    client_logger.addHandler(logging.FileHandler(requests_log))
    player = ResponseStore(store_path, "replay")
    replayed = run_all(settings, player)
    player.close()
    replay_requests = requests_log.read_text().count("Alpaca request:")
    check(
        replay_requests == 0 and player.fetched_live == 0,
        f"network requests during replay: {replay_requests} (live fetches {player.fetched_live})",
    )
    for symbol in SCRIPT_SYMBOLS:
        check(
            replayed[symbol] == recorded[symbol],
            f"{symbol} replay equals record: "
            f"{replayed[symbol] if isinstance(replayed[symbol], str) else 'same bars'}",
        )
    check(
        hashlib.sha256(store_path.read_bytes()).hexdigest() == sha_before,
        "store sha256 unchanged by the replay",
    )
    check(len(Provider.received) == len(received), "the fake provider received nothing more")

    say(f"\nstore sha256 {sha_before}")
    say(f"RESULT: {'PASS' if failures == 0 else 'FAIL'}")
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
