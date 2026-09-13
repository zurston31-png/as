"""Tiny JSON-over-HTTP helper (stdlib only, so the tool installs with no deps)."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

USER_AGENT = "valcoach/0.1 (+https://github.com/zurston31-png/as)"


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 2.0,
    insecure: bool = False,
) -> Any:
    """GET a JSON document, retrying on 429/5xx and transient network errors."""
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            url = f"{url}?{urllib.parse.urlencode(clean)}"

    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    req_headers.update(headers or {})
    ctx = None
    if insecure:  # only used for the Valorant client's self-signed localhost cert
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        request = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ctx) as resp:
                body = resp.read().decode("utf-8", "replace")
            return json.loads(body) if body.strip() else None
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            retryable = exc.code in (408, 425, 429, 500, 502, 503, 504)
            last = HttpError(exc.code, url, body)
            if not retryable or attempt == retries - 1:
                raise last
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt == retries - 1:
                raise
        time.sleep(backoff * (2 ** attempt))
    if last:
        raise last
    return None
