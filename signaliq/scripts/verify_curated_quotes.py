#!/usr/bin/env python3
"""Check every curated SEC quote against the live SEC document (read-only, no cache).

    python scripts/verify_curated_quotes.py

Each quote must appear word for word in its document after markup and spacing are
removed. Exits with status 1 if any quote is missing.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.universe import sec
from app.universe import sec_history as sh
from app.universe.curated import (
    CURATED_DEALS,
    CURATED_FACTS,
    CURATED_IDENTITY_LINKS,
    CURATED_RENAMES,
)


def main() -> int:
    docs: dict[str, str] = {}
    found_total = quotes_total = 0
    with httpx.Client(headers={"User-Agent": sec.USER_AGENT}, timeout=60) as client:
        for kind, entries in (
            ("rename", CURATED_RENAMES),
            ("same", CURATED_IDENTITY_LINKS),
            ("fact", CURATED_FACTS),
            ("deal", CURATED_DEALS),
        ):
            for e in entries:
                if kind in ("rename", "same"):
                    label = f"{e.old_symbol} -> {e.new_symbol}"
                else:
                    symbol = getattr(e, "symbol", getattr(e, "target_symbol", ""))
                    label = f"{symbol} {e.last_trade}"
                found = 0
                for q in e.quotes:
                    if q.url not in docs:
                        time.sleep(sec.PAUSE_SECONDS)
                        r = client.get(q.url)
                        docs[q.url] = sh.plain_text(r.text) if r.status_code == 200 else ""
                    if " ".join(q.text.split()) in docs[q.url]:
                        found += 1
                    else:
                        print(f"  MISSING in {q.url}: {q.text[:90]}")
                found_total += found
                quotes_total += len(e.quotes)
                print(f"{kind:6} {label:26} {found}/{len(e.quotes)} quotes found")
    print(f"TOTAL: {found_total} of {quotes_total} quotes found in {len(docs)} live SEC documents")
    return 0 if found_total == quotes_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
