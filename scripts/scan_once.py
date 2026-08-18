#!/usr/bin/env python3
"""Run exactly one scanner cycle and show what it found, stage by stage.

    python scripts/scan_once.py            # discover + pre-screen only (no trades)
    python scripts/scan_once.py --trade    # full pipeline, opens PAPER positions

Use this first on a real server before leaving the scanner running
unattended. Discovery talks to DexScreener (and Birdeye, if you set
BIRDEYE_API_KEY), and those API shapes are the one part of
app/scanner/discovery.py that could not be verified from the development
sandbox - this script is how you confirm they actually match.

Without --trade nothing is executed at all: it discovers, pre-screens, and
prints, so you can tune SCANNER_MIN_* thresholds against real data safely.
With --trade it runs the full pipeline through the same rug check, signal
score, and risk gates a TradingView alert goes through - which in paper
mode (the default) still costs nothing real.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.scanner.discovery import discover_tokens  # noqa: E402
from app.scanner.filters import prescreen  # noqa: E402
from app.scanner.loop import scan_once  # noqa: E402


async def discover_only() -> None:
    print(f"Discovering newly listed tokens on {settings.CHAIN}...")
    print(f"  DexScreener: always on")
    print(f"  Birdeye:     {'on (API key set)' if settings.BIRDEYE_API_KEY else 'off (no BIRDEYE_API_KEY)'}")
    print()

    tokens = await discover_tokens()
    if not tokens:
        print("  No tokens returned.")
        print()
        print("  If you expected some, the API response shape may not match what")
        print("  app/scanner/discovery.py assumes - check the server log for a")
        print("  'token discovery request failed' or 'unrecognised ... shape' warning.")
        return

    passed, rejected = [], []
    for token in tokens:
        (passed if prescreen(token).passed else rejected).append(token)

    print(f"  Found {len(tokens)} token(s): {len(passed)} passed pre-screen, {len(rejected)} rejected.")
    print()

    if passed:
        print("  PASSED pre-screen (these would go on to the rug check + signal score):")
        for t in passed:
            print(f"    {t.symbol:<12} {t.token_address}")
            print(f"      liquidity ${t.liquidity_usd or 0:>12,.0f} | 24h vol ${t.volume_24h_usd or 0:>12,.0f} "
                  f"| age {t.age_hours or 0:>6.1f}h | {t.buys_24h or 0} buys / {t.sells_24h or 0} sells")
        print()

    print("  REJECTED on pre-screen (first 15):")
    for t in rejected[:15]:
        print(f"    {t.symbol:<12} {prescreen(t).reason}")
    if len(rejected) > 15:
        print(f"    ... and {len(rejected) - 15} more")

    print()
    print("  Tune SCANNER_MIN_LIQUIDITY_USD / SCANNER_MIN_VOLUME_24H_USD /")
    print("  SCANNER_MIN_TOKEN_AGE_HOURS / SCANNER_MIN_TXNS_24H in .env if these")
    print("  thresholds are letting through too much or too little.")


async def full_cycle() -> None:
    init_db()
    mode = "LIVE" if settings.LIVE_TRADING else "PAPER"
    print(f"Running one FULL scanner cycle in {mode} mode...")
    print("  (discover -> pre-screen -> signal score -> rug check -> size -> execute)")
    print()

    db = SessionLocal()
    try:
        summary = await scan_once(db)
    finally:
        db.close()

    if summary.get("skipped"):
        print(f"  Cycle skipped: {summary['skipped']}")
        return

    print(f"  discovered            {summary['discovered']}")
    print(f"  skipped (seen recently){summary['skipped_recent']:>5}")
    print(f"  rejected on pre-screen {summary['prescreen_rejected']:>5}")
    print(f"  fully evaluated        {summary['evaluated']:>5}")
    print(f"  traded                 {summary['traded']:>5}")
    print()
    print("  Open the dashboard to see any positions, and /journal for the full")
    print("  per-token record of why each candidate was or wasn't traded.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trade", action="store_true",
                        help="run the full pipeline (opens paper positions) instead of discovery only")
    args = parser.parse_args()

    print("=" * 70)
    print("  TOKEN SCANNER — single cycle")
    print("=" * 70)
    print()

    asyncio.run(full_cycle() if args.trade else discover_only())

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
