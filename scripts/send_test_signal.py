#!/usr/bin/env python3
"""Send a fake TradingView alert to your running bot.

Lets you watch the bot make decisions without setting up TradingView.
The alerts this sends are identical in format to real ones - they go
through the exact same webhook, security checks, and risk limits.

The bot must already be running (via START_HERE.bat or setup_and_run.py).

Windows users: double-click SEND_TEST_SIGNAL.bat instead.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
URL = "http://127.0.0.1:8000/webhook/tradingview"

IS_WINDOWS = os.name == "nt"


def pause_and_exit(code: int = 0) -> None:
    if IS_WINDOWS:
        input("\nPress Enter to close this window...")
    sys.exit(code)


def read_secret() -> str:
    if not ENV_FILE.exists():
        print("Couldn't find your .env file.")
        print("Run START_HERE.bat first to set the bot up.")
        pause_and_exit(1)
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("WEBHOOK_SECRET="):
            return line.split("=", 1)[1].strip()
    print("No WEBHOOK_SECRET found in .env.")
    pause_and_exit(1)


def post(payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        print(f"\n  Bot accepted the alert (signal #{body.get('signal_id')}).")
        print("  Now refresh the dashboard to see what it decided:")
        print("      http://127.0.0.1:8000")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("\n  Rejected: the webhook secret didn't match.")
        else:
            print(f"\n  The bot returned an error (HTTP {exc.code}).")
    except urllib.error.URLError:
        print("\n  Couldn't reach the bot.")
        print("  Is it running? Double-click START_HERE.bat and wait for")
        print("  the dashboard to open, then try this again.")


def main() -> None:
    secret = read_secret()

    print("=" * 68)
    print("  SEND A TEST SIGNAL")
    print("=" * 68)
    print()
    print("  This pretends to be TradingView sending your bot an alert.")
    print("  Nothing here involves real money - the bot is in paper mode.")
    print()
    print("  1) Buy signal, no contract address")
    print("     The bot should BLOCK this. It refuses to buy any token it")
    print("     can't run scam checks on. Good way to see the safety net.")
    print()
    print("  2) Buy signal with a contract address you provide")
    print("     The bot runs REAL scam checks (liquidity, holders, mint")
    print("     authority, honeypot). Most memecoins fail these - that's")
    print("     normal and is the filter working, not a bug.")
    print()
    print("  3) Sell signal")
    print("     Closes an open position, if there is one for that symbol.")
    print()

    choice = input("  Pick 1, 2 or 3: ").strip()

    if choice == "1":
        post({
            "secret": secret,
            "symbol": "TESTCOIN",
            "signal": "buy",
            "price": 0.001,
            "rsi": 42.0,
        })

    elif choice == "2":
        print()
        print("  Find a token on https://dexscreener.com and copy its")
        print("  contract address (a long string of letters and numbers).")
        print()
        address = input("  Paste the contract address: ").strip()
        if not address:
            print("\n  No address given, nothing sent.")
            pause_and_exit()
        symbol = input("  Give it a short name (e.g. WIF): ").strip() or "TESTCOIN"
        chain = input("  Chain [solana]: ").strip() or "solana"
        post({
            "secret": secret,
            "symbol": symbol,
            "token_address": address,
            "chain": chain,
            "signal": "buy",
            "price": 0.001,
            "rsi": 42.0,
        })

    elif choice == "3":
        symbol = input("\n  Which symbol to sell? [TESTCOIN]: ").strip() or "TESTCOIN"
        post({
            "secret": secret,
            "symbol": symbol,
            "signal": "sell",
            "price": 0.0015,
        })

    else:
        print("\n  Not a valid choice, nothing sent.")

    pause_and_exit()


if __name__ == "__main__":
    main()
