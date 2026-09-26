#!/usr/bin/env python3
"""Expose the running bot to the internet so TradingView can send it alerts.

Starts an ngrok tunnel to the local bot, verifies end to end that a signal
sent to the PUBLIC address actually reaches it, then prints the exact two
values to paste into TradingView.

Everything stays in paper trading mode. Real alerts arriving over this
tunnel are simulated exactly like the ones from SEND_TEST_SIGNAL.

Windows users: double-click CONNECT_TRADINGVIEW.bat instead.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

BOT_URL = "http://127.0.0.1:8000"
NGROK_API = "http://127.0.0.1:4040/api/tunnels"
NGROK_DOWNLOAD = "https://ngrok.com/download"
NGROK_TOKEN_PAGE = "https://dashboard.ngrok.com/get-started/your-authtoken"

IS_WINDOWS = os.name == "nt"


def pause_and_exit(code: int = 0) -> None:
    if IS_WINDOWS:
        try:
            input("\nPress Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(code)


def ask(prompt: str, default: str = "") -> str:
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def read_env(key: str) -> str | None:
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def get_json(url: str, timeout: int = 5):
    req = urllib.request.Request(url, headers={"User-Agent": "memecoin-bot"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def bot_is_running() -> bool:
    try:
        get_json(f"{BOT_URL}/health", timeout=3)
        return True
    except Exception:
        return False


def find_ngrok() -> str | None:
    local = REPO_ROOT / ("ngrok.exe" if IS_WINDOWS else "ngrok")
    if local.exists():
        return str(local)
    return shutil.which("ngrok")


def existing_tunnel() -> str | None:
    """Reuse a tunnel if ngrok is already running."""
    try:
        data = get_json(NGROK_API, timeout=3)
    except Exception:
        return None
    for tunnel in data.get("tunnels", []):
        url = tunnel.get("public_url", "")
        if url.startswith("https://"):
            return url
    return None


def wait_for_tunnel(timeout_s: int = 25) -> str | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        url = existing_tunnel()
        if url:
            return url
        time.sleep(1)
    return None


def explain_missing_ngrok() -> None:
    print("\n" + "=" * 68)
    print("  NGROK IS NOT INSTALLED")
    print("=" * 68)
    print("\n  ngrok gives your bot a temporary public web address, so")
    print("  TradingView can reach it. It's free.")
    print("\n  1. Your browser will open the ngrok download page")
    print("  2. Download the Windows version" if IS_WINDOWS else "  2. Download the version for your OS")
    print("  3. Unzip it - you'll get a file called "
          + ("ngrok.exe" if IS_WINDOWS else "ngrok"))
    print(f"  4. Put that file in THIS folder:\n       {REPO_ROOT}")
    print("  5. Run this again")
    print("\n  You'll also need a free ngrok account for the login token.")
    print("=" * 68)
    try:
        webbrowser.open(NGROK_DOWNLOAD)
    except Exception:
        pass


def ensure_authtoken(ngrok: str) -> bool:
    """ngrok needs a one-time token from a free account."""
    result = subprocess.run([ngrok, "config", "check"], capture_output=True, text=True)
    if result.returncode == 0 and "authtoken" not in (result.stderr or "").lower():
        return True

    print("\n  ngrok needs a one-time login token (free account).")
    print(f"  Your browser will open: {NGROK_TOKEN_PAGE}")
    print("  Sign up / log in, then copy the token shown there.\n")
    try:
        webbrowser.open(NGROK_TOKEN_PAGE)
    except Exception:
        pass

    token = ask("  Paste your ngrok authtoken (or Enter to skip): ")
    if not token:
        return False

    saved = subprocess.run([ngrok, "config", "add-authtoken", token], capture_output=True, text=True)
    if saved.returncode != 0:
        print(f"\n  ngrok rejected that token:\n    {saved.stderr.strip()[:300]}")
        return False
    print("  Token saved.")
    return True


def self_test(public_url: str, secret: str) -> bool:
    """Prove a signal sent to the PUBLIC address reaches the bot.

    Sends a deliberately incomplete buy (no token address). The bot accepts
    and records it, then its rug filter rejects the trade - which is the
    correct outcome and confirms the whole path works without opening a
    position.
    """
    payload = json.dumps({
        "secret": secret,
        "symbol": "TUNNELTEST",
        "signal": "buy",
        "price": 0.001,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{public_url}/webhook/tradingview",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "memecoin-bot-selftest"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        print(f"     success - the bot received it (signal #{body.get('signal_id')})")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("     the bot rejected the secret. Is another copy of the bot running?")
        else:
            print(f"     the bot returned HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001
        print(f"     could not reach the bot through the tunnel: {exc}")
    return False


def main() -> None:
    print("=" * 68)
    print("  CONNECT TRADINGVIEW  (paper trading - no real money)")
    print("=" * 68)

    # ---- 1. bot running? ----
    print("\n  [1/4] Checking the bot is running...")
    if not bot_is_running():
        print("\n  The bot isn't running.")
        print("  Double-click START_HERE.bat first, wait for the dashboard to")
        print("  open, then run this again. Leave BOTH windows open.")
        pause_and_exit(1)
    print("        running")

    secret = read_env("WEBHOOK_SECRET")
    if not secret:
        print("\n  No WEBHOOK_SECRET found in .env. Run START_HERE.bat first.")
        pause_and_exit(1)

    # ---- 2. ngrok ----
    print("\n  [2/4] Setting up the public address...")
    public_url = existing_tunnel()
    proc = None

    if public_url:
        print(f"        reusing tunnel already running: {public_url}")
    else:
        ngrok = find_ngrok()
        if not ngrok:
            explain_missing_ngrok()
            pause_and_exit(1)

        if not ensure_authtoken(ngrok):
            print("\n  Can't continue without an ngrok token.")
            pause_and_exit(1)

        print("        starting tunnel...")
        proc = subprocess.Popen(
            [ngrok, "http", "8000", "--log=stdout"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        public_url = wait_for_tunnel()
        if not public_url:
            print("\n  ngrok didn't produce a public address in time.")
            print("  Try running this in a terminal to see the error:")
            print(f"      {ngrok} http 8000")
            if proc:
                proc.terminate()
            pause_and_exit(1)
        print(f"        tunnel up: {public_url}")

    webhook_url = f"{public_url}/webhook/tradingview"

    # ---- 3. prove it works ----
    print("\n  [3/4] Testing the public address end to end...")
    ok = self_test(public_url, secret)

    # ---- 4. what to paste into TradingView ----
    print("\n" + "=" * 68)
    print("  PASTE THESE INTO TRADINGVIEW" if ok else "  TUNNEL IS UP (self-test did not confirm)")
    print("=" * 68)
    print("\n  Webhook URL  (goes in the alert's 'Webhook URL' box):")
    print(f"\n      {webhook_url}\n")
    print("  Webhook Secret  (goes in the Pine script's settings):")
    print(f"\n      {secret}\n")
    print("-" * 68)
    print("""
  IN TRADINGVIEW:

   1. Open a chart for a coin you want to trade.

   2. Pine Editor (bottom of the screen) -> paste the contents of
      pine/memecoin_signal_strategy.pine -> "Add to chart".

   3. Click the indicator's settings (gear icon) and fill in:
        Webhook Secret        the secret printed above
        Token/Contract Addr   the coin's address from dexscreener.com
        Chain                 solana

   4. Right-click the chart -> Add alert
        Condition   your indicator's name
        then pick   "Any alert() function call"
        Webhook URL the URL printed above
        (tick "Webhook URL" under Notifications)
      -> Create

   5. Repeat per coin. One alert per chart.

  NOTE: webhook alerts need a PAID TradingView plan. They are not
  available on the free tier.
""")
    print("-" * 68)
    print("\n  KEEP THIS WINDOW OPEN. Closing it kills the public address")
    print("  and TradingView's alerts stop arriving.")
    print("\n  The address changes every time you restart ngrok, so you'd")
    print("  need to update your alerts. A rented server with a real domain")
    print("  fixes that - see deploy/vps_setup.md when you're ready.")
    print("\n  Press Ctrl+C to stop.")
    print("=" * 68)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n  Shutting down the tunnel.")
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
