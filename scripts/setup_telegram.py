#!/usr/bin/env python3
"""Set up Telegram notifications.

Getting a chat ID normally means pasting a URL with your bot token into a
browser and picking a number out of raw JSON. This does that for you: it
validates the token, waits for you to message the bot, reads the chat ID
off that message, saves both to .env, and sends a test message to prove it
works.

Windows users: double-click SETUP_TELEGRAM.bat instead.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
BOTFATHER = "https://t.me/BotFather"

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


def api(token: str, method: str, **params):
    url = f"https://api.telegram.org/bot{token}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "memecoin-bot"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def extract_chat(update: dict) -> tuple[str, str] | None:
    """Pull (chat_id, who) out of whichever update shape arrived."""
    for key in ("message", "edited_message", "channel_post", "my_chat_member"):
        payload = update.get(key)
        if not isinstance(payload, dict):
            continue
        chat = payload.get("chat")
        if isinstance(chat, dict) and chat.get("id") is not None:
            who = chat.get("username") or chat.get("first_name") or chat.get("title") or "you"
            return str(chat["id"]), str(who)
    return None


def write_env(token: str, chat_id: str) -> None:
    """Set the two Telegram keys, leaving everything else untouched."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    wanted = {"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": chat_id}
    seen = set()

    out = []
    for line in lines:
        replaced = False
        for key, value in wanted.items():
            if line.startswith(f"{key}="):
                out.append(f"{key}={value}")
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)

    for key, value in wanted.items():
        if key not in seen:
            out.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 68)
    print("  TELEGRAM NOTIFICATIONS SETUP")
    print("=" * 68)
    print("\n  Get a message on your phone for every trade, every rejection,")
    print("  and a daily profit/loss summary - so you don't have to keep")
    print("  checking the dashboard.\n")

    if not ENV_FILE.exists():
        print("  No .env file found. Run START_HERE.bat first.")
        pause_and_exit(1)

    # ---- 1. the bot token ----
    print("-" * 68)
    print("""
  STEP 1 - Create a Telegram bot (one minute)

    1. Open Telegram on your phone or computer
    2. Search for:  @BotFather   (it has a blue tick)
    3. Send it:     /newbot
    4. It asks for a name      -> anything, e.g. My Trading Bot
    5. It asks for a username  -> must end in "bot",
                                  e.g. lerbry_trading_bot
    6. It replies with a token that looks like:
         123456789:AAHdqTcvbXd8-the-rest-is-random
""")
    print("-" * 68)

    if ask("\n  Open BotFather in your browser now? [y/N]: ").lower().startswith("y"):
        try:
            webbrowser.open(BOTFATHER)
        except Exception:
            print(f"  Couldn't open a browser. Go to {BOTFATHER}")

    token = ask("\n  Paste the token from BotFather: ")
    if not token:
        print("\n  No token given, nothing changed.")
        pause_and_exit()

    print("\n  Checking the token...")
    try:
        info = api(token, "getMe")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("\n  Telegram rejected that token. Copy the whole thing from")
            print("  BotFather, including the numbers before the colon.")
        else:
            print(f"\n  Telegram returned HTTP {exc.code}.")
        pause_and_exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Couldn't reach Telegram: {exc}")
        pause_and_exit(1)

    username = (info.get("result") or {}).get("username", "your bot")
    print(f"        token is valid - bot is @{username}")

    # ---- 2. the chat id ----
    print("\n" + "-" * 68)
    print(f"""
  STEP 2 - Say hello to your bot

    Telegram won't let a bot message you until you message it first.

    1. In Telegram, search for:  @{username}
    2. Open the chat and press START (or just send "hi")

  Waiting for your message...
""")
    print("-" * 68)

    chat_id = None
    who = ""
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            updates = api(token, "getUpdates", timeout=0, limit=20)
        except Exception:
            time.sleep(2)
            continue
        for update in reversed(updates.get("result") or []):
            found = extract_chat(update)
            if found:
                chat_id, who = found
                break
        if chat_id:
            break
        time.sleep(2)

    if not chat_id:
        print("\n  Didn't see a message after 3 minutes.")
        print(f"  Make sure you opened @{username} in Telegram and pressed START,")
        print("  then run this again.")
        pause_and_exit(1)

    print(f"\n        got it - chat with {who} (id {chat_id})")

    # ---- 3. save and prove it ----
    write_env(token, chat_id)
    print("        saved to .env")

    print("\n  Sending a test message...")
    try:
        api(token, "sendMessage", chat_id=chat_id,
            text="✅ Your memecoin trading bot is connected.\n\n"
                 "You'll get a message here for every trade, every rejected "
                 "token, and a daily profit/loss summary. Still paper trading "
                 "- no real money.")
        print("        sent - check your phone")
    except Exception as exc:  # noqa: BLE001
        print(f"        couldn't send: {exc}")

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)
    print("\n  RESTART THE BOT for this to take effect:")
    print("    1. Close the bot's console window")
    print("    2. Double-click START_HERE.bat again")
    print("\n  After that it messages you on its own. No need to watch the")
    print("  dashboard.")
    print("=" * 68)

    pause_and_exit()


if __name__ == "__main__":
    main()
