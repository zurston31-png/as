#!/usr/bin/env python3
"""One-command setup + launch for local paper trading.

Does everything needed to get the bot running on a fresh machine:
  1. checks the Python version
  2. creates a virtual environment (venv/)
  3. installs dependencies into it
  4. writes a .env with freshly generated secrets (if one doesn't exist)
  5. creates the database
  6. starts the server and opens the dashboard in a browser

Safe to re-run: an existing venv or .env is reused, never overwritten.

Windows users: double-click START_HERE.bat instead of running this directly.
Mac/Linux users: python3 scripts/setup_and_run.py
"""
import os
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / "venv"
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

MIN_PYTHON = (3, 10)
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"

IS_WINDOWS = os.name == "nt"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(n: int, total: int, msg: str) -> None:
    say(f"\n[{n}/{total}] {msg}")


def fail(msg: str) -> None:
    say("\n" + "=" * 68)
    say("SETUP STOPPED")
    say("=" * 68)
    say(msg)
    say("")
    if IS_WINDOWS:
        input("Press Enter to close this window...")
    sys.exit(1)


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"This needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.\n"
            f"You have Python {sys.version_info.major}.{sys.version_info.minor}.\n\n"
            "Download a newer version from https://www.python.org/downloads/\n"
            'On Windows, tick "Add python.exe to PATH" on the first install screen.'
        )


def create_venv() -> None:
    if venv_python().exists():
        say("    already exists, reusing it")
        return
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0 or not venv_python().exists():
        fail(
            "Could not create the virtual environment.\n\n"
            "On Debian/Ubuntu you may need:  sudo apt install python3-venv"
        )


def install_dependencies() -> None:
    say("    this can take a couple of minutes the first time...")
    subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        cwd=REPO_ROOT,
    )
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "--quiet", "-r", "requirements.txt"],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        fail(
            "Installing dependencies failed.\n\n"
            "Most common cause is no internet connection or a firewall blocking pip.\n"
            "Scroll up for the specific error."
        )


def write_env_file() -> str | None:
    """Create .env with generated secrets. Returns the dashboard password,
    or None if .env already existed (so we don't clobber real settings)."""
    if ENV_FILE.exists():
        say("    .env already exists - keeping your existing settings")
        return None

    if not ENV_EXAMPLE.exists():
        fail(f"Missing {ENV_EXAMPLE.name}. Re-download the project files.")

    webhook_secret = secrets.token_urlsafe(32)
    dashboard_password = secrets.token_urlsafe(12)

    lines = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("WEBHOOK_SECRET="):
            line = f"WEBHOOK_SECRET={webhook_secret}"
        elif line.startswith("DASHBOARD_PASSWORD="):
            line = f"DASHBOARD_PASSWORD={dashboard_password}"
        lines.append(line)

    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say("    created .env with freshly generated secrets")
    return dashboard_password


def init_database() -> None:
    result = subprocess.run([str(venv_python()), "scripts/init_db.py"], cwd=REPO_ROOT)
    if result.returncode != 0:
        fail("Could not create the database. Scroll up for the error.")


def read_dashboard_credentials() -> tuple[str, str]:
    username, password = "admin", "changeme"
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("DASHBOARD_USERNAME="):
                username = line.split("=", 1)[1].strip()
            elif line.startswith("DASHBOARD_PASSWORD="):
                password = line.split("=", 1)[1].strip()
    return username, password


def open_browser_when_ready() -> None:
    """Poll the health endpoint, then open the dashboard."""
    import urllib.request

    for _ in range(60):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=1) as resp:
                if resp.status == 200:
                    webbrowser.open(URL)
                    return
        except Exception:
            continue


def main() -> None:
    say("=" * 68)
    say("  MEMECOIN TRADING BOT - LOCAL SETUP")
    say("  Paper trading mode: no real money, no wallet, no crypto involved.")
    say("=" * 68)

    total = 5
    step(1, total, "Checking your Python version...")
    check_python_version()
    say(f"    Python {sys.version_info.major}.{sys.version_info.minor} - good")

    step(2, total, "Creating an isolated environment for the bot...")
    create_venv()

    step(3, total, "Installing the software it needs...")
    install_dependencies()

    step(4, total, "Writing your configuration...")
    generated_password = write_env_file()

    step(5, total, "Setting up the database...")
    init_database()

    username, password = read_dashboard_credentials()

    say("\n" + "=" * 68)
    say("  READY")
    say("=" * 68)
    say(f"\n  Dashboard:  {URL}")
    say(f"  Username:   {username}")
    say(f"  Password:   {password}")
    if generated_password:
        say("\n  (These were generated for you and saved in the .env file.)")
    say("\n  Your browser should open automatically in a few seconds.")
    say("\n  To STOP the bot: press Ctrl+C here, or just close this window.")
    say("=" * 68 + "\n")

    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    try:
        subprocess.run(
            [
                str(venv_python()), "-m", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", str(PORT),
            ],
            cwd=REPO_ROOT,
        )
    except KeyboardInterrupt:
        pass

    say("\nBot stopped.")
    if IS_WINDOWS:
        input("Press Enter to close this window...")


if __name__ == "__main__":
    main()
