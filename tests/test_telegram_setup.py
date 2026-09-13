"""Tests for the Telegram setup helper.

The .env rewrite is the risky part: it edits the file holding the webhook
secret and the LIVE_TRADING switch, so a careless write could silently
change trading behaviour while appearing to configure notifications.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import setup_telegram as st  # noqa: E402


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "LIVE_TRADING=false\n"
        "WEBHOOK_SECRET=super-secret-value\n"
        "TELEGRAM_BOT_TOKEN=\n"
        "TELEGRAM_CHAT_ID=\n"
        "DASHBOARD_PASSWORD=hunter2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(st, "ENV_FILE", path)
    return path


def value_of(text: str, key: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    return None


# --- chat id extraction -----------------------------------------------------

@pytest.mark.parametrize("update,expected_id", [
    ({"message": {"chat": {"id": 111, "username": "someone"}}}, "111"),
    ({"edited_message": {"chat": {"id": 222, "first_name": "L"}}}, "222"),
    ({"channel_post": {"chat": {"id": -333, "title": "Chan"}}}, "-333"),
    ({"my_chat_member": {"chat": {"id": 444, "username": "x"}}}, "444"),
])
def test_extract_chat_handles_each_update_shape(update, expected_id):
    found = st.extract_chat(update)
    assert found is not None
    assert found[0] == expected_id


def test_extract_chat_ignores_unrelated_updates():
    assert st.extract_chat({"poll": {"id": "abc"}}) is None
    assert st.extract_chat({}) is None


def test_extract_chat_handles_missing_chat_id():
    assert st.extract_chat({"message": {"text": "hi"}}) is None


# --- .env rewriting ---------------------------------------------------------

def test_write_env_sets_both_keys(env_file):
    st.write_env("123:TOKEN", "987")
    text = env_file.read_text(encoding="utf-8")
    assert value_of(text, "TELEGRAM_BOT_TOKEN") == "123:TOKEN"
    assert value_of(text, "TELEGRAM_CHAT_ID") == "987"


def test_write_env_preserves_other_settings(env_file):
    st.write_env("123:TOKEN", "987")
    text = env_file.read_text(encoding="utf-8")
    assert value_of(text, "WEBHOOK_SECRET") == "super-secret-value"
    assert value_of(text, "LIVE_TRADING") == "false"
    assert value_of(text, "DASHBOARD_PASSWORD") == "hunter2"


def test_write_env_is_idempotent(env_file):
    st.write_env("123:TOKEN", "987")
    st.write_env("456:OTHER", "654")
    text = env_file.read_text(encoding="utf-8")
    assert text.count("TELEGRAM_BOT_TOKEN=") == 1
    assert text.count("TELEGRAM_CHAT_ID=") == 1
    assert value_of(text, "TELEGRAM_BOT_TOKEN") == "456:OTHER"


def test_write_env_appends_keys_that_are_absent(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("LIVE_TRADING=false\n", encoding="utf-8")
    monkeypatch.setattr(st, "ENV_FILE", path)

    st.write_env("123:TOKEN", "987")
    text = path.read_text(encoding="utf-8")
    assert value_of(text, "TELEGRAM_BOT_TOKEN") == "123:TOKEN"
    assert value_of(text, "TELEGRAM_CHAT_ID") == "987"
    assert value_of(text, "LIVE_TRADING") == "false"


def test_write_env_never_enables_live_trading(env_file):
    """A notifications helper must not be able to flip the safety switch."""
    st.write_env("123:TOKEN", "987")
    assert value_of(env_file.read_text(encoding="utf-8"), "LIVE_TRADING") == "false"
