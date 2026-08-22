"""Round 3 of the external review. Two findings held up, one did not.

HELD UP

  1. `LIVE_TRADING = true` - with spaces around the separator - defeated
     the operator guards. python-dotenv trims that whitespace, so the
     application read the flag as ENABLED while the guard, matching
     `FLAG=` as a literal prefix, reported the deployment paper-only.
     Second fail-open found in this same guard, after the boolean-parsing
     one in round 2. Both had the same shape: the guard reimplemented a
     parser instead of matching the one that actually loads the file.

  2. The post-trade halt checks ran before the filled sell leg was
     flushed. `SessionLocal` is `autoflush=False` and
     `evaluate_consecutive_losses` queries `models.Trade`, so the streak
     was computed WITHOUT the trade that had just closed - the halt fires
     one trade late, and the run that hits its limit takes one more
     position before stopping.

DID NOT HOLD UP

  A build-request failure in the Jupiter swap path was said to escape
  `_execute_swap` and skip the failed-trade record. It does not:
  `app/services/http.py` catches `Exception` on every path in
  `request_json` and returns None, and `_execute_swap` already turns None
  into a failed SwapResult. Not fixed, deliberately - see
  test_a_failed_swap_build_still_produces_a_result below, which pins the
  behaviour so the claim does not need re-litigating.
"""
import asyncio
import datetime as dt
import importlib.util
import pathlib

import pytest

from app import models
from app.database import SessionLocal

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# 1. whitespace around the .env separator
# ---------------------------------------------------------------------------

GUARD_SCRIPTS = ("scripts/send_test_signal.py", "scripts/setup_and_run.py")


def _load(path: str):
    """Import a stdlib-only operator script by path.

    They are scripts, not modules, and deliberately import nothing from
    `app` - they run before the venv exists. Loading them by spec is the
    only way to test the real function rather than a copy of it.
    """
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Every spelling python-dotenv accepts as an assignment to LIVE_TRADING.
# The parser trims whitespace around both the key and the separator, so
# all of these enable the flag as far as the application is concerned.
WHITESPACE_SPELLINGS = [
    "LIVE_TRADING=true",
    "LIVE_TRADING =true",
    "LIVE_TRADING= true",
    "LIVE_TRADING = true",
    "LIVE_TRADING\t=\ttrue",
    "  LIVE_TRADING = true",
    "export LIVE_TRADING = true",
]


def test_dotenv_really_does_trim_whitespace_around_the_separator(tmp_path):
    """The premise, checked against the installed parser rather than
    taken on faith. If a future python-dotenv stopped accepting these,
    the guard below would be stricter than it needs to be - which is the
    safe direction, but this test would say so."""
    from dotenv import dotenv_values

    for spelling in WHITESPACE_SPELLINGS:
        env = tmp_path / "env"
        env.write_text(spelling + "\n", encoding="utf-8")
        parsed = dotenv_values(env)
        assert parsed.get("LIVE_TRADING") == "true", (
            f"{spelling!r} parsed as {parsed!r}"
        )


@pytest.mark.parametrize("script", GUARD_SCRIPTS)
@pytest.mark.parametrize("spelling", WHITESPACE_SPELLINGS)
def test_the_guards_refuse_every_spelling_the_loader_accepts(script, spelling, tmp_path):
    """The guard must agree with the parser that actually loads the file.
    Any spelling the application reads as live has to stop the script -
    otherwise the script prints its paper-mode banner over a live
    configuration, which is worse than no guard at all because it
    reassures."""
    module = _load(script)
    env = tmp_path / ".env"
    env.write_text(spelling + "\n", encoding="utf-8")

    assert module._live_flags_enabled_in_env(env) == ["LIVE_TRADING"], (
        f"{script} did not see {spelling!r} as live"
    )


@pytest.mark.parametrize("script", GUARD_SCRIPTS)
def test_a_commented_out_flag_is_still_ignored(script, tmp_path):
    """The complementary half. Loosening the key match must not make a
    commented line count - that would refuse to launch a correctly
    configured paper deployment, and an operator who cannot start the bot
    edits the guard out."""
    module = _load(script)
    env = tmp_path / ".env"
    env.write_text(
        "# LIVE_TRADING = true\n"
        "#LIVE_TRADING=true\n"
        "LIVE_TRADING = false\n",
        encoding="utf-8",
    )
    assert module._live_flags_enabled_in_env(env) == []


@pytest.mark.parametrize("script", GUARD_SCRIPTS)
def test_a_key_that_merely_starts_with_the_flag_name_is_not_the_flag(script, tmp_path):
    """`partition("=")` plus a trimmed EQUALITY check, not a prefix match.
    The old code would have matched `LIVE_TRADING_NOTES=true` had it been
    written `LIVE_TRADING=...`; the new code must not match a different
    key that happens to share the prefix."""
    module = _load(script)
    env = tmp_path / ".env"
    env.write_text("LIVE_TRADING_NOTES = true\nLIVE_TRADINGX=true\n", encoding="utf-8")
    assert module._live_flags_enabled_in_env(env) == []


@pytest.mark.parametrize("script", GUARD_SCRIPTS)
def test_a_line_with_no_separator_is_skipped_not_crashed_on(script, tmp_path):
    """A hand-edited .env can contain junk. The guard is a refusal path;
    it has to survive reading a malformed file rather than raising and
    taking the launcher down with it."""
    module = _load(script)
    env = tmp_path / ".env"
    env.write_text("LIVE_TRADING\njust some text\n\nLIVE_TRADING = 1\n", encoding="utf-8")
    assert module._live_flags_enabled_in_env(env) == ["LIVE_TRADING"]


# ---------------------------------------------------------------------------
# 2. the halt checks ran against an unflushed session
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    session = SessionLocal()

    def wipe():
        for row in session.query(models.Trade).filter(
            models.Trade.symbol.like("R3%")
        ).all():
            session.delete(row)
        session.commit()

    wipe()
    try:
        yield session
    finally:
        wipe()
        session.close()


def _losing_trade(db, symbol, minutes_ago, pnl_usd=-1.0):
    """A closed losing leg.

    The default loss is deliberately tiny. The daily-loss check runs
    BEFORE the streak check and returns early, so a fixture that loses
    real money halts on the daily limit and never reaches the streak
    logic the test is aiming at - which is exactly what the first draft
    of this test did.
    """
    now = dt.datetime.now(dt.timezone.utc)
    return models.Trade(
        symbol=symbol, side="sell", chain="solana",
        status=models.TradeStatus.FILLED.value,
        size_usd=100.0, qty=100.0, entry_price=1.0, exit_price=0.99,
        pnl_usd=pnl_usd, pnl_pct=-0.01,
        closed_at=now - dt.timedelta(minutes=minutes_ago),
        created_at=now - dt.timedelta(minutes=minutes_ago),
    )


async def test_the_loss_streak_counts_the_trade_that_just_closed(db, monkeypatch):
    """The regression.

    Build a streak one short of the limit, add the trade that completes
    it WITHOUT flushing - exactly the state both exit paths leave the
    session in - and check the halt fires. Before the fix the query could
    not see the pending row, returned only the earlier losses, and the
    streak came up one short: the bot took another position before
    stopping.
    """
    from app.risk.manager import RiskManager
    from app.services import trading_service

    rm = RiskManager()
    limit = rm.max_consecutive_losses

    for i in range(limit - 1):
        db.add(_losing_trade(db, f"R3STREAK{i}", minutes_ago=limit - i))
    db.commit()

    # The trade that completes the streak: added, deliberately NOT flushed.
    db.add(_losing_trade(db, "R3STREAKLAST", minutes_ago=0))

    halted = {}

    def fake_halt(session, reason):
        halted["reason"] = reason

    async def no_notify(*_a, **_k):
        return None

    monkeypatch.setattr(trading_service, "halt_trading", fake_halt)
    monkeypatch.setattr(trading_service.notifier, "notify_risk_halt", no_notify)

    await trading_service._check_halt_conditions(db)

    assert "reason" in halted, (
        "the halt did not fire - the pending sell leg was invisible to the "
        "streak query, so the limit is enforced one trade late"
    )
    assert "consecutive losing trades" in halted["reason"], (
        f"halted for the wrong reason: {halted['reason']}"
    )


async def test_the_daily_loss_check_also_sees_the_unflushed_trade(db, monkeypatch):
    """The flush fixes both halt checks, not just the streak.

    The daily-loss check runs first and returns early, so it is the one
    that decides whether the streak check is reached at all. A loss large
    enough to breach the daily limit must trip it while still pending,
    for the same reason: one more position would otherwise be taken
    first.
    """
    from app.services import trading_service

    db.add(_losing_trade(db, "R3DAILY", minutes_ago=0, pnl_usd=-100_000.0))

    halted = {}
    monkeypatch.setattr(trading_service, "halt_trading",
                        lambda session, reason: halted.setdefault("reason", reason))

    async def no_notify(*_a, **_k):
        return None

    monkeypatch.setattr(trading_service.notifier, "notify_risk_halt", no_notify)
    await trading_service._check_halt_conditions(db)

    assert "reason" in halted
    assert "daily" in halted["reason"].lower()


async def test_the_halt_check_flushes_rather_than_commits(db, monkeypatch):
    """A flush makes pending rows visible to queries in the same
    transaction; a commit would end the transaction the caller still
    owns, and an exit path that raised after the halt check could no
    longer be rolled back as a unit."""
    from app.services import trading_service

    commits = []
    monkeypatch.setattr(type(db), "commit",
                        lambda self: commits.append(1), raising=False)

    async def no_notify(*_a, **_k):
        return None

    monkeypatch.setattr(trading_service.notifier, "notify_risk_halt", no_notify)
    await trading_service._check_halt_conditions(db)

    assert commits == [], "_check_halt_conditions committed the caller's transaction"


# ---------------------------------------------------------------------------
# 3. the finding that did not hold up
# ---------------------------------------------------------------------------

async def test_a_failed_swap_build_still_produces_a_result(monkeypatch):
    """Pins why the third finding was not acted on.

    The claim was that a raising `http.post_json` would escape
    `_execute_swap`, so the caller would skip its failed-trade record and
    its notification. It cannot: `request_json` catches Exception on
    every path and returns None, and `_execute_swap` maps None onto a
    failed SwapResult. This test asserts the contract the claim depends
    on, so a future change that DID let the exception through would fail
    here rather than silently losing an audit record.
    """
    from app.services import http

    class ExplodingClient:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            raise RuntimeError("connection reset mid-flight")

        async def __aexit__(self, *_a):
            return False

    monkeypatch.setattr(http.httpx, "AsyncClient", ExplodingClient)

    result = await http.post_json("https://example.invalid/swap", json={},
                                  label="test", idempotent=True)
    assert result is None, "post_json propagated instead of returning None"
