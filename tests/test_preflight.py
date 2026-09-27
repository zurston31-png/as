"""Tests for app/analysis/preflight.py.

This command exists because of a real failure mode: the app booted, the
workers started, the scanner logged a cycle every interval, and not one
price came back. The bot degrades quietly by design - a missing price must
never take down the position monitor - so a completely non-functional
deployment looks exactly like a quiet market.

The tests that matter are therefore the ones proving it reports NOT READY
when that is true, and that it never reports ready on a live-armed
configuration.
"""
import pytest

from app.analysis import preflight
from app.analysis.preflight import FAIL, PASS, WARN, Check, Preflight
from app.config import settings


def named(report, name):
    return next(c for c in report.checks if c.name == name)


@pytest.fixture(autouse=True)
def paper_defaults(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_TRADING", False)
    monkeypatch.setattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", False)
    for name in ("SCANNER_ENABLED", "SHADOW_ENABLED", "SHADOW_RESOLVER_ENABLED",
                 "FORWARD_RETURNS_ENABLED"):
        monkeypatch.setattr(settings, name, True)


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------

def test_a_live_armed_configuration_is_never_ready(monkeypatch):
    """Every other check is about data quality. This one is about whether
    the run can spend money, so reporting "ready" would be actively
    dangerous rather than merely wrong."""
    monkeypatch.setattr(settings, "LIVE_TRADING", True)

    report = preflight.run(probe_upstreams=False)
    check = named(report, "paper-only")
    assert check.status == FAIL
    assert check.fatal is True
    assert report.blocking
    assert "NOT READY" in report.verdict()


def test_an_acknowledged_live_flag_alone_still_blocks(monkeypatch):
    """Both flags are part of the arming sequence. Treating only the first
    as dangerous would let a half-armed deployment pass."""
    monkeypatch.setattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", True)

    assert named(preflight.run(probe_upstreams=False), "paper-only").status == FAIL


# ---------------------------------------------------------------------------
# the silent failure this exists to catch
# ---------------------------------------------------------------------------

def _stub_probes(monkeypatch, *, price=180.0, candles=20, security=True):
    async def fake_price(mint):
        return price

    async def fake_candles(chain, mint, symbol, timeframe, limit):
        if candles is None:
            return None
        return [object()] * candles

    async def fake_security(chain, mint):
        if not security:
            raise RuntimeError("GoPlus did not answer")
        return {"ok": True}

    monkeypatch.setattr("app.services.price_feed.get_price_usd", fake_price)
    monkeypatch.setattr("app.data.live_provider.fetch_candles", fake_candles)
    monkeypatch.setattr("app.rugcheck.goplus.fetch_token_security", fake_security)


def test_healthy_upstreams_report_ready(monkeypatch):
    _stub_probes(monkeypatch)
    report = preflight.run()
    assert named(report, "price feed").status == PASS
    assert named(report, "candles").status == PASS
    assert named(report, "security screening").status == PASS
    assert report.blocking == []


def test_a_blocked_price_feed_blocks_the_run(monkeypatch):
    """The exact failure this command was written for: reachable-looking
    process, unreachable API, empty database a week later."""
    _stub_probes(monkeypatch, price=None)

    report = preflight.run()
    check = named(report, "price feed")
    assert check.status == FAIL
    assert "unreachable, blocked or rate-limited" in check.detail
    # Every failure carries its consequence, not just what broke.
    assert "forward return depends on it" in check.detail
    assert "NOT READY" in report.verdict()


def test_missing_candles_block_the_run(monkeypatch):
    _stub_probes(monkeypatch, candles=None)

    assert named(preflight.run(), "candles").status == FAIL


def test_a_dead_security_api_blocks_the_run(monkeypatch):
    """The security gate fails CLOSED, so an unreachable screening API does
    not produce risky trades - it produces zero trades, silently."""
    _stub_probes(monkeypatch, security=False)

    check = named(preflight.run(), "security screening")
    assert check.status == FAIL
    # A raised LookupFailed is the COMMON case for an upstream, and a bare
    # "GoPlus did not answer" says what broke without saying what it costs.
    assert "GoPlus did not answer" in check.detail
    assert "fails CLOSED" in check.detail


def test_a_slow_upstream_counts_as_a_failure(monkeypatch):
    """Too slow to be usable is a failure even if it eventually answers -
    a scanner on a short cycle cannot wait."""
    import asyncio

    async def crawl(mint):
        await asyncio.sleep(5)
        return 1.0

    monkeypatch.setattr(preflight, "PROBE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("app.services.price_feed.get_price_usd", crawl)

    async def fake_candles(*a, **k):
        return [object()]

    async def fake_security(*a, **k):
        return {"ok": True}

    monkeypatch.setattr("app.data.live_provider.fetch_candles", fake_candles)
    monkeypatch.setattr("app.rugcheck.goplus.fetch_token_security", fake_security)

    check = named(preflight.run(), "price feed")
    assert check.status == FAIL
    assert "within" in check.detail


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("worker", [
    "SCANNER_ENABLED", "SHADOW_ENABLED", "SHADOW_RESOLVER_ENABLED", "FORWARD_RETURNS_ENABLED",
])
def test_any_disabled_worker_blocks_the_run(monkeypatch, worker):
    """Each one off means a silently incomplete dataset rather than a
    visible failure."""
    monkeypatch.setattr(settings, worker, False)

    check = named(preflight.run(probe_upstreams=False), "collection workers")
    assert check.status == FAIL
    assert worker in check.detail


def test_malformed_challenger_json_is_caught_on_day_zero(monkeypatch):
    """A malformed entry is skipped with a log line by design - the
    alternative is refusing to start over a research feature. The cost is
    an arm that silently never runs."""
    monkeypatch.setattr(settings, "SHADOW_CHALLENGERS", "[{not json at all}]")

    check = named(preflight.run(probe_upstreams=False), "challengers")
    assert check.status == FAIL
    assert "malformed" in check.detail


def test_no_challengers_is_a_warning_not_a_blocker(monkeypatch):
    """Champion-only recording is a valid baseline. It just produces no
    comparison, which is worth saying and not worth blocking on."""
    monkeypatch.setattr(settings, "SHADOW_CHALLENGERS", "")

    report = preflight.run(probe_upstreams=False)
    check = named(report, "challengers")
    assert check.status == WARN
    assert check.fatal is False
    assert report.blocking == []


def test_the_shipped_challengers_pass(monkeypatch):
    check = named(preflight.run(probe_upstreams=False), "challengers")
    assert check.status == PASS
    assert "strict-70" in check.detail and "loose-60" in check.detail


def test_the_database_check_sees_the_newest_table():
    """A database that predates the newest migration would answer every
    older query fine and fail only on the table the run depends on."""
    assert named(preflight.run(probe_upstreams=False), "database").status == PASS


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------

def test_a_non_fatal_failure_does_not_block():
    report = Preflight(checks=[
        Check("backups", WARN, "ephemeral disk", fatal=False),
        Check("paper-only", PASS, "ok"),
    ])
    assert report.blocking == []
    assert "READY, with caveats" in report.verdict()


def test_everything_green_says_ready_plainly():
    report = Preflight(checks=[Check("paper-only", PASS, "ok")])
    assert report.verdict().startswith("READY.")


def test_preflight_writes_nothing():
    """Safe to run against a live deployment at any time - including one
    that is mid-collection, which is exactly when someone will reach for
    it."""
    import pathlib

    body = "\n".join(
        line for line in pathlib.Path("app/analysis/preflight.py").read_text().splitlines()
        if not line.strip().startswith("#")
    )
    for forbidden in ("db.add(", "db.commit(", "db.delete(", "execute("):
        assert forbidden not in body
