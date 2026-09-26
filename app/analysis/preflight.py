"""Can this machine actually run a collection, right now?

Written after a session where the market-data APIs were blocked by a
network policy. Everything looked healthy: the app booted, the workers
started, the scanner logged a cycle every interval. It just never got a
single price, and the only trace was a warning line in a log nobody was
reading. Left alone it would have "collected" for a week and produced an
empty database.

That failure has a shape worth naming: the bot is deliberately built to
degrade quietly rather than crash, because a missing price must never take
down a position monitor guarding real risk. The cost of that choice is
that a completely non-functional deployment looks exactly like a quiet
market. This module is the counterweight - the one place that goes and
LOOKS, and says so loudly.

WHAT IT ACTUALLY DOES

Probes each upstream for real, with a live request. Not the recorded
health table, which starts empty and therefore reports nothing wrong on a
fresh box - the precise moment the answer matters most.

RUN IT BEFORE WALKING AWAY

    python scripts/research.py preflight

Non-zero exit if anything that would waste the run is broken. It is safe
to run at any time: read-only, no orders, no writes.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import os
from dataclasses import dataclass, field

from app import models
from app.config import settings
from app.database import SessionLocal

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

# A probe slower than this is not usable for a scanner on a short cycle,
# even if it eventually answers.
PROBE_TIMEOUT_SECONDS = 20.0

# Well-known mints used only as probe subjects. Chosen because they are
# certain to exist and to have a pool - a probe that fails on an obscure
# token tells you nothing about the API.
PROBE_SOLANA_MINT = "So11111111111111111111111111111111111111112"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fatal: bool = True

    @property
    def blocking(self) -> bool:
        return self.status == FAIL and self.fatal

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "fatal": self.fatal}


@dataclass
class Preflight:
    checks: list[Check] = field(default_factory=list)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.blocking]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status in (WARN,) or
                (c.status == FAIL and not c.fatal)]

    def verdict(self) -> str:
        if self.blocking:
            names = ", ".join(c.name for c in self.blocking)
            return (
                f"NOT READY: {names}. Starting a collection run now would produce days of "
                "empty or unusable data while the logs looked busy."
            )
        if self.warnings:
            names = ", ".join(c.name for c in self.warnings)
            return f"READY, with caveats: {names}."
        return "READY. Every upstream answered and every gate is set for paper collection."

    def as_dict(self) -> dict:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "blocking": [c.name for c in self.blocking],
            "ready": not self.blocking,
            "verdict": self.verdict(),
        }


# ---------------------------------------------------------------------------
# safety - these must be true before anything else is worth checking
# ---------------------------------------------------------------------------

def _check_paper_only() -> Check:
    """Live trading is off and unacknowledged.

    First, and fatal. Every other check is about data quality; this one is
    about whether the run can spend money, and a preflight that reported
    "ready" on a live-armed deployment would be actively dangerous.
    """
    live = bool(getattr(settings, "LIVE_TRADING", False))
    ack = bool(getattr(settings, "LIVE_EXECUTION_ACKNOWLEDGED", False))
    if live or ack:
        return Check(
            "paper-only", FAIL,
            f"LIVE_TRADING={live}, LIVE_EXECUTION_ACKNOWLEDGED={ack}. This run is supposed to "
            "be paper-only. Stop and fix the configuration before starting anything.",
        )
    return Check("paper-only", PASS, "LIVE_TRADING and LIVE_EXECUTION_ACKNOWLEDGED are both false")


def _check_workers() -> Check:
    """Every loop the collection depends on is switched on.

    Any one of these off means a silently incomplete dataset rather than a
    visible failure: no scanner is no opportunities, no forward returns is
    no answer key, no resolver is decisions with no outcomes.
    """
    required = {
        "SCANNER_ENABLED": "nothing would be discovered",
        "SHADOW_ENABLED": "no champion or challenger decisions would be recorded",
        "SHADOW_RESOLVER_ENABLED": "hypothetical positions would never resolve",
        "FORWARD_RETURNS_ENABLED": "there would be no answer key for calibration",
    }
    off = [f"{name} ({why})" for name, why in required.items()
           if not bool(getattr(settings, name, False))]
    if off:
        return Check("collection workers", FAIL, "switched off: " + "; ".join(off))
    return Check("collection workers", PASS, "scanner, shadow, resolver and forward returns all on")


def _check_challengers() -> Check:
    """The challengers parse, and there are some.

    A malformed SHADOW_CHALLENGERS entry is skipped with a log line, by
    design - the alternative is the bot refusing to start over a research
    feature. The cost is an arm that silently never runs, which is exactly
    what this catches on day zero instead of week three.
    """
    from app.shadow.challengers import enabled

    try:
        challengers = enabled()
    except Exception as exc:                                # pragma: no cover - defensive
        return Check("challengers", FAIL, f"SHADOW_CHALLENGERS did not parse: {exc}")

    declared = (settings.SHADOW_CHALLENGERS or "").strip()
    if declared and not challengers:
        return Check("challengers", FAIL,
                     "SHADOW_CHALLENGERS is set but no challenger loaded - the JSON is malformed "
                     "and every entry was skipped")
    if not challengers:
        return Check("challengers", WARN,
                     "no challengers configured - the champion will be recorded alone, which is "
                     "a valid baseline but produces no comparison", fatal=False)
    names = ", ".join(f"{c.strategy_id}@{c.threshold():g}" for c in challengers)
    return Check("challengers", PASS,
                 f"{len(challengers)} loaded ({names}) against champion "
                 f"@{settings.MIN_SIGNAL_SCORE_TO_ENTER:g}")


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def _check_database() -> Check:
    """The database exists, is writable, and carries the current schema."""
    db = SessionLocal()
    try:
        db.query(models.ShadowHorizonReturn).count()      # newest table
        db.query(models.ShadowDecision).count()
        db.query(models.ForwardReturn).count()
        return Check("database", PASS, "readable and carrying the current schema")
    except Exception as exc:
        return Check("database", FAIL,
                     f"{type(exc).__name__}: {exc}. Run scripts/init_db.py, or start the app "
                     "once so migrations apply.")
    finally:
        db.close()


def _check_backups() -> Check:
    """Snapshots land somewhere that survives a restart.

    A backup written inside an ephemeral container is not a backup. It is
    disk I/O that produces a feeling.
    """
    from app import backup

    if not settings.BACKUP_ENABLED:
        return Check("backups", WARN,
                     "BACKUP_ENABLED=false - a wiped disk loses every observation collected so "
                     "far, and there is no way to get them back", fatal=False)
    pointless = backup.warn_if_backups_are_pointless()
    if pointless:
        return Check("backups", WARN, pointless, fatal=False)
    return Check("backups", PASS, f"writing to {settings.BACKUP_DIR}")


def _check_version_registered() -> Check:
    """One strategy version, and it is the one running now."""
    from app.strategy.version import current_label

    label = current_label()
    db = SessionLocal()
    try:
        versions = [v[0] for v in db.query(models.ShadowDecision.strategy_version)
                    .distinct().all() if v[0]]
    finally:
        db.close()
    stale = [v for v in versions if v != label]
    if stale:
        return Check("strategy version", WARN,
                     f"running {label}, but the database already holds observations from "
                     f"{', '.join(stale)}. Do not pool them - filter to one version before "
                     "judging any challenger.", fatal=False)
    return Check("strategy version", PASS, f"running {label}")


# ---------------------------------------------------------------------------
# upstreams - the check that would have caught the silent failure
# ---------------------------------------------------------------------------

async def _probe(name: str, coro, *, ok, healthy, consequence: str) -> Check:
    """Run one live probe, bounded in time.

    Any exception is a failure. This is the one place in the codebase where
    a broad except is right: the question is "did a usable answer come
    back", and every way of not answering is the same answer.

    `consequence` is attached to EVERY failure, raised or empty alike. An
    upstream usually fails by raising, and a bare "LookupFailed: GoPlus did
    not answer" tells a reader what broke without telling them what it
    costs - which is the entire job of this command.
    """
    try:
        result = await asyncio.wait_for(coro, timeout=PROBE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return Check(name, FAIL,
                     f"no response within {PROBE_TIMEOUT_SECONDS:g}s - too slow to be usable "
                     f"even if it eventually answers. {consequence}")
    except Exception as exc:
        return Check(name, FAIL, f"{type(exc).__name__}: {exc}. {consequence}")
    if ok(result):
        return Check(name, PASS, healthy(result))
    return Check(name, FAIL, f"answered, but with nothing usable. {consequence}")


async def _check_price_feed() -> Check:
    from app.services import price_feed

    return await _probe(
        "price feed",
        price_feed.get_price_usd(PROBE_SOLANA_MINT),
        ok=lambda price: bool(price and price > 0),
        healthy=lambda price: f"quoted a known mint at ${price:,.2f}",
        consequence=(
            "The probe mint certainly has a price, so this is the API being unreachable, "
            "blocked or rate-limited - not a quiet market. Every fill, exit and forward return "
            "depends on it."
        ),
    )


async def _check_candles() -> Check:
    from app.data.candles import Timeframe
    from app.data.live_provider import fetch_candles

    return await _probe(
        "candles",
        fetch_candles("solana", PROBE_SOLANA_MINT, "SOL", Timeframe.M5, 20),
        ok=lambda series: bool(series and len(series)),
        healthy=lambda series: f"returned {len(series)} bars for a known mint",
        consequence=(
            "Without OHLCV nothing is scored, no hypothetical position resolves, and the "
            "scanner rejects every candidate for missing history."
        ),
    )


async def _check_security_api() -> Check:
    from app.rugcheck import goplus

    return await _probe(
        "security screening",
        goplus.fetch_token_security("solana", PROBE_SOLANA_MINT),
        ok=bool,
        healthy=lambda data: "answered a token security query",
        consequence=(
            "The security gate fails CLOSED, so this does not produce risky trades - it "
            "produces zero trades, silently, for as long as it lasts."
        ),
    )


async def _upstream_checks() -> list[Check]:
    return list(await asyncio.gather(
        _check_price_feed(), _check_candles(), _check_security_api()
    ))


# ---------------------------------------------------------------------------

def run(*, probe_upstreams: bool = True) -> Preflight:
    """Every check. Read-only; safe to run against a live deployment."""
    report = Preflight()
    report.checks = [
        _check_paper_only(),
        _check_workers(),
        _check_challengers(),
        _check_database(),
        _check_backups(),
        _check_version_registered(),
    ]
    if probe_upstreams:
        report.checks.extend(asyncio.run(_upstream_checks()))
    return report


def environment_summary() -> dict:
    """Context worth printing next to the result, not graded.

    Included because the commonest cause of a confusing preflight is a
    process reading a different .env than the person running it.
    """
    return {
        "database": settings.DATABASE_URL,
        "chain": settings.CHAIN,
        "backup_dir": settings.BACKUP_DIR,
        "cwd": os.getcwd(),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
