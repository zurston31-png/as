"""Simple server-rendered dashboard: open positions, trade history,
performance stats, and manual halt/resume controls. Protected by HTTP
Basic Auth (DASHBOARD_USERNAME/DASHBOARD_PASSWORD).
"""
import datetime as dt
import secrets

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from app import backup, models
from app.analysis.funnel import build_funnel
from app.analysis.calibration import HORIZONS_MINUTES, build_calibration
from app.analysis.forward_returns import coverage as forward_coverage
from app.analysis.research_report import build_research_report
from app.analysis.score_distribution import build_score_distribution
from app.analysis.stage_funnel import build_stage_funnel
from app.pipeline import MARKET_QUALITY, SECURITY, TECHNICAL_SCORE
from app.analysis.early_calibration import (
    build_early_calibration, build_false_positives, build_lead_time,
)
from app.early import watchlist as wl
from app.safety import killswitch
from app.analysis.report import build_performance_report
from app.analysis.token_detail import build_token_detail
from app.analysis.trade_analytics import MIN_TRADES_FOR_A_MEANINGFUL_BUCKET
from app.config import settings
from app.dashboard.analytics import compute_equity_curve, compute_portfolio_stats
from app.dashboard.charts import equity_curve_svg
from app.database import SessionLocal
from app.risk.manager import halt_trading, is_trading_halted, resume_trading
from app.scanner.loop import scanner_blocked_reason
from app.services import api_health, portfolio, price_feed
from app.startup_checks import check_config_coherence

router = APIRouter()
templates = Jinja2Templates(directory="app/dashboard/templates")
security = HTTPBasic()


PLACEHOLDER_PASSWORD = "changeme"

# Rotates on restart, which is fine: it only has to outlive a session, and a
# restart simply forces a page reload before halt/resume can be used again.
CSRF_TOKEN = secrets.token_urlsafe(32)


def check_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    # Refuse the shipped default outright, the same way app/security.py
    # refuses the placeholder webhook secret. docker-compose publishes port
    # 8000 and .env.example sets HOST=0.0.0.0, so a deployer who changes
    # WEBHOOK_SECRET (forced - alerts are rejected otherwise) but overlooks
    # this one would otherwise expose the dashboard and its halt/resume
    # controls on admin/changeme.
    if not settings.DASHBOARD_PASSWORD or settings.DASHBOARD_PASSWORD == PLACEHOLDER_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "dashboard password is still the default - set DASHBOARD_PASSWORD "
                "in .env to a value of your own, then restart the bot"
            ),
            headers={"WWW-Authenticate": "Basic"},
        )

    ok_user = secrets.compare_digest(credentials.username, settings.DASHBOARD_USERNAME)
    ok_pass = secrets.compare_digest(credentials.password, settings.DASHBOARD_PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def check_csrf(csrf_token: str = Form(default="")) -> None:
    """Guard state-changing endpoints against cross-site form submission.

    Basic Auth alone is not enough: browsers attach cached Basic credentials
    to cross-origin form posts, and unlike cookies there is no SameSite to
    lean on. Without this, a page the operator merely visits could POST to
    /api/resume and restart a bot the daily-loss limit had halted.
    """
    if not secrets.compare_digest(csrf_token or "", CSRF_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid or missing CSRF token - reload the dashboard and try again",
        )


async def _enrich_open_positions(db, open_positions: list[models.Position]) -> list[dict]:
    """Attach current price, unrealized P&L, age, the rug risk score, and
    the live signal score recorded at entry to each open position, for the
    dashboard's "what am I actually holding right now" view - the raw
    Position row alone answers "what did I buy", not "how is it doing" or
    "why did the bot buy it". Both scores are entry-time snapshots (from
    the Signal/RugCheckResult rows the entry trade is linked to), not a
    live re-score on every dashboard refresh - re-running the signal score
    per open position per page load would mean another GeckoTerminal call
    per position per refresh, which buys nothing (the entry decision is
    already made) at the cost of latency and rate-limit pressure.
    """
    entry_trade_ids = [p.entry_trade_id for p in open_positions if p.entry_trade_id]
    signal_ids_by_trade = {}
    if entry_trade_ids:
        for trade in db.query(models.Trade).filter(models.Trade.id.in_(entry_trade_ids)).all():
            signal_ids_by_trade[trade.id] = trade.signal_id

    signal_ids = [sid for sid in signal_ids_by_trade.values() if sid]
    rug_by_signal = {}
    signals_by_id = {}
    if signal_ids:
        for check in db.query(models.RugCheckResult).filter(models.RugCheckResult.signal_id.in_(signal_ids)).all():
            rug_by_signal[check.signal_id] = check
        for sig in db.query(models.Signal).filter(models.Signal.id.in_(signal_ids)).all():
            signals_by_id[sig.id] = sig

    now = dt.datetime.now(dt.timezone.utc)
    enriched = []
    for p in open_positions:
        current_price = None
        if p.token_address:
            try:
                current_price = await price_feed.get_price_usd(p.token_address)
            except Exception:
                current_price = None
        current_price = current_price or p.entry_price

        unrealized_pnl_usd = (current_price - p.entry_price) * p.qty
        unrealized_pnl_pct = (current_price / p.entry_price - 1) if p.entry_price else 0.0

        opened_at = p.opened_at
        if opened_at and opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=dt.timezone.utc)
        age_hours = (now - opened_at).total_seconds() / 3600 if opened_at else None

        signal_id = signal_ids_by_trade.get(p.entry_trade_id)
        rug_check = rug_by_signal.get(signal_id) if signal_id else None
        entry_signal = signals_by_id.get(signal_id) if signal_id else None

        enriched.append({
            "position": p,
            "current_price": current_price,
            "unrealized_pnl_usd": unrealized_pnl_usd,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "age_hours": age_hours,
            "rug_risk_score": rug_check.rug_risk_score if rug_check else None,
            "rug_risk_level": rug_check.rug_risk_level if rug_check else None,
            "signal_score": entry_signal.signal_score if entry_signal else None,
        })
    return enriched


@router.get("/")
async def dashboard(request: Request, user: str = Depends(check_auth)):
    db = SessionLocal()
    try:
        open_positions = (
            db.query(models.Position)
            .filter_by(status=models.PositionStatus.OPEN.value)
            .order_by(models.Position.opened_at.desc())
            .all()
        )
        enriched_positions = await _enrich_open_positions(db, open_positions)
        current_exposure_usd = sum(
            e["position"].qty * e["current_price"] for e in enriched_positions
        )

        recent_trades = db.query(models.Trade).order_by(models.Trade.id.desc()).limit(50).all()
        recent_rejections = (
            db.query(models.RiskEvent).order_by(models.RiskEvent.id.desc()).limit(20).all()
        )

        # Every signal received, whether or not it led to anything. Without
        # this, "no alerts are arriving" and "alerts arrive and nothing
        # happens" look identical from the dashboard — the main question
        # while the bot runs unattended.
        recent_signals = (
            db.query(models.Signal).order_by(models.Signal.id.desc()).limit(25).all()
        )
        checks_by_signal = {
            row.signal_id: row
            for row in db.query(models.RugCheckResult)
            .filter(models.RugCheckResult.signal_id.in_([s.id for s in recent_signals] or [0]))
            .all()
        }

        # What the auto-scanner has been doing. Without this, "the scanner is
        # running but never trades" is indistinguishable from "the scanner
        # isn't running" — the same visibility gap the Signals panel closes
        # for TradingView alerts.
        scanned_tokens = (
            db.query(models.ScannedToken)
            .order_by(models.ScannedToken.last_evaluated_at.desc())
            .limit(25)
            .all()
        )
        scanner_summary = {
            "tracked": db.query(models.ScannedToken).count(),
            "traded": db.query(models.ScannedToken).filter(models.ScannedToken.times_traded > 0).count(),
            "blocked_reason": scanner_blocked_reason(),
        }

        # Surfaced on the page, not just at boot: "why isn't it trading" is
        # asked while looking at the dashboard, not while reading startup logs.
        config_warnings = check_config_coherence()

        all_trades = db.query(models.Trade).all()
        portfolio_stats = compute_portfolio_stats(all_trades, settings.PORTFOLIO_STARTING_BALANCE_USD)
        equity_curve = compute_equity_curve(all_trades, settings.PORTFOLIO_STARTING_BALANCE_USD)
        equity_svg = equity_curve_svg(equity_curve)

        today = dt.datetime.now(dt.timezone.utc).date()
        realized_today_usd = sum(
            t.pnl_usd or 0.0
            for t in all_trades
            if t.pnl_usd is not None and t.closed_at is not None and t.closed_at.date() == today
        )

        portfolio_value = await portfolio.get_portfolio_value_usd(db)

        stats = {
            "portfolio_value": portfolio_value,
            "starting_balance": settings.PORTFOLIO_STARTING_BALANCE_USD,
            "total_trades": portfolio_stats.trade_count,
            "win_rate": portfolio_stats.win_rate,
            "total_pnl": portfolio_value - settings.PORTFOLIO_STARTING_BALANCE_USD,
            "realized_today_usd": realized_today_usd,
            "unrealized_pnl_usd": sum(e["unrealized_pnl_usd"] for e in enriched_positions),
            "current_exposure_usd": current_exposure_usd,
            "exposure_pct_of_portfolio": (current_exposure_usd / portfolio_value * 100) if portfolio_value else 0.0,
            "open_positions": len(open_positions),
            "mode": "LIVE" if settings.LIVE_TRADING else "PAPER",
            "chain": settings.CHAIN,
            "execution_backend": settings.EXECUTION_BACKEND,
            "halted": is_trading_halted(db),
            "watchlist": settings.SYMBOLS_WATCHLIST,
            "scanner_interval_seconds": settings.SCANNER_INTERVAL_SECONDS,
            "profit_factor_display": (
                "∞" if portfolio_stats.profit_factor == float("inf")
                else (f"{portfolio_stats.profit_factor:.2f}" if portfolio_stats.profit_factor is not None else "-")
            ),
            "expectancy_usd": portfolio_stats.expectancy_usd,
            "max_drawdown_pct": portfolio_stats.max_drawdown_pct,
            "current_streak": portfolio_stats.current_streak,
            "longest_winning_streak": portfolio_stats.longest_winning_streak,
            "longest_losing_streak": portfolio_stats.longest_losing_streak,
        }
        # Starlette's request-first signature. The older
        # TemplateResponse(name, {"request": ...}) form is deprecated, and on
        # current Starlette it silently treats the context dict as the
        # template name ("unhashable type: 'dict'").
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "stats": stats,
                "csrf_token": CSRF_TOKEN,
                "positions": enriched_positions,
                "trades": recent_trades,
                "events": recent_rejections,
                "signals": recent_signals,
                "checks": checks_by_signal,
                "equity_svg": equity_svg,
                "scanned_tokens": scanned_tokens,
                "scanner": scanner_summary,
                "config_warnings": config_warnings,
            },
        )
    finally:
        db.close()


@router.get("/journal")
async def journal(request: Request, user: str = Depends(check_auth)):
    """One row per position (a full round trip, or still open): entry
    context (indicators from the triggering signal, rug risk score),
    every exit leg with its own reason (full or partial - Trade.close_reason,
    Stage 8), and the net realized P&L. This is what "study why the bot won
    or lost" means in practice - the pieces already existed on separate
    dashboard tables, this assembles them into one record per trade.
    """
    db = SessionLocal()
    try:
        positions = (
            db.query(models.Position).order_by(models.Position.opened_at.desc()).limit(100).all()
        )

        entry_trade_ids = [p.entry_trade_id for p in positions if p.entry_trade_id]
        entry_trades = {
            t.id: t
            for t in db.query(models.Trade).filter(models.Trade.id.in_(entry_trade_ids or [0])).all()
        }

        signal_ids = [t.signal_id for t in entry_trades.values() if t.signal_id]
        signals_by_id = {
            s.id: s for s in db.query(models.Signal).filter(models.Signal.id.in_(signal_ids or [0])).all()
        }
        rug_checks_by_signal = {
            c.signal_id: c
            for c in db.query(models.RugCheckResult)
            .filter(models.RugCheckResult.signal_id.in_(signal_ids or [0]))
            .all()
        }

        position_ids = [p.id for p in positions]
        exit_trades_by_position: dict[int, list[models.Trade]] = {}
        if position_ids:
            exit_trades = (
                db.query(models.Trade)
                .filter(
                    models.Trade.position_id.in_(position_ids),
                    models.Trade.side == "sell",
                    models.Trade.status == models.TradeStatus.FILLED.value,
                )
                .order_by(models.Trade.created_at.asc())
                .all()
            )
            for t in exit_trades:
                exit_trades_by_position.setdefault(t.position_id, []).append(t)

        rows = []
        for p in positions:
            entry_trade = entry_trades.get(p.entry_trade_id)
            signal = signals_by_id.get(entry_trade.signal_id) if entry_trade and entry_trade.signal_id else None
            rug_check = (
                rug_checks_by_signal.get(entry_trade.signal_id) if entry_trade and entry_trade.signal_id else None
            )
            exits = exit_trades_by_position.get(p.id, [])
            total_realized_pnl = (p.realized_pnl_usd or 0.0) + sum(t.pnl_usd or 0.0 for t in exits)
            rows.append({
                "position": p,
                "entry_trade": entry_trade,
                "signal": signal,
                "rug_check": rug_check,
                "exit_trades": exits,
                "total_realized_pnl": total_realized_pnl,
            })

        return templates.TemplateResponse(request, "journal.html", {"rows": rows})
    finally:
        db.close()


@router.get("/performance")
async def performance(
    request: Request, version: str | None = None, user: str = Depends(check_auth)
):
    """The page that answers "is this record strong enough to believe?".

    Deliberately leads with the validation verdict rather than the return.
    "+34%" and "on 12 trades" mean very different things, and the second is
    the part people skip - so the status banner is the first thing on the
    page and the P&L is below it.

    `?version=<label>` reports on one strategy configuration. Without it the
    report pools every version and says so, loudly, in its warnings.
    """
    db = SessionLocal()
    try:
        report = build_performance_report(db, strategy_version=version)
        return templates.TemplateResponse(
            request,
            "performance.html",
            {
                "r": report,
                "inf": float("inf"),
                "min_bucket": MIN_TRADES_FOR_A_MEANINGFUL_BUCKET,
            },
        )
    finally:
        db.close()


@router.get("/api/performance")
async def api_performance(version: str | None = None, user: str = Depends(check_auth)):
    db = SessionLocal()
    try:
        return JSONResponse(build_performance_report(db, strategy_version=version).as_dict())
    finally:
        db.close()


@router.get("/pipeline")
async def pipeline(request: Request, hours: float = 24.0, user: str = Depends(check_auth)):
    """The scanner funnel plus system health, on one page.

    These belong together: a funnel that suddenly narrows and an upstream
    that just went down look identical from the trade log, and putting the
    health panel next to the stage counts is what lets an operator tell
    "the market is quiet" from "DexScreener has been 429ing for an hour".
    """
    db = SessionLocal()
    try:
        api_health.persist(db)
        db.commit()

        window = hours if hours > 0 else None
        funnel = build_funnel(db, window_hours=window)
        # The stage funnel is the newer, per-token instrumented view. It
        # sits alongside the RiskEvent-derived one rather than replacing it
        # because the older view still covers databases recorded before the
        # pipeline log existed, and a page that silently showed zero for
        # that history would read as a broken scanner.
        stages = build_stage_funnel(db, window_hours=window)
        integrity = await killswitch.may_open_position(db)
        recent = (
            db.query(models.ScannedToken)
            .order_by(models.ScannedToken.last_evaluated_at.desc())
            .limit(60)
            .all()
        )
        return templates.TemplateResponse(
            request,
            "pipeline.html",
            {
                "funnel": funnel,
                "stages": stages,
                "integrity": integrity,
                "health": [h.as_dict() for h in api_health.snapshot()],
                "recent_tokens": recent,
                "hours": hours,
                "scanner_blocked": scanner_blocked_reason(),
            },
        )
    finally:
        db.close()


@router.get("/token/{token_address}")
async def token_detail(request: Request, token_address: str, user: str = Depends(check_auth)):
    """Everything the bot knows about one mint, on one timeline.

    Keyed on the mint address, never the symbol - symbols are not unique
    and are trivially spoofed, so looking one up by symbol could merge two
    unrelated assets into a single history.
    """
    db = SessionLocal()
    try:
        detail = build_token_detail(db, token_address)
        return templates.TemplateResponse(request, "token.html", {"d": detail})
    finally:
        db.close()


@router.get("/api/pipeline")
async def api_pipeline(hours: float = 24.0, user: str = Depends(check_auth)):
    db = SessionLocal()
    try:
        return JSONResponse({
            "funnel": build_funnel(db, window_hours=hours if hours > 0 else None).as_dict(),
            "stages": build_stage_funnel(db, window_hours=hours if hours > 0 else None).as_dict(),
            "health": [h.as_dict() for h in api_health.snapshot()],
            "scanner_blocked": scanner_blocked_reason(),
        })
    finally:
        db.close()


@router.get("/research")
async def research(request: Request, user: str = Depends(check_auth)):
    """Is this strategy any good, and how would we know?

    Deliberately a separate page from /performance. That one shows what the
    portfolio DID; this one shows how much of it can be believed. Putting
    the two on one screen makes the return the headline and the sample size
    a footnote, which is the reading order that gets people hurt.
    """
    db = SessionLocal()
    try:
        report = build_research_report(db)
        distributions = [
            build_score_distribution(db, stage=stage)
            for stage in (TECHNICAL_SCORE, MARKET_QUALITY, SECURITY)
        ]
        calibrations = [build_calibration(db, horizon_minutes=h) for h in HORIZONS_MINUTES]
        return templates.TemplateResponse(
            request,
            "research.html",
            {
                "report": report,
                "distributions": distributions,
                "calibrations": calibrations,
                "coverage": forward_coverage(db),
            },
        )
    finally:
        db.close()


@router.get("/api/research")
async def api_research(user: str = Depends(check_auth)):
    db = SessionLocal()
    try:
        return JSONResponse({
            "report": build_research_report(db).as_dict(),
            "distribution": build_score_distribution(db).as_dict(),
            "calibration": [
                build_calibration(db, horizon_minutes=h).as_dict() for h in HORIZONS_MINUTES
            ],
            "forward_return_coverage": forward_coverage(db),
        })
    finally:
        db.close()


@router.get("/early")
async def early_signals(request: Request, user: str = Depends(check_auth)):
    """The Early Signals watchlist and what it has produced so far.

    Separate from /research because they answer different questions: this
    is "what is the engine looking at right now", that is "is the engine
    any good". Mixing them would put a live, tempting list of high-scoring
    tokens next to the numbers that say the score is unvalidated, and the
    list would win.
    """
    db = SessionLocal()
    try:
        watching = wl.active(db)
        resolved = (
            db.query(models.WatchlistEntry)
            .filter(models.WatchlistEntry.state.in_(list(wl.TERMINAL_STATES)))
            .order_by(models.WatchlistEntry.last_evaluated_at.desc())
            .limit(40)
            .all()
        )
        return templates.TemplateResponse(
            request,
            "early.html",
            {
                "watching": watching,
                "resolved": resolved,
                "false_positives": build_false_positives(db),
                "lead_time": build_lead_time(db),
                "calibration": [
                    build_early_calibration(db, horizon_minutes=h) for h in (15, 60, 240)
                ],
                "may_trade": settings.EARLY_SIGNAL_MAY_TRADE,
                "enabled": settings.EARLY_SIGNAL_ENABLED,
                "watch_threshold": settings.EARLY_SIGNAL_WATCH_THRESHOLD,
                "confirm_threshold": settings.EARLY_SIGNAL_CONFIRM_THRESHOLD,
                "alerts": _early_alerts(watching),
            },
        )
    finally:
        db.close()


def _early_alerts(watching: list) -> list[dict]:
    """Dashboard-only alerts. Research signals, never execution triggers.

    Each one names a state change worth a human glance: a score crossing
    the confirm threshold, a score improving fast, or a watched token
    deteriorating. Nothing here places or cancels anything.
    """
    alerts: list[dict] = []
    for entry in watching:
        score = entry.early_score or 0.0
        history = entry.score_history or []

        if entry.state == wl.CONFIRMED:
            alerts.append({
                "level": "good", "symbol": entry.symbol, "token": entry.token_address,
                "message": f"CONFIRMED at early score {score:.0f}",
            })
        elif score >= settings.EARLY_SIGNAL_CONFIRM_THRESHOLD:
            alerts.append({
                "level": "good", "symbol": entry.symbol, "token": entry.token_address,
                "message": f"early score {score:.0f} crossed the confirm threshold",
            })

        if len(history) >= 3:
            earlier = history[-3].get("early")
            if earlier is not None and score - earlier >= 10:
                alerts.append({
                    "level": "good", "symbol": entry.symbol, "token": entry.token_address,
                    "message": f"early score improving fast: {earlier:.0f} -> {score:.0f}",
                })

        if (entry.late_entry_risk or 0) >= 45:
            alerts.append({
                "level": "warn", "symbol": entry.symbol, "token": entry.token_address,
                "message": f"late-entry risk {entry.late_entry_risk:.0f} - the window is closing",
            })

        if wl.deteriorating(entry):
            alerts.append({
                "level": "warn", "symbol": entry.symbol, "token": entry.token_address,
                "message": (
                    f"score fading: {entry.best_early_score:.0f} -> {score:.0f}"
                ),
            })
    return alerts[:25]


@router.get("/api/early")
async def api_early(user: str = Depends(check_auth)):
    db = SessionLocal()
    try:
        return JSONResponse({
            "may_trade": settings.EARLY_SIGNAL_MAY_TRADE,
            "watching": [
                {
                    "symbol": e.symbol, "token_address": e.token_address, "state": e.state,
                    "stage": e.stage, "early_score": e.early_score,
                    "technical_score": e.technical_score,
                    "security_score": e.security_score,
                    "late_entry_risk": e.late_entry_risk,
                    "momentum_class": e.momentum_class, "reason": e.reason,
                    "evaluations": e.evaluations,
                }
                for e in wl.active(db)
            ],
            "false_positives": build_false_positives(db).as_dict(),
            "lead_time": build_lead_time(db).as_dict(),
            "calibration": [
                build_early_calibration(db, horizon_minutes=h).as_dict() for h in (15, 60, 240)
            ],
        })
    finally:
        db.close()


@router.get("/api/stats")
async def api_stats(user: str = Depends(check_auth)):
    db = SessionLocal()
    try:
        value = await portfolio.get_portfolio_value_usd(db)
        return JSONResponse(
            {
                "portfolio_value_usd": value,
                "mode": "LIVE" if settings.LIVE_TRADING else "PAPER",
                "halted": is_trading_halted(db),
            }
        )
    finally:
        db.close()


@router.post("/api/halt")
async def api_halt(user: str = Depends(check_auth), _csrf: None = Depends(check_csrf)):
    db = SessionLocal()
    try:
        halt_trading(db, "manually halted via dashboard")
        db.commit()
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    finally:
        db.close()


@router.post("/api/resume")
async def api_resume(user: str = Depends(check_auth), _csrf: None = Depends(check_csrf)):
    db = SessionLocal()
    try:
        resume_trading(db)
        db.commit()
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# database snapshots
# ---------------------------------------------------------------------------

@router.get("/backup")
async def backup_status(user: str = Depends(check_auth)):
    """What snapshots exist, and whether they will survive a reset."""
    snapshots = backup.list_snapshots()
    return JSONResponse({
        "enabled": settings.BACKUP_ENABLED,
        "directory": str(backup.backup_dir()),
        "interval_minutes": settings.BACKUP_INTERVAL_MINUTES,
        "keep": settings.BACKUP_KEEP,
        "restore_on_empty": settings.BACKUP_RESTORE_ON_EMPTY,
        "warning": backup.warn_if_backups_are_pointless(),
        "snapshots": [s.as_dict() for s in snapshots],
        "download": "/backup/download",
    })


@router.post("/backup/now")
async def backup_now(user: str = Depends(check_auth)):
    """Take a snapshot immediately.

    POST rather than GET: it writes a file and rotates old ones, so it must
    not be something a browser prefetch or a link crawler can trigger.
    """
    snapshot = backup.take_snapshot(reason="manual")
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="snapshot failed - see the logs. Nothing was rotated.",
        )
    return JSONResponse(snapshot.as_dict())


@router.get("/backup/download")
async def backup_download(user: str = Depends(check_auth)):
    """Download the newest snapshot.

    The escape hatch for a host with no persistent disk: when BACKUP_DIR is
    wiped on every deploy, pulling the file through the browser is the only
    way the dataset leaves the box. Takes a fresh snapshot first so the
    download is current rather than up to an hour stale.
    """
    snapshot = backup.snapshot_for_download()
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no snapshot available - backups may be disabled, or the database "
                   "may not be SQLite (Postgres has its own backup tooling).",
        )
    return FileResponse(
        snapshot.path,
        media_type="application/vnd.sqlite3",
        filename=snapshot.path.name,
    )
