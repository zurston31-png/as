"""Simple server-rendered dashboard: open positions, trade history,
performance stats, and manual halt/resume controls. Protected by HTTP
Basic Auth (DASHBOARD_USERNAME/DASHBOARD_PASSWORD).
"""
import datetime as dt
import secrets

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from app import models
from app.config import settings
from app.dashboard.analytics import compute_equity_curve, compute_portfolio_stats
from app.dashboard.charts import equity_curve_svg
from app.database import SessionLocal
from app.risk.manager import halt_trading, is_trading_halted, resume_trading
from app.scanner.loop import scanner_blocked_reason
from app.services import portfolio, price_feed
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
