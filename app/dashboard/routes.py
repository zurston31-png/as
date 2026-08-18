"""Simple server-rendered dashboard: open positions, trade history,
performance stats, and manual halt/resume controls. Protected by HTTP
Basic Auth (DASHBOARD_USERNAME/DASHBOARD_PASSWORD).
"""
import secrets

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from app import models
from app.config import settings
from app.database import SessionLocal
from app.risk.manager import halt_trading, is_trading_halted, resume_trading
from app.services import portfolio

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
        closed_trades = db.query(models.Trade).filter(models.Trade.pnl_usd.isnot(None)).all()
        wins = [t for t in closed_trades if (t.pnl_usd or 0) > 0]
        total_pnl = sum(t.pnl_usd or 0 for t in closed_trades)
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0
        portfolio_value = await portfolio.get_portfolio_value_usd(db)

        stats = {
            "portfolio_value": portfolio_value,
            "starting_balance": settings.PORTFOLIO_STARTING_BALANCE_USD,
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "open_positions": len(open_positions),
            "mode": "LIVE" if settings.LIVE_TRADING else "PAPER",
            "chain": settings.CHAIN,
            "execution_backend": settings.EXECUTION_BACKEND,
            "halted": is_trading_halted(db),
            "watchlist": settings.SYMBOLS_WATCHLIST,
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
                "positions": open_positions,
                "trades": recent_trades,
                "events": recent_rejections,
                "signals": recent_signals,
                "checks": checks_by_signal,
            },
        )
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
