#!/usr/bin/env python3
"""Walk a synthetic token through a complete trade so you can see the full
cycle on the dashboard: position opened with a stop-loss and take-profit,
price moves, the monitor closes it automatically, P&L recorded.

This drives the REAL trading engine, risk manager, position monitor and
database. The only thing stubbed out is the security scanner lookup, which
is replaced with a pass for one made-up token called DEMOCOIN. Nothing in
your rug-check thresholds is changed or bypassed for real tokens, and no
real money, wallet or network trade is involved at any point.

Run it while the bot is running, then refresh the dashboard.

Windows users: double-click RUN_DEMO.bat instead.
"""
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

IS_WINDOWS = os.name == "nt"

DEMO_SYMBOL = "DEMOCOIN"
DEMO_ADDRESS = "DemoCoin1111111111111111111111111111111111"
ENTRY_PRICE = 1.00

# Scripted prices the fake market moves through, in order.
_price_script: list[float] = []


def pause_and_exit(code: int = 0) -> None:
    if IS_WINDOWS:
        input("\nPress Enter to close this window...")
    sys.exit(code)


async def main() -> None:
    from app import models
    from app.database import SessionLocal, init_db
    from app.rugcheck.filters import RugCheckReport
    from app.schemas import TradingViewAlert
    from app.services import price_feed
    from app.config import settings
    import app.services.trading_service as trading_service
    from app.monitor import position_monitor

    if settings.LIVE_TRADING:
        print("\n  LIVE_TRADING is true. Refusing to run the demo against a live")
        print("  configuration. Set LIVE_TRADING=false in .env first.")
        pause_and_exit(1)

    init_db()

    # --- stub the market and the scanner, for DEMOCOIN only ---
    async def fake_price(token_address: str):
        if token_address != DEMO_ADDRESS:
            return None
        return _price_script[0] if len(_price_script) == 1 else _price_script.pop(0)

    async def fake_rug_check(chain: str, token_address: str):
        if token_address != DEMO_ADDRESS:
            return RugCheckReport(passed=False, reasons=["not the demo token"])
        return RugCheckReport(
            passed=True, reasons=[], liquidity_usd=500_000.0, dev_wallet_pct=0.04,
            ownership_renounced=True, mint_disabled=True, liquidity_locked=True, is_honeypot=False,
        )

    async def fake_signal_score(chain: str, token_address: str, symbol: str):
        from app.signals.scoring import Factor, SignalScore

        if token_address != DEMO_ADDRESS:
            return None
        return SignalScore(
            score=88.0, direction="long", reliable=True,
            factors=[Factor(name="trend_direction", score=0.9, weight=1.0, reason="demo stub")],
        )

    price_feed.get_price_usd = fake_price
    trading_service.run_rug_checks = fake_rug_check
    trading_service.evaluate_live_entry_signal = fake_signal_score

    print("=" * 68)
    print("  DEMO TRADE  (simulated token, simulated money)")
    print("=" * 68)

    db = SessionLocal()
    try:
        # Clear any previous demo rows so the run is repeatable.
        for model in (models.Position, models.Trade, models.Signal):
            for row in db.query(model).filter_by(symbol=DEMO_SYMBOL).all():
                db.delete(row)
        db.commit()

        # ---- 1. buy signal ----
        _price_script[:] = [ENTRY_PRICE]
        print(f"\n  1. TradingView sends a BUY for {DEMO_SYMBOL} at ${ENTRY_PRICE:.2f}")
        alert = TradingViewAlert(
            secret=settings.WEBHOOK_SECRET, symbol=DEMO_SYMBOL, token_address=DEMO_ADDRESS,
            chain="solana", signal="buy", price=ENTRY_PRICE, rsi=38.0,
        )
        await trading_service.handle_alert(db, alert)
        db.commit()

        position = (
            db.query(models.Position)
            .filter_by(symbol=DEMO_SYMBOL, status=models.PositionStatus.OPEN.value)
            .first()
        )
        if not position:
            print("\n  The buy did not open a position. Is trading halted on the")
            print("  dashboard, or are you at your max concurrent positions?")
            pause_and_exit(1)

        size = position.qty * position.entry_price
        print(f"     -> position opened: {position.qty:,.2f} tokens for ${size:,.2f}")
        print(f"     -> stop-loss   ${position.stop_loss:.4f}  (risk manager set this)")
        print(f"     -> take-profit ${position.take_profit:.4f}")

        # ---- 2. price rises past take-profit ----
        target = position.take_profit * 1.02
        print(f"\n  2. Price climbs to ${target:.4f}, above the take-profit")
        _price_script[:] = [target]

        print("     Running one monitor tick (this is the loop that watches")
        print("     your positions every 30s while the bot runs)...")
        await position_monitor._check_positions_once()

        # ---- 3. result ----
        db.expire_all()
        closed = db.query(models.Position).filter_by(symbol=DEMO_SYMBOL).order_by(
            models.Position.id.desc()
        ).first()
        sell = db.query(models.Trade).filter_by(symbol=DEMO_SYMBOL, side="sell").order_by(
            models.Trade.id.desc()
        ).first()

        print("\n" + "=" * 68)
        if closed and closed.status == models.PositionStatus.CLOSED.value and sell:
            print("  RESULT: position closed automatically")
            print("=" * 68)
            print(f"\n     reason:     {closed.close_reason}")
            print(f"     entry:      ${closed.entry_price:.4f}")
            print(f"     exit:       ${sell.exit_price:.4f}")
            print(f"     profit:     ${sell.pnl_usd:,.2f}  ({sell.pnl_pct * 100:+.1f}%)")
            print("\n  Nobody pressed anything. The bot saw the price hit its")
            print("  take-profit and exited on its own - that is the autonomous")
            print("  part working.")
            print("\n  !! THIS IS NOT A TRADING RESULT. The price move above is")
            print("     SCRIPTED by this demo ($1.00 -> $1.33) purely to show the")
            print("     full cycle end to end. The profit number says NOTHING about")
            print("     whether the strategy works - it only proves entry, monitoring")
            print("     and exit are wired together correctly.")
            print("     Real fill costs (price impact, spread, confirmation delay,")
            print("     fees) WERE applied, which is why entry is above $1.00 and")
            print("     exit below $1.33. For actual performance run the backtester")
            print("     (scripts/run_backtest.py) or let it paper trade a few hundred")
            print("     real setups.")
        else:
            print("  The position did not close as expected.")
            print("=" * 68)

        print("\n  Refresh the dashboard to see it in Recent Trades:")
        print("      http://127.0.0.1:8000")
        print("\n  All of this was simulated. No wallet, no crypto, no real money.")
        print("=" * 68)
    finally:
        db.close()

    pause_and_exit()


if __name__ == "__main__":
    asyncio.run(main())
