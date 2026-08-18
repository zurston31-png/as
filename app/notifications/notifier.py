"""Telegram + Discord notifications. Both are optional and independent —
configure either, both, or neither via .env. Every send is best-effort:
a notification failure never blocks or fails a trade.
"""
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class Notifier:
    async def _send_telegram(self, text: str) -> None:
        if not (settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID):
            return
        url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": text})
        except Exception:
            logger.exception("failed to send Telegram notification")

    async def _send_discord(self, text: str) -> None:
        if not settings.DISCORD_WEBHOOK_URL:
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(settings.DISCORD_WEBHOOK_URL, json={"content": text[:2000]})
        except Exception:
            logger.exception("failed to send Discord notification")

    async def _broadcast(self, text: str) -> None:
        await self._send_telegram(text)
        await self._send_discord(text)

    async def notify_trade_executed(self, trade: "models.Trade", extra: str = "") -> None:  # noqa: F821
        mode = "\U0001f9ea PAPER" if trade.mode == "paper" else "\U0001f534 LIVE"
        text = f"{mode} {trade.side.upper()} {trade.symbol}\nqty: {trade.qty}\n{extra}".strip()
        await self._broadcast(text)

    async def notify_rejection(self, signal: "models.Signal", reason: str) -> None:  # noqa: F821
        text = f"\U0001f6ab Trade rejected: {signal.symbol}\nreason: {reason}"
        await self._broadcast(text)

    async def notify_risk_halt(self, reason: str) -> None:
        text = f"\U0001f6d1 TRADING HALTED\nreason: {reason}\nResume manually via the dashboard when ready."
        await self._broadcast(text)

    async def notify_error(self, text: str) -> None:
        await self._broadcast(f"⚠️ {text}")

    async def notify_daily_summary(self, summary: dict) -> None:
        mode = "LIVE" if settings.LIVE_TRADING else "PAPER"
        text = (
            "\U0001f4ca Daily summary\n"
            f"mode: {mode}\n"
            f"trades closed today: {summary['trades_count']}\n"
            f"realized P&L: ${summary['realized_pnl_usd']:,.2f}\n"
            f"portfolio value: ${summary['portfolio_value_usd']:,.2f}"
        )
        await self._broadcast(text)


notifier = Notifier()
