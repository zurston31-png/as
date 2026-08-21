"""Telegram + Discord notifications. Both are optional and independent —
configure either, both, or neither via .env. Every send is best-effort:
a notification failure never blocks or fails a trade.
"""
import logging
import time

from app.config import settings
from app.services import http

logger = logging.getLogger(__name__)

# How long a background worker stays quiet about a failure it already
# reported. Every worker loop swallows its own exceptions and keeps running
# - correct, since one bad tick must not take the whole process down - but
# swallowed silently means the only record was a server log nobody not
# already SSH'd in is reading. The operator finds out a cycle is failing
# minutes to hours after it started, from a dashboard number that stopped
# moving, or not at all.
#
# The failure mode this throttle exists for is the opposite one: a loop
# that fails EVERY tick would otherwise send a message every
# SCANNER_INTERVAL_SECONDS (as low as 30-60s) forever, which trains the
# operator to ignore the channel that exists specifically to be trusted.
# One notification on the way in, silence while it stays broken, then a
# reminder if it is still broken after the window - never a flood.
WORKER_FAILURE_THROTTLE_SECONDS = 900.0


class Notifier:
    """Both sends go through app/services/http.py for RATE LIMITING, not
    for reliability.

    Telegram and Discord both rate-limit aggressively, and a burst of
    rejections during a busy scan is exactly the situation the shared
    backoff exists for. But posting a message is NOT idempotent: a send
    that times out may already have been delivered, and retrying it would
    double-post. So `idempotent=False` - only a 429 retries, because that
    is the one response that says the message was definitely not sent.

    A duplicate alert is only annoying, not dangerous. It is still not
    worth having, and getting it wrong here would teach the same reflex in
    a place where it does matter.
    """

    def __init__(self) -> None:
        # worker name -> monotonic time of the last failure notification
        # sent for it. Process-local and reset on restart, which is fine:
        # a restart is itself worth learning about again if the worker is
        # still broken afterward.
        self._last_worker_failure: dict[str, float] = {}

    async def _send_telegram(self, text: str) -> None:
        if not (settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID):
            return
        url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
        try:
            await http.post_json(
                url,
                json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": text},
                timeout=10, label="telegram sendMessage", service="telegram",
            )
        except Exception:
            logger.exception("failed to send Telegram notification")

    async def _send_discord(self, text: str) -> None:
        if not settings.DISCORD_WEBHOOK_URL:
            return
        try:
            await http.post_json(
                settings.DISCORD_WEBHOOK_URL,
                json={"content": text[:2000]},
                timeout=10, label="discord webhook", service="discord",
            )
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

    async def notify_worker_failure(self, worker: str, error: BaseException) -> None:
        """A whole pass of a background loop failed - not one bad
        candidate inside it (those already notify individually via
        notify_rejection/notify_error), but the pass itself: discovery
        raised, a query failed, the loop's own bookkeeping broke.

        Throttled per worker name at WORKER_FAILURE_THROTTLE_SECONDS so a
        loop failing every tick sends one message, not one per tick
        forever. The exception's type and message are included; the
        traceback is not - that already went to the server log via
        logger.exception right before this is called, and a Telegram
        message is not where a stack trace belongs.
        """
        now = time.monotonic()
        last = self._last_worker_failure.get(worker)
        if last is not None and (now - last) < WORKER_FAILURE_THROTTLE_SECONDS:
            return
        self._last_worker_failure[worker] = now
        text = (
            f"🐛 {worker} failed: {type(error).__name__}: {error}\n"
            "Check `docker compose logs bot` for the traceback."
        )
        await self._broadcast(text)

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
