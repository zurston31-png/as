"""The desktop-facing app: dashboard, live signal push, chat, webhook receiver.

It's a local web app rather than a native window on purpose - it runs the same
on macOS, Windows and Linux, you can put it on a second monitor next to
TradingView, and the same process can receive TradingView's webhooks. Bind it
to 127.0.0.1 (the default) and it never leaves your machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..orchestrator import Orchestrator

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(config: Config, orchestrator: Optional[Orchestrator] = None) -> FastAPI:
    bot = orchestrator or Orchestrator(config)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        bot.start()
        try:
            yield
        finally:
            await bot.stop()

    app = FastAPI(title="Trading Bot", version="0.1.0", docs_url=None,
                  redoc_url=None, lifespan=lifespan)
    app.state.bot = bot
    app.state.config = config

    # ------------------------------------------------------------- pages

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # --------------------------------------------------------------- api

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return bot.state()

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return bot.status()

    @app.get("/api/candles")
    async def candles(limit: int = 200) -> dict[str, Any]:
        rows = list(bot.series.candles)[-max(1, min(limit, 2000)):]
        return {"symbol": bot.series.symbol, "timeframe": bot.series.timeframe,
                "candles": [c.to_dict() for c in rows]}

    @app.post("/api/kill")
    async def kill(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        reason = body.get("reason") or "manual: stopped from the dashboard"
        return {"ok": True, "risk": bot.kill(str(reason)[:200])}

    @app.post("/api/resume")
    async def resume() -> dict[str, Any]:
        return {"ok": True, "risk": bot.resume()}

    @app.post("/api/flatten")
    async def flatten() -> dict[str, Any]:
        return {"ok": True, "closed": bot.flatten()}

    @app.post("/api/evaluate")
    async def evaluate() -> dict[str, Any]:
        """Force a re-evaluation of the current bar - useful while tuning rules."""
        signal = await bot.evaluate()
        return {"signal": signal.to_dict() if signal else None}

    # ----------------------------------------------------------- webhook

    @app.post("/webhook/tradingview")
    async def tradingview(request: Request) -> JSONResponse:
        payload = await _json_body(request)
        if not payload:
            raise HTTPException(status_code=400, detail="expected a JSON body")

        secret = config.feed.webhook_secret
        if secret:
            supplied = str(payload.get("secret") or request.headers.get("x-webhook-secret") or "")
            if not hmac.compare_digest(supplied, secret):
                log.warning("rejected webhook with a bad secret")
                raise HTTPException(status_code=401, detail="bad secret")
        payload.pop("secret", None)

        result = bot.submit_webhook(payload)
        return JSONResponse(result, status_code=200 if result.get("accepted") else 422)

    # ---------------------------------------------------------- websocket

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)

        async def writer() -> None:
            while True:
                message = await outbox.get()
                await socket.send_text(json.dumps(message, default=str))

        async def forwarder() -> None:
            async for event in bot.bus.listen():
                _offer(outbox, event)

        writer_task = asyncio.create_task(writer())
        forward_task = asyncio.create_task(forwarder())
        _offer(outbox, {"type": "state", "data": bot.state()})

        try:
            while True:
                raw = await socket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    _offer(outbox, {"type": "error", "data": {"message": "invalid JSON"}})
                    continue
                await _handle_client_message(bot, message, outbox)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("websocket error")
        finally:
            for task in (writer_task, forward_task):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return app


def _offer(queue: asyncio.Queue, message: dict[str, Any]) -> None:
    """Never block the producer on a slow socket."""
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        with contextlib.suppress(asyncio.QueueEmpty, asyncio.QueueFull):
            queue.get_nowait()
            queue.put_nowait(message)


async def _handle_client_message(
    bot: Orchestrator, message: dict[str, Any], outbox: asyncio.Queue
) -> None:
    action = str(message.get("action") or "").lower()

    if action == "chat":
        text = str(message.get("message") or "").strip()
        if not text:
            return
        turn_id = uuid.uuid4().hex[:8]
        _offer(outbox, {"type": "chat_start", "data": {"id": turn_id}})
        try:
            async for chunk in bot.chat.ask(text[:8000]):
                _offer(outbox, {"type": "chat_delta", "data": {"id": turn_id, "text": chunk}})
        except Exception as exc:  # noqa: BLE001 - a chat failure stays in the chat panel
            log.warning("chat turn failed: %s", exc)
            _offer(outbox, {"type": "chat_delta",
                            "data": {"id": turn_id, "text": f"\n[chat error: {type(exc).__name__}]"}})
        _offer(outbox, {"type": "chat_end", "data": {"id": turn_id}})

    elif action == "state":
        _offer(outbox, {"type": "state", "data": bot.state()})

    elif action == "kill":
        reason = str(message.get("reason") or "manual: stopped from the dashboard")[:200]
        _offer(outbox, {"type": "state", "data": {**bot.state(), "risk": bot.kill(reason)}})

    elif action == "resume":
        bot.resume()
        _offer(outbox, {"type": "state", "data": bot.state()})

    elif action == "flatten":
        closed = bot.flatten()
        _offer(outbox, {"type": "flattened", "data": {"closed": closed}})
        _offer(outbox, {"type": "state", "data": bot.state()})

    elif action == "reset_chat":
        bot.chat.reset()
        _offer(outbox, {"type": "chat_reset", "data": {}})

    elif action == "ping":
        _offer(outbox, {"type": "pong", "data": {}})


async def _json_body(request: Request) -> dict[str, Any]:
    """Accept JSON, and also the bare text TradingView sends when you forget."""
    try:
        body = await request.body()
    except Exception:  # noqa: BLE001
        return {}
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"text": body.decode("utf-8", "replace")[:2000]}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
