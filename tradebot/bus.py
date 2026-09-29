"""A tiny async fan-out bus.

The orchestrator publishes; websocket clients and anything else subscribe. Slow
subscribers get dropped messages rather than being allowed to stall the trading
loop - a lagging browser tab must never delay a signal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, queue_size: int = 256) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self.queue_size = queue_size
        self.dropped = 0

    def publish(self, event_type: str, payload: Any) -> None:
        message = {"type": event_type, "data": payload}
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self.dropped += 1
                try:  # drop the oldest so a stalled client still gets fresh data
                    queue.get_nowait()
                    queue.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        queue = self.subscribe()
        try:
            while True:
                yield await queue.get()
        finally:
            self.unsubscribe(queue)
