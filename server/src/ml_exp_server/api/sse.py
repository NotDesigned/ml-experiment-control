"""In-process pub/sub bridging index updates to SSE clients."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish_threadsafe(self, payload: dict) -> None:
        """Publish from any thread (indexer runs in a worker thread)."""
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._publish, payload)

    def _publish(self, payload: dict) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(payload)

    async def stream(self) -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            self._subscribers.discard(queue)
