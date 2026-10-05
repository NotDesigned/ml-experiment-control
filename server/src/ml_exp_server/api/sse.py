"""Bounded, best-effort index invalidations; HTTP snapshots are authoritative."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from typing import AsyncIterator
from sse_starlette import EventSourceResponse


class IndexEventResponse(EventSourceResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Disconnect can cancel a send while the generator is suspended
            # at yield, rather than in queue.get(). Close both cases promptly.
            await self.body_iterator.aclose()


class EventBroker:
    def __init__(self, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: deque[dict | None] = deque()
        self._handoff = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish_threadsafe(self, payload: dict) -> None:
        """Bound both the thread handoff and each subscriber's backlog."""
        with self._handoff:
            if self._loop is None or self._loop.is_closed():
                return
            if not self._pending:
                self._loop.call_soon_threadsafe(self._drain)
            if len(self._pending) >= self._capacity:
                self._pending.clear()
                self._pending.append(None)
            # Once overflowed, one resync indication covers later changes.
            if not self._pending or self._pending[0] is not None:
                self._pending.append(payload)

    def _drain(self) -> None:
        with self._handoff:
            pending = tuple(self._pending)
            self._pending.clear()
        for payload in pending:
            self._publish(payload)

    def _publish(self, payload: dict | None) -> None:
        for queue in tuple(self._subscribers):
            if payload is None or queue.full():
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)
                self._subscribers.discard(queue)
            else:
                queue.put_nowait(payload)

    async def stream(self) -> AsyncIterator[dict]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._capacity)
        self._subscribers.add(queue)
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    yield {"data": json.dumps({"type": "resync_required"})}
                    return
                yield {"data": json.dumps(payload)}
        finally:
            self._subscribers.discard(queue)
