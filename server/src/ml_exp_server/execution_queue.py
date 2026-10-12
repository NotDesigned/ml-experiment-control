"""Bounded background admission, reserved before durable execution claims."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
import threading


class QueueUnavailable(RuntimeError):
    """No work was admitted; the caller may retry without reconciling effects."""

    def __init__(self, lane: str):
        self.lane = lane
        super().__init__(f"{lane} execution capacity is unavailable")


_admission: ContextVar[Reservation | None] = ContextVar("execution_admission", default=None)


class Reservation:
    def __init__(self, owner: BoundedExecutor):
        self.owner = owner
        self.active = True
        self._context_token = None
        self._transferred = False

    def __enter__(self):
        self._context_token = _admission.set(self)
        return self

    def __exit__(self, *_exc):
        _admission.reset(self._context_token)
        if not self._transferred:
            self.release()

    def release(self):
        with self.owner._condition:
            if self.active:
                self.active = False
                self.owner._reserved -= 1
                self.owner._in_use -= 1
                self.owner._condition.notify_all()
                self.owner._close_if_ready()

    def submit(self, call, *args):
        with self.owner._condition:
            if not self.active:
                raise RuntimeError("execution admission was already consumed")
            # shutdown defers closing the underlying executor until every
            # reservation has either submitted or released. A durable claim
            # therefore never loses its admitted slot to concurrent shutdown.
            future = self.owner._executor.submit(call, *args)
            self.active = False
            self.owner._reserved -= 1
            future.add_done_callback(self.owner._completed)
            self.owner._condition.notify_all()
            self.owner._close_if_ready()
            return future


class BoundedExecutor:
    """Capacity counts running, queued, and reserved work together."""

    def __init__(self, lane: str, *, workers: int, capacity: int):
        if workers < 1 or capacity < workers:
            raise ValueError("execution capacity must be at least its positive worker count")
        self.lane = lane
        self.capacity = capacity
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=lane + "-executor")
        self._condition = threading.Condition(threading.RLock())
        self._in_use = 0
        self._reserved = 0
        self._closing = False
        self._closed = False
        self._cancel_futures = False
        self._handoffs = set()

    @property
    def busy(self):
        with self._condition:
            return bool(self._in_use)

    def reserve(self):
        with self._condition:
            if self._closing or self._in_use >= self.capacity:
                raise QueueUnavailable(self.lane)
            self._in_use += 1
            self._reserved += 1
            return Reservation(self)

    def submit(self, call, *args):
        admitted = _admission.get()
        if admitted is not None and admitted.owner is self and admitted.active:
            return admitted.submit(call, *args)
        with self.reserve() as reservation:
            return reservation.submit(call, *args)

    def _completed(self, _future: Future):
        with self._condition:
            self._in_use -= 1
            self._condition.notify_all()

    async def drain_handoffs(self):
        # Only durable begin + enqueue runs here, never the provider wait.
        with self._condition:
            tasks = tuple(self._handoffs)
        if tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)

    def _close_if_ready(self):
        if self._closing and not self._reserved and not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=self._cancel_futures)

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._condition:
            self._closing = True
            self._cancel_futures = cancel_futures
            self._close_if_ready()
            if wait:
                while self._reserved:
                    self._condition.wait()
        if wait:
            self._executor.shutdown(wait=True, cancel_futures=cancel_futures)


async def admitted_handoff(call):
    """Keep an admitted durable claim and its enqueue alive after disconnect."""
    admitted = _admission.get()
    if admitted is None:
        return await call()
    # The middleware owns the context token; this task now owns its capacity.
    # In particular, middleware cancellation must not release the slot while
    # run_in_threadpool is still committing a durable execution claim.
    admitted._transferred = True

    async def finish():
        try:
            return await call()
        finally:
            admitted.release()

    task = asyncio.create_task(finish())
    with admitted.owner._condition:
        admitted.owner._handoffs.add(task)

    def completed(item):
        with admitted.owner._condition:
            admitted.owner._handoffs.discard(item)
        # Retrieve exceptions even when the HTTP waiter has disconnected.
        if not item.cancelled():
            item.exception()

    task.add_done_callback(completed)
    return await asyncio.shield(task)


def job_lane(call):
    owner = getattr(call, "__self__", None)
    return {
        "ContainerExecutionService": "build",
        "DataDeliveryService": "data",
        "ResultCollectionService": "collection",
    }.get(type(owner).__name__, "preparation")
