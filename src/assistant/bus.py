"""Async pub/sub event bus — the only channel modules use to talk to each other.

The asyncio side subscribes with `subscribe()` and consumes events as an async
iterator. Sync threads (the pygame render thread) publish with `publish()` from
any thread and drain command events through an `Inbox` without touching the loop.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Subscription:
    """Async-iterator view of matching bus events; consume on the bus loop."""

    def __init__(self, bus: EventBus, types: frozenset[str]) -> None:
        self._bus = bus
        self.types = types
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    def _deliver(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def get(self) -> Event:
        return await self._queue.get()

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> Event:
        return await self._queue.get()

    def close(self) -> None:
        self._bus._drop(self)


class Inbox:
    """Thread-safe mailbox so a sync thread can drain bus events without the loop."""

    def __init__(self, bus: EventBus, types: frozenset[str]) -> None:
        self._bus = bus
        self.types = types
        self._queue: queue.SimpleQueue[Event] = queue.SimpleQueue()

    def _deliver(self, event: Event) -> None:
        self._queue.put(event)

    def drain(self) -> list[Event]:
        events: list[Event] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                return events

    def get(self, timeout: float | None = None) -> Event | None:
        """Blocking get for worker threads; None when no event arrives in time."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._bus._drop(self)


class EventBus:
    """Routes published events to async subscriptions and thread-safe inboxes.

    Delivery always happens on the attached asyncio loop; `publish()` from any
    other thread hops onto it via `call_soon_threadsafe`, so subscribers see
    events in publish order and `asyncio.Queue` is only touched loop-side.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._sinks: list[Subscription | Inbox] = []

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, *types: str) -> Subscription:
        """Subscribe to the given event types; no arguments means all events."""
        sub = Subscription(self, frozenset(types))
        with self._lock:
            self._sinks.append(sub)
        return sub

    def open_inbox(self, *types: str) -> Inbox:
        """Like subscribe(), but for sync consumers; no arguments means all events."""
        inbox = Inbox(self, frozenset(types))
        with self._lock:
            self._sinks.append(inbox)
        return inbox

    def publish(self, event: Event) -> None:
        loop = self._loop
        if loop is None:
            raise RuntimeError("EventBus.publish() called before attach_loop()")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._dispatch(event)
        else:
            loop.call_soon_threadsafe(self._dispatch, event)

    def emit(self, type: str, payload: dict[str, Any] | None = None) -> None:
        self.publish(Event(type, payload or {}))

    def _dispatch(self, event: Event) -> None:
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            if not sink.types or event.type in sink.types:
                sink._deliver(event)

    def _drop(self, sink: Subscription | Inbox) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)
