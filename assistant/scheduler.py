"""Background scheduler. Polls the Store for due events and fires them.

Restart-safe because it reads the DB every tick rather than holding events in
memory. Fires via the `on_fire(Event)` callback you pass in — typically: set
AppState.SPEAKING and announce the alarm via TTS.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from .store import Event, Store


class Scheduler:
    def __init__(self, store: Store, on_fire: Callable[[Event], None],
                 poll_seconds: float = 1.0) -> None:
        self._store = store
        self._on_fire = on_fire
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            for ev in self._store.due():
                try:
                    self._on_fire(ev)
                finally:
                    # mark fired even if announce failed, so it can't loop forever
                    self._store.mark_fired(ev.id)
            self._stop.wait(self._poll)
