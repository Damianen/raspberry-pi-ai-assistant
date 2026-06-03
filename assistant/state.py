"""Shared application state.

THE CONTRACT (read this before touching anything):
- `AppState` is the single source of truth for what the assistant is doing.
- The eyes/UI ONLY READ state. They contain no logic and never mutate it.
- The pipeline/scheduler ONLY WRITE state. They never draw.
- All access goes through `SharedState`, which is thread-safe.

If you find yourself wanting the eyes to "decide" something, or the pipeline
to "draw" something, stop — you're about to break the architecture.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class AppState(Enum):
    IDLE = auto()       # calm, blinking, glancing around
    LISTENING = auto()  # mic hot, wide bright eyes
    THINKING = auto()   # processing (STT / intent / LLM)
    SPEAKING = auto()   # TTS playing
    CONFIRM = auto()    # command succeeded (happy eyes) — transient
    ERROR = auto()      # didn't catch / failure (red shake) — transient


@dataclass
class StateSnapshot:
    state: AppState
    # free-form metadata the UI may read (e.g. live mic level 0..1 for a
    # reactive pulse, or a confirm message). Eyes may read, never require.
    meta: dict[str, Any] = field(default_factory=dict)


class SharedState:
    """Thread-safe holder. One instance is shared across UI + worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = AppState.IDLE
        self._meta: dict[str, Any] = {}

    def set(self, state: AppState, **meta: Any) -> None:
        with self._lock:
            self._state = state
            self._meta = dict(meta)

    def update_meta(self, **meta: Any) -> None:
        with self._lock:
            self._meta.update(meta)

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot(self._state, dict(self._meta))

    @property
    def state(self) -> AppState:
        with self._lock:
            return self._state
