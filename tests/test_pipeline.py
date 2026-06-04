"""Tests for the pipeline's tap-to-interrupt wiring (slice 6).

No audio here — these pin the *decision* on_tap makes, which is the tricky
concurrency bit: a tap interrupts only while SPEAKING, and never while a command
is being captured (LISTENING) or processed (THINKING), so a half-spoken command
can't be dropped by a stray tap. The chunked playback that actually reads the
event lives in audio_io and is exercised on the Pi.
"""
from __future__ import annotations

import threading

from assistant.pipeline import Pipeline
from assistant.state import AppState, SharedState
from assistant.store import Store


def _pipeline() -> Pipeline:
    # in-memory DB: no file, no audio, just enough to construct the pipeline.
    return Pipeline(SharedState(), Store(":memory:"), {})


def test_tap_interrupts_only_while_speaking():
    p = _pipeline()
    p._busy.acquire()                 # stand in for an in-flight interaction
    try:
        p.shared.set(AppState.SPEAKING)
        p.on_tap()
        assert p._interrupt.is_set()
    finally:
        p._busy.release()


def test_tap_ignored_while_thinking():
    p = _pipeline()
    p._busy.acquire()
    try:
        p.shared.set(AppState.THINKING)
        p.on_tap()
        assert not p._interrupt.is_set()
    finally:
        p._busy.release()


def test_tap_ignored_while_listening():
    p = _pipeline()
    p._busy.acquire()
    try:
        p.shared.set(AppState.LISTENING)
        p.on_tap()
        assert not p._interrupt.is_set()
    finally:
        p._busy.release()


def test_idle_tap_starts_work_without_interrupting(monkeypatch):
    p = _pipeline()
    started = threading.Event()
    # The real _run needs a mic; swap it so we only prove a tap when idle kicks off
    # a worker thread and does NOT set the interrupt.
    monkeypatch.setattr(p, "_run", started.set)
    p.on_tap()
    assert started.wait(timeout=1.0)
    assert not p._interrupt.is_set()
