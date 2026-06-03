"""Tests for UI input routing (assistant.ui.dispatch_event).

These guard the one piece of UI behaviour that's easy to get wrong on the panel:
a single physical tap must fire on_tap exactly once. SDL synthesizes a
MOUSEBUTTONDOWN from every touch, so naive handling double-fires.

No display is needed — dispatch_event is pure routing over plain Event objects.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from assistant.ui import dispatch_event


def _counter():
    calls = {"tap": 0, "keys": []}

    def on_tap() -> None:
        calls["tap"] += 1

    def on_key(k: int) -> None:
        calls["keys"].append(k)

    return calls, on_tap, on_key


def test_finger_tap_fires_once():
    calls, on_tap, on_key = _counter()
    ev = pygame.event.Event(pygame.FINGERDOWN, {"x": 0.5, "y": 0.5})
    assert dispatch_event(ev, on_tap, on_key) is True
    assert calls["tap"] == 1


def test_touch_synthesized_mouse_is_ignored():
    # A real tap = one FINGERDOWN + one SDL-synthesized MOUSEBUTTONDOWN(touch=True).
    # That pair must fire on_tap once, not twice.
    calls, on_tap, on_key = _counter()
    finger = pygame.event.Event(pygame.FINGERDOWN, {"x": 0.5, "y": 0.5})
    synth = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (0, 0), "touch": True})
    dispatch_event(finger, on_tap, on_key)
    dispatch_event(synth, on_tap, on_key)
    assert calls["tap"] == 1


def test_real_mouse_click_still_fires():
    # Desktop dev: a genuine mouse click has touch=False and must still trigger.
    calls, on_tap, on_key = _counter()
    ev = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (0, 0), "touch": False})
    dispatch_event(ev, on_tap, on_key)
    assert calls["tap"] == 1


def test_escape_requests_quit():
    calls, on_tap, on_key = _counter()
    ev = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})
    assert dispatch_event(ev, on_tap, on_key) is False
    assert calls["keys"] == []  # ESC is consumed by the loop, not forwarded


def test_quit_event_requests_quit():
    calls, on_tap, on_key = _counter()
    assert dispatch_event(pygame.event.Event(pygame.QUIT), on_tap, on_key) is False


def test_other_keys_forwarded():
    calls, on_tap, on_key = _counter()
    ev = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_1})
    assert dispatch_event(ev, on_tap, on_key) is True
    assert calls["keys"] == [pygame.K_1]
    assert calls["tap"] == 0


def test_on_key_optional():
    # run.py passes no on_key; a keypress must not blow up.
    calls, on_tap, _ = _counter()
    ev = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_1})
    assert dispatch_event(ev, on_tap, None) is True
