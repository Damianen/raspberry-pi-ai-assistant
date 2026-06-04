"""Tests for the v3 face renderer (assistant.eyes).

The *look* (colours, drift, micro-expressions) is eyeballed on the panel — these
tests pin the LOGIC that's easy to break and matters: (1) the idle-life state
machine reaches drowsy after the right amount of idle, (2) THE WAKING RULE — any
exit from IDLE snaps the face to full brightness the same frame so drowsiness can
never mute or soften an alarm, (3) colour crossfades rather than cutting, and
(4) the double-blink follow-up schedules and fires.

Eyes builds pygame Surfaces, so we init pygame with the dummy drivers — no real
display or audio device needed (same trick the rest of the suite uses).
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

import assistant.eyes as em
from assistant.eyes import DROWSY_AFTER, DROWSY_COLOR, DOUBLE_BLINK_GAP, Eyes, PALETTE
from assistant.state import AppState


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


def _eyes() -> Eyes:
    return Eyes(800, 480)


def _idle_for(e: Eyes, frames: int) -> None:
    for _ in range(frames):
        e.update()


def test_goes_drowsy_after_two_minutes_idle():
    e = _eyes()
    # just before the threshold it is still awake...
    _idle_for(e, DROWSY_AFTER - 1)
    assert e._idle_mode == "normal"
    # ...and crosses into drowsy a couple of frames past it (no yawn can pre-empt
    # it: the first yawn is scheduled minutes later than DROWSY_AFTER).
    _idle_for(e, 5)
    assert e._idle_mode == "drowsy"


def test_alarm_while_drowsy_wakes_to_full_brightness_same_frame():
    # The exact slice test: let it fall asleep, then fire an announcement
    # (SPEAKING, as a due alarm would). The face must be at full SPEAKING
    # brightness the SAME frame — drowsiness is cosmetic and must never soften it.
    e = _eyes()
    _idle_for(e, DROWSY_AFTER + 40)
    assert e._idle_mode == "drowsy"
    assert e.p["glow"] < 0.8          # dimmed while asleep
    assert e.p["openness"] < 0.7      # heavy lids

    e.set_state(AppState.SPEAKING)    # alarm fires
    # snapped THIS frame, before any further update()
    assert e.p["glow"] == pytest.approx(1.35)
    assert e.p["openness"] == pytest.approx(0.95)
    assert e._idle_mode == "normal"

    e.update(0.0)                     # the next frame keeps it bright + open
    assert e.p["glow"] == pytest.approx(1.35, abs=1e-3)
    assert e.p["openness"] > 0.9


def test_wake_mid_blink_snaps_eyes_open():
    # A tap landing exactly during a blink must not flash a closed-eyed face: any
    # exit from IDLE snaps openness to the new target, not just the drowsy case.
    e = _eyes()
    e._blink_until = e.t + 10
    _idle_for(e, 8)                   # openness lerps down toward the blink (0.06)
    assert e.p["openness"] < 0.5

    e.set_state(AppState.LISTENING)
    assert e.p["openness"] == pytest.approx(1.15)


def test_colour_crossfades_not_snaps():
    e = _eyes()
    start = list(e._col)              # idle cyan
    e.set_state(AppState.ERROR)
    assert e._col == start            # set_state never touches the hue

    e.update()
    assert e._col != start            # one step toward red...
    target = PALETTE[AppState.ERROR]
    assert abs(e._col[0] - target[0]) > 1   # ...but not all the way (crossfade)

    for _ in range(80):
        e.update()                    # eventually converges
    assert e._col[0] == pytest.approx(target[0], abs=1.0)
    assert e._col[2] == pytest.approx(target[2], abs=1.0)


def test_drowsy_wears_the_drowsy_palette():
    e = _eyes()
    _idle_for(e, DROWSY_AFTER + 80)
    assert e._col[0] == pytest.approx(DROWSY_COLOR[0], abs=3)
    assert e._col[1] == pytest.approx(DROWSY_COLOR[1], abs=3)
    assert e._col[2] == pytest.approx(DROWSY_COLOR[2], abs=3)


def test_double_blink_rolls_a_second(monkeypatch):
    # Force the per-blink roll under the 18% threshold -> a follow-up is scheduled.
    e = _eyes()
    monkeypatch.setattr(em.random, "random", lambda: 0.05)
    e._next_blink = e.t               # make a blink fire on the next idle tick
    e.update()
    assert e._blink_until > e.t                       # primary blink armed
    assert e._second_blink_at == e.t + DOUBLE_BLINK_GAP


def test_scheduled_second_blink_fires(monkeypatch):
    # Once t passes the scheduled follow-up, it re-arms the blink and clears.
    e = _eyes()
    monkeypatch.setattr(em.random, "random", lambda: 0.9)   # no NEW double blinks
    _idle_for(e, 5)                   # get t off zero so the flag is truthy
    e._second_blink_at = e.t
    e._next_blink = e.t + 10_000      # keep primary blinks out of the way
    e.update()
    assert e._blink_until > e.t
    assert e._second_blink_at == 0


def test_draw_runs_in_every_state():
    # Smoke: each state must draw without blowing up (geometry / colour / aura).
    e = _eyes()
    surf = pygame.Surface((800, 480))
    for st in AppState:
        e.set_state(st)
        e.update(0.4)
        e.draw(surf)                  # no exception == pass
