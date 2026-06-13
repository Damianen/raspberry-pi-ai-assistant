"""Presence debounce and gaze throttling."""

from __future__ import annotations

from assistant.perception.logic import GazeThrottle, PresenceTracker, clamp01

APPEAR_FRAMES = 3
ABSENCE_TIMEOUT_S = 25.0
FRAME_DT = 0.2  # detection cadence at 15 fps / every 3rd frame


def make_tracker() -> PresenceTracker:
    return PresenceTracker(APPEAR_FRAMES, ABSENCE_TIMEOUT_S)


def test_appears_after_exactly_three_consecutive_face_frames() -> None:
    tracker = make_tracker()
    assert tracker.update(True, 0.0) is None
    assert tracker.update(True, FRAME_DT) is None
    assert tracker.update(True, 2 * FRAME_DT) == "person_appeared"
    assert tracker.present


def test_broken_streak_resets_the_debounce() -> None:
    tracker = make_tracker()
    tracker.update(True, 0.0)
    tracker.update(True, 0.2)
    assert tracker.update(False, 0.4) is None  # streak broken
    assert tracker.update(True, 0.6) is None
    assert tracker.update(True, 0.8) is None
    assert tracker.update(True, 1.0) == "person_appeared"


def test_no_repeat_appeared_while_present() -> None:
    tracker = make_tracker()
    for i in range(10):
        event = tracker.update(True, i * FRAME_DT)
        if i == APPEAR_FRAMES - 1:
            assert event == "person_appeared"
        else:
            assert event is None


def test_leaves_only_after_absence_timeout() -> None:
    tracker = make_tracker()
    for i in range(APPEAR_FRAMES):
        tracker.update(True, i * FRAME_DT)
    last_face_at = (APPEAR_FRAMES - 1) * FRAME_DT
    assert tracker.update(False, last_face_at + ABSENCE_TIMEOUT_S - 0.1) is None
    assert tracker.present
    assert tracker.update(False, last_face_at + ABSENCE_TIMEOUT_S) == "person_left"
    assert not tracker.present


def test_no_left_event_when_never_appeared() -> None:
    tracker = make_tracker()
    tracker.update(True, 0.0)  # one glimpse, below the debounce
    assert tracker.update(False, 100.0) is None


def test_brief_absence_does_not_leave() -> None:
    tracker = make_tracker()
    for i in range(APPEAR_FRAMES):
        tracker.update(True, i * FRAME_DT)
    assert tracker.update(False, 5.0) is None
    assert tracker.update(True, 5.2) is None  # back; no new appeared event
    assert tracker.present


def test_reappearance_cycle_after_leaving() -> None:
    tracker = make_tracker()
    for i in range(APPEAR_FRAMES):
        tracker.update(True, i * FRAME_DT)
    assert tracker.update(False, 30.0) == "person_left"
    assert tracker.update(True, 31.0) is None
    assert tracker.update(True, 31.2) is None
    assert tracker.update(True, 31.4) == "person_appeared"


def test_gaze_throttle_limits_rate() -> None:
    throttle = GazeThrottle(0.15)
    assert throttle.ready(0.0)
    assert not throttle.ready(0.1)
    assert throttle.ready(0.16)
    assert not throttle.ready(0.2)
    assert throttle.ready(0.31)


def test_clamp01() -> None:
    assert clamp01(-0.5) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(1.5) == 1.0
