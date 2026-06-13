"""Pure presence/gaze logic — no OpenCV, time injected, fully unit-testable."""

from __future__ import annotations

PERSON_APPEARED = "person_appeared"
PERSON_LEFT = "person_left"


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


class PresenceTracker:
    """Debounces per-frame detection results into appeared/left transitions.

    Appearing requires `appear_frames` consecutive face-frames; leaving requires
    `absence_timeout_s` without any face while present. A single no-face frame
    resets the appearance streak.
    """

    def __init__(self, appear_frames: int, absence_timeout_s: float) -> None:
        self._appear_frames = appear_frames
        self._absence_timeout_s = absence_timeout_s
        self._consecutive = 0
        self._last_face_at: float | None = None
        self.present = False

    def update(self, face_seen: bool, now: float) -> str | None:
        """Feed one detection result; returns the transition event name, if any."""
        if face_seen:
            self._consecutive += 1
            self._last_face_at = now
            if not self.present and self._consecutive >= self._appear_frames:
                self.present = True
                return PERSON_APPEARED
            return None
        self._consecutive = 0
        if (
            self.present
            and self._last_face_at is not None
            and now - self._last_face_at >= self._absence_timeout_s
        ):
            self.present = False
            return PERSON_LEFT
        return None


class GazeThrottle:
    """Rate-limits gaze events to at most one per interval."""

    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._last_at: float | None = None

    def ready(self, now: float) -> bool:
        if self._last_at is not None and now - self._last_at < self._interval_s:
            return False
        self._last_at = now
        return True
