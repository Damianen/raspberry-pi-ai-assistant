"""Pure face-animation math: blinking, gaze, and per-frame geometry.

No pygame here. Everything is driven by an injected clock (`now` in seconds,
monotonic) and an injected RNG, so pytest can cover it deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from assistant.config import BlinkConfig, FaceConfig, GazeConfig

if TYPE_CHECKING:
    from assistant.face.styles import Color, FaceStyle

# Face layout — fixed art-design fractions of the display (what the face *is*);
# behavioral tunables live in config/<profile>.yaml.
EYE_OFFSET_X = 0.205  # eye center distance from screen center, × display width
EYE_CENTER_Y = 0.48  # eye row, × display height
MIN_LID_HEIGHT = 0.018  # closed-lid line height, × display height
PUPIL_TRAVEL = 0.80  # usable fraction of the eye interior for pupil travel
PUPIL_MAX_OF_HEIGHT = 0.42  # pupil radius cap, × current open eye height
PUPIL_MIN_OPEN = 0.2  # pupil fades out below this lid openness
GAZE_DRIFT_AMOUNT = 0.25  # slow drift modulation, fraction of the gaze bias

_SCHEDULE_EPS = 1e-9  # tolerance for float error in time comparisons


class Rng(Protocol):
    def uniform(self, a: float, b: float) -> float: ...
    def random(self) -> float: ...


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def smooth_toward(current: float, target: float, factor: float) -> float:
    """One frame of exponential smoothing: cover `factor` of the remaining distance."""
    return current + (target - current) * factor


def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


class BlinkScheduler:
    """Schedules randomized blinks; yields lid openness as a 1 → 0 → 1 factor.

    The next interval is sampled when a blink completes, so a tempo change
    mid-wait only takes effect from the following blink — acceptable drift.
    A double blink's second closure never rolls another double.
    """

    def __init__(self, cfg: BlinkConfig, rng: Rng, now: float = 0.0) -> None:
        self._cfg = cfg
        self._rng = rng
        self._interval_scale = 1.0
        self._duration_scale = 1.0
        self._blink_start: float | None = None
        self._double_pending = False
        self._followup = False
        self._next_at = now + self._next_interval()

    def _next_interval(self) -> float:
        base = self._rng.uniform(self._cfg.min_interval_s, self._cfg.max_interval_s)
        return base * self._interval_scale

    def set_tempo(self, interval_scale: float, duration_scale: float) -> None:
        self._interval_scale = interval_scale
        self._duration_scale = duration_scale

    def force(self, now: float) -> None:
        """Start a blink immediately; no-op if one is already in flight."""
        if self._blink_start is None:
            self._blink_start = now
            self._double_pending = False
            self._followup = False

    def update(self, now: float, enabled: bool = True) -> float:
        """Advance to `now` and return lid openness in [0, 1].

        When disabled (sleeping) an in-flight blink still completes — the
        sleeping style's near-zero openness masks it — but no new blink starts.
        """
        if self._blink_start is None:
            if not enabled or now < self._next_at - _SCHEDULE_EPS:
                return 1.0
            self._blink_start = now
            if self._followup:
                self._followup = False
            else:
                self._double_pending = self._rng.random() < self._cfg.double_chance
        duration = (self._cfg.close_open_ms / 1000.0) * self._duration_scale
        t = (now - self._blink_start) / duration
        if t >= 1.0 - _SCHEDULE_EPS:
            self._blink_start = None
            if self._double_pending:
                self._double_pending = False
                self._followup = True
                self._next_at = now + self._cfg.double_gap_ms / 1000.0
            else:
                self._next_at = now + self._next_interval()
            return 1.0
        return 1.0 - math.sin(math.pi * t)


class GazeController:
    """Smooths the pupils toward a commanded target; wanders when commands stop.

    Smoothing is applied per `update()` call, which the render loop makes once
    per frame at a locked 60 fps — a documented simplification; revisit if the
    pi drops frames.
    """

    def __init__(self, cfg: GazeConfig, rng: Rng, now: float = 0.0) -> None:
        self._cfg = cfg
        self._rng = rng
        self.x = 0.5
        self.y = 0.5
        self._target = (0.5, 0.5)
        self._last_command = now
        self._idle = False
        self._wander = (0.5, 0.5)
        self._next_glance = math.inf
        self._next_drift = math.inf

    def set_target(self, x: float, y: float, now: float) -> None:
        self._target = (clamp01(x), clamp01(y))
        self._last_command = now
        self._idle = False

    def is_idle(self, now: float) -> bool:
        return now - self._last_command >= self._cfg.idle_after_s

    def update(self, now: float) -> tuple[float, float]:
        target = self._wander_target(now) if self.is_idle(now) else self._target
        self.x = smooth_toward(self.x, target[0], self._cfg.smoothing)
        self.y = smooth_toward(self.y, target[1], self._cfg.smoothing)
        return self.x, self.y

    def _wander_target(self, now: float) -> tuple[float, float]:
        cfg = self._cfg
        lo, hi = cfg.glance_margin, 1.0 - cfg.glance_margin
        if not self._idle:
            self._idle = True
            self._wander = (clamp(self.x, lo, hi), clamp(self.y, lo, hi))
            self._next_glance = now + self._rng.uniform(cfg.glance_min_s, cfg.glance_max_s)
            self._next_drift = now + cfg.drift_interval_s
        if now >= self._next_glance:
            self._wander = (self._rng.uniform(lo, hi), self._rng.uniform(lo, hi))
            self._next_glance = now + self._rng.uniform(cfg.glance_min_s, cfg.glance_max_s)
            self._next_drift = now + cfg.drift_interval_s
        elif now >= self._next_drift:
            dx = self._rng.uniform(-cfg.drift_amount, cfg.drift_amount)
            dy = self._rng.uniform(-cfg.drift_amount, cfg.drift_amount)
            self._wander = (
                clamp(self._wander[0] + dx, lo, hi),
                clamp(self._wander[1] + dy, lo, hi),
            )
            self._next_drift = now + cfg.drift_interval_s
        return self._wander


@dataclass(frozen=True)
class EyeFrame:
    """One eye's geometry for a single frame, in display pixels (floats)."""

    center_x: float
    center_y: float
    width: float
    height: float
    pupil_x: float
    pupil_y: float
    pupil_r: float
    curve: float  # 0..1 upward bottom-lid arc depth


@dataclass(frozen=True)
class FaceFrame:
    """Everything the renderer needs to draw one frame."""

    bg: Color
    eye_color: Color
    iris_color: Color
    eyes: tuple[EyeFrame, EyeFrame]


def compute_frame(
    style: FaceStyle,
    blink_openness: float,
    gaze: tuple[float, float],
    now: float,
    cfg: FaceConfig,
    size: tuple[int, int],
) -> FaceFrame:
    """Compose the per-eye geometry for one frame: oscillators, lids, pupils."""
    w, h = size
    breath = 1.0 + style.breathe * math.sin(math.tau * cfg.breathing_hz * now)
    bounce = style.bounce * math.sin(math.tau * cfg.bounce_hz * now) * style.eye_h * h
    drift = 1.0 + GAZE_DRIFT_AMOUNT * math.sin(math.tau * cfg.drift_hz * now)
    open_total = max(style.openness * blink_openness, 0.0)

    eyes = []
    for side in (-1, 1):  # left, right
        scale = style.scale * breath * (style.asym if side > 0 else 1.0)
        eye_w = style.eye_w * w * scale
        eye_h = max(style.eye_h * h * scale * open_total, MIN_LID_HEIGHT * h)
        cx = w / 2 + side * EYE_OFFSET_X * w
        cy = EYE_CENTER_Y * h + side * style.tilt * h + bounce
        # The happy squint swallows the pupil instead of leaving a notch in the
        # arc, and closing lids fade it out so nothing peeks through them.
        pupil_r = (
            min(style.pupil * eye_w, PUPIL_MAX_OF_HEIGHT * eye_h)
            * (1.0 - style.curve)
            * clamp01(open_total / PUPIL_MIN_OPEN)
        )
        gx = clamp((gaze[0] - 0.5) * 2.0 + style.gaze_bias_x, -1.0, 1.0)
        gy = clamp((gaze[1] - 0.5) * 2.0 + style.gaze_bias_y * drift, -1.0, 1.0)
        max_dx = max(eye_w / 2 - pupil_r, 0.0) * PUPIL_TRAVEL
        max_dy = max(eye_h / 2 - pupil_r, 0.0) * PUPIL_TRAVEL
        eyes.append(
            EyeFrame(
                center_x=cx,
                center_y=cy,
                width=eye_w,
                height=eye_h,
                pupil_x=cx + gx * max_dx,
                pupil_y=cy + gy * max_dy,
                pupil_r=pupil_r,
                curve=style.curve,
            )
        )
    return FaceFrame(
        bg=style.bg, eye_color=style.eye, iris_color=style.iris, eyes=(eyes[0], eyes[1])
    )
