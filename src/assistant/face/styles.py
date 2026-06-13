"""Per-state face styles: the palette/shape table and easing between states.

This table is the face's art design — what each state *looks like* — so it
lives in code rather than the profile YAMLs: the values are not per-device
tunables, and mirroring ~9 states × 15 fields across both profiles would
invite drift. Behavioral timings (blink, gaze, transition speed, oscillator
rates) are real tunables and live in config/<profile>.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from assistant.face.logic import lerp, smoothstep

Color = tuple[float, float, float]

# CLAUDE.md order — laptop debug keys 1-9 map onto this.
FACE_STATES = (
    "sleeping",
    "drowsy",
    "neutral",
    "alert",
    "listening",
    "thinking",
    "speaking",
    "happy",
    "curious",
)


@dataclass(frozen=True)
class FaceStyle:
    bg: Color  # background
    eye: Color  # accent: the eye body
    iris: Color  # pupil/iris
    eye_w: float  # eye width, × display width
    eye_h: float  # eye height when fully open, × display height
    openness: float  # lid openness (0 closed .. ~1.15 wide)
    pupil: float  # pupil radius, × eye width
    scale: float  # overall eye scale
    curve: float  # upward bottom-lid arc (happy squint), 0..1
    tilt: float  # vertical eye skew (head tilt), × display height
    asym: float  # extra scale on the right eye (curious)
    bounce: float  # rhythmic vertical bounce amplitude (speaking), × eye height
    breathe: float  # slow breathing scale amplitude (sleeping)
    gaze_bias_x: float  # constant pupil offset, -1..1
    gaze_bias_y: float
    blink_interval_scale: float  # multiplier on the blink interval
    blink_speed_scale: float  # multiplier on the blink close-open duration


_BASE: dict[str, float] = {
    "eye_w": 0.17,
    "eye_h": 0.42,
    "openness": 1.0,
    "pupil": 0.20,
    "scale": 1.0,
    "curve": 0.0,
    "tilt": 0.0,
    "asym": 1.0,
    "bounce": 0.0,
    "breathe": 0.0,
    "gaze_bias_x": 0.0,
    "gaze_bias_y": 0.0,
    "blink_interval_scale": 1.0,
    "blink_speed_scale": 1.0,
}


def _style(bg: Color, eye: Color, iris: Color, **overrides: float) -> FaceStyle:
    return FaceStyle(bg=bg, eye=eye, iris=iris, **{**_BASE, **overrides})


STYLES: dict[str, FaceStyle] = {
    # Closed lids, slow breathing; blinking is disabled by the face module.
    "sleeping": _style(
        (6, 8, 20), (64, 78, 130), (90, 110, 170), openness=0.0, breathe=0.05
    ),
    # Half-lids and long, languid blinks.
    "drowsy": _style(
        (10, 12, 28), (120, 140, 200), (45, 55, 105),
        openness=0.45, pupil=0.18, blink_interval_scale=1.5, blink_speed_scale=3.0,
    ),
    "neutral": _style((14, 16, 36), (205, 232, 255), (35, 80, 150)),
    # Wide open, slightly enlarged, quick frequent blinks.
    "alert": _style(
        (18, 22, 48), (238, 246, 255), (25, 115, 205),
        openness=1.12, pupil=0.22, scale=1.06,
        blink_interval_scale=0.7, blink_speed_scale=0.8,
    ),
    # Slightly enlarged and steady: blinks come rarely.
    "listening": _style(
        (12, 24, 34), (165, 240, 215), (12, 100, 84),
        openness=1.05, pupil=0.24, scale=1.10, blink_interval_scale=1.6,
    ),
    # Narrowed, gaze biased upward; logic adds a slow drift on the bias.
    "thinking": _style(
        (18, 14, 40), (190, 172, 255), (75, 52, 145),
        openness=0.58, eye_w=0.18, gaze_bias_y=-0.45, blink_interval_scale=1.2,
    ),
    "speaking": _style(
        (24, 18, 34), (255, 206, 120), (155, 84, 28),
        openness=0.95, scale=1.02, bounce=0.05, blink_interval_scale=1.2,
    ),
    # Squinted upward-curving arcs.
    "happy": _style(
        (26, 20, 30), (255, 219, 92), (185, 116, 24),
        openness=0.78, scale=1.05, curve=0.85,
    ),
    # One eye larger, head-tilt skew, big pupils.
    "curious": _style(
        (20, 14, 32), (255, 172, 210), (165, 42, 112),
        openness=1.05, pupil=0.26, scale=1.03, tilt=0.045, asym=1.16,
        gaze_bias_y=-0.08,
    ),
}


def blend_styles(a: FaceStyle, b: FaceStyle, t: float) -> FaceStyle:
    if t >= 1.0:
        return b
    values: dict[str, object] = {}
    for f in fields(FaceStyle):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if isinstance(va, tuple):
            values[f.name] = tuple(lerp(ca, cb, t) for ca, cb in zip(va, vb))
        else:
            values[f.name] = lerp(va, vb, t)
    return FaceStyle(**values)  # type: ignore[arg-type]


class StyleBlender:
    """Eases between state styles over a fixed transition window.

    Retargeting mid-transition snapshots the current blend as the new start,
    so there is never a visual pop.
    """

    def __init__(self, initial: str, transition_s: float, now: float = 0.0) -> None:
        self.state = initial
        self._transition_s = transition_s
        self._from = STYLES[initial]
        self._start = now - transition_s  # begin settled

    def set_state(self, state: str, now: float) -> None:
        if state == self.state:
            return
        self._from = self.current(now)
        self.state = state
        self._start = now

    def current(self, now: float) -> FaceStyle:
        if self._transition_s <= 0:
            return STYLES[self.state]
        t = (now - self._start) / self._transition_s
        return blend_styles(self._from, STYLES[self.state], smoothstep(t))
