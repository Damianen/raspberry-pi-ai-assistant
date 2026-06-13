"""Pure-logic tests for the face: blink scheduling, gaze math, style blending.

Nothing here imports pygame — assistant.face.logic and .styles are plain math.
"""

import pytest

from assistant.config import BlinkConfig, GazeConfig, load_config
from assistant.face.logic import (
    BlinkScheduler,
    GazeController,
    compute_frame,
    lerp,
    smooth_toward,
)
from assistant.face.styles import FACE_STATES, STYLES, StyleBlender, blend_styles

SIZE = (800, 480)


class ScriptedRng:
    """Pops scripted values; deterministic fallbacks (midpoint / 1.0) after."""

    def __init__(self, uniforms=(), randoms=()):
        self.uniforms = list(uniforms)
        self.randoms = list(randoms)

    def uniform(self, a: float, b: float) -> float:
        return self.uniforms.pop(0) if self.uniforms else (a + b) / 2

    def random(self) -> float:
        return self.randoms.pop(0) if self.randoms else 1.0


def blink_cfg() -> BlinkConfig:
    return BlinkConfig(
        min_interval_s=2.0,
        max_interval_s=7.0,
        close_open_ms=120,
        double_chance=0.15,
        double_gap_ms=180,
    )


def gaze_cfg(smoothing: float = 1.0) -> GazeConfig:
    return GazeConfig(
        smoothing=smoothing,
        idle_after_s=5.0,
        drift_amount=0.05,
        drift_interval_s=0.8,
        glance_min_s=2.0,
        glance_max_s=6.0,
        glance_margin=0.15,
    )


# --- helpers ---------------------------------------------------------------


def test_lerp_and_smooth_toward() -> None:
    assert lerp(0.0, 10.0, 0.25) == 2.5
    assert smooth_toward(0.0, 1.0, 0.15) == pytest.approx(0.15)
    assert smooth_toward(0.5, 0.5, 0.15) == 0.5


# --- blink scheduler ---------------------------------------------------------


def test_blink_closes_and_reopens_on_schedule() -> None:
    b = BlinkScheduler(blink_cfg(), ScriptedRng(uniforms=[3.0], randoms=[1.0]), now=0.0)
    assert b.update(0.0) == 1.0
    assert b.update(2.99) == 1.0
    assert b.update(3.0) == pytest.approx(1.0)  # blink starts, lids not yet moving
    assert b.update(3.06) == pytest.approx(0.0, abs=1e-6)  # fully closed at midpoint
    assert b.update(3.12) == 1.0  # reopened, next blink scheduled
    # Next interval falls back to the midpoint of 2..7 s => 4.5 s after reopening.
    assert b.update(7.0) == 1.0
    assert b.update(7.63) == 1.0  # blink starts on the first update past the schedule
    assert b.update(7.69) == pytest.approx(0.0, abs=1e-6)


def test_force_blink_and_noop_while_in_flight() -> None:
    b = BlinkScheduler(blink_cfg(), ScriptedRng(uniforms=[5.0]), now=0.0)
    b.force(1.0)
    assert b.update(1.06) == pytest.approx(0.0, abs=1e-6)
    b.force(1.07)  # already blinking: ignored, does not restart the phase
    assert b.update(1.12) == 1.0


def test_double_blink_fires_after_gap_and_never_chains() -> None:
    # random 0.1 < 0.15 triggers a double; a second 0.1 must NOT chain another.
    b = BlinkScheduler(blink_cfg(), ScriptedRng(uniforms=[3.0], randoms=[0.1, 0.1]), now=0.0)
    assert b.update(3.0) == 1.0
    assert b.update(3.06) == pytest.approx(0.0, abs=1e-6)
    assert b.update(3.12) == 1.0  # first closure done, follow-up due 180 ms later
    assert b.update(3.25) == 1.0
    assert b.update(3.30) == 1.0  # follow-up starts
    assert b.update(3.36) == pytest.approx(0.0, abs=1e-6)
    assert b.update(3.42) == 1.0
    # No chaining: nothing happens until a full interval (fallback 4.5 s) later.
    assert b.update(4.0) == 1.0
    assert b.update(7.0) == 1.0


def test_no_new_blinks_while_disabled_but_in_flight_completes() -> None:
    b = BlinkScheduler(blink_cfg(), ScriptedRng(uniforms=[2.0], randoms=[1.0]), now=0.0)
    assert b.update(2.0) == 1.0  # blink starts while enabled
    assert b.update(2.06, enabled=False) == pytest.approx(0.0, abs=1e-6)  # completes
    assert b.update(2.12, enabled=False) == 1.0
    assert b.update(60.0, enabled=False) == 1.0  # long sleep: nothing starts
    assert b.update(60.1) == 1.0  # re-enabled: wake blink starts
    assert b.update(60.16) == pytest.approx(0.0, abs=1e-6)


def test_blink_tempo_scales_duration() -> None:
    b = BlinkScheduler(blink_cfg(), ScriptedRng(uniforms=[2.0], randoms=[1.0]), now=0.0)
    b.set_tempo(1.0, 3.0)  # drowsy-style languid blink: 360 ms close-open
    assert b.update(2.0) == 1.0
    assert b.update(2.18) == pytest.approx(0.0, abs=1e-6)  # midpoint at 180 ms
    assert b.update(2.30) < 1.0  # still in flight where a normal blink would be done
    assert b.update(2.36) == 1.0


# --- gaze controller ---------------------------------------------------------


def test_gaze_smoothing_step_math() -> None:
    g = GazeController(gaze_cfg(smoothing=0.15), ScriptedRng(), now=0.0)
    g.set_target(1.0, 0.0, now=0.0)
    x, y = g.update(0.016)
    assert x == pytest.approx(0.5 + 0.15 * 0.5)
    assert y == pytest.approx(0.5 - 0.15 * 0.5)
    x, y = g.update(0.033)
    assert x == pytest.approx(0.575 + 0.15 * (1.0 - 0.575))


def test_gaze_idle_wander_drift_and_glance() -> None:
    rng = ScriptedRng(uniforms=[2.0, 0.04, -0.04, 0.7, 0.3])
    g = GazeController(gaze_cfg(smoothing=1.0), rng, now=0.0)
    g.set_target(0.8, 0.2, now=0.0)
    assert g.update(4.9) == (0.8, 0.2)
    assert not g.is_idle(4.9)
    assert g.is_idle(5.0)
    # Idle entry: wander anchors at the current position, timers armed.
    assert g.update(5.0) == (0.8, 0.2)
    # Drift tick at +0.8 s nudges by the scripted (0.04, -0.04).
    assert g.update(5.8) == (pytest.approx(0.84), pytest.approx(0.16))
    # Glance at +2.0 s jumps to the scripted random point.
    assert g.update(7.0) == (pytest.approx(0.7), pytest.approx(0.3))
    # A new command exits idle immediately.
    g.set_target(0.5, 0.5, now=7.2)
    assert not g.is_idle(7.3)
    assert g.update(7.3) == (0.5, 0.5)


def test_gaze_wander_stays_within_margins() -> None:
    rng = ScriptedRng(uniforms=[2.0, 0.2, 0.2])  # oversized drift gets clamped
    g = GazeController(gaze_cfg(smoothing=1.0), rng, now=0.0)
    g.set_target(0.84, 0.84, now=0.0)
    g.update(0.1)
    g.update(5.0)  # idle entry
    x, y = g.update(5.8)  # drift tick
    assert x == pytest.approx(0.85)  # clamped to 1 - glance_margin
    assert y == pytest.approx(0.85)


# --- styles and blending -----------------------------------------------------


def test_face_states_match_claude_md_order_and_styles() -> None:
    assert FACE_STATES == (
        "sleeping", "drowsy", "neutral", "alert", "listening",
        "thinking", "speaking", "happy", "curious",
    )
    assert set(FACE_STATES) == set(STYLES)


def test_style_blender_eases_and_completes() -> None:
    blender = StyleBlender("neutral", 0.25, now=0.0)
    assert blender.current(0.0) == STYLES["neutral"]  # starts settled
    blender.set_state("happy", now=1.0)
    mid = blender.current(1.125)  # halfway: smoothstep(0.5) == 0.5
    expected = (STYLES["neutral"].openness + STYLES["happy"].openness) / 2
    assert mid.openness == pytest.approx(expected)
    assert mid.curve == pytest.approx(STYLES["happy"].curve / 2)
    assert blender.current(1.3) == STYLES["happy"]


def test_style_blender_retarget_mid_flight_snapshots() -> None:
    blender = StyleBlender("neutral", 0.25, now=0.0)
    blender.set_state("happy", now=1.0)
    mid = blender.current(1.125)
    blender.set_state("alert", now=1.125)
    assert blender.current(1.125) == mid  # no pop: blend starts from the snapshot
    assert blender.current(1.4) == STYLES["alert"]


def test_blend_styles_interpolates_colors() -> None:
    a, b = STYLES["neutral"], STYLES["happy"]
    mid = blend_styles(a, b, 0.5)
    for ca, cb, cm in zip(a.bg, b.bg, mid.bg):
        assert cm == pytest.approx((ca + cb) / 2)


# --- frame composition -------------------------------------------------------


def test_pupil_fits_inside_narrowed_eye() -> None:
    cfg = load_config("laptop").face
    frame = compute_frame(STYLES["thinking"], 1.0, (0.5, 0.5), 0.0, cfg, SIZE)
    for eye in frame.eyes:
        assert eye.pupil_r <= eye.height / 2


def test_pupil_travel_stays_inside_eye() -> None:
    cfg = load_config("laptop").face
    for corner in ((0.0, 0.0), (1.0, 1.0)):
        frame = compute_frame(STYLES["curious"], 1.0, corner, 0.0, cfg, SIZE)
        for eye in frame.eyes:
            assert eye.pupil_x - eye.pupil_r >= eye.center_x - eye.width / 2
            assert eye.pupil_x + eye.pupil_r <= eye.center_x + eye.width / 2
            assert eye.pupil_y - eye.pupil_r >= eye.center_y - eye.height / 2
            assert eye.pupil_y + eye.pupil_r <= eye.center_y + eye.height / 2


def test_closed_lids_hide_pupil() -> None:
    cfg = load_config("laptop").face
    frame = compute_frame(STYLES["sleeping"], 1.0, (0.5, 0.5), 0.0, cfg, SIZE)
    for eye in frame.eyes:
        assert eye.pupil_r == 0.0


def test_sleeping_breathes() -> None:
    cfg = load_config("laptop").face
    rest = compute_frame(STYLES["sleeping"], 1.0, (0.5, 0.5), 0.0, cfg, SIZE)
    # Quarter period of the 0.1 Hz breathing cycle: peak inhale.
    peak = compute_frame(STYLES["sleeping"], 1.0, (0.5, 0.5), 2.5, cfg, SIZE)
    assert peak.eyes[0].width > rest.eyes[0].width


def test_happy_curve_shrinks_pupil() -> None:
    cfg = load_config("laptop").face
    happy = compute_frame(STYLES["happy"], 1.0, (0.5, 0.5), 0.0, cfg, SIZE)
    neutral = compute_frame(STYLES["neutral"], 1.0, (0.5, 0.5), 0.0, cfg, SIZE)
    assert happy.eyes[0].pupil_r < neutral.eyes[0].pupil_r


def test_curious_right_eye_is_larger_and_tilted() -> None:
    cfg = load_config("laptop").face
    frame = compute_frame(STYLES["curious"], 1.0, (0.5, 0.5), 0.0, cfg, SIZE)
    left, right = frame.eyes
    assert right.width > left.width
    assert right.center_y != left.center_y


# --- config ------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["laptop", "pi"])
def test_profiles_parse_face_config(profile: str) -> None:
    cfg = load_config(profile)
    assert cfg.face.transition_ms == 250
    assert cfg.face.blink.min_interval_s < cfg.face.blink.max_interval_s
    assert 0 < cfg.face.gaze.smoothing <= 1
    assert cfg.face.debug_controls == (profile == "laptop")
