"""Pixel-eye renderer for pygame. Ported from the approved HTML prototype (v3:
`pixel_face_prototype_v3.html`).

It READS state and draws. It contains no logic and never mutates AppState. Same
parameter model as the prototype: per-state targets for openness / scale / glow /
happy, pupil offset + dilation, a mouth (openness / curve / side-shift), AND a
per-state colour — each interpolated toward its target every frame.

v3 adds two things, both render-side:
  1. Per-state COLOUR. Each AppState (plus a drowsy variant) maps to an RGB in
     PALETTE; the displayed colour crossfades toward the target at COLOR_LERP, so
     state changes fade smoothly instead of cutting. The bloom halo (the port's
     stand-in for the prototype's listening "aura") and the optional listening
     aura both derive from this live colour.
  2. An IDLE "life" pack — pupil drift, double blinks, micro-expressions, a rare
     yawn, and a drowsy mode after a long idle. All of it is driven purely by
     time + randomness inside the IDLE path; none of it ever touches AppState.

THE WAKING RULE (must be exact — see set_state): any change OUT of IDLE drops the
idle sub-behaviour and SNAPS the dimming params (glow, openness) to the new
state's target the same frame. Drowsiness is cosmetic; it must never be able to
mute or soften an alarm. An announcement that fires while the face is asleep lands
at full brightness with zero ramp. (This deliberately goes further than the
prototype, which lets glow ramp up — on the real device a dim alarm is wrong.)

The one external input is the live audio level (0..1) from playback, read from the
snapshot's meta and passed into update(); only SPEAKING uses it (mouth openness).
The eyes SMOOTH and DRAW that number — they never compute it.

NOTE: tune CELL, PALETTE, the timing constants and proportions on the actual
800x480 DSI panel — the geometry below comes from the 500x300 prototype (absolute
offsets scaled by sx=w/500, sy=h/300 so the look is resolution-independent), and
the idle timings assume the ui loop's 60 fps. They're a starting point.
"""
from __future__ import annotations

import math
import random

import pygame

from .state import AppState

# ---- Palette (tune the hues here) ------------------------------------------
# cyan idle / blue listening / purple thinking / gold speaking / green confirm /
# red error — matching the prototype's legend.
PALETTE: dict[AppState, tuple[int, int, int]] = {
    AppState.IDLE:      (95, 243, 232),
    AppState.LISTENING: (90, 167, 255),
    AppState.THINKING:  (176, 123, 255),
    AppState.SPEAKING:  (255, 201, 102),
    AppState.CONFIRM:   (109, 255, 138),
    AppState.ERROR:     (255, 90, 77),
}
DROWSY_COLOR = (53, 143, 134)   # dim teal worn only while asleep (IDLE + drowsy)
COLOR_LERP = 0.12               # per-frame crossfade of the displayed hue

# ---- Idle-life timing (tune here) ------------------------------------------
# In FRAMES at the ui loop's 60 fps. The HTML prototype uses shorter demo values
# (e.g. 25 s to drowsy) "for the demo"; these are the real-device durations the
# slice asks for. If the panel ever runs at a different fps, rescale FPS.
FPS = 60
DROWSY_AFTER = 120 * FPS            # 2 min of unbroken idle -> drowsy
YAWN_MIN, YAWN_RANGE = 180 * FPS, 180 * FPS   # next yawn 3..6 min out
YAWN_LEN = int(2.5 * FPS)           # a yawn lasts ~2.5 s
MICRO_MIN, MICRO_RANGE = 6 * FPS, 8 * FPS     # micro-expression every 6..14 s
STARE_LEN = 110                     # held sideways stare (frames)
EXPR_LEN = 70                       # brief smile / squint (frames)
BLINK_LEN = 6                       # normal blink duration
DROWSY_BLINK_LEN = 14               # slow heavy blink while asleep
DOUBLE_BLINK_CHANCE = 0.18          # ~18% of blinks get a follow-up
DOUBLE_BLINK_GAP = 16               # 2nd blink ~0.27 s after the 1st

# The prototype canvas the absolute pixel offsets (saccade range, drift, thinking
# drift, mouth shift, mouth-curve depth) were tuned on. We scale them to the panel.
_PROTO_W, _PROTO_H = 500.0, 300.0


def _lerp(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


def _ease(x: float) -> float:
    """Smoothstep, clamped to [0,1] — the prototype's `ease` (yawn ramp)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3 - 2 * x)


def _shade(col: tuple[int, int, int], a: float) -> tuple[int, int, int]:
    """Premultiply a colour by alpha against the black background.

    The prototype draws each cell with an rgba alpha (eye gradient ~0.82..1.0,
    pupil 0.13, mouth lip 0.85, interior 0.16). The eye layer is composited onto
    black with black colour-keyed transparent, so premultiplying == alpha-blending
    onto black, and the dimmer cells also dim the bloom for free. Every palette
    colour has all-nonzero channels, so even the 0.13 pupil stays above (0,0,0)
    and survives the colour-key."""
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
    return int(col[0] * a), int(col[1] * a), int(col[2] * a)


class Eyes:
    def __init__(self, width: int, height: int, cell: int = 12) -> None:
        self.w, self.h, self.cell = width, height, cell
        self.cols, self.rows = width // cell, height // cell
        # Proto-pixel -> panel-pixel scale for the absolute offsets. Equal on both
        # axes while the panel stays 5:3, but kept separate so it survives a reshape.
        self._sx = width / _PROTO_W
        self._sy = height / _PROTO_H
        self._curve_amp = 16.0 * self._sy   # mouth smile/frown arc depth (proto: 16px)
        self.t = 0
        self.state = AppState.IDLE
        self._entered = 0
        self._level = 0.0           # latest audio level from meta (set in update)
        # displayed colour, crossfaded toward the target each frame (floats)
        self._col = [float(c) for c in PALETTE[AppState.IDLE]]
        # blinks (+ the optional double-blink follow-up)
        self._blink_until = 0
        self._next_blink = 60
        self._second_blink_at = 0
        # saccades / pupil
        self._next_saccade = 60
        self._sac = (0.0, 0.0)      # current idle pupil target (saccade / stare)
        self._shake = 0.0
        # idle-life sub-state: 'normal' | 'yawning' | 'drowsy'
        self._idle_enter = 0
        self._idle_mode = "normal"
        self._yawn_start = 0
        self._next_yawn = YAWN_MIN + int(random.random() * YAWN_RANGE)
        self._micro: str | None = None    # 'smile' | 'squint' | 'stare'
        self._micro_until = 0
        self._next_micro = MICRO_MIN + int(random.random() * MICRO_RANGE)
        # slow pupil drift — random phase so two units don't drift in lockstep
        self._drift1 = random.random() * 7.0
        self._drift2 = random.random() * 7.0
        # current interpolated params (superset of the prototype's CUR object)
        self.p = dict(openness=1.0, scale=1.0, glow=1.0, happy=0.0,
                      px=0.0, py=0.0, dil=1.0,              # pupil offset + dilation
                      mopen=0.0, mcurve=0.25, mshift=0.0)   # mouth
        # Offscreen layer for the crisp eyes. Black is colour-keyed transparent
        # so the bloom halo behind the eyes survives the final composite.
        self._eye_layer = pygame.Surface((width, height))
        self._eye_layer.set_colorkey((0, 0, 0))
        # Bloom works on a downscaled copy (shrink then grow == a cheap blur).
        # Smaller divisor -> wider, softer halo. Tune on the panel.
        self._bloom_size = (max(1, width // 8), max(1, height // 8))
        # Translucent pulsing halo for LISTENING (the prototype's "aura"). Drawn
        # only while listening, so its per-frame cost is paid only then.
        self._aura = pygame.Surface((width, height), pygame.SRCALPHA)

    def set_state(self, state: AppState) -> None:
        """Called every frame by the ui loop with the live AppState; acts only on
        a real transition. THE WAKING RULE lives here."""
        if state is self.state:
            return
        leaving_idle = self.state is AppState.IDLE
        self.state = state
        self._entered = self.t
        self._shake = 0.0
        if state is AppState.IDLE:
            self._idle_enter = self.t
            return
        if leaving_idle:
            # Drop any idle sub-behaviour (drowsy / yawn / micro / mid-blink) so it
            # can't bleed into the new state, then SNAP the dimming params to the
            # new state's target this very frame. A sleepy or mid-blink face must
            # never mute or soften what comes next — an alarm announcement lands at
            # full brightness with zero ramp. Colour still crossfades (cosmetic).
            self._idle_mode = "normal"
            self._micro = None
            self._blink_until = 0
            tg = self._targets()
            self.p["glow"] = tg["glow"]
            self.p["openness"] = tg["openness"]

    # ---- idle "life": runs only while IDLE (ported from the prototype) -------
    def _idle_tick(self) -> None:
        t, sx, sy = self.t, self._sx, self._sy
        rnd = random.random
        if self._idle_mode == "normal":
            if t - self._idle_enter > DROWSY_AFTER:
                self._idle_mode = "drowsy"          # dim, heavy-lidded, slow blinks
                return
            if t > self._next_yawn:
                self._idle_mode = "yawning"
                self._yawn_start = t
                self._next_yawn = t + YAWN_MIN + int(rnd() * YAWN_RANGE)
                return
            if t > self._next_micro and self._micro is None:
                pick = ("smile", "squint", "stare")[int(rnd() * 3)]
                self._micro = pick
                self._micro_until = t + (STARE_LEN if pick == "stare" else EXPR_LEN)
                if pick == "stare":     # hold a sideways gaze for a beat
                    self._sac = ((-1 if rnd() < 0.5 else 1) * 34 * sx,
                                 (rnd() * 2 - 1) * 8 * sy)
                self._next_micro = t + MICRO_MIN + int(rnd() * MICRO_RANGE)
            if self._micro is not None and t > self._micro_until:
                self._micro = None
            # saccades move the pupils; suppressed mid-stare so the gaze holds
            if t > self._next_saccade and self._micro != "stare":
                self._sac = ((rnd() * 2 - 1) * 24 * sx, (rnd() * 2 - 1) * 10 * sy)
                self._next_saccade = t + 70 + int(rnd() * 120)
            if t > self._next_blink:
                self._blink_until = t + BLINK_LEN
                if rnd() < DOUBLE_BLINK_CHANCE:
                    self._second_blink_at = t + DOUBLE_BLINK_GAP
                self._next_blink = t + 90 + int(rnd() * 150)
        elif self._idle_mode == "drowsy":
            if t > self._next_blink:                # slow, heavy blinks
                self._blink_until = t + DROWSY_BLINK_LEN
                self._next_blink = t + 220 + int(rnd() * 200)
            if t > self._next_saccade:              # pupils settle low and barely move
                self._sac = ((rnd() * 2 - 1) * 8 * sx, (4 + rnd() * 4) * sy)
                self._next_saccade = t + 200 + int(rnd() * 200)
        elif self._idle_mode == "yawning":
            if t - self._yawn_start > YAWN_LEN:
                self._idle_mode = "normal"
                self._second_blink_at = t + 10      # a double blink after the yawn
        # the double-blink follow-up (and the post-yawn blink) fires here
        if self._second_blink_at and t > self._second_blink_at:
            self._blink_until = t + BLINK_LEN
            self._second_blink_at = 0

    # ---- targets per state (ported 1:1 from the prototype's targets()) ------
    def _targets(self) -> dict:
        age = self.t - self._entered
        sx, sy = self._sx, self._sy
        s = self.state
        if s is AppState.IDLE:
            # slow drift layered under the saccades for a "living" gaze
            dx = math.sin(self.t * 0.013 + self._drift1) * 8 * sx
            dy = math.cos(self.t * 0.011 + self._drift2) * 5 * sy
            tg = dict(openness=1.0, scale=1.0, glow=1.0, happy=0.0,
                      px=self._sac[0] + dx, py=self._sac[1] + dy, dil=1.0,
                      mopen=0.0, mcurve=0.25, mshift=0.0,
                      col=PALETTE[AppState.IDLE])
            if self._micro == "smile":
                tg["mcurve"] = 0.6
            elif self._micro == "squint":
                tg["openness"] = 0.78
            # 'stare' is realised purely through the held _sac above.
            if self._idle_mode == "yawning":
                a = (self.t - self._yawn_start) / YAWN_LEN          # 0..1
                if a < 0.35:
                    o = _ease(a / 0.35)
                elif a > 0.65:
                    o = 1 - _ease((a - 0.65) / 0.35)
                else:
                    o = 1.0
                tg["mopen"] = o
                tg["mcurve"] = 0.0
                tg["openness"] = 1 - 0.85 * o      # eyes squeeze shut as mouth opens
                tg["py"] = 6 * sy * o
            if self._idle_mode == "drowsy":
                tg["openness"] = 0.55              # heavy lids (~55%)
                tg["glow"] = 0.5                   # dim to ~50%
                tg["py"] = 6 * sy                  # pupils settle low
                tg["px"] = dx * 0.4
                tg["mcurve"] = 0.1
                tg["col"] = DROWSY_COLOR
            return tg
        if s is AppState.LISTENING:
            return dict(openness=1.15, scale=1 + 0.04 * math.sin(self.t * 0.18),
                        glow=1.5, happy=0.0, px=0.0, py=0.0, dil=1.3,
                        mopen=0.12, mcurve=0.1, mshift=0.0,
                        col=PALETTE[AppState.LISTENING])
        if s is AppState.THINKING:
            return dict(openness=0.8, scale=1.0, glow=0.9, happy=0.0,
                        px=math.cos(self.t * 0.1) * 14 * sx,
                        py=-10 * sy + math.sin(self.t * 0.1) * 4 * sy, dil=0.95,
                        mopen=0.0, mcurve=0.0, mshift=12 * sx,
                        col=PALETTE[AppState.THINKING])
        if s is AppState.SPEAKING:
            lvl = self._level
            return dict(openness=0.95 + lvl * 0.12, scale=1.01, glow=1.35,
                        happy=0.0, px=0.0, py=0.0, dil=1.05,
                        mopen=lvl, mcurve=0.1, mshift=0.0,
                        col=PALETTE[AppState.SPEAKING])
        if s is AppState.CONFIRM:
            return dict(openness=0.6,
                        scale=1 + (0.1 * math.sin(age * 0.5) if age < 14 else 0),
                        glow=2.4 if age < 8 else 1.5, happy=1.0,
                        px=0.0, py=0.0, dil=1.0,
                        mopen=0.0, mcurve=1.0, mshift=0.0,
                        col=PALETTE[AppState.CONFIRM])
        # ERROR
        self._shake = math.sin(age * 0.9) * 6 * sx if age < 26 else 0.0
        return dict(openness=0.7, scale=1.0, glow=1.4, happy=0.0,
                    px=0.0, py=0.0, dil=0.85,
                    mopen=0.0, mcurve=-0.7, mshift=0.0,
                    col=PALETTE[AppState.ERROR])

    def update(self, level: float = 0.0) -> None:
        """Advance one frame. `level` is the live playback level (0..1) the eyes
        read from meta; only SPEAKING uses it (mouth openness). Clamped here so a
        bad meta value can't blow up the geometry."""
        self.t += 1
        self._level = 0.0 if level < 0.0 else (1.0 if level > 1.0 else level)
        if self.state is AppState.IDLE:
            self._idle_tick()
        elif self.state is not AppState.ERROR:
            self._shake = 0.0

        tg = self._targets()
        if self.state is AppState.IDLE and self.t < self._blink_until:
            tg["openness"] = 0.06       # blink overrides whatever idle was doing

        k = 0.5 if self.state is AppState.SPEAKING else 0.22
        self.p["openness"] = _lerp(self.p["openness"], tg["openness"], k)
        self.p["scale"] = _lerp(self.p["scale"], tg["scale"], 0.22)
        self.p["glow"] = _lerp(self.p["glow"], tg["glow"], 0.18)
        self.p["happy"] = _lerp(self.p["happy"], tg["happy"], 0.25)
        self.p["px"] = _lerp(self.p["px"], tg["px"], 0.20)
        self.p["py"] = _lerp(self.p["py"], tg["py"], 0.20)
        self.p["dil"] = _lerp(self.p["dil"], tg["dil"], 0.15)
        # mouth openness tracks the audio: fast ATTACK so a syllable pops the mouth
        # open, slower DECAY so it eases shut in the gaps (and fully closes in the
        # pauses between sentences — that pause behaviour is the proof the level is
        # real, not a constant flap).
        m_k = 0.6 if tg["mopen"] > self.p["mopen"] else 0.35
        self.p["mopen"] = _lerp(self.p["mopen"], tg["mopen"], m_k)
        self.p["mcurve"] = _lerp(self.p["mcurve"], tg["mcurve"], 0.20)
        self.p["mshift"] = _lerp(self.p["mshift"], tg["mshift"], 0.20)
        # colour crossfade — the one knob that makes state changes fade, not cut.
        for i in range(3):
            self._col[i] = _lerp(self._col[i], tg["col"][i], COLOR_LERP)

    # ---- shape tests (same as prototype) ----
    @staticmethod
    def _inside_eye(px, py, cx, cy, hw, hh) -> bool:
        r = min(hw, hh) * 0.55
        qx = abs(px - cx) - (hw - r)
        qy = abs(py - cy) - (hh - r)
        ox, oy = max(qx, 0), max(qy, 0)
        d = math.hypot(ox, oy) + min(max(qx, qy), 0) - r
        return d <= 0

    @staticmethod
    def _inside_happy(px, py, cx, cy, hw, hh) -> bool:
        if abs(px - cx) > hw:
            return False
        nx = (px - cx) / hw
        arch = cy + nx * nx * hh * 1.3 - hh * 0.55
        band = max(6, hh * 0.55)
        return arch <= py <= arch + band

    def _inside_mouth(self, px, py, cx, cy, hw, openness, curve) -> bool:
        nx = (px - cx) / hw
        if abs(nx) > 1:
            return False
        y_c = cy + curve * self._curve_amp * (0.5 - nx * nx)
        half_open = (self.cell * 0.65
                     + openness * self.h * 0.07 * math.sqrt(max(0.0, 1 - nx * nx)))
        return abs(py - y_c) <= half_open

    def _eye_bbox(self, cx: float, cy: float, hw: float, hh: float,
                  full_hh: float) -> tuple[int, int, int, int]:
        """Grid-cell range covering one eye, so we don't scan the whole panel.

        Generous on the vertical: the happy arch dips well below cy, so the box
        must reach down to ~cy + 1.35*full_hh to include it.
        """
        cell = self.cell
        left = cx - hw - cell
        right = cx + hw + cell
        top = cy - max(hh, full_hh * 0.6) - cell
        bot = cy + max(hh, full_hh * 1.35) + cell
        gx0 = max(0, int(left // cell))
        gx1 = min(self.cols, int(right // cell) + 1)
        gy0 = max(0, int(top // cell))
        gy1 = min(self.rows, int(bot // cell) + 1)
        return gx0, gx1, gy0, gy1

    def _mouth_bbox(self, cx: float, cy: float, hw: float,
                    openness: float, curve: float) -> tuple[int, int, int, int]:
        """Grid-cell range covering the mouth band, so the mouth costs ~one row of
        cells per frame instead of a full-grid scan (the eyes and mouth don't
        overlap, so scanning each feature's box is strictly cheaper)."""
        cell = self.cell
        vy = (abs(curve) * self._curve_amp * 0.5      # arc rise/dip of the lip line
              + cell * 0.65 + openness * self.h * 0.07  # max half-openness
              + cell)
        gx0 = max(0, int((cx - hw - cell) // cell))
        gx1 = min(self.cols, int((cx + hw + cell) // cell) + 1)
        gy0 = max(0, int((cy - vy) // cell))
        gy1 = min(self.rows, int((cy + vy) // cell) + 1)
        return gx0, gx1, gy0, gy1

    def draw(self, surf: pygame.Surface) -> None:
        surf.fill((0, 0, 0))
        breathe = 1 + 0.012 * math.sin(self.t * 0.05)
        s = self.p["scale"] * breathe
        hw = self.w * 0.155 * s
        full_hh = self.h * 0.27 * s
        hh = full_hh * max(0.05, self.p["openness"])
        gap = self.w * 0.20
        eye_y = self.h * 0.40                       # eyes up high (v2) to free room
        shake = self._shake
        cxl = self.w * 0.5 - gap + shake
        cxr = self.w * 0.5 + gap + shake
        mouth_y = self.h * 0.78
        mouth_hw = self.w * 0.115 * (1 + self.p["happy"] * 0.25)
        mouth_x = self.w * 0.5 + self.p["mshift"] + shake

        r, g, b = (int(c) for c in self._col)       # live crossfaded colour
        col = (r, g, b)
        # glow < 1 dims the EYES too (not just the bloom): this is what makes the
        # drowsy face actually look dim instead of merely losing its halo.
        dim = min(1.0, self.p["glow"])
        happy = self.p["happy"] > 0.5
        pup_r = hw * 0.40 * self.p["dil"]
        px_off, py_off = self.p["px"], self.p["py"]

        cell = self.cell
        layer = self._eye_layer
        layer.fill((0, 0, 0))

        # --- Eyes (+ pupils) -------------------------------------------------
        for cx in (cxl, cxr):
            gx0, gx1, gy0, gy1 = self._eye_bbox(cx, eye_y, hw, hh, full_hh)
            for gx in range(gx0, gx1):
                for gy in range(gy0, gy1):
                    px, py = gx * cell + cell / 2, gy * cell + cell / 2
                    on = (self._inside_happy(px, py, cx, eye_y, hw, full_hh)
                          if happy else
                          self._inside_eye(px, py, cx, eye_y, hw, hh))
                    if not on:
                        continue
                    # pupil: a dark block (low alpha, not skipped); hidden while the
                    # happy crescent is up. Vertical follow is damped to 0.6. The
                    # pupil keeps its 0.13 alpha even when dim, so it stays readable.
                    in_pupil = (not happy and
                                math.hypot(px - (cx + px_off),
                                           py - (eye_y + py_off * 0.6)) < pup_r)
                    if in_pupil:
                        a = 0.13
                    else:
                        a = (0.82 + 0.18 * (1 - (py - (eye_y - hh)) / (2 * hh))) * dim
                    pygame.draw.rect(layer, _shade(col, a),
                                     (gx * cell, gy * cell, cell - 1, cell - 1),
                                     border_radius=2)

        # --- Mouth -----------------------------------------------------------
        m_open, m_curve = self.p["mopen"], self.p["mcurve"]
        gx0, gx1, gy0, gy1 = self._mouth_bbox(mouth_x, mouth_y, mouth_hw,
                                              m_open, m_curve)
        for gx in range(gx0, gx1):
            for gy in range(gy0, gy1):
                px, py = gx * cell + cell / 2, gy * cell + cell / 2
                if not self._inside_mouth(px, py, mouth_x, mouth_y, mouth_hw,
                                          m_open, m_curve):
                    continue
                # open interior is darker than the lip edge, for depth
                nx = (px - mouth_x) / mouth_hw
                y_c = mouth_y + m_curve * self._curve_amp * (0.5 - nx * nx)
                interior = m_open > 0.12 and abs(py - y_c) < m_open * self.h * 0.05
                a = (0.16 if interior else 0.85) * dim
                pygame.draw.rect(layer, _shade(col, a),
                                 (gx * cell, gy * cell, cell - 1, cell - 1),
                                 border_radius=2)

        # --- Listening aura (behind everything) ------------------------------
        # The prototype's soft pulsing halo, in the live colour. Only paid for
        # while listening; sits under the bloom + crisp eyes.
        if self.state is AppState.LISTENING:
            aura = 0.5 + 0.5 * math.sin(self.t * 0.18)
            a = max(0, min(255, int(255 * (0.10 + 0.10 * aura))))
            self._aura.fill((0, 0, 0, 0))
            for cx in (cxl, cxr):
                rw, rh = hw * (1.7 + 0.3 * aura), hh * (1.5 + 0.3 * aura)
                rect = pygame.Rect(0, 0, int(rw * 2), int(rh * 2))
                rect.center = (int(cx), int(eye_y))
                pygame.draw.ellipse(self._aura, (r, g, b, a), rect)
            surf.blit(self._aura, (0, 0))

        # Glow: shrink the eye layer then grow it back == a cheap blur (no
        # per-pixel work), scaled by the state's glow and blended additively so
        # the eyes "light up". Crisp eyes go on top (black is keyed out, so the
        # halo behind them survives).
        g_glow = self.p["glow"]
        if g_glow > 0.01:
            small = pygame.transform.smoothscale(layer, self._bloom_size)
            bloom = pygame.transform.smoothscale(small, (self.w, self.h))
            m = max(0, min(255, int(80 * g_glow)))
            bloom.fill((m, m, m), special_flags=pygame.BLEND_RGB_MULT)
            surf.blit(bloom, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        surf.blit(layer, (0, 0))
