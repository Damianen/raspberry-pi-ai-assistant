"""Pixel-eye renderer for pygame. Ported from the approved HTML prototype (v2).

It READS state and draws. It contains no logic. Same parameter model as the
prototype: per-state targets for openness / scale / glow / happy / warn, plus the
v2 additions — pupil offset + dilation and a mouth (openness / curve / side-shift)
— each interpolated toward its target every frame.

v2 adds one external input: the live audio level (0..1) from playback, read from
the snapshot's meta and passed into update(). The eyes only SMOOTH and DRAW that
number (mouth openness during SPEAKING). They never compute audio — tts produces
the level, the eyes consume it. That one-way flow is the whole point of the slice.

NOTE: tune CELL, colours and proportions on the actual 800x480 DSI panel — the
numbers below come straight from the 500x300 prototype (absolute pixel offsets are
scaled by sx=w/500, sy=h/300 so the look is resolution-independent), but they're a
starting point, not gospel.
"""
from __future__ import annotations

import math
import random

import pygame

from .state import AppState

EYE = (95, 243, 232)
WARN = (255, 90, 77)

# The prototype canvas the absolute pixel offsets (saccade range, thinking drift,
# mouth shift, mouth-curve depth) were tuned on. We scale them to the real panel.
_PROTO_W, _PROTO_H = 500.0, 300.0


def _lerp(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


def _shade(col: tuple[int, int, int], a: float) -> tuple[int, int, int]:
    """Premultiply a colour by alpha against the black background.

    The prototype draws each cell with an rgba alpha (eye gradient ~0.82..1.0,
    pupil 0.13, mouth lip 0.85, interior 0.16). The eye layer is composited onto
    black with black colour-keyed transparent, so premultiplying == alpha-blending
    onto black, and the dimmer cells also dim the bloom for free. The minimum alpha
    (0.13) keeps even the pupil above (0,0,0), so it survives the colour-key."""
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
        self._blink_until = 0
        self._next_blink = 60
        self._next_saccade = 60
        self._sac = (0.0, 0.0)      # current idle pupil target (saccade)
        self._shake = 0.0
        # current interpolated params (superset of the prototype's CUR object)
        self.p = dict(openness=1.0, scale=1.0, glow=1.0, happy=0.0, warn=0.0,
                      px=0.0, py=0.0, dil=1.0,             # pupil offset + dilation
                      mopen=0.0, mcurve=0.25, mshift=0.0)  # mouth
        # Offscreen layer for the crisp eyes. Black is colour-keyed transparent
        # so the bloom halo behind the eyes survives the final composite.
        self._eye_layer = pygame.Surface((width, height))
        self._eye_layer.set_colorkey((0, 0, 0))
        # Bloom works on a downscaled copy (shrink then grow == a cheap blur).
        # Smaller divisor -> wider, softer halo. Tune on the panel.
        self._bloom_size = (max(1, width // 8), max(1, height // 8))

    def set_state(self, state: AppState) -> None:
        if state is not self.state:
            self.state = state
            self._entered = self.t

    # ---- targets per state (ported 1:1 from the prototype's targets()) ----
    def _targets(self) -> dict:
        age = self.t - self._entered
        sx, sy = self._sx, self._sy
        s = self.state
        if s is AppState.IDLE:
            return dict(openness=1.0, scale=1.0, glow=1.0, happy=0.0, warn=0.0,
                        px=self._sac[0], py=self._sac[1], dil=1.0,
                        mopen=0.0, mcurve=0.25, mshift=0.0)
        if s is AppState.LISTENING:
            return dict(openness=1.15, scale=1 + 0.04 * math.sin(self.t * 0.18),
                        glow=1.5, happy=0.0, warn=0.0,
                        px=0.0, py=0.0, dil=1.3,
                        mopen=0.12, mcurve=0.1, mshift=0.0)
        if s is AppState.THINKING:
            return dict(openness=0.8, scale=1.0, glow=0.9, happy=0.0, warn=0.0,
                        px=math.cos(self.t * 0.1) * 14 * sx,
                        py=-10 * sy + math.sin(self.t * 0.1) * 4 * sy, dil=0.95,
                        mopen=0.0, mcurve=0.0, mshift=12 * sx)
        if s is AppState.SPEAKING:
            lvl = self._level
            return dict(openness=0.95 + lvl * 0.12, scale=1.01, glow=1.35,
                        happy=0.0, warn=0.0, px=0.0, py=0.0, dil=1.05,
                        mopen=lvl, mcurve=0.1, mshift=0.0)
        if s is AppState.CONFIRM:
            return dict(openness=0.6,
                        scale=1 + (0.1 * math.sin(age * 0.5) if age < 14 else 0),
                        glow=2.4 if age < 8 else 1.5, happy=1.0, warn=0.0,
                        px=0.0, py=0.0, dil=1.0,
                        mopen=0.0, mcurve=1.0, mshift=0.0)
        # ERROR
        self._shake = math.sin(age * 0.9) * 6 * sx if age < 26 else 0.0
        return dict(openness=0.7, scale=1.0, glow=1.4, happy=0.0, warn=1.0,
                    px=0.0, py=0.0, dil=0.85,
                    mopen=0.0, mcurve=-0.7, mshift=0.0)

    def update(self, level: float = 0.0) -> None:
        """Advance one frame. `level` is the live playback level (0..1) the eyes
        read from meta; only SPEAKING uses it (mouth openness). Clamped here so a
        bad meta value can't blow up the geometry."""
        self.t += 1
        self._level = 0.0 if level < 0.0 else (1.0 if level > 1.0 else level)
        if self.state is AppState.IDLE:
            if self.t > self._next_blink:
                self._blink_until = self.t + 6
                self._next_blink = self.t + 90 + int(random.random() * 150)
            if self.t > self._next_saccade:
                # saccades now move the PUPILS, not the whole eye (v2)
                self._sac = ((random.random() * 2 - 1) * 24 * self._sx,
                             (random.random() * 2 - 1) * 10 * self._sy)
                self._next_saccade = self.t + 70 + int(random.random() * 120)
        elif self.state is not AppState.ERROR:
            self._shake = 0.0

        tg = self._targets()
        if self.state is AppState.IDLE and self.t < self._blink_until:
            tg["openness"] = 0.06

        k = 0.5 if self.state is AppState.SPEAKING else 0.22
        self.p["openness"] = _lerp(self.p["openness"], tg["openness"], k)
        self.p["scale"] = _lerp(self.p["scale"], tg["scale"], 0.22)
        self.p["glow"] = _lerp(self.p["glow"], tg["glow"], 0.18)
        self.p["happy"] = _lerp(self.p["happy"], tg["happy"], 0.25)
        self.p["warn"] = _lerp(self.p["warn"], tg["warn"], 0.20)
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

        col = tuple(int(_lerp(EYE[i], WARN[i], self.p["warn"])) for i in range(3))
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
                    # happy crescent is up. Vertical follow is damped to 0.6.
                    in_pupil = (not happy and
                                math.hypot(px - (cx + px_off),
                                           py - (eye_y + py_off * 0.6)) < pup_r)
                    if in_pupil:
                        a = 0.13
                    else:
                        a = 0.82 + 0.18 * (1 - (py - (eye_y - hh)) / (2 * hh))
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
                a = 0.16 if interior else 0.85
                pygame.draw.rect(layer, _shade(col, a),
                                 (gx * cell, gy * cell, cell - 1, cell - 1),
                                 border_radius=2)

        # Glow: shrink the eye layer then grow it back == a cheap blur (no
        # per-pixel work), scaled by the state's glow and blended additively so
        # the eyes "light up". Crisp eyes go on top (black is keyed out, so the
        # halo behind them survives).
        g = self.p["glow"]
        if g > 0.01:
            small = pygame.transform.smoothscale(layer, self._bloom_size)
            bloom = pygame.transform.smoothscale(small, (self.w, self.h))
            m = max(0, min(255, int(80 * g)))
            bloom.fill((m, m, m), special_flags=pygame.BLEND_RGB_MULT)
            surf.blit(bloom, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        surf.blit(layer, (0, 0))
