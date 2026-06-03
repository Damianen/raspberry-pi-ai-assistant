"""Pixel-eye renderer for pygame. Ported from the approved HTML prototype.

It READS state and draws. It contains no logic. Same parameter model as the
prototype: openness / scale / look offset / glow / happy / warn, interpolated
toward per-state targets each frame.

NOTE: tune CELL, colours and proportions on the actual 800x480 DSI panel — the
numbers below are a sensible starting point, not gospel.
"""
from __future__ import annotations

import math
import random

import pygame

from .state import AppState

EYE = (95, 243, 232)
WARN = (255, 90, 77)


def _lerp(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


class Eyes:
    def __init__(self, width: int, height: int, cell: int = 12) -> None:
        self.w, self.h, self.cell = width, height, cell
        self.cols, self.rows = width // cell, height // cell
        self.t = 0
        self.state = AppState.IDLE
        self._entered = 0
        self._blink_until = 0
        self._next_blink = 60
        self._next_saccade = 60
        self._sac = (0.0, 0.0)
        self._shake = 0.0
        # current interpolated params
        self.p = dict(openness=1.0, scale=1.0, offx=0.0, offy=0.0,
                      glow=1.0, happy=0.0, warn=0.0)
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

    # ---- targets per state ----
    def _targets(self) -> dict:
        age = self.t - self._entered
        s = self.state
        if s is AppState.IDLE:
            return dict(openness=1.0, scale=1.0, offx=self._sac[0], offy=self._sac[1],
                        glow=1.0, happy=0.0, warn=0.0)
        if s is AppState.LISTENING:
            return dict(openness=1.18, scale=1 + 0.05 * math.sin(self.t * 0.18),
                        offx=0.0, offy=-1.0, glow=1.5, happy=0.0, warn=0.0)
        if s is AppState.THINKING:
            return dict(openness=0.78, scale=1.0,
                        offx=math.cos(self.t * 0.10) * 16,
                        offy=-12 + math.sin(self.t * 0.10) * 6,
                        glow=0.9, happy=0.0, warn=0.0)
        if s is AppState.SPEAKING:
            flutter = 0.85 + 0.32 * abs(math.sin(self.t * 0.42)) * (0.6 + 0.4 * random.random())
            return dict(openness=flutter, scale=1.02, offx=0.0,
                        offy=math.sin(self.t * 0.42) * 4, glow=1.4, happy=0.0, warn=0.0)
        if s is AppState.CONFIRM:
            return dict(openness=0.6, scale=1 + (0.10 * math.sin(age * 0.5) if age < 14 else 0),
                        offx=0.0, offy=0.0, glow=2.4 if age < 8 else 1.5,
                        happy=1.0, warn=0.0)
        # ERROR
        self._shake = math.sin(age * 0.9) * 6 if age < 26 else 0.0
        return dict(openness=0.7, scale=1.0, offx=0.0, offy=0.0,
                    glow=1.4, happy=0.0, warn=1.0)

    def update(self) -> None:
        self.t += 1
        if self.state is AppState.IDLE:
            if self.t > self._next_blink:
                self._blink_until = self.t + 6
                self._next_blink = self.t + 90 + int(random.random() * 150)
            if self.t > self._next_saccade:
                self._sac = ((random.random() * 2 - 1) * 22, (random.random() * 2 - 1) * 10)
                self._next_saccade = self.t + 70 + int(random.random() * 120)
        else:
            self._shake = self._shake if self.state is AppState.ERROR else 0.0

        tg = self._targets()
        if self.state is AppState.IDLE and self.t < self._blink_until:
            tg["openness"] = 0.06

        k = 0.55 if self.state is AppState.SPEAKING else 0.22
        self.p["openness"] = _lerp(self.p["openness"], tg["openness"], k)
        self.p["scale"] = _lerp(self.p["scale"], tg["scale"], 0.22)
        self.p["offx"] = _lerp(self.p["offx"], tg["offx"], 0.18)
        self.p["offy"] = _lerp(self.p["offy"], tg["offy"], 0.18)
        self.p["glow"] = _lerp(self.p["glow"], tg["glow"], 0.18)
        self.p["happy"] = _lerp(self.p["happy"], tg["happy"], 0.25)
        self.p["warn"] = _lerp(self.p["warn"], tg["warn"], 0.20)

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

    def draw(self, surf: pygame.Surface) -> None:
        surf.fill((0, 0, 0))
        breathe = 1 + 0.012 * math.sin(self.t * 0.05)
        s = self.p["scale"] * breathe
        base_hw = self.w * 0.165 * s
        base_hh = self.h * 0.30 * s * max(0.05, self.p["openness"])
        gap = self.w * 0.21
        cy = self.h * 0.5 + self.p["offy"]
        cxl = self.w * 0.5 - gap + self.p["offx"] + self._shake
        cxr = self.w * 0.5 + gap + self.p["offx"] + self._shake

        col = tuple(int(_lerp(EYE[i], WARN[i], self.p["warn"])) for i in range(3))
        happy = self.p["happy"] > 0.5
        full_hh = self.h * 0.30 * s

        # Draw the crisp eyes onto the offscreen layer (only the cells each eye
        # actually covers — not the whole grid).
        cell = self.cell
        layer = self._eye_layer
        layer.fill((0, 0, 0))
        for cx in (cxl, cxr):
            gx0, gx1, gy0, gy1 = self._eye_bbox(cx, cy, base_hw, base_hh, full_hh)
            for gx in range(gx0, gx1):
                for gy in range(gy0, gy1):
                    px, py = gx * cell + cell / 2, gy * cell + cell / 2
                    on = (self._inside_happy(px, py, cx, cy, base_hw, full_hh)
                          if happy else
                          self._inside_eye(px, py, cx, cy, base_hw, base_hh))
                    if on:
                        pygame.draw.rect(
                            layer, col,
                            (gx * cell, gy * cell, cell - 1, cell - 1),
                            border_radius=2,
                        )

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
