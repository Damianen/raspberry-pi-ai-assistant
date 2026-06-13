"""The Face module: consumes face_state / face_gaze commands, renders every frame.

Runs entirely on the pygame main thread. Commands arrive through a
thread-safe bus Inbox; debug input is published back through the bus so the
production command path is exercised even in development.
"""

from __future__ import annotations

import logging
import random
import time

import pygame

from assistant.bus import EventBus
from assistant.config import Config
from assistant.face.logic import (
    BlinkScheduler,
    FaceFrame,
    GazeController,
    clamp01,
    compute_frame,
)
from assistant.face.render import draw_face
from assistant.face.styles import FACE_STATES, STYLES, StyleBlender

log = logging.getLogger("assistant.face")

INITIAL_STATE = "neutral"
DEBUG_STATE_KEYS = "123456789"  # matched against event.unicode, layout-proof


class Face:
    def __init__(self, config: Config, bus: EventBus, *, rng: random.Random | None = None) -> None:
        self._bus = bus
        self._cfg = config.face
        self._size = (config.display.width, config.display.height)
        self._inbox = bus.open_inbox("face_state", "face_gaze")
        now = time.monotonic()
        rng = rng or random.Random()
        self._blink = BlinkScheduler(self._cfg.blink, rng, now)
        self._gaze = GazeController(self._cfg.gaze, rng, now)
        self._blender = StyleBlender(INITIAL_STATE, self._cfg.transition_ms / 1000.0, now)
        self._gaze_publish_interval = 1.0 / self._cfg.debug_gaze_hz
        self._last_gaze_publish = 0.0
        self._caption_state: str | None = None
        self._frame: FaceFrame
        self.update()

    @property
    def state(self) -> str:
        return self._blender.state

    def handle_event(self, event: pygame.event.Event) -> None:
        """Debug controls: 1-9 set states, mouse sets gaze, B forces a blink."""
        if not self._cfg.debug_controls:
            return
        if event.type == pygame.KEYDOWN:
            index = self._state_key_index(event)
            if index is not None:
                self._bus.emit("face_state", {"state": FACE_STATES[index]})
            elif event.key == pygame.K_b:
                self._blink.force(time.monotonic())
        elif event.type == pygame.MOUSEMOTION:
            now = time.monotonic()
            if now - self._last_gaze_publish >= self._gaze_publish_interval:
                self._last_gaze_publish = now
                x = event.pos[0] / max(self._size[0] - 1, 1)
                y = event.pos[1] / max(self._size[1] - 1, 1)
                self._bus.emit("face_gaze", {"x": clamp01(x), "y": clamp01(y)})

    @staticmethod
    def _state_key_index(event: pygame.event.Event) -> int | None:
        # event.unicode respects the keyboard layout; keysyms are the fallback.
        if event.unicode and event.unicode in DEBUG_STATE_KEYS:
            return int(event.unicode) - 1
        if pygame.K_1 <= event.key <= pygame.K_9:
            return event.key - pygame.K_1
        return None

    def update(self) -> None:
        """Advance one frame: drain commands, blend the style, step blink and gaze."""
        now = time.monotonic()
        for event in self._inbox.drain():
            if event.type == "face_state":
                self._apply_state(event.payload.get("state"), now)
            elif event.type == "face_gaze":
                self._apply_gaze(event.payload, now)
        style = self._blender.current(now)
        self._blink.set_tempo(style.blink_interval_scale, style.blink_speed_scale)
        blink = self._blink.update(now, enabled=self.state != "sleeping")
        gaze = self._gaze.update(now)
        self._frame = compute_frame(style, blink, gaze, now, self._cfg, self._size)

    def _apply_state(self, state: object, now: float) -> None:
        if isinstance(state, str) and state in STYLES:
            self._blender.set_state(state, now)
        else:
            log.warning("ignoring unknown face state: %r", state)

    def _apply_gaze(self, payload: dict[str, object], now: float) -> None:
        try:
            x = float(payload["x"])  # type: ignore[arg-type]
            y = float(payload["y"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            log.warning("ignoring malformed face_gaze payload: %r", payload)
            return
        self._gaze.set_target(x, y, now)

    def render(self, screen: pygame.Surface) -> None:
        draw_face(screen, self._frame)
        if self._cfg.debug_controls and self.state != self._caption_state:
            self._caption_state = self.state
            pygame.display.set_caption(f"assistant — {self.state}")
