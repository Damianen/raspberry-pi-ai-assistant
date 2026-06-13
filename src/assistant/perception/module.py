"""The Perception module: webcam frames in, presence/identity/gaze bus events out.

Owns a dedicated worker thread because camera reads block. YuNet detection
runs on every Nth captured frame; SFace recognition feeds a majority vote per
identity decision. Publishing is thread-safe through the bus. A perception
failure only disables this module — the face never freezes because of it.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from assistant.bus import EventBus
from assistant.config import Config
from assistant.memory.db import connect
from assistant.memory.people import PeopleStore
from assistant.perception.camera import open_capture
from assistant.perception.identity import Decision, IdentityTracker, best_match
from assistant.perception.logic import (
    PERSON_APPEARED,
    PERSON_LEFT,
    GazeThrottle,
    PresenceTracker,
    clamp01,
)
from assistant.perception.vision import FaceFinder, create_recognizer, embed_face

log = logging.getLogger("assistant.perception")

# Internal resilience plumbing — behavioral tunables live in config.
READ_FAILURE_SLEEP_S = 0.1
READ_FAILURES_BEFORE_REOPEN = 50
DEBUG_WINDOW = "assistant perception"
DEBUG_BOX_COLOR = (0, 255, 0)  # BGR


class Perception:
    def __init__(self, config: Config, bus: EventBus, *, show_debug: bool | None = None) -> None:
        self._config = config
        self._pcfg = config.perception
        self._bus = bus
        self._show = self._pcfg.show_debug if show_debug is None else show_debug
        self._presence = PresenceTracker(self._pcfg.presence_frames, self._pcfg.absence_timeout_s)
        self._gaze_throttle = GazeThrottle(self._pcfg.gaze_throttle_ms / 1000.0)
        self._identity = IdentityTracker(self._pcfg.vote_samples, self._pcfg.reverify_interval_s)
        self._people: list[tuple[str, np.ndarray]] = []
        self._people_stale = True  # reload enrolled people before the next vote
        self._last_face: np.ndarray | None = None
        self._last_decision: Decision | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="assistant-perception", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    # --- worker thread -----------------------------------------------------

    def _run(self) -> None:
        # Everything stateful (camera, models, sqlite) lives on this thread.
        try:
            self._finder = FaceFinder(
                self._pcfg, (self._config.camera.width, self._config.camera.height)
            )
            self._recognizer = create_recognizer()
            self._store = PeopleStore(connect())
            self._cap = open_capture(self._config.camera)
        except Exception:
            log.exception("perception setup failed; module disabled")
            return
        try:
            self._loop()
        except Exception:
            log.exception("perception crashed; module disabled")
        finally:
            self._cap.release()
            if self._show:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

    def _loop(self) -> None:
        failures = 0
        frame_index = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                failures += 1
                # The absence timeout must keep running while the camera is
                # down, or a present person could never "leave".
                self._publish_presence(self._presence.update(False, now))
                if failures >= READ_FAILURES_BEFORE_REOPEN:
                    failures = 0
                    log.warning("camera reads failing; reopening")
                    self._cap.release()
                    try:
                        self._cap = open_capture(self._config.camera)
                    except Exception:
                        log.exception("camera reopen failed; retrying")
                time.sleep(READ_FAILURE_SLEEP_S)
                continue
            failures = 0
            frame_index += 1
            if frame_index % self._pcfg.detect_every_n_frames == 0:
                self._process(frame, now)
            if self._show:
                self._draw_debug(frame)

    def _process(self, frame: np.ndarray, now: float) -> None:
        face = self._finder.largest(frame)
        self._last_face = face
        self._publish_presence(self._presence.update(face is not None, now))
        if face is None:
            return
        self._publish_gaze(face, frame.shape, now)
        if self._presence.present and self._identity.sampling(now):
            self._sample_identity(frame, face, now)

    def _publish_presence(self, transition: str | None) -> None:
        if transition is None:
            return
        log.info("presence: %s", transition)
        self._bus.emit(transition)
        if transition == PERSON_LEFT:
            self._identity.reset()
            self._last_decision = None
            self._people_stale = True

    def _publish_gaze(self, face: np.ndarray, frame_shape: tuple[int, ...], now: float) -> None:
        if not self._gaze_throttle.ready(now):
            return
        height, width = frame_shape[:2]
        x = (float(face[0]) + float(face[2]) / 2.0) / max(width, 1)
        y = (float(face[1]) + float(face[3]) / 2.0) / max(height, 1)
        if self._pcfg.gaze_mirror:
            x = 1.0 - x
        self._bus.emit("gaze", {"x": clamp01(x), "y": clamp01(y)})

    def _sample_identity(self, frame: np.ndarray, face: np.ndarray, now: float) -> None:
        if self._people_stale:
            # Refreshed once per vote window so a new enrollment is picked up
            # without restarting the app.
            self._people = self._store.all_embeddings()
            self._people_stale = False
            log.info("identity: matching against %d enrolled people", len(self._people))
        if self._people:
            embedding = embed_face(self._recognizer, frame, face)
            match = best_match(embedding, self._people, self._pcfg.match_threshold)
        else:
            match = None
        name, score = match if match is not None else (None, 0.0)
        decision = self._identity.observe(name, score, now)
        if decision is None:
            return
        self._last_decision = decision
        self._people_stale = True
        if decision.name is None:
            log.info("identity vote: unknown face")
            self._bus.emit("face_unknown")
        else:
            log.info("identity vote: %s (score %.3f)", decision.name, decision.score)
            self._bus.emit("face_recognized", {"name": decision.name, "score": decision.score})
            self._store.touch_last_seen(decision.name)

    def _draw_debug(self, frame: np.ndarray) -> None:
        try:
            if self._last_face is not None:
                x, y, w, h = (int(v) for v in self._last_face[:4])
                cv2.rectangle(frame, (x, y), (x + w, y + h), DEBUG_BOX_COLOR, 2)
                if self._last_decision is None:
                    label = "..."
                else:
                    label = self._last_decision.name or "unknown"
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    DEBUG_BOX_COLOR,
                    2,
                )
            cv2.imshow(DEBUG_WINDOW, frame)
            cv2.waitKey(1)
        except cv2.error:
            log.exception("debug window failed; disabling --show")
            self._show = False
