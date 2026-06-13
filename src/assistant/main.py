"""Entry point: pygame render loop on the main thread, asyncio on a background thread."""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading

import pygame

from assistant.bus import EventBus
from assistant.config import Config, load_config
from assistant.face.module import Face
from assistant.perception.module import Perception
from assistant.voice.module import Voice

log = logging.getLogger("assistant")

FPS = 60
HEARTBEAT_INTERVAL_S = 5.0
HIGH_RATE_EVENTS = frozenset({"gaze", "face_gaze"})  # logged at DEBUG to avoid spam


class AsyncRuntime:
    """Owns the asyncio loop on a background thread; the render loop never blocks on it."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._loop = asyncio.new_event_loop()
        self._stop = asyncio.Event()
        self._thread = threading.Thread(target=self._run, name="assistant-async", daemon=True)
        bus.attach_loop(self._loop)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        tasks = [
            asyncio.create_task(self._heartbeat(), name="heartbeat"),
            asyncio.create_task(self._log_events(), name="bus-logger"),
            asyncio.create_task(self._placeholder_reflexes(), name="placeholder-reflexes"),
        ]
        await self._stop.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _heartbeat(self) -> None:
        while True:
            self._bus.emit("heartbeat")
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    async def _log_events(self) -> None:
        async for event in self._bus.subscribe():
            level = logging.DEBUG if event.type in HIGH_RATE_EVENTS else logging.INFO
            log.log(level, "bus: %s ts=%.3f payload=%s", event.type, event.ts, event.payload)

    async def _placeholder_reflexes(self) -> None:
        """Stand-in for the brain (slice 4): perception drives the face directly,
        and heard speech is echoed back so the voice loop is testable end to end."""
        events = self._bus.subscribe(
            "person_appeared", "person_left", "gaze", "speech_heard", "speaking_finished"
        )
        async for event in events:
            if event.type == "person_appeared":
                self._bus.emit("face_state", {"state": "alert"})
            elif event.type == "person_left":
                self._bus.emit("face_state", {"state": "drowsy"})
            elif event.type == "gaze":
                self._bus.emit("face_gaze", dict(event.payload))
            elif event.type == "speech_heard":
                self._bus.emit("say", {"text": f"You said: {event.payload.get('text', '')}"})
            elif event.type == "speaking_finished":
                self._bus.emit("face_state", {"state": "neutral"})


def _render_loop(config: Config, bus: EventBus) -> None:
    pygame.init()
    flags = pygame.FULLSCREEN if config.display.fullscreen else 0
    screen = pygame.display.set_mode((config.display.width, config.display.height), flags)
    pygame.display.set_caption("assistant")
    clock = pygame.time.Clock()
    face = Face(config, bus)

    running = True
    while running:
        for pg_event in pygame.event.get():
            if pg_event.type == pygame.QUIT:
                running = False
            else:
                face.handle_event(pg_event)
        face.update()
        face.render(screen)
        pygame.display.flip()
        clock.tick(FPS)


def run() -> None:
    parser = argparse.ArgumentParser(prog="assistant")
    parser.add_argument(
        "--show", action="store_true", help="open the OpenCV perception debug window"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config()
    log.info(
        "starting profile=%s display=%dx%d fullscreen=%s",
        config.profile,
        config.display.width,
        config.display.height,
        config.display.fullscreen,
    )

    bus = EventBus()
    runtime = AsyncRuntime(bus)
    perception = Perception(config, bus, show_debug=True if args.show else None)
    voice = Voice(config, bus)
    runtime.start()
    perception.start()
    voice.start()
    try:
        _render_loop(config, bus)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        voice.stop()
        perception.stop()
        runtime.stop()
        pygame.quit()
        log.info("clean shutdown")


if __name__ == "__main__":
    run()
