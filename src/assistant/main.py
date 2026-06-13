"""Entry point: pygame render loop on the main thread, asyncio on a background thread."""

from __future__ import annotations

import asyncio
import logging
import threading

import pygame

from assistant.bus import EventBus
from assistant.config import Config, load_config
from assistant.face.module import Face

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
    runtime.start()
    try:
        _render_loop(config, bus)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        runtime.stop()
        pygame.quit()
        log.info("clean shutdown")


if __name__ == "__main__":
    run()
