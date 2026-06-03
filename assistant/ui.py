"""pygame main loop. Owns the screen and input; reads SharedState; never does logic.

v1 trigger: a tap anywhere starts the pipeline (wake word comes later). The loop
calls `on_tap()` which should kick the pipeline off in a worker thread and return
immediately — do NOT block the render loop.
"""
from __future__ import annotations

import os
from typing import Callable

import pygame

from .eyes import Eyes
from .state import SharedState

# 4.3" DSI panel native resolution
SCREEN_W, SCREEN_H = 800, 480


def run_ui(shared: SharedState, on_tap: Callable[[], None],
           fullscreen: bool = True, fps: int = 60) -> None:
    pygame.init()
    pygame.mouse.set_visible(False)
    flags = pygame.FULLSCREEN | pygame.SCALED if fullscreen else pygame.SCALED
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
    pygame.display.set_caption("Desk Assistant")
    clock = pygame.time.Clock()
    eyes = Eyes(SCREEN_W, SCREEN_H)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            # touch (FINGERDOWN) or mouse click both = tap-to-listen
            elif ev.type in (pygame.FINGERDOWN, pygame.MOUSEBUTTONDOWN):
                on_tap()

        eyes.set_state(shared.state)
        eyes.update()
        eyes.draw(screen)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


# Allows `SDL_VIDEODRIVER=dummy python -m assistant.ui` for a headless smoke test.
if __name__ == "__main__":
    import time
    from .state import AppState

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    shared = SharedState()

    def fake_tap() -> None:
        shared.set(AppState.LISTENING)

    # cycle through states so you can eyeball transitions headlessly
    import threading

    def cycler() -> None:
        seq = [AppState.IDLE, AppState.LISTENING, AppState.THINKING,
               AppState.SPEAKING, AppState.CONFIRM, AppState.ERROR]
        for st in seq:
            shared.set(st)
            time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=cycler, daemon=True).start()
    run_ui(shared, fake_tap, fullscreen=False)
