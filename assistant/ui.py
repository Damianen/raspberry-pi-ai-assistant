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


def dispatch_event(ev: pygame.event.Event, on_tap: Callable[[], None],
                   on_key: Callable[[int], None] | None) -> bool:
    """Route one pygame event to the right callback. Pure input routing — no
    drawing, no app logic. Returns False if the app should quit.

    A real finger tap is the trigger. SDL also synthesizes a MOUSEBUTTONDOWN for
    every touch (ev.touch == True); we ignore those so one physical tap fires
    once, while real mouse clicks (desktop dev) still work.
    """
    if ev.type == pygame.QUIT:
        return False
    if ev.type == pygame.KEYDOWN:
        if ev.key == pygame.K_ESCAPE:
            return False
        if on_key is not None:
            on_key(ev.key)
    elif ev.type == pygame.FINGERDOWN:
        on_tap()
    elif ev.type == pygame.MOUSEBUTTONDOWN and not getattr(ev, "touch", False):
        on_tap()
    return True


def run_ui(shared: SharedState, on_tap: Callable[[], None],
           fullscreen: bool = True, fps: int = 60,
           on_key: Callable[[int], None] | None = None) -> None:
    # The app does ALL audio via sounddevice (record + beep) and Piper (TTS) —
    # pygame needs no audio. Force SDL's dummy audio driver so SDL doesn't open
    # and HOLD the output device. On the Pi the case speaker is single-client
    # HDMI0 audio (no dmix); if SDL grabs it at init, sounddevice can't open it
    # when an alarm fires (PaError "querying device" / default resolves to -1).
    # Must be set before pygame.init().
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
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
            if not dispatch_event(ev, on_tap, on_key):
                running = False

        eyes.set_state(shared.state)
        eyes.update()
        eyes.draw(screen)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


# `python -m assistant.ui` — show the eyes on the panel and cycle states by
# tapping (or number keys 1-6 / SPACE). For a headless CI smoke test instead:
#   SDL_VIDEODRIVER=dummy python -m assistant.ui   (auto-cycles once, then exits)
# Pass --windowed to run in a window on a dev desktop instead of fullscreen.
if __name__ == "__main__":
    import sys
    from .state import AppState

    SEQ = [AppState.IDLE, AppState.LISTENING, AppState.THINKING,
           AppState.SPEAKING, AppState.CONFIRM, AppState.ERROR]
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    shared = SharedState()

    if headless:
        # CI: cycle through every state once so a crash in any of them shows up,
        # then exit. No window, no input.
        import threading
        import time

        def cycler() -> None:
            for st in SEQ:
                shared.set(st)
                time.sleep(0.4)
            os._exit(0)

        threading.Thread(target=cycler, daemon=True).start()
        run_ui(shared, lambda: None, fullscreen=False)
    else:
        # On the panel: tap (or SPACE) advances to the next state; number keys
        # 1-6 jump straight to one. ESC quits. This stands in for pipeline.on_tap
        # so we can eyeball every state on the real display before wiring audio.
        def advance() -> None:
            cur = shared.state
            shared.set(SEQ[(SEQ.index(cur) + 1) % len(SEQ)])

        def on_key(key: int) -> None:
            if key == pygame.K_SPACE:
                advance()
            elif pygame.K_1 <= key <= pygame.K_6:
                shared.set(SEQ[key - pygame.K_1])

        fullscreen = "--windowed" not in sys.argv
        run_ui(shared, advance, fullscreen=fullscreen, on_key=on_key)
