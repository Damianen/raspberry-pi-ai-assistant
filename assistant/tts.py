"""Text-to-speech. IMPLEMENT + TEST ON THE PI.

Interface:
    speak(text: str) -> None     # synthesize + play, blocking

Notes:
- Use Piper. Pick a voice you like (e.g. en_US/en_GB medium). Pre-load the model.
- Caller sets AppState.SPEAKING before calling and IDLE after.
- Later: stream sentence-by-sentence so long LLM answers start speaking sooner.
"""
from __future__ import annotations

import time


def speak(text: str) -> None:
    # SLICE-3 PLACEHOLDER — NOT Piper. Real speak() will synthesize and block for
    # the duration of speech; that blocking is what holds CONFIRM/SPEAKING on the
    # eyes. The print proves the flow end-to-end; the sleep simulates that
    # blocking so the eye states actually dwell during testing.
    # TODO(slice4): delete the sleep and synthesize with Piper.
    print(f"TTS(stub): {text}")
    time.sleep(min(4.0, max(1.5, len(text) / 14)))  # ~14 chars/sec speaking rate
