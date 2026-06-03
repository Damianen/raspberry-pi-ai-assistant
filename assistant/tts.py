"""Text-to-speech. IMPLEMENT + TEST ON THE PI.

Interface:
    speak(text: str) -> None     # synthesize + play, blocking

Notes:
- Use Piper. Pick a voice you like (e.g. en_US/en_GB medium). Pre-load the model.
- Caller sets AppState.SPEAKING before calling and IDLE after.
- Later: stream sentence-by-sentence so long LLM answers start speaking sooner.
"""
from __future__ import annotations


def speak(text: str) -> None:
    raise NotImplementedError("Implement with Piper on the Pi.")
