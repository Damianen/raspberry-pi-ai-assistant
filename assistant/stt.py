"""Speech-to-text. IMPLEMENT + TEST ON THE PI.

Interface:
    transcribe(audio: bytes) -> str

Notes:
- Use whisper.cpp via `pywhispercpp`, model `base.en` (good speed/accuracy on
  Pi 5; `tiny.en` if too slow, `small.en` if accuracy is poor and you can spare
  the time). Keep the model loaded once, reuse across calls.
- Runs inside the worker thread, while AppState is THINKING.
"""
from __future__ import annotations


def transcribe(audio: bytes) -> str:
    raise NotImplementedError("Implement with pywhispercpp on the Pi.")
