"""Audio capture + playback. IMPLEMENT + TEST ON THE PI.

Interface the pipeline expects:
    record_until_silence(max_seconds=8) -> bytes        # 16k mono PCM/WAV
    play(audio: bytes) -> None                          # blocking playback

Implementation notes:
- Use `sounddevice` (PortAudio). Pick the USB mic + case-speaker device indices
  from config — do NOT rely on defaults; the case has multiple audio endpoints
  and the speaker/HDMI/3.5mm jumper must be set for speakers first.
- Endpointing for v1: simple energy threshold + trailing-silence timeout is fine.
  (webrtcvad later for robustness; full echo cancellation is a LATER milestone.)
- Capture at 16 kHz mono — that's what whisper.cpp wants.
"""
from __future__ import annotations


def record_until_silence(max_seconds: float = 8.0) -> bytes:
    raise NotImplementedError("Implement with sounddevice on the Pi.")


def play(audio: bytes) -> None:
    raise NotImplementedError("Implement with sounddevice/aplay on the Pi.")
