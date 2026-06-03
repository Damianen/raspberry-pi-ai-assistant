"""Audio capture + playback.

Slice 2: capture is IMPLEMENTED; `play()` is still a stub (do it in the TTS slice).

`record_until_silence` returns a 1-D numpy float32 array, 16 kHz mono, in [-1, 1]
(PortAudio's native float range — we do NOT peak-normalize; whisper.cpp wants
natural levels, and rescaling would just amplify room noise). Endpointing is a
simple energy threshold + trailing-silence timeout, with a fixed-window fallback
so the slice always completes even if the mic never trips the threshold.

`sounddevice` is imported LAZILY inside the function: importing it loads native
PortAudio, which is absent on dev machines. Keeping the import local means this
module (and the offline command path + tests) load fine without an audio stack.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# --- Fixed capture format (whisper.cpp wants 16 kHz mono float32) ---
SAMPLE_RATE = 16_000
CHANNELS = 1
FRAME_MS = 30                                    # energy-analysis frame length
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 480 samples per frame
_FRAME_SEC = FRAME_MS / 1000.0

# --- TUNE THESE ON THE MIC (single far-field USB mic, expect a noisy room) ---
# RMS is computed on a float32 frame in [-1, 1]. START_RMS must be > SILENCE_RMS:
# that gap is hysteresis, so a frame hovering near the threshold can't flip us
# between "speech" and "silence" every 30 ms.
START_RMS = 0.020            # frame energy that counts as "speech has started"
SILENCE_RMS = 0.010          # frame energy below this counts as silence
TRAILING_SILENCE_SEC = 1.2   # stop once we see this much silence after speech
PREROLL_SEC = 0.3            # audio kept BEFORE onset so word 1 isn't clipped
LISTEN_TIMEOUT_SEC = 6.0     # no speech within this -> fixed-window fallback
FALLBACK_SECONDS = 5.0       # length of that fixed-window fallback


def record_until_silence(
    max_seconds: float = 8.0, device: int | str | None = None
) -> np.ndarray:
    """Record one utterance; return 16 kHz mono float32 in [-1, 1].

    Capture begins when frame energy crosses START_RMS (keeping PREROLL_SEC of
    lead-in) and stops after TRAILING_SILENCE_SEC of trailing silence OR at
    max_seconds, whichever comes first. If no speech starts within
    LISTEN_TIMEOUT_SEC, falls back to a fixed FALLBACK_SECONDS window and says so.

    `device` may be a PortAudio index, a name substring, or None (default device).
    """
    import sounddevice as sd  # lazy: needs native PortAudio (absent on dev boxes)

    device = _resolve_device(device)
    info = sd.query_devices(device, "input")
    label = "default" if device is None else device
    print(f"[audio] input device [{label}]: {info['name']!r} "
          f"(native {info['default_samplerate']:.0f} Hz, capturing at {SAMPLE_RATE} Hz)")

    preroll_frames = max(1, round(PREROLL_SEC / _FRAME_SEC))
    preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
    collected: list[np.ndarray] = []
    started = False
    silence_sec = 0.0
    waited = 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                        device=device, blocksize=FRAME_SAMPLES) as stream:
        while True:
            frame, _overflow = stream.read(FRAME_SAMPLES)
            frame = frame[:, 0]                       # (N, 1) -> (N,) mono
            rms = float(np.sqrt(np.mean(frame * frame)))

            if not started:
                preroll.append(frame)                 # rolling lead-in buffer
                waited += _FRAME_SEC
                if rms >= START_RMS:
                    started = True
                    collected.extend(preroll)         # keep the lead-in + onset
                elif waited >= LISTEN_TIMEOUT_SEC:
                    print(f"[audio] no speech in {LISTEN_TIMEOUT_SEC:.0f}s — "
                          f"falling back to a fixed {FALLBACK_SECONDS:.0f}s window.")
                    collected.extend(preroll)
                    _fill_fixed_window(stream, collected)
                    break
                continue

            collected.append(frame)
            if rms < SILENCE_RMS:
                silence_sec += _FRAME_SEC
                if silence_sec >= TRAILING_SILENCE_SEC:
                    break
            else:
                silence_sec = 0.0
            if len(collected) * _FRAME_SEC >= max_seconds:
                break

    if not collected:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(collected).astype(np.float32, copy=False)
    return np.clip(audio, -1.0, 1.0)


def _fill_fixed_window(stream, collected: list[np.ndarray]) -> None:
    """Top up `collected` to ~FALLBACK_SECONDS total (includes existing preroll)."""
    target = round(FALLBACK_SECONDS / _FRAME_SEC)
    while len(collected) < target:
        frame, _overflow = stream.read(FRAME_SAMPLES)
        collected.append(frame[:, 0])


def _resolve_device(device: int | str | None) -> int | str | None:
    """Normalize config's device value: ''/None -> default; digit-string -> index."""
    if device is None:
        return None
    if isinstance(device, str):
        s = device.strip()
        if not s:
            return None
        return int(s) if s.isdigit() else s
    return device


def play(audio: bytes) -> None:
    raise NotImplementedError("Implement with sounddevice/aplay on the Pi (TTS slice).")
