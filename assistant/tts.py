"""Text-to-speech via Piper.

Interface:
    configure(voice, output_device)   # set once at startup (from config); cheap
    speak(text) -> None               # synthesize + play, blocking

The PiperVoice is a lazy module-level singleton — loaded on the first speak() and
reused for every utterance after (same rule as the whisper model in stt.py).
Loading the ONNX model costs real time; doing it per call would make the assistant
feel broken.

speak() BLOCKS until playback finishes. That blocking is load-bearing: the
pipeline sets CONFIRM/SPEAKING before calling and IDLE after, so the eyes only
dwell on those states for as long as speak() is talking.

Playback goes through audio_io.play(), which resamples the voice's native rate to
the output device's rate and tiles to its channels — so this module never touches
sounddevice. Piper emits float32 mono in [-1, 1], which is exactly what play()
wants, and the rate comes from the voice config (the .onnx.json), never hardcoded.

`piper` is imported lazily inside the loader so this module stays importable on
machines without onnxruntime / a downloaded voice (e.g. the offline test path).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import audio_io

# Voice files live in models/ (gitignored). A voice is the .onnx + matching
# .onnx.json pair, e.g. models/en_US-amy-medium.onnx{,.json}.
MODELS_DIR = Path("models")
_DEFAULT_VOICE = "en_US-amy-medium"

_voice_name = _DEFAULT_VOICE
_output_device: int | str | None = None

_voice = None                      # PiperVoice, loaded once
_load_lock = threading.Lock()      # speak() runs on both the pipeline AND the
                                   # scheduler thread; guard the one-time load.


def configure(voice: str | None, output_device: int | str | None = None) -> None:
    """Record the voice name + output device from config. Does NOT load the model
    (that happens lazily on the first speak), so it's safe to call at startup."""
    global _voice_name, _output_device
    _voice_name = voice or _DEFAULT_VOICE
    _output_device = output_device


def _get_voice():
    """Load (once) and return the PiperVoice, reusing it across calls."""
    global _voice
    if _voice is None:
        with _load_lock:
            if _voice is None:   # double-check: another thread may have loaded it
                from piper import PiperVoice  # lazy: pulls in onnxruntime

                onnx = MODELS_DIR / f"{_voice_name}.onnx"
                conf = MODELS_DIR / f"{_voice_name}.onnx.json"
                if not onnx.exists() or not conf.exists():
                    raise FileNotFoundError(
                        f"Piper voice not found: expected {onnx} and {conf}. "
                        f"Download the {_voice_name} .onnx + .onnx.json into "
                        f"{MODELS_DIR}/."
                    )
                _voice = PiperVoice.load(onnx, config_path=conf)
    return _voice


def speak(text: str) -> None:
    """Synthesize `text` with Piper and play it on the output device. Blocks until
    playback finishes. No-op on empty text."""
    text = text.strip()
    if not text:
        return

    voice = _get_voice()
    # synthesize() yields one AudioChunk per sentence; audio_float_array is float32
    # mono in [-1, 1]. Concatenate so the whole phrase plays as one stream.
    chunks = [chunk.audio_float_array for chunk in voice.synthesize(text)]
    if not chunks:
        return
    samples = np.concatenate(chunks).astype(np.float32, copy=False)

    # Rate from the voice config (the .onnx.json) — NOT a hardcoded 22050.
    audio_io.play(samples, voice.config.sample_rate, device=_output_device)
