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
_shared = None                     # SharedState | None — set via configure(); the
                                   # ONLY place TTS writes app state (the live level).

_voice = None                      # PiperVoice, loaded once
_load_lock = threading.Lock()      # speak() runs on both the pipeline AND the
                                   # scheduler thread; guard the one-time load.

# --- Mouth-sync level. We turn each played chunk's RMS into a 0..1 openness and
# stash it in SharedState meta; the eyes read+smooth it (they never compute audio).
# Normalize against a ROLLING PEAK so a quiet voice animates as fully as a loud one
# and the absolute output volume doesn't matter — only the envelope shape does. The
# peak floors at _LEVEL_FLOOR so true silence (room noise) maps to ~0 instead of
# being amplified into a flapping mouth, and decays so the peak tracks the current
# passage rather than the single loudest syllable of the whole story. Tune on the Pi.
_LEVEL_PEAK_DECAY = 0.97           # rolling peak's forgetting factor, per ~100 ms chunk
_LEVEL_FLOOR = 1e-3                # RMS below this counts as silence (mouth closed)


def configure(voice: str | None, output_device: int | str | None = None,
              shared=None) -> None:
    """Record the voice name + output device from config, and the SharedState the
    live mouth-sync level is published into. Does NOT load the model (that happens
    lazily on the first speak), so it's safe to call at startup. `shared` may be
    None (tests / offline command path) — then level reporting is simply skipped."""
    global _voice_name, _output_device, _shared
    _voice_name = voice or _DEFAULT_VOICE
    _output_device = output_device
    _shared = shared


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


def speak(text: str, *, stop_event: "threading.Event | None" = None) -> None:
    """Synthesize `text` with Piper and play it on the output device. Blocks until
    playback finishes. No-op on empty text.

    `stop_event`: forwarded to audio_io.play. When set mid-playback, playback bails
    out (tap-to-interrupt during SPEAKING). The pipeline passes its interrupt event
    only for spoken answers/stories; confirmations and alarm announcements pass
    None so they always finish.

    NOTE: synthesis happens up front — the whole utterance is built before any
    audio plays. A tap DURING synthesis (the silent gap before speech) sets the
    event, so play() aborts on its first chunk and nothing is spoken. That's the
    intended "stop" behaviour, but it's also why a long story has a noticeable gap
    before it starts: sentence-streaming TTS (deferred) is what removes that gap.
    """
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

    # Mouth-sync only on the interruptible (chunked) SPEAKING path: confirmations and
    # alarm announcements pass stop_event=None (one blocking write, no chunks) and
    # show CONFIRM's smile rather than a talking mouth, so they need no level. The
    # finally resets level to 0 so the mouth always closes on a clean end, an
    # interrupt, or a synth/playback error — never freezes mid-flap.
    level_cb = (_make_level_cb()
                if stop_event is not None and _shared is not None else None)
    try:
        # Rate from the voice config (the .onnx.json) — NOT a hardcoded 22050.
        audio_io.play(samples, voice.config.sample_rate, device=_output_device,
                      stop_event=stop_event, level_cb=level_cb)
    finally:
        if _shared is not None:
            _shared.update_meta(level=0.0)


def _make_level_cb():
    """Build a per-utterance chunk->level reporter with its own rolling-peak state.

    Fresh per speak() so one utterance's loudness can't bias the next. Each call
    publishes meta['level'] in [0, 1]; the eyes do the visual smoothing."""
    shared = _shared
    peak = _LEVEL_FLOOR

    def report(chunk: np.ndarray) -> None:
        nonlocal peak
        if chunk.size == 0:
            return
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        # Peak jumps to a loud chunk instantly, forgets slowly; floored so silence
        # can't divide us up to a full-open mouth.
        peak = max(peak * _LEVEL_PEAK_DECAY, rms, _LEVEL_FLOOR)
        shared.update_meta(level=min(1.0, rms / peak))

    return report
