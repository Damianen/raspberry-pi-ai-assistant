"""Speech-to-text via whisper.cpp (pywhispercpp).

`transcribe(audio)` takes the 16 kHz mono float32 array from `audio_io` and
returns the transcript string. pywhispercpp accepts a raw float32 ndarray
directly, so there's no WAV round-trip.

The model is a lazy module-level singleton: it loads on the first call and is
reused for every utterance afterwards. Loading base.en costs seconds — doing it
per tap would make the assistant feel broken. Runs on the worker thread while
AppState is THINKING.

`pywhispercpp` is imported lazily inside the loader so this module stays
importable on machines without the native whisper.cpp build.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "base.en"

_model = None         # pywhispercpp Model instance, loaded once
_model_name: str | None = None


def _get_model(name: str):
    """Load (once) and return the whisper model, reusing it across calls."""
    global _model, _model_name
    if _model is None or _model_name != name:
        from pywhispercpp.model import Model  # lazy: native lib, slow import
        _model = Model(name, print_realtime=False, print_progress=False)
        _model_name = name
    return _model


def transcribe(audio: np.ndarray, model: str = DEFAULT_MODEL) -> str:
    """Return the transcript of a 16 kHz mono float32 array. Empty in -> empty out."""
    if audio is None or audio.size == 0:
        return ""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    segments = _get_model(model).transcribe(audio)
    return " ".join(seg.text.strip() for seg in segments).strip()
